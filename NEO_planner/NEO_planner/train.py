#!/usr/bin/env python3
"""NEO-Planner NN trainer on the KAIROS dataset, with the training configuration of a KAIROS run.

Network and loss are NEO's (~/Training_model_FM_planner/nn_trainer.py, the port of the original
github.com/Amos-Chen98/neo-planner trainer): PlannerNet = ResNet18 (pretrained, frozen, conv1 replaced by a
1-channel conv, fc -> 24) + motion MLP 24 -> 24 + fusion MLP -> 9, LeakyReLU, MSELoss(mean); input = depth_u8 as
float32 (0..255, no /255) concatenated with the 24 motion values. The image is the KAIROS rig panorama 128 x 400.

Everything else is taken from the KAIROS run given by --kairos_run (default runs/2nov_exc3/B), so that NEO and
KAIROS see the same data the same way:
  * files: its split.json (train / val episodes) and config.yaml data.dirs
  * optimizer: AdamW (train.lr, train.weight_decay), per-step cosine schedule with train.warmup_epochs of linear
    warm-up down to train.lr_min_ratio x lr (kairos train.lr_lambda), gradient clipping train.grad_clip
  * train.epochs, train.batch_size, a new random order of the train samples every epoch, train.seed
  * augmentation of kairos.data.augment_batch, in its order: lateral mirror with probability 0.5 (data.mirror),
    depth noise / holes / range cut in meters (data.depth_aug, kairos.data.augment_depth itself), then the
    drone velocity + N(0, data.state_noise) (it enters NEO's motion vector three times: drone_local_vel, and
    subtracted in init_vel_body and tgt_vel_body). NEO's depth normalization comes after the augmentation, as at
    deployment. data.obs_aug has nothing to act on: NEO has no obstacle inputs.
  * validation every train.eval_every epochs and at the end; best.pth = the lowest validation MSE (KAIROS picks
    its best checkpoint by its own validation score)

    python3 train.py --out_dir ~/saved_net/neo_kairos
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import neo_data as nd
from planner_net import IMG_HEIGHT, IMG_WIDTH, MOTION_INPUT_SIZE, PlannerNet


def lr_lambda(total, warmup, min_ratio):
    """kairos train.lr_lambda: linear warm-up, then cosine down to min_ratio."""
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))
    return f


def depth_u8(d):
    """neo_data.depth_to_u8 on a (B, H, W) batch in meters: d / max valid depth * 255, truncated to 0..255."""
    mx = torch.where(d > 0, d, torch.zeros_like(d)).flatten(1).max(1).values[:, None, None]
    return torch.where(mx > 0, (d / (mx + 1e-6) * 255).clamp(0, 255).floor(), torch.zeros_like(d))


def make_batch(depth_mm, motion, label, idx, device, dcfg=None, augment_depth=None):
    d = torch.as_tensor(depth_mm[idx].astype(np.float32), device=device) * 1e-3
    m = torch.as_tensor(motion[idx], device=device).clone()
    y = torch.as_tensor(label[idx], device=device).clone()
    if dcfg is not None:
        B = d.shape[0]
        if dcfg.mirror:
            f = torch.rand(B, device=device) < 0.5
            d = torch.where(f[:, None, None], d.flip(-1), d)
            sm = torch.ones(MOTION_INPUT_SIZE, device=device)
            sm[nd.MIRROR_MOTION_NEG] = -1.0
            so = torch.ones(y.shape[1], device=device)
            so[nd.MIRROR_OUTPUT_NEG] = -1.0
            m = torch.where(f[:, None], m * sm, m)
            y = torch.where(f[:, None], y * so, y)
        if dcfg.depth_aug.enable:
            d = augment_depth(d, dcfg.depth_aug)
        if dcfg.state_noise > 0:
            n = torch.randn(B, 3, device=device) * dcfg.state_noise
            m[:, nd.MOTION_VEL] += n
            m[:, nd.MOTION_INIT_VEL] -= n
            m[:, nd.MOTION_TGT_VEL] -= n
    x = torch.cat([depth_u8(d).flatten(1), m], 1)
    return x, y


def load(paths, cache):
    if cache and os.path.isfile(cache):
        z = np.load(cache)
        print(f"[train] cache {cache}: {len(z['depth'])} samples", flush=True)
        return z["depth"], z["motion"], z["label"]
    depth, motion, label = nd.load_files(paths)
    if cache:
        np.savez(cache, depth=depth, motion=motion, label=label)
    return depth, motion, label


def evaluate(net, data, crit, device, batch=256):
    net.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(data[0]), batch):
            idx = np.arange(s, min(s + batch, len(data[0])))
            x, y = make_batch(*data, idx, device)
            total += crit(net(x), y).item() * len(idx)
            n += len(idx)
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser(description="NEO-Planner NN trainer, KAIROS dataset and training configuration")
    ap.add_argument("--kairos_run", default="~/kairosV3/runs/2nov_exc3/B",
                    help="KAIROS run: split.json, data.dirs, train.* and data augmentation are taken from it")
    ap.add_argument("--kairos_root", default="~/kairosV3", help="folder of the kairos package")
    ap.add_argument("--data_root", default="", help="folder of the episode dirs ('' = parent of the run's first dir)")
    ap.add_argument("--out_dir", default="~/saved_net/neo_kairos")
    args = ap.parse_args()

    run = os.path.expanduser(args.kairos_run)
    sys.path.insert(0, os.path.expanduser(args.kairos_root))
    from kairos.config import load_config
    from kairos.data import augment_depth
    kc = load_config(os.path.join(run, "config.yaml"))
    tc, dcfg = kc.train, kc.data
    torch.manual_seed(tc.seed)
    rng = np.random.default_rng(tc.seed)
    split, dirs = nd.load_split(run, os.path.expanduser(args.data_root) or None)
    out = os.path.join(os.path.expanduser(args.out_dir), f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    cache_dir = os.path.expanduser(args.out_dir)
    tag = os.path.basename(run.rstrip("/"))
    tr = load(split["train"], os.path.join(cache_dir, f"cache_{tag}_train_mm.npz"))
    va = load(split["val"], os.path.join(cache_dir, f"cache_{tag}_val_mm.npz"))
    assert tr[0].shape[1:] == (IMG_HEIGHT, IMG_WIDTH) and tr[1].shape[1] == MOTION_INPUT_SIZE
    setup = dict(kairos_run=run, epochs=tc.epochs, batch_size=tc.batch_size, optimizer="AdamW", lr=tc.lr,
                 weight_decay=tc.weight_decay, warmup_epochs=tc.warmup_epochs, lr_min_ratio=tc.lr_min_ratio,
                 grad_clip=tc.grad_clip, eval_every=tc.eval_every, seed=tc.seed, mirror=dcfg.mirror,
                 state_noise=dcfg.state_noise, depth_aug=vars(dcfg.depth_aug) if hasattr(dcfg.depth_aug, "__dict__")
                 else dict(dcfg.depth_aug), loss="MSELoss(mean)",
                 split_files=dict(train=len(split["train"]), val=len(split["val"])),
                 samples=dict(train=len(tr[0]), val=len(va[0])), data_dirs=dirs)
    with open(os.path.join(out, "train_config.json"), "w") as fh:
        json.dump(setup, fh, indent=1, default=str)
    print(f"Train/val: {len(tr[0])}/{len(va[0])} samples ({len(split['train'])}/{len(split['val'])} files of "
          f"{run}/split.json)\nKAIROS training configuration: " +
          ", ".join(f"{k} {setup[k]}" for k in ("epochs", "batch_size", "optimizer", "lr", "weight_decay",
                                               "warmup_epochs", "lr_min_ratio", "grad_clip", "eval_every", "seed",
                                               "mirror", "state_noise", "depth_aug")), flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    net = PlannerNet(pretrained=True).to(device)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
    B = tc.batch_size
    steps = math.ceil(len(tr[0]) / B)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps * tc.epochs, steps * tc.warmup_epochs,
                                                             tc.lr_min_ratio))
    crit = nn.MSELoss(reduction="mean")
    hist, best, best_ep = [], np.inf, 0
    t0 = time.time()
    for ep in range(1, tc.epochs + 1):
        te = time.time()
        net.train()
        perm = rng.permutation(len(tr[0]))
        run_loss = 0.0
        for s in range(0, len(perm), B):
            x, y = make_batch(*tr, perm[s:s + B], device, dcfg, augment_depth)
            opt.zero_grad()
            loss = crit(net(x), y)
            loss.backward()
            if tc.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
            opt.step()
            sched.step()
            run_loss += loss.item()
        tl = run_loss / steps
        row = [ep, sched.get_last_lr()[0], tl, np.nan]
        msg = f"Epoch {ep} loss: {tl:.4f} | lr {row[1]:.2e} | {time.time() - te:.1f}s"
        if ep % tc.eval_every == 0 or ep == tc.epochs:
            vl = evaluate(net, va, crit, device)
            row[3] = vl
            msg += f" | val: {vl:.4f}"
            if vl < best:
                best, best_ep = vl, ep
                torch.save(net.state_dict(), os.path.join(out, "best.pth"))
                msg += "  (best)"
        hist.append(row)
        np.savetxt(os.path.join(out, "history.csv"), np.array(hist), delimiter=",", header="epoch,lr,train,val",
                   comments="")
        torch.save(net.state_dict(), os.path.join(out, "planner_net.pth"))
        print(msg, flush=True)
    m, s = divmod(time.time() - t0, 60)
    print(f"Training time {int(m // 60)}:{int(m % 60):02d}:{int(s):02d}; best val {best:.4f} at epoch {best_ep}; "
          f"saved in {out}", flush=True)


if __name__ == "__main__":
    main()
