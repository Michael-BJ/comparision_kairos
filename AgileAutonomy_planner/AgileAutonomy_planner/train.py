#!/usr/bin/env python3
"""Agile Autonomy network trainer on the KAIROS dataset, with the training configuration of a KAIROS run.

Network and losses are Agile Autonomy's (aa_net.PlaNet, aa_loss: MixtureSpaceLoss + TrajectoryCostLoss, summed
as in PlanLearner.train_step); the image is the KAIROS rig panorama 128 x 400.

Everything else is taken from the KAIROS run given by --kairos_run (default runs/2nov_exc3/B), as for NEO
(../NEO_planner/NEO_planner/train.py):
  * files: its split.json (train / val episodes) and config.yaml data.dirs
  * optimizer: AdamW (train.lr, train.weight_decay), per-step cosine schedule with train.warmup_epochs of linear
    warm-up down to train.lr_min_ratio x lr, gradient clipping train.grad_clip (Agile Autonomy: Adam, cosine
    decay with restarts, each gradient clipped to norm 1)
  * train.epochs, train.batch_size, a new random order every epoch, train.seed (Agile Autonomy: 150 epochs, 8)
  * augmentation of kairos.data.augment_batch: lateral mirror with probability 0.5 (image, state, labels and the
    world geometry), depth noise / holes / range cut in meters (kairos.data.augment_depth), the start velocity
    + N(0, data.state_noise); Agile Autonomy's depth preprocessing after the augmentation, as at deployment
  * validation every train.eval_every epochs and at the end; best.pth = lowest validation loss (space + cost,
    Agile Autonomy's checkpoint criterion)
The collision cost uses the drone radius of the test config (planner.drone_radius), threshold radius + 0.5.

    python3 train.py --out_dir ~/saved_net/aa_kairos
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aa_data as ad
from aa_loss import mixture_space_loss, trajectory_cost_loss
from aa_net import PlaNet, depth_input

GEO_KEYS = ("boxes", "cyls", "spheres", "radii")


def lr_lambda(total, warmup, min_ratio):
    """kairos train.lr_lambda: linear warm-up, then cosine down to min_ratio."""
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))
    return f


def make_batch(data, idx, device, dcfg=None, augment_depth=None):
    """-> depth3 (B, 3, H, W), state (B, 15), label (B, 3, 30), geometry tuple, all on device."""
    t = lambda k: torch.as_tensor(data[k][idx], device=device).clone()
    d = torch.as_tensor(data["depth"][idx].astype(np.float32), device=device) * 1e-3
    s, y = t("state"), t("label")
    geo = [t(k) for k in GEO_KEYS]
    if dcfg is not None:
        B = d.shape[0]
        if dcfg.mirror:
            f = torch.rand(B, device=device) < 0.5
            d = torch.where(f[:, None, None], d.flip(-1), d)
            ss = torch.ones(s.shape[1], device=device)
            ss[ad.MIRROR_STATE_NEG] = -1.0
            sy = torch.ones(y.shape[2], device=device)
            sy[ad.MIRROR_LABEL_NEG] = -1.0
            s = torch.where(f[:, None], s * ss, s)
            y = torch.where(f[:, None, None], y * sy, y)
            boxes, cyls, sph, rad = geo
            mb = torch.ones(7, device=device)
            mb[[1, 6]] = -1.0
            mc = torch.ones(5, device=device)
            mc[1] = -1.0
            ms = torch.tensor([1.0, -1.0, 1.0], device=device)
            geo = [torch.where(f[:, None, None], boxes * mb, boxes), torch.where(f[:, None, None], cyls * mc, cyls),
                   torch.where(f[:, None, None, None], sph * ms, sph), rad]
        if dcfg.depth_aug.enable:
            d = augment_depth(d, dcfg.depth_aug)
        if dcfg.state_noise > 0:
            s[:, ad.STATE_VEL] += torch.randn(B, 3, device=device) * dcfg.state_noise
    return depth_input(d), s, y, tuple(geo)


def load(paths, cache):
    if cache and os.path.isfile(cache):
        z = np.load(cache)
        data = {k: z[k] for k in z.files}
        print(f"[train] cache {cache}: {len(data['depth'])} samples", flush=True)
        return data
    data = ad.load_files(paths)
    if cache:
        np.savez(cache, **data)
    return data


def evaluate(net, data, device, size, thr, batch=128):
    net.eval()
    tot_s, tot_c, n = 0.0, 0.0, 0
    with torch.no_grad():
        for s in range(0, len(data["depth"]), batch):
            idx = np.arange(s, min(s + batch, len(data["depth"])))
            x, st, y, geo = make_batch(data, idx, device)
            out = net(x, st)
            tot_s += mixture_space_loss(y, out).item() * len(idx)
            tot_c += trajectory_cost_loss(out, geo, size, thr).item() * len(idx)
            n += len(idx)
    return tot_s / max(1, n), tot_c / max(1, n)


def main():
    ap = argparse.ArgumentParser(description="Agile Autonomy trainer, KAIROS dataset and training configuration")
    ap.add_argument("--kairos_run", default="~/kairosV3/runs/2nov_exc3/B",
                    help="KAIROS run: split.json, data.dirs, train.* and data augmentation are taken from it")
    ap.add_argument("--kairos_root", default="~/kairosV3", help="folder of the kairos package")
    ap.add_argument("--data_root", default="", help="folder of the episode dirs ('' = parent of the run's first dir)")
    ap.add_argument("--test_config", default="~/kairos_test/20261102_compare/config/expert_m4.yaml",
                    help="dyn_expert config of the test: planner.drone_radius sets the collision cost size")
    ap.add_argument("--out_dir", default="~/saved_net/aa_kairos")
    args = ap.parse_args()

    import yaml
    run = os.path.expanduser(args.kairos_run)
    sys.path.insert(0, os.path.expanduser(args.kairos_root))
    from kairos.config import load_config
    from kairos.data import augment_depth
    kc = load_config(os.path.join(run, "config.yaml"))
    tc, dcfg = kc.train, kc.data
    with open(os.path.expanduser(args.test_config)) as fh:
        size = float(yaml.safe_load(fh)["planner"]["drone_radius"])
    thr = size + 0.5
    torch.manual_seed(tc.seed)
    rng = np.random.default_rng(tc.seed)
    split, dirs = ad.load_split(run, os.path.expanduser(args.data_root) or None)
    out = os.path.join(os.path.expanduser(args.out_dir), f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    cache_dir = os.path.expanduser(args.out_dir)
    tag = os.path.basename(run.rstrip("/"))
    tr = load(split["train"], os.path.join(cache_dir, f"cache_{tag}_train.npz"))
    va = load(split["val"], os.path.join(cache_dir, f"cache_{tag}_val.npz"))
    assert tr["depth"].shape[1:] == (ad.PANO_H, ad.PANO_W) and tr["state"].shape[1] == ad.STATE_SIZE
    setup = dict(kairos_run=run, epochs=tc.epochs, batch_size=tc.batch_size, optimizer="AdamW", lr=tc.lr,
                 weight_decay=tc.weight_decay, warmup_epochs=tc.warmup_epochs, lr_min_ratio=tc.lr_min_ratio,
                 grad_clip=tc.grad_clip, eval_every=tc.eval_every, seed=tc.seed, mirror=dcfg.mirror,
                 state_noise=dcfg.state_noise, depth_aug=vars(dcfg.depth_aug) if hasattr(dcfg.depth_aug, "__dict__")
                 else dict(dcfg.depth_aug), loss="MixtureSpaceLoss + TrajectoryCostLoss",
                 cost_size=size, cost_threshold=thr, label_dt=ad.LABEL_DT, out_seq_len=ad.OUT_SEQ_LEN,
                 modes=ad.TOP_TRAJ, split_files=dict(train=len(split["train"]), val=len(split["val"])),
                 samples=dict(train=len(tr["depth"]), val=len(va["depth"])), data_dirs=dirs)
    with open(os.path.join(out, "train_config.json"), "w") as fh:
        json.dump(setup, fh, indent=1, default=str)
    print(f"Train/val: {len(tr['depth'])}/{len(va['depth'])} samples ({len(split['train'])}/{len(split['val'])} "
          f"files of {run}/split.json)\nKAIROS training configuration: " +
          ", ".join(f"{k} {setup[k]}" for k in ("epochs", "batch_size", "optimizer", "lr", "weight_decay",
                                               "warmup_epochs", "lr_min_ratio", "grad_clip", "eval_every", "seed",
                                               "mirror", "state_noise", "depth_aug", "cost_size")), flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    net = PlaNet(pretrained=True).to(device)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
    B = tc.batch_size
    N = len(tr["depth"])
    steps = math.ceil(N / B)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps * tc.epochs, steps * tc.warmup_epochs,
                                                             tc.lr_min_ratio))
    hist, best, best_ep = [], np.inf, 0
    t0 = time.time()
    for ep in range(1, tc.epochs + 1):
        te = time.time()
        net.train()
        perm = rng.permutation(N)
        run_s = run_c = 0.0
        for s in range(0, N, B):
            x, st, y, geo = make_batch(tr, perm[s:s + B], device, dcfg, augment_depth)
            outp = net(x, st)
            ls = mixture_space_loss(y, outp)
            lc = trajectory_cost_loss(outp, geo, size, thr)
            opt.zero_grad()
            (ls + lc).backward()
            if tc.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
            opt.step()
            sched.step()
            run_s += ls.item()
            run_c += lc.item()
        row = [ep, sched.get_last_lr()[0], run_s / steps, run_c / steps, np.nan, np.nan]
        msg = (f"Epoch {ep} space {row[2]:.4f} cost {row[3]:.4f} | lr {row[1]:.2e} | {time.time() - te:.1f}s")
        if ep % tc.eval_every == 0 or ep == tc.epochs:
            vs, vc = evaluate(net, va, device, size, thr)
            row[4], row[5] = vs, vc
            msg += f" | val space {vs:.4f} cost {vc:.4f}"
            if vs + vc < best:
                best, best_ep = vs + vc, ep
                torch.save(net.state_dict(), os.path.join(out, "best.pth"))
                msg += "  (best)"
        hist.append(row)
        np.savetxt(os.path.join(out, "history.csv"), np.array(hist), delimiter=",",
                   header="epoch,lr,train_space,train_cost,val_space,val_cost", comments="")
        torch.save(net.state_dict(), os.path.join(out, "planner_net.pth"))
        print(msg, flush=True)
    m, s = divmod(time.time() - t0, 60)
    print(f"Training time {int(m // 60)}:{int(m % 60):02d}:{int(s):02d}; best val {best:.4f} at epoch {best_ep}; "
          f"saved in {out}", flush=True)


if __name__ == "__main__":
    main()
