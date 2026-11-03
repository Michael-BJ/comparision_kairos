#!/usr/bin/env python3
"""Open-loop obstacle prediction: KAIROS head 2 against the predictor of Xu et al., ICRA 2023
("A real-time dynamic obstacle tracking and mapping system for UAV navigation and collision avoidance
with an RGB-D camera", arXiv 2209.08258), on the same sightings and the same GT futures.

    python3 compare_pred.py --set mix                    # test_022 mix worlds (not used in training)
    python3 compare_pred.py --set smooth --out results   # test_022 smooth worlds

Predictors, all from the same sightings of each seen slot (the J = 4 slots head 2 sees):
    head2      KAIROS-Net head 2 (checkpoint --ckpt), CV + learned residual, 3-D
    cv_hist    constant velocity fitted to obs_hist (the CV baseline of kairos/metrics.py)
    ctrv       constant turn rate and speed fitted to obs_frames (kairos/metrics.py, no learning)
    kf_cv      their tracker: Kalman filter [x, y, vx, vy] on obs_frames (onboard_detector
               kalmanFilterMatrixVel / getKalmanObservationVel, detector_param.yaml noise values),
               linear extrapolation as in their released planner (bsplineTraj.h: pos + vel * t)
    markov     their predictor (paper Sec. III-D, Eqs. 10-11): path library of constant-speed arcs,
               Markov chain over the paths updated at every frame with the environment probability
               Softmax(distance to the first static collision), most probable path
    markov_or  oracle: the library path closest to the truth (upper bound of any path selection)

The static map of the environment term is the exact geometry of the scenario (scenario_yaml), not an
onboard map: this favours markov. Errors are counted where k_pred continues the history
(k_pred_valid), as in kairos/metrics.py.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import h5py
import numpy as np
import torch
import yaml

KAIROS_ROOT = os.path.expanduser("~/kairosV3")
sys.path.insert(0, KAIROS_ROOT)

from kairos.data import EventStore, scan_files                   # noqa: E402
from kairos.metrics import cv_prediction                          # noqa: E402
from kairos.model import ctrv_extrapolate, ctrv_state, frames_state, load_checkpoint  # noqa: E402
from kairos.safety import tracker_summary                         # noqa: E402

SETS = {
    "mix": ["~/dynamic_39_world_1nov_exc3/dyn_kairos2_test_022_mix",
            "~/dynamic_39_world_1nov_exc3/dyn_kairos2_fullDyn_test_022_mix",
            "~/dynamic_39_world_1nov_exc3/dyn_kairos2_N_test_022_mix"],
    "smooth": ["~/dynamic_39_world_28oct_smooth/dyn_kairos2_test_022_smooth",
               "~/dynamic_39_world_28oct_smooth/dyn_kairos2_fullDyn_test_022_smooth",
               "~/dynamic_39_world_28oct_smooth/dyn_kairos2_N_test_022_smooth"],
}
KINDS = ["constant", "accel", "stopgo", "turn", "accel_sine", "sinusoid", "trefoil", "static"]
RECENT_AGE = 1.0                      # s, as kairos/metrics.py

# onboard_detector cfg/detector_param.yaml (commit 4b4be99)
KF = dict(e_p=0.25, e_q_pos=0.01, e_q_vel=0.05, e_r_pos=0.04, e_r_vel=0.3, avg_time=0.33)
# avg_time: kalman_filter_averaging_frames 10 at their time_difference 0.033 s; our sightings come at
# ~15 Hz, so the velocity is measured over the same ~0.33 s (5 frames) instead of over 10 frames.
MARKOV = dict(n_paths=5, w_max=0.4, sig_init=1.0, sig_trans=1.0, horizon=4.0, samples=20, env=True)
# n_paths 5: the example of the paper (Fig. 5); the paper gives no values. w_max 0.4 rad/s: the largest
# turn rate of the smooth worlds (scenario edits), i.e. a library that spans our obstacles' turns.


# ----------------------------------------------------------------------------- data
def load_slots(model, cfg, dirs, device):
    """Every seen slot (head 2's mask) of every event: inputs, GT future, predictions of the KAIROS side."""
    infos = scan_files(dirs, cfg.data.min_events, cfg.data.variant)
    store = EventStore(infos, cfg.data, model.M, log=lambda *a: None)
    J = cfg.data.j_max
    kinds, scen = [], {}
    for fi, info in enumerate(store.files):
        with h5py.File(info["path"], "r") as h:
            k = h["events"]["obs_kind"][:]
            scen[fi] = yaml.safe_load(h.attrs["scenario_yaml"])
        sel = store.event_idx[store.file_idx == fi]
        k = k[sel][:, :J]
        if k.shape[1] < J:
            k = np.concatenate([k, np.full((k.shape[0], J - k.shape[1]), -1, k.dtype)], 1)
        kinds.append(k)
    kinds = np.concatenate(kinds, 0)
    out = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(store), 128):
            ids = np.arange(s, min(s + 128, len(store)))
            b = store.batch(ids, device)
            t_k = b["k_pred_t"] - b["foresee_dt"][:, None]                       # s after t0
            k_cv, kmask = cv_prediction(b["obs_hist"], b["obs_visible"], b["obs_mask"], b["foresee_dt"], t_k,
                                        model.obs_enc.track_window)
            _, pred = model.encode_all(*[b[k] for k in model.input_order])
            k_h2, _ = model.predict(pred, t_k)
            st = ctrv_state(b["obs_frames"], b["obs_frames_mask"], getattr(model, "ctrv_window", 0.5),
                            getattr(model, "ctrv_w_max", 1.2))
            fs = frames_state(b["obs_frames"], b["obs_frames_mask"], getattr(model.obs_enc, "frames_window", 0.5))
            tau = b["foresee_dt"][:, None, None] + t_k[:, None, :]
            kf_ = fs["p_last"][:, :, None, :] + fs["vel"][:, :, None, :] * (tau - fs["t_last"][..., None])[..., None]
            k_ct = ctrv_extrapolate(st, tau - st["t_last"][..., None])
            k_ct = torch.where(st["ok"][..., None, None] > 0.5, k_ct, kf_)
            age = -tracker_summary(model, b)["t_last"]
            m = (kmask > 0.5).cpu().numpy()
            ev, sl = np.nonzero(m)
            g = lambda x: x.cpu().numpy()[ev, sl]
            out["event"].append(ids[ev])
            out["file"].append(store.file_idx[ids[ev]])
            out["kind"].append(kinds[ids[ev], sl])
            out["k_pred"].append(g(b["k_pred"]))
            out["valid"].append(g(b["k_pred_valid"]))
            out["radius"].append(g(b["obs_radius"]))
            out["frames"].append(g(b["obs_frames"]))
            out["fmask"].append(g(b["obs_frames_mask"]) > 0.5)
            vis = b["obs_visible"].cpu().numpy()[ev, sl]
            hist = b["obs_hist"].cpu().numpy()[ev, sl]
            out["hist"].append(hist)
            out["hmask"].append(vis > 0.5)
            out["tau"].append(tau.expand(-1, m.shape[1], -1).cpu().numpy()[ev, sl])
            out["age"].append(g(age))
            out["origin"].append(store.a["frame_origin"][ids[ev]])
            out["yaw"].append(store.a["frame_yaw"][ids[ev]])
            for name, x in (("head2", k_h2), ("cv_hist", k_cv), ("ctrv", k_ct)):
                out["pred_" + name].append(g(x))
    d = {k: np.concatenate(v, 0) for k, v in out.items()}
    d["file_names"] = [f["name"] for f in store.files]
    d["t_rel"] = store.a["k_pred_t"][0] - store.a["foresee_dt"][0]
    d["n_events"] = len(store)
    return d, scen


def sightings(d):
    """Per slot the detections in time order, compressed to the front: obs_frames (camera rate), or the
    visible obs_hist samples (0.1 s) where a slot has fewer than 2 frames. -> t (N, F), p (N, F, 3), n (N,)."""
    N = len(d["kind"])
    F = max(d["frames"].shape[1], d["hist"].shape[1])
    T, P, n = np.zeros((N, F)), np.zeros((N, F, 3)), np.zeros(N, int)
    use_hist = d["fmask"].sum(1) < 2
    for i in range(N):
        src, msk = (d["hist"][i], d["hmask"][i]) if use_hist[i] else (d["frames"][i], d["fmask"][i])
        r = src[msk]
        r = r[np.argsort(r[:, 0])]
        n[i] = len(r)
        T[i, :n[i]], P[i, :n[i]] = r[:, 0], r[:, 1:4]
    return T, P, n, use_hist


# ----------------------------------------------------------------------------- their tracker
def kalman_vel(T, P, n, c=KF):
    """onboard_detector kalmanFilterMatrixVel + getKalmanObservationVel, vectorized over slots.
    Returns the state [x, y, vx, vy] after every detection (N, F, 4)."""
    N, F = T.shape
    X = np.zeros((N, F, 4))
    x = np.zeros((N, 4))
    x[:, :2] = P[:, 0, :2]
    Pc = np.tile(np.eye(4) * c["e_p"], (N, 1, 1))
    Q = np.diag([c["e_q_pos"]] * 2 + [c["e_q_vel"]] * 2)
    R = np.diag([c["e_r_pos"]] * 2 + [c["e_r_vel"]] * 2)
    X[:, 0] = x
    dt_med = np.median(np.diff(T, axis=1)[np.arange(F - 1)[None] < (n[:, None] - 1)]) if (n > 1).any() else 0.067
    kavg = max(1, int(round(c["avg_time"] / dt_med)))
    I4 = np.eye(4)
    for f in range(1, F):
        on = f < n
        if not on.any():
            break
        dt = np.where(on, T[:, f] - T[:, f - 1], 0.0)
        A = np.tile(I4, (N, 1, 1))
        A[:, 0, 2] = A[:, 1, 3] = dt
        xp = np.einsum("nij,nj->ni", A, x)
        Pp = A @ Pc @ A.transpose(0, 2, 1) + Q
        k = np.minimum(kavg, f)                                   # history entries before this detection
        j = f - k
        ref, tref = X[np.arange(N), j, :2], T[np.arange(N), j]
        z = np.concatenate([P[:, f, :2], (P[:, f, :2] - ref) / np.maximum(T[:, f] - tref, 1e-3)[:, None]], 1)
        K = Pp @ np.linalg.inv(Pp + R)                            # H = I
        xn = xp + np.einsum("nij,nj->ni", K, z - xp)
        Pn = (I4 - K) @ Pp
        x = np.where(on[:, None], xn, x)
        Pc = np.where(on[:, None, None], Pn, Pc)
        X[:, f] = x
    return X


# ----------------------------------------------------------------------------- static map
def static_geometry(sc):
    boxes, cyls = [], []
    for s in sc.get("static", []):
        if s["type"] == "box":
            boxes.append([*s["center"], *s["size"], s.get("yaw", 0.0)])
        elif s["type"] == "cylinder":
            cyls.append([*s["xy"], s["radius"], s["height"]])
        else:
            raise ValueError(f"static type {s['type']} not handled")
    return np.array(boxes, float).reshape(-1, 7), np.array(cyls, float).reshape(-1, 4)


def static_distance(q, boxes, cyls):
    """q (..., 3) world -> distance to the nearest static surface (...)."""
    d = np.full(q.shape[:-1], np.inf)
    for c in boxes:
        v = q - c[:3]
        cy, sy = np.cos(c[6]), np.sin(c[6])
        lx, ly = cy * v[..., 0] + sy * v[..., 1], -sy * v[..., 0] + cy * v[..., 1]
        e = np.stack([np.abs(lx) - c[3] / 2, np.abs(ly) - c[4] / 2, np.abs(v[..., 2]) - c[5] / 2], -1)
        d = np.minimum(d, np.linalg.norm(np.maximum(e, 0.0), axis=-1))
    for c in cyls:
        rho = np.hypot(q[..., 0] - c[0], q[..., 1] - c[1])
        er, ez = np.maximum(rho - c[2], 0.0), np.maximum(np.maximum(q[..., 2] - c[3], -q[..., 2]), 0.0)
        d = np.minimum(d, np.hypot(er, ez))
    return d


# ----------------------------------------------------------------------------- their predictor
def arcs(p0, theta, speed, w, tau):
    """Constant-speed arcs: p0 (N, 3), theta, speed (N,), w (L,) rad/s, tau (N, Q) -> (N, L, Q, 3)."""
    th = theta[:, None, None]
    wq = w[None, :, None]
    t = tau[:, None, :]
    s = speed[:, None, None]
    small = np.abs(wq) < 1e-6
    wsafe = np.where(small, 1.0, wq)
    dx = np.where(small, s * t * np.cos(th), s / wsafe * (np.sin(th + wsafe * t) - np.sin(th)))
    dy = np.where(small, s * t * np.sin(th), s / wsafe * (np.cos(th) - np.cos(th + wsafe * t)))
    z = np.broadcast_to(p0[:, None, None, 2], dx.shape)
    return np.stack([p0[:, None, None, 0] + dx, p0[:, None, None, 1] + dy, z], -1)


def discrete_gauss(L, center, sig):
    x = np.arange(L)
    g = np.exp(-0.5 * ((x[None] - np.asarray(center)[..., None]) / sig) ** 2)
    return g / g.sum(-1, keepdims=True)


def to_world(p, origin, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    sh = (slice(None),) + (None,) * (p.ndim - 2)
    x = c[sh] * p[..., 0] - s[sh] * p[..., 1] + origin[sh + (0,)]
    y = s[sh] * p[..., 0] + c[sh] * p[..., 1] + origin[sh + (1,)]
    return np.stack([x, y, p[..., 2] + origin[sh + (2,)]], -1)


def env_probabilities(d, T, P, n, X, scen, mk):
    """P_env (N, F, L) at every detection (Eq. 10), the library built from the KF state at that frame.
    Dist_i = path length to the first point closer than the obstacle radius to a static surface."""
    N, F = T.shape
    L = mk["n_paths"]
    w = np.linspace(mk["w_max"], -mk["w_max"], L)                      # left turn ... straight ... right turn
    ts = np.linspace(0.0, mk["horizon"], mk["samples"] + 1)[1:]
    Penv = np.full((N, F, L), 1.0 / L)
    geo = {fi: static_geometry(sc) for fi, sc in scen.items()}
    for fi, (boxes, cyls) in geo.items():
        rows = np.nonzero(d["file"] == fi)[0]
        if not len(rows):
            continue
        for f in range(1, F):
            r = rows[f < n[rows]]
            if not len(r):
                break
            v = X[r, f, 2:4]
            sp = np.linalg.norm(v, axis=1)
            th = np.arctan2(v[:, 1], v[:, 0])
            p0 = np.concatenate([X[r, f, :2], P[r, f, 2:3]], 1)
            pts = arcs(p0, th, sp, w, np.tile(ts, (len(r), 1)))          # (n, L, S, 3) local
            pw = to_world(pts, d["origin"][r], d["yaw"][r])
            hit = static_distance(pw, boxes, cyls) < d["radius"][r][:, None, None]
            first = np.where(hit.any(-1), hit.argmax(-1), len(ts))          # index of the first collision
            tcol = np.concatenate([[0.0], ts])[first]                       # last free sample time
            tcol = np.where(first == len(ts), mk["horizon"], tcol)
            dist = sp[:, None] * tcol
            e = np.exp(dist - dist.max(1, keepdims=True))
            Penv[r, f] = e / e.sum(1, keepdims=True)
    return Penv, w


def markov_chain(n, Penv, mk):
    """Eq. 11: P^{t+1} = (P^t P_trans) * P_env^{t+1}, from P_init at the first velocity estimate."""
    N, F, L = Penv.shape
    Ptr = discrete_gauss(L, np.arange(L), mk["sig_trans"])
    p = np.tile(discrete_gauss(L, L // 2, mk["sig_init"]), (N, 1))
    for f in range(1, F):
        on = f < n
        if not on.any():
            break
        env = Penv[:, f] if mk["env"] else np.full((N, L), 1.0 / L)
        q = (p @ Ptr) * env
        q /= q.sum(1, keepdims=True)
        p = np.where(on[:, None], q, p)
    return p


def their_predictions(d, scen, mk=MARKOV, cache=None):
    T, P, n, use_hist = sightings(d)
    X = kalman_vel(T, P, n)
    last = np.maximum(n - 1, 0)
    xl = X[np.arange(len(n)), last]
    tl = T[np.arange(len(n)), last]
    zl = P[np.arange(len(n)), last, 2]
    tau = d["tau"] - tl[:, None]                                        # s after the last detection
    p0 = np.concatenate([xl[:, :2], zl[:, None]], 1)
    kf = p0[:, None, :] + np.concatenate([xl[:, 2:4], np.zeros((len(n), 1))], 1)[:, None, :] * tau[..., None]
    key = (mk["n_paths"], mk["w_max"], mk["horizon"], mk["samples"])
    if cache is not None and key in cache:
        Penv, w = cache[key]
    else:
        Penv, w = env_probabilities(d, T, P, n, X, scen, mk)
        if cache is not None:
            cache[key] = (Penv, w)
    prob = markov_chain(n, Penv, mk)
    sp = np.linalg.norm(xl[:, 2:4], axis=1)
    th = np.arctan2(xl[:, 3], xl[:, 2])
    lib = arcs(p0, th, sp, w, tau)                                      # (N, L, Q, 3)
    mc = lib[np.arange(len(n)), prob.argmax(1)]
    e_lib = np.linalg.norm(lib - d["k_pred"][:, None], axis=-1)
    v = d["valid"][:, None].astype(float)
    best = ((e_lib * v).sum(-1) / np.maximum(v.sum(-1), 1)).argmin(1)
    return dict(kf_cv=kf, markov=mc, markov_or=lib[np.arange(len(n)), best]), dict(
        use_hist=float(use_hist.mean()), straight_share=float((prob.argmax(1) == mk["n_paths"] // 2).mean()),
        kavg=int(max(1, round(KF["avg_time"] / np.median(np.diff(T, axis=1)[np.arange(T.shape[1] - 1)[None] < (n[:, None] - 1)])))))


# ----------------------------------------------------------------------------- metrics
def errors(d, preds):
    return {k: np.linalg.norm(p - d["k_pred"], axis=-1) for k, p in preds.items()}


def summarize(d, E, sel, t_rel):
    v = d["valid"][sel].astype(float)
    out = {"n_slots": int(sel.sum())}
    if not sel.any() or v.sum() == 0:
        return out
    qs = {f"@{s:.0f}s": int(np.abs(t_rel - s).argmin()) for s in (1.0, 2.0, 4.0)}
    for k, e in E.items():
        e = e[sel]
        r = {"ade": float((e * v).sum() / v.sum())}
        for tag, q in qs.items():
            r[tag] = float((e[:, q] * v[:, q]).sum() / max(v[:, q].sum(), 1))
        out[k] = r
    return out


def bootstrap_diff(d, E, sel, a, b, reps=2000, seed=0):
    """ADE(a) - ADE(b) with a 95 % CI over episodes (slots of one episode are correlated)."""
    v = d["valid"].astype(float)
    files = np.unique(d["file"][sel])
    sa = np.array([(E[a][sel & (d["file"] == f)] * v[sel & (d["file"] == f)]).sum() for f in files])
    sb = np.array([(E[b][sel & (d["file"] == f)] * v[sel & (d["file"] == f)]).sum() for f in files])
    nv = np.array([v[sel & (d["file"] == f)].sum() for f in files])
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(files), size=(reps, len(files)))
    diff = (sa[idx].sum(1) - sb[idx].sum(1)) / np.maximum(nv[idx].sum(1), 1)
    return float((sa.sum() - sb.sum()) / nv.sum()), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=os.path.join(KAIROS_ROOT, "runs/2nov_exc3/B/best.pt"))
    ap.add_argument("--set", choices=sorted(SETS), default="mix")
    ap.add_argument("--data-dir", nargs="+", help="instead of --set")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--sweep", action="store_true", help="also sweep the Markov parameters (best on this set)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, cfg, _ = load_checkpoint(args.ckpt, args.device)
    dirs = args.data_dir or SETS[args.set]
    d, scen = load_slots(model, cfg, dirs, args.device)
    t_rel = d["t_rel"]                                                  # s after t0
    cache = {}
    theirs, info = their_predictions(d, scen, MARKOV, cache)
    preds = {k: d["pred_" + k] for k in ("head2", "cv_hist", "ctrv")}
    preds.update(theirs)
    E = errors(d, preds)

    recent = d["age"] <= RECENT_AGE
    groups = {"all": np.ones(len(recent), bool), "recent": recent}
    for ki, kn in enumerate(KINDS):
        s = d["kind"] == ki
        if s.any():
            groups[kn] = s
            groups[kn + "_recent"] = s & recent
    res = dict(set=args.set if not args.data_dir else "custom", dirs=dirs, ckpt=os.path.abspath(args.ckpt),
               n_events=d["n_events"], n_episodes=len(d["file_names"]), markov=MARKOV, kf=KF, info=info,
               groups={g: summarize(d, E, s, t_rel) for g, s in groups.items()},
               diff_markov_minus_head2={g: bootstrap_diff(d, E, s, "markov", "head2") for g, s in groups.items()
                                        if s.sum() > 0},
               diff_kfcv_minus_head2={g: bootstrap_diff(d, E, s, "kf_cv", "head2") for g, s in groups.items()
                                      if s.sum() > 0})
    if args.sweep:
        sw = []
        for L in (5, 9):
            for wm in (0.2, 0.4, 0.8):
                for st in (0.5, 1.0, 2.0):
                    for env in (True, False):
                        mk = dict(MARKOV, n_paths=L, w_max=wm, sig_trans=st, env=env)
                        p, _ = their_predictions(d, scen, mk, cache)
                        e = errors(d, {"markov": p["markov"], "markov_or": p["markov_or"]})
                        s = summarize(d, e, groups["all"], t_rel)
                        sr = summarize(d, e, groups["recent"], t_rel)
                        sw.append(dict(n_paths=L, w_max=wm, sig_trans=st, env=env, ade=s["markov"]["ade"],
                                       ade_recent=sr["markov"]["ade"], oracle_ade=s["markov_or"]["ade"]))
        res["sweep"] = sorted(sw, key=lambda r: r["ade"])
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"pred_compare_{res['set']}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)

    names = ["head2", "markov", "markov_or", "kf_cv", "ctrv", "cv_hist"]
    print(f"[{res['set']}] {res['n_episodes']} episodes, {res['n_events']} events; info {info}")
    print(f"{'group':18s} {'slots':>6s} " + " ".join(f"{n:>10s}" for n in names) + "   ADE (m); markov - head2 [95% CI]")
    for g, s in res["groups"].items():
        if "head2" not in s:
            continue
        dm = res["diff_markov_minus_head2"][g]
        print(f"{g:18s} {s['n_slots']:6d} " + " ".join(f"{s[n]['ade']:10.3f}" for n in names)
              + f"   {dm[0]:+.3f} [{dm[1]:+.3f}, {dm[2]:+.3f}]")
    if args.sweep:
        print("Markov sweep (best 5 by ADE):")
        for r in res["sweep"][:5]:
            print("  ", r)
    print(f"written {path}")


if __name__ == "__main__":
    main()
