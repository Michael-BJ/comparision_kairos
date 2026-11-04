"""Analysis of sando_probe.py output: lidar returns on the spheres, SANDO's tracking of them, dynamic contacts.

  python3 analyze_probe.py <scenario.yaml> <probe.npz>

1) returns per sphere per scan by distance, and inside / outside the Mid-360's vertical field of view;
2) share of scans in which a SANDO track (/NX01/predicted_trajs, last 0.5 s) lies within 1 m of the true sphere, by
   distance and when the lidar has >= 5 returns on it; track position error, speed estimate, tracks on no sphere;
3) every dynamic contact (true pose, drone radius 0.35 m): the sphere's speed against SANDO's obst_max_vel 0.5, and
   how often the lidar saw it and SANDO tracked it in the 3 s before.
"""
import sys

import numpy as np

from dyn_expert.core.scenario import Scenario

scn = Scenario.load(sys.argv[1])
d = np.load(sys.argv[2])
S, T, O = d["scans"], d["tracks"], d["odom"]
DYN = scn.dynamic
K = len(DYN)
RAD = d["radius"]
t_s = S[:, 0]
N = S[:, 4:4 + K]
DIST = S[:, 4 + K:4 + 2 * K]
ELEV = S[:, 4 + 2 * K:4 + 3 * K]
print(f"{len(S)} lidar scans over {t_s[-1] - t_s[0]:.0f} s sim time, {len(T)} track messages, {K} dynamic spheres")

print("\n1) Lidar returns on a sphere per scan")
print(f"{'distance':>10} {'samples':>8} {'in FOV':>7} {'>=1 ret':>8} {'>=5 ret':>8} {'median (hit)':>13}")
fov = (ELEV >= -7.2 - 5) & (ELEV <= 55.2 + 5)
for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 15), (15, 20)):
    m = (DIST >= lo) & (DIST < hi)
    n = N[m]
    hit = n[n > 0]
    print(f"{lo:>4}-{hi:<3} m {m.sum():>8} {100 * fov[m].mean():>6.0f}% {100 * (n >= 1).mean():>7.0f}% "
          f"{100 * (n >= 5).mean():>7.0f}% {np.median(hit) if len(hit) else 0:>13.0f}")
m = (DIST < 10) & fov
print(f"within 10 m and inside the vertical FOV: {100 * (N[m] >= 1).mean():.0f}% of the scans see the sphere")
m = (DIST < 10) & ~fov
print(f"within 10 m, outside the vertical FOV: {m.sum()} samples, {100 * (N[m] >= 1).mean() if m.sum() else 0:.0f}% seen")

true_at = lambda t: np.array([o.motion.pos(t)[0] for o in DYN])
tt = T[:, 0] if len(T) else np.zeros(0)
tracked = np.zeros_like(N, dtype=bool)
err, vrat = [], []
for i, t in enumerate(t_s):
    sel = (tt > t - 0.5) & (tt <= t + 0.05)
    if not sel.any():
        continue
    C = true_at(t)
    P = T[sel, 2:5]
    dd = np.linalg.norm(C[:, None, :] - P[None, :, :], axis=2)
    j = dd.argmin(axis=1)
    ok = dd[np.arange(K), j] < 1.0
    tracked[i] = ok
    for k in np.flatnonzero(ok & (DIST[i] < 10)):
        err.append(dd[k, j[k]])
        vt = np.linalg.norm(DYN[k].motion.vel(t)[0])
        vk = np.linalg.norm(T[sel][j[k], 5:8])
        if vt > 0.2:
            vrat.append((vk, vt))
print("\n2) SANDO's tracking of the true spheres (track within 1 m in the last 0.5 s)")
for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 15)):
    m = (DIST >= lo) & (DIST < hi)
    mm = m & (N >= 5)
    print(f"{lo:>4}-{hi:<3} m: tracked {100 * tracked[m].mean():>4.0f}% of the scans; "
          f"with >= 5 lidar returns on it: {100 * tracked[mm].mean() if mm.sum() else 0:>4.0f}%")
if err:
    vrat = np.array(vrat)
    print(f"track position error (within 10 m): median {np.median(err):.2f} m; speed estimate median "
          f"{np.median(vrat[:, 0]):.2f} m/s against true {np.median(vrat[:, 1]):.2f} m/s")
if len(T):
    false = sum(np.linalg.norm(true_at(r[0]) - r[2:5], axis=1).min() > 1.5 for r in T)
    print(f"track messages not on any true sphere (> 1.5 m): {false} of {len(T)} ({100 * false / len(T):.0f}%)")

print("\n3) Dynamic contacts (true pose, drone radius 0.35 m)")
to, po = O[:, 0], O[:, 1:4]
step = max(1, len(to) // 20000)
events = []
for i in range(0, len(to), step):
    C = true_at(to[i])
    gap = np.linalg.norm(C - po[i], axis=1) - RAD - 0.35
    for k in np.flatnonzero(gap < 0):
        if not events or events[-1][1] != k or to[i] - events[-1][0] > 1.0:
            events.append([to[i], k])
        else:
            events[-1][0] = to[i]
for t0, k in events:
    pre = (t_s > t0 - 3.0) & (t_s <= t0)
    seen, trk, el = N[pre, k], tracked[pre, k], ELEV[pre, k]
    v = np.linalg.norm(DYN[k].motion.vel(t0)[0])
    print(f"t {t0:7.1f} s  sphere {DYN[k].id:>3}  speed {v:.2f} m/s (SANDO bound 0.5)  last 3 s: lidar saw it in "
          f"{100 * (seen > 0).mean() if len(seen) else 0:.0f}% of scans, tracked in {100 * trk.mean() if len(trk) else 0:.0f}%, "
          f"elevation {np.nanmin(el) if len(el) else 0:.0f}..{np.nanmax(el) if len(el) else 0:.0f} deg")
