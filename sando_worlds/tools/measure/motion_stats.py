"""Obstacle motion statistics: training worlds (split train of runs/2nov_exc3/B) vs SANDO worlds vs our test world."""
import json
import os
import sys
from collections import Counter

import numpy as np

from dyn_expert.core.scenario import Scenario

SC = os.path.expanduser("~/Dynamic_obstacle/install/dyn_gz_worlds/share/dyn_gz_worlds/scenarios")
split = json.load(open(os.path.expanduser("~/kairosV3/runs/2nov_exc3/B/split.json")))
train_worlds = sorted({p.split("/")[0] for p in split["train"]})
T = np.arange(0.0, 120.0, 0.05)


def stats(worlds):
    sp, acc, turn, vz, zr, rad, kinds, peak = [], [], [], [], [], [], Counter(), []
    for w in worlds:
        scn = Scenario.load(os.path.join(SC, w + ".yaml"))
        for d in scn.dynamic:
            m = d.motion
            p = m.pos(T)
            v = np.gradient(p, T, axis=0)
            a = np.gradient(v, T, axis=0)
            ok = np.all(np.abs(np.diff(p, axis=0, prepend=p[:1])) < 1.0, axis=1)    # drop respawn jumps
            ok &= p[:, 2] > -5                                                       # hidden (z -10) phases
            if ok.sum() < 10:
                continue
            v, a, p = v[ok], a[ok], p[ok]
            s = np.linalg.norm(v, axis=1)
            vh = v[:, :2]
            sh = np.linalg.norm(vh, axis=1)
            w_ = np.abs(vh[:, 0] * a[:, 1] - vh[:, 1] * a[:, 0]) / np.maximum(sh, 0.2) ** 2   # horizontal turn rate
            sp.append(s); acc.append(np.linalg.norm(a, axis=1)); turn.append(w_[sh > 0.2]); vz.append(np.abs(v[:, 2]))
            zr.append(p[:, 2].max() - p[:, 2].min()); rad.append(d.radius); peak.append(s.max())
            kinds[getattr(m, "kind", type(m).__name__)] += 1
    c = lambda xs: np.concatenate(xs)
    q = lambda x: f"{np.percentile(x, 50):.2f} / {np.percentile(x, 90):.2f} / {np.percentile(x, 99):.2f}"
    return {
        "obstacles": len(rad), "kinds": dict(kinds.most_common()),
        "radius min/med/max": f"{min(rad):.2f} / {np.median(rad):.2f} / {max(rad):.2f}",
        "speed p50/p90/p99 [m/s]": q(c(sp)), "peak speed per obstacle med/max": f"{np.median(peak):.2f} / {max(peak):.2f}",
        "|acc| p50/p90/p99 [m/s2]": q(c(acc)), "turn rate p50/p90/p99 [rad/s]": q(c(turn)),
        "|vz| p50/p90/p99 [m/s]": q(c(vz)), "share of time |vz|>0.2": f"{100 * (c(vz) > 0.2).mean():.0f}%",
        "z range per obstacle med/max [m]": f"{np.median(zr):.2f} / {max(zr):.2f}",
    }


groups = {"TRAINING (%d worlds)" % len(train_worlds): train_worlds,
          "TEST dyn_long200_smooth_021_clear2": ["dyn_long200_smooth_021_clear2"],
          "SANDO easy+medium+hard_dense": ["dyn_sando_easy", "dyn_sando_medium", "dyn_sando_hard_dense"]}
res = {k: stats(v) for k, v in groups.items()}
keys = list(next(iter(res.values())).keys())
for k in keys:
    print(f"{k:34s}" + "".join(f" | {str(res[g][k]):38s}" for g in res))
print("columns:", " | ".join(res))
