"""SANDO's own dynamic benchmark worlds (seed 0; easy / medium / hard = 50 / 100 / 200 obstacles, dynamic ratio 0.65)
as SANDO generates and moves them: top and side views and the obstacles' speed statistics.

  python3 sando_layout_plot.py <out.png> [seed]

Draws from from_sando.sando_obstacles (the same draws as sando/launch/dyn_obstacles.launch.py). Motions as SANDO
evaluates its expression strings: x = sx/6 (sin(w t + o) + 2 sin(2 w t + o)) + x0, y = sy/5 (cos(w t + o) -
2 cos(2 w t + o)) + y0, z = sz/2 (-sin(3 w t + o)) + z0, w = 1/slower, o = offset (the same phase o in every
harmonic: "2*t/slower+offset" without parentheses). Statics as SANDO's: pillars 0.4 x 0.4 x 4 m, bars 0.4 x 4 x 0.4 m.
"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from from_sando import sando_obstacles  # noqa: E402


def trefoil(o, t):
    w, ph = 1.0 / o["slower"], o["offset"]
    s = np.array([o["sx"] / 6.0, o["sy"] / 5.0, o["sz"] / 2.0])
    q1, q2, q3 = w * t + ph, 2 * w * t + ph, 3 * w * t + ph
    p = np.stack([np.sin(q1) + 2 * np.sin(q2), np.cos(q1) - 2 * np.cos(q2), -np.sin(q3)], -1)
    v = np.stack([w * (np.cos(q1) + 4 * np.cos(q2)), w * (-np.sin(q1) + 4 * np.sin(q2)), -3 * w * np.cos(q3)], -1)
    return np.array([o["x0"], o["y0"], o["z0"]]) + s * p, s * v


out = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
cases = [("easy", 50), ("medium", 100), ("hard", 200)]
fig, axes = plt.subplots(3, 2, figsize=(18, 11))
tt = np.linspace(0, 2 * np.pi * 6.0, 600)
stats = {}
for row, (name, n) in enumerate(cases):
    obs = sando_obstacles(n, seed)
    axt, axs = axes[row]
    peak = []
    for o in obs:
        if o["dynamic"]:
            p, v = trefoil(o, tt)
            peak.append(np.linalg.norm(v, axis=1).max())
            axt.plot(p[:, 0], p[:, 1], lw=0.6, color="tab:red", alpha=0.45)
            axs.plot(p[:, 0], p[:, 2], lw=0.6, color="tab:red", alpha=0.45)
            p0, _ = trefoil(o, np.array([0.0]))
            axt.add_patch(Rectangle((p0[0, 0] - 0.4, p0[0, 1] - 0.4), 0.8, 0.8, color="tab:red"))
            axs.add_patch(Rectangle((p0[0, 0] - 0.4, p0[0, 2] - 0.4), 0.8, 0.8, color="tab:red"))
        else:
            x, y, z = o["x"], o["y"], o["z"]
            if o["kind"] == "pillar":
                (bx, by, bz), z, col = (0.4, 0.4, 4.0), 2.0, "tab:gray"
            else:
                (bx, by, bz), col = (0.4, 4.0, 0.4), "tab:blue"
            axt.add_patch(Rectangle((x - bx / 2, y - by / 2), bx, by, color=col))
            axs.add_patch(Rectangle((x - bx / 2, z - bz / 2), bx, bz, color=col))
    nd = sum(o["dynamic"] for o in obs)
    nv = sum((not o["dynamic"]) and o["kind"] == "pillar" for o in obs)
    stats[name] = dict(total=n, dynamic=nd, pillars=nv, bars=n - nd - nv,
                       peak_speed_min_med_max=[round(min(peak), 2), round(float(np.median(peak)), 2), round(max(peak), 2)])
    for ax, ylab, ylim in ((axt, "y [m]", (-11, 11)), (axs, "z [m]", (-2, 7.5))):
        ax.plot([0], [0 if ylab == "y [m]" else 2], "g^", ms=12)
        ax.plot([105], [0 if ylab == "y [m]" else 2], "g*", ms=16)
        ax.set_xlim(-3, 110)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axt.set_title(f"{name}: {n} obstacles ({nd} dynamic, {nv} pillars, {n - nd - nv} floating bars) - top view")
    axs.set_title(f"{name} - side view (x-z)")
fig.suptitle(f"SANDO dynamic benchmark worlds (seed {seed}), as SANDO moves them. Red: 0.8 m cubes at t=0 and their "
             "paths; gray: pillars; blue: floating bars; green: start (0,0,2) -> goal (105,0,2)", fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=110)
print(json.dumps(stats, indent=1))
