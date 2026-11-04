"""Save the user's Gazebo edit of dyn_sando_easy: scenario rebuilt from the SDF Gazebo loaded, static poses from the
live /world/<w>/pose/info dump. Writes dyn_sando_easy, _static and _trefoil (statics replaced) into the package src.

  python3 save_edit.py <loaded.sdf> <poseinfo.txt> <pkg dir>
"""
import math
import os
import re
import sys

import yaml

from dyn_expert.core.scenario import Scenario
from dyn_expert.make_world import world_sdf

sdf_path, pose_path, pkg = sys.argv[1:4]
sdf = open(sdf_path).read()
fl = lambda s: [float(v) for v in s.split()]

# --- statics from the loaded SDF
statics = {}
for m in re.finditer(r"<model name='(static_\d+)'><static>true</static><pose>([^<]+)</pose>(.*?)</model>", sdf, re.S):
    name, pose, body = m.group(1), fl(m.group(2)), m.group(3)
    sizes = [fl(s) for s in re.findall(r"<box><size>([^<]+)</size>", body)]
    layers = len(sizes)
    size = [sizes[0][0], sizes[0][1], round(sum(s[2] for s in sizes), 4)]
    statics[name] = {"type": "box", "kind": "brick", "center": pose[:3], "size": size, "yaw": pose[5],
                     "layers": layers, "group": f"user_edit_{name}"}

# --- dynamics from the KinematicMover parameters (exactly what Gazebo moves)
dynamic = []
for m in re.finditer(r"<model name='dyn_obs_(\d+)'>(.*?)</model>", sdf, re.S):
    i, body = int(m.group(1)), m.group(2)
    r = float(re.search(r"<sphere><radius>([^<]+)</radius>", body).group(1))
    g = lambda tag: re.search(rf"<{tag}>([^<]*)</{tag}>", body).group(1)
    t = g("motion_type")
    if t == "table":
        motion = {"type": "table", "period": float(g("table_period")), "z": float(g("table_z")),
                  "x": fl(g("table_x")), "y": fl(g("table_y"))}
        kind = "table"
    elif t == "sinusoid":
        motion = {"type": "sinusoid", "center": fl(g("center")), "amp": fl(g("amp")), "omega": fl(g("omega_vec")),
                  "phase": fl(g("phase_vec"))}
        kind = "loop" if min(abs(a) for a in motion["amp"][:2]) > 1e-6 else "line"
    else:
        raise SystemExit(f"dyn_obs_{i}: motion {t} not handled")
    dynamic.append({"id": i, "shape": "sphere", "radius": r, "height": 0.0, "kind": kind, "vertical": False,
                    "motion": motion})

# --- the user's poses
txt = open(pose_path).read()
moved = []
for m in re.finditer(r'pose \{\s*name: "(static_\d+)"\s*id: \d+\s*position \{(.*?)\}\s*orientation \{(.*?)\}', txt, re.S):
    name = m.group(1)
    g = lambda s, k, d=0.0: float(re.search(k + r": ([-\d.e+]+)", s).group(1)) if re.search(k + r": ([-\d.e+]+)", s) else d
    p = [g(m.group(2), k) for k in "xyz"]
    o = m.group(3)
    q = [g(o, "x"), g(o, "y"), g(o, "z"), g(o, "w", 1.0)]
    yaw = math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    s = statics[name]
    if math.dist(p, s["center"]) > 0.01 or abs(yaw - s["yaw"]) > 0.01:
        moved.append((name, [round(c, 2) for c in s["center"]], [round(c, 2) for c in p]))
    s["center"] = [round(c, 4) for c in p]
    s["yaw"] = round(yaw, 4)
static = [statics[f"static_{k:03d}"] for k in range(len(statics))]

base = {
    "seed": 0, "generator": "sando_dyn_obstacles (from_sando.py), edited by the user in Gazebo 2026-11-04",
    "bounds": {"lo": [-2.0, -9.0, 0.0], "hi": [107.0, 9.0, 7.0]}, "spawn": [0.0, 0.0, 0.0],
    "route": {"a": [0.0, 0.0], "b": [105.0, 0.0], "jitter": 0.0, "z": 2.0, "z_band": 1.0},
    "layout": {"case": "easy", "half_width": 6.0, "n_static": len(static), "static_obstacles": True,
               "note": "the from_sando.py easy world of 2026-11-04 20:2x (statics and spheres mixed, gap 1.75 m), "
                       f"{len(moved)} statics moved by the user in Gazebo; dynamic motions as loaded"},
    "edits": [f"user edit in Gazebo 2026-11-04: moved {[n for n, _, _ in moved]}"],
}

# --- checks: static gaps (footprints, yaw 0), sphere clearance to statics and between spheres over 120 s
scn = Scenario(dict(base, static=static, dynamic=dynamic))
rect = lambda s: (s["center"][0] - s["size"][0] / 2, s["center"][1] - s["size"][1] / 2,
                  s["center"][0] + s["size"][0] / 2, s["center"][1] + s["size"][1] / 2)
gapf = lambda a, b: math.hypot(max(a[0] - b[2], b[0] - a[2], 0), max(a[1] - b[3], b[1] - a[3], 0))
R = [rect(s) for s in static]
close = [(i, j, round(gapf(R[i], R[j]), 2)) for i in range(len(R)) for j in range(i + 1, len(R)) if gapf(R[i], R[j]) < 1.75]
import numpy as np  # noqa: E402
T = np.arange(0, 120, 0.1)
P = [d.motion.pos(T) for d in scn.dynamic]
dyn_static = [(d.id, round(float(scn.static_distance(p).min() - d.radius), 2)) for d, p in zip(scn.dynamic, P)]
dyn_dyn = min(float(np.linalg.norm(P[a] - P[b], axis=1).min() - scn.dynamic[a].radius - scn.dynamic[b].radius)
              for a in range(len(P)) for b in range(a + 1, len(P)))

out = {"moved": moved, "static_pairs_below_1.75": close,
       "sphere_to_static_min_gap": sorted(dyn_static, key=lambda x: x[1])[:5], "sphere_sphere_min_gap": round(dyn_dyn, 2)}
print(yaml.safe_dump(out, sort_keys=False))

# --- write
tref_path = os.path.join(pkg, "scenarios", "dyn_sando_easy_trefoil.yaml")
tref = yaml.safe_load(open(tref_path))
for name, scn_d in (("dyn_sando_easy", dict(base, name="dyn_sando_easy", static=static, dynamic=dynamic)),
                    ("dyn_sando_easy_static", dict(base, name="dyn_sando_easy_static", static=static, dynamic=[])),
                    ("dyn_sando_easy_trefoil", dict(base, name="dyn_sando_easy_trefoil", static=static,
                                                    dynamic=tref["dynamic"]))):
    with open(os.path.join(pkg, "scenarios", name + ".yaml"), "w") as fh:
        yaml.safe_dump(scn_d, fh, sort_keys=False)
    with open(os.path.join(pkg, "worlds", name + ".sdf"), "w") as fh:
        fh.write(world_sdf(scn_d))
# the rebuilt (unmoved) world must equal the loaded SDF in its obstacle models
check = world_sdf(dict(base, name="dyn_sando_easy", static=[dict(s) for s in static], dynamic=dynamic))
print("written:", [n for n in ("dyn_sando_easy", "dyn_sando_easy_static", "dyn_sando_easy_trefoil")])
