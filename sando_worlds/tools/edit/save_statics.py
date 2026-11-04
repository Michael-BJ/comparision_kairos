"""Save a Gazebo edit of a dyn_sando world's statics (moved / deleted), heights and spheres unchanged.
Writes <name>, <name>_static, <name>_trefoil (same statics) into the package src.

  python3 save_statics.py <saved scenario.yaml (== the loaded world)> <poseinfo.txt> <pkg dir> <name>
"""
import math
import os
import re
import sys

import yaml

from dyn_expert.make_world import world_sdf

base_path, pose_path, pkg, name = sys.argv[1:5]
scn = yaml.safe_load(open(base_path))
txt = open(pose_path).read()
live = {}
for m in re.finditer(r'pose \{\s*name: "(static_\d+)"\s*id: \d+\s*position \{(.*?)\}\s*orientation \{(.*?)\}', txt, re.S):
    f = lambda x, k, d=0.0: float(re.search(k + r": ([-\d.e+]+)", x).group(1)) if re.search(k + r": ([-\d.e+]+)", x) else d
    o = m.group(3)
    q = [f(o, "x"), f(o, "y"), f(o, "z"), f(o, "w", 1.0)]
    live[m.group(1)] = ([f(m.group(2), k) for k in "xyz"],
                        math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
static, deleted, moved = [], [], []
for i, s in enumerate(scn["static"]):
    n = f"static_{i:03d}"
    if n not in live:
        deleted.append(n)
        continue
    p, yaw = live[n]
    if math.dist(p, s["center"]) > 0.01 or abs(yaw - s.get("yaw", 0.0)) > 0.01:
        moved.append(n)
    s["center"] = [round(c, 4) for c in p]
    s["yaw"] = round(yaw, 4)
    static.append(s)
edit = f"2026-11-04 user edit in Gazebo: deleted {deleted}, moved {moved}"
for variant in (name, name + "_static", name + "_trefoil"):
    path = os.path.join(pkg, "scenarios", variant + ".yaml")
    out = scn if variant == name else yaml.safe_load(open(path))
    out["static"] = static
    out["layout"] = dict(out["layout"], n_static=len(static))
    out.setdefault("edits", []).append(edit)
    with open(path, "w") as fh:
        yaml.safe_dump(out, fh, sort_keys=False)
    with open(os.path.join(pkg, "worlds", variant + ".sdf"), "w") as fh:
        fh.write(world_sdf(out))
print(f"deleted {len(deleted)} {deleted}; moved {len(moved)} {moved}; statics now {len(static)}")
