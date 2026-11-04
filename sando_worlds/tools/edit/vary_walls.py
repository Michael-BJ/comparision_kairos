"""Vary the heights of the brick statics of a dyn_sando world (positions kept), in the training worlds' ranges:
pillars (footprint 0.6 x 0.6) 1.5..3.9 m, walls (4 m long) 1.5..2.1 m, multiples of the 0.3 m brick course.
Applied to <name>, <name>_static and <name>_trefoil (same statics), SDFs rewritten.

  python3 vary_heights.py <pkg dir> <name> [seed]
"""
import math
import os
import sys

import numpy as np
import yaml

from dyn_expert.core.scenario import Scenario
from dyn_expert.make_world import world_sdf

pkg, name = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rng = np.random.default_rng(seed + 2000)
PILLAR_H = np.array([4.0])
WALL_H = np.round(np.arange(1.5, 2.1 + 1e-9, 0.3), 2)

main = yaml.safe_load(open(os.path.join(pkg, "scenarios", name + ".yaml")))
heights = []
for s in main["static"]:
    pillar = max(s["size"][0], s["size"][1]) < 1.0
    h = float(rng.choice(PILLAR_H if pillar else WALL_H))
    heights.append(h)
note = (f"heights varied 2026-11-04 (user): pillars 4.0 m, walls {WALL_H.min()}.."
        f"{WALL_H.max()} m, steps of 0.3 m (training worlds' ranges), rng seed {seed + 2000}")

for variant in (name, name + "_static", name + "_trefoil"):
    p = os.path.join(pkg, "scenarios", variant + ".yaml")
    scn = yaml.safe_load(open(p))
    assert [s["center"][:2] for s in scn["static"]] == [s["center"][:2] for s in main["static"]], variant
    for s, h in zip(scn["static"], heights):
        s["size"] = [s["size"][0], s["size"][1], h]
        s["center"] = [s["center"][0], s["center"][1], round(h / 2.0, 4)]
        s["layers"] = int(round(h / 0.3))
    scn.setdefault("edits", []).append(note)
    with open(p, "w") as fh:
        yaml.safe_dump(scn, fh, sort_keys=False)
    with open(os.path.join(pkg, "worlds", variant + ".sdf"), "w") as fh:
        fh.write(world_sdf(scn))

kinds = ["pillar" if max(s["size"][0], s["size"][1]) < 1.0 else "wall" for s in main["static"]]
for k in ("pillar", "wall"):
    hs = sorted(h for h, kk in zip(heights, kinds) if kk == k)
    print(f"{k}s: {len(hs)}, heights {hs}")
scn = Scenario(yaml.safe_load(open(os.path.join(pkg, "scenarios", name + ".yaml"))))
T = np.arange(0, 120, 0.1)
g = sorted((d.id, round(float(scn.static_distance(d.motion.pos(T)).min() - d.radius), 2)) for d in scn.dynamic)
print("sphere-to-static min gap (3D):", sorted(g, key=lambda x: x[1])[:5])
