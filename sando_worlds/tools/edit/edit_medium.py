"""dyn_sando_medium, 2026-11-04 (user): save the Gazebo edit (moved / deleted statics), vary every static's height,
add one sphere to each arena. Writes dyn_sando_medium, _static, _trefoil (same statics) into the package src.

  python3 edit_medium.py <installed scenario.yaml> <poseinfo.txt> <pkg dir>
"""
import math
import os
import re
import sys

import numpy as np
import yaml

pkg = sys.argv[3]
sys.path.insert(0, os.path.join(pkg, "tools"))
sys.argv = sys.argv[:1] + sys.argv[1:3]
import from_sando as F  # noqa: E402
from dyn_expert.core.scenario import Scenario  # noqa: E402
from dyn_expert.make_world import world_sdf  # noqa: E402

inst, pose_path = sys.argv[1], sys.argv[2]
scn = yaml.safe_load(open(inst))
rng = np.random.default_rng(3000)

# 1) the user's edit
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

# 2) heights: walls 1.5 / 1.8 / 2.1 m, pillars 3.0 .. 3.9 m (tall, as the user wants them), 0.3 m courses
WALL_H = [1.5, 1.8, 2.1]
PILLAR_H = [3.0, 3.3, 3.6, 3.9]
for s in static:
    pillar = max(s["size"][0], s["size"][1]) < 1.0
    h = float(rng.choice(PILLAR_H if pillar else WALL_H))
    s["size"] = [s["size"][0], s["size"][1], h]
    s["center"] = [s["center"][0], s["center"][1], round(h / 2.0, 4)]
    s["layers"] = int(round(h / 0.3))

# 3) one more sphere in every arena, training motions, GAP to statics and spheres (from_sando rules)
zones = scn["zones"]
arenas = [i for i, z in enumerate(zones) if z["kind"] == "arena"]
sdist = Scenario(dict(static=static)).static_distance
clearance = lambda P: sdist(P) - (F.GAP - 0.05)
names = list(F.MIX_KINDS)
p = np.asarray([F.MIX_KINDS[k] for k in names], float)
p /= p.sum()
dyn = scn["dynamic"]
next_id = max(d["id"] for d in dyn) + 1
added, failed = [], []
sep0 = F.SG.SEP_MARGIN
F.SG.SEP_MARGIN = F.GAP
try:
    for zi in arenas:
        z = zones[zi]
        others = [(F.smooth_track(d["motion"]), d["radius"]) for d in dyn if d.get("zone") == zi]
        box = (z["x0"] + 0.1, z["x1"] - 0.1, -F.HALF_WIDTH, F.HALF_WIDTH)
        zc = round(float(rng.uniform(1.0, 3.0)), 3)
        kind0 = str(rng.choice(names, p=p))
        motion = None
        for kind in [kind0] + (["line"] if kind0 in F.TABLE_KINDS else []):
            for _ in range(3):
                try:
                    if kind in F.TABLE_KINDS:
                        motion, sp = F._table_path(rng, kind, F.R_MIX, box, (0.5, 2.5), 2.0, 1.5, clearance, zc,
                                                   others=others)
                    else:
                        motion, sp = F._smooth_path(rng, kind, F.R_MIX, box, (0.8, 2.0), 1.0, 0.4, clearance, zc,
                                                    others=others)
                    break
                except RuntimeError:
                    motion = None
            if motion is not None:
                break
        if motion is None:
            failed.append(zi)
            continue
        dyn.append(dict(id=next_id, shape="sphere", radius=F.R_MIX, height=0.0, speed=round(float(sp.mean()), 4),
                        speed_range=[round(float(sp.min()), 4), round(float(sp.max()), 4)], kind=kind, vertical=False,
                        zone=zi, motion=motion))
        added.append((zi, next_id, kind))
        next_id += 1
finally:
    F.SG.SEP_MARGIN = sep0

edit = (f"2026-11-04 user edit in Gazebo: deleted {deleted}, moved {moved}; heights varied (walls {WALL_H}, pillars "
        f"{PILLAR_H}); one sphere added per arena {[(a, i, k) for a, i, k in added]}, no room in arenas {failed}")
scn["static"] = static
scn["dynamic"] = dyn
scn["layout"] = dict(scn["layout"], n_static=len(static), n_dynamic=len(dyn))
scn.setdefault("edits", []).append(edit)

# checks
S = Scenario(scn)
T = np.arange(0, 120, 0.1)
P = [d.motion.pos(T) for d in S.dynamic]
ds = sorted((d.id, round(float(S.static_distance(pp).min() - d.radius), 2)) for d, pp in zip(S.dynamic, P))
dd = min(float(np.linalg.norm(P[a] - P[b], axis=1).min() - S.dynamic[a].radius - S.dynamic[b].radius)
         for a in range(len(P)) for b in range(a + 1, len(P)))
rect = lambda s: (s["center"][0] - s["size"][0] / 2, s["center"][1] - s["size"][1] / 2,
                  s["center"][0] + s["size"][0] / 2, s["center"][1] + s["size"][1] / 2)
R = [rect(s) for s in static]
close = [(i, j, round(F._rect_gap(R[i], R[j]), 2)) for i in range(len(R)) for j in range(i + 1, len(R))
         if F._rect_gap(R[i], R[j]) < F.GAP]

for variant in ("dyn_sando_medium", "dyn_sando_medium_static", "dyn_sando_medium_trefoil"):
    if variant == "dyn_sando_medium":
        out = scn
    else:
        out = yaml.safe_load(open(os.path.join(pkg, "scenarios", variant + ".yaml")))
        out["static"] = static
        out.setdefault("edits", []).append(edit.split("; one sphere")[0] + " (statics as dyn_sando_medium)")
    with open(os.path.join(pkg, "scenarios", variant + ".yaml"), "w") as fh:
        yaml.safe_dump(out, fh, sort_keys=False)
    with open(os.path.join(pkg, "worlds", variant + ".sdf"), "w") as fh:
        fh.write(world_sdf(out))

kinds = {}
for d in dyn:
    kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
print(f"deleted {deleted}; moved {len(moved)}; statics {len(static)}")
print("heights: walls", sorted(s["size"][2] for s in static if max(s["size"][:2]) >= 1.0))
print("         pillars", sorted(s["size"][2] for s in static if max(s["size"][:2]) < 1.0))
print(f"spheres {len(dyn)} (+{len(added)}: {added}; no room in arenas {failed}); kinds {kinds}")
print(f"static pairs below {F.GAP} m: {close}")
print("sphere-to-static smallest gaps:", sorted(ds, key=lambda x: x[1])[:4], "| sphere-sphere smallest gap:", round(dd, 2))
