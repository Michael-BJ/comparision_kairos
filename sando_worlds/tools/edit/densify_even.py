"""Spheres added to a zoned dyn_sando world until it holds TOTAL, spread evenly over the arenas (the least occupied
arena first), with the given motion kinds and sphere-sphere gap. Written for dyn_sando_medium (2026-11-06, user:
medium like long200 = dyn_long200_smooth_021_clear2: 0.30 spheres per m of route, smooth `loop` / `line` motions
only, spheres down to 0.3 m apart; radius 0.45 m as the other spheres of these worlds).

  python3 densify_even.py <pkg dir> <world> <TOTAL> [kinds, default loop,line] [SEP m, default 0.3] [out name]

Rules as densify.py / from_sando.py otherwise: altitude drawn from 1..3 m in 0.25 m steps, the training generator's
smooth-path limits (peak speed 0.8..2.0 m/s, |a| <= 1, turn <= 0.4 rad/s), GAP = 1.75 m to the statics, SEP to every
other sphere at every time. The world is written in place unless an out name is given.
"""
import os
import sys

import numpy as np
import yaml

pkg, src, total = sys.argv[1], sys.argv[2], int(sys.argv[3])
kinds = (sys.argv[4] if len(sys.argv) > 4 else "loop,line").split(",")
SEP = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
dst = sys.argv[6] if len(sys.argv) > 6 else src
sys.path.insert(0, os.path.join(pkg, "tools"))
sys.argv = sys.argv[:1]
import from_sando as F  # noqa: E402
from dyn_expert.core.scenario import Scenario  # noqa: E402
from dyn_expert.make_world import world_sdf  # noqa: E402

scn = yaml.safe_load(open(os.path.join(pkg, "scenarios", src + ".yaml")))
scn["name"] = dst
zones, dyn = scn["zones"], scn["dynamic"]
rng = np.random.default_rng(7000)
p = np.asarray([F.MIX_KINDS[k] for k in kinds], float)
p /= p.sum()                                          # training shares among the allowed kinds
sdist = Scenario(dict(static=scn["static"])).static_distance
clearance = lambda P: sdist(P) - (F.GAP - 0.05)
arenas = [i for i, z in enumerate(zones) if z["kind"] == "arena"]
full = set()
next_id = max(d["id"] for d in dyn) + 1
added = []
sep0, F.SG.SEP_MARGIN = F.SG.SEP_MARGIN, SEP
try:
    while len(dyn) < total and len(full) < len(arenas):
        count = {zi: sum(d.get("zone") == zi for d in dyn) for zi in arenas if zi not in full}
        zi = min(count, key=lambda k: (count[k], k))
        z = zones[zi]
        box = (z["x0"] + 0.1, z["x1"] - 0.1, -F.HALF_WIDTH, F.HALF_WIDTH)
        others = [(F.smooth_track(d["motion"]), d["radius"]) for d in dyn if d.get("zone") == zi]
        motion = None
        for _ in range(40):
            kind = str(rng.choice(kinds, p=p))
            zc = float(rng.choice(np.round(np.arange(1.0, 3.01, 0.25), 2)))
            try:
                motion, sp = F._smooth_path(rng, kind, F.R_MIX, box, (0.8, 2.0), 1.0, 0.4, clearance, zc, others=others)
                break
            except RuntimeError:
                motion = None
        if motion is None:
            full.add(zi)
            print(f"arena {zi}: no room after {count[zi]} spheres", flush=True)
            continue
        dyn.append(dict(id=next_id, shape="sphere", radius=F.R_MIX, height=0.0, speed=round(float(sp.mean()), 4),
                        speed_range=[round(float(sp.min()), 4), round(float(sp.max()), 4)], kind=kind,
                        vertical=False, zone=zi, motion=motion))
        added.append((zi, next_id, kind, zc))
        print(f"arena {zi}: sphere {next_id} {kind} at z {zc}", flush=True)
        next_id += 1
finally:
    F.SG.SEP_MARGIN = sep0

course = float(np.hypot(*(np.asarray(scn["route"]["b"]) - np.asarray(scn["route"]["a"]))))
scn["layout"] = dict(scn["layout"], n_dynamic=len(dyn))
scn["edits"] = list(scn.get("edits", [])) + [
    f"2026-11-06 (user: like long200): densify_even.py, {len(added)} spheres added ({len(dyn)} in all, "
    f"{len(dyn) / course:.2f} per m), kinds {kinds}, sphere-sphere gap >= {SEP} m for the new spheres, "
    f">= {F.GAP} m to the statics"]
S = Scenario(scn)
T = np.arange(0, 300, 0.05)
P = {d.id: d.motion.pos(T) for d in S.dynamic}
R = {d.id: d.radius for d in S.dynamic}
new = {a[1] for a in added}
ids = list(P)
dd = min(float(np.linalg.norm(P[a] - P[b], axis=1).min() - R[a] - R[b])
         for i, a in enumerate(ids) for b in ids[i + 1:] if a in new or b in new) if added else float("nan")
ds = min(float(S.static_distance(P[i]).min() - R[i]) for i in new) if added else float("nan")
with open(os.path.join(pkg, "scenarios", dst + ".yaml"), "w") as fh:
    yaml.safe_dump(scn, fh, sort_keys=False)
with open(os.path.join(pkg, "worlds", dst + ".sdf"), "w") as fh:
    fh.write(world_sdf(scn))
per = {zi: sum(d.get("zone") == zi for d in dyn) for zi in arenas}
print(f"{dst}: spheres {len(dyn)} (added {len(added)}, {len(dyn) / course:.2f} per m), per arena {per}; "
      f"new spheres over 300 s: min gap to spheres {dd:.2f} m, to statics {ds:.2f} m (rules {SEP} / {F.GAP})")
