"""Add one sphere to each listed arena of a saved dyn_sando world (statics and other spheres unchanged): every MIX
kind, at several altitudes in the training range, GAP to statics and spheres (from_sando rules).

  python3 add_spheres.py <pkg dir> <world name> <arena zone index> [...]
"""
import os
import sys

import numpy as np
import yaml

pkg, name, arena_list = sys.argv[1], sys.argv[2], [int(a) for a in sys.argv[3:]]
sys.path.insert(0, os.path.join(pkg, "tools"))
sys.argv = sys.argv[:1]
import from_sando as F  # noqa: E402
from dyn_expert.core.scenario import Scenario  # noqa: E402
from dyn_expert.make_world import world_sdf  # noqa: E402

path = os.path.join(pkg, "scenarios", name + ".yaml")
scn = yaml.safe_load(open(path))
rng = np.random.default_rng(4000)
zones, dyn = scn["zones"], scn["dynamic"]
sdist = Scenario(dict(static=scn["static"])).static_distance
clearance = lambda P: sdist(P) - (F.GAP - 0.05)
names = list(F.MIX_KINDS)
p = np.asarray([F.MIX_KINDS[k] for k in names], float)
p /= p.sum()
next_id = max(d["id"] for d in dyn) + 1
added, failed = [], []
F.SG.SEP_MARGIN, sep0 = F.GAP, F.SG.SEP_MARGIN
try:
    for zi in arena_list:
        z = zones[zi]
        others = [(F.smooth_track(d["motion"]), d["radius"]) for d in dyn if d.get("zone") == zi]
        alts = [float(np.mean([d["motion"].get("z", d["motion"].get("center", [0, 0, 2])[2]) for d in dyn
                               if d.get("zone") == zi] or [2.0]))]
        # altitudes away from the arena's spheres first
        cand_z = sorted(np.round(np.arange(1.0, 3.01, 0.25), 2), key=lambda zz: -min(abs(zz - a) for a in alts))
        kinds = [str(k) for k in rng.choice(names, size=len(names), replace=False, p=p)]
        box = (z["x0"] + 0.1, z["x1"] - 0.1, -F.HALF_WIDTH, F.HALF_WIDTH)
        motion = None
        for zc in cand_z:
            for kind in kinds:
                try:
                    if kind in F.TABLE_KINDS:
                        motion, sp = F._table_path(rng, kind, F.R_MIX, box, (0.5, 2.5), 2.0, 1.5, clearance, float(zc),
                                                   others=others)
                    else:
                        motion, sp = F._smooth_path(rng, kind, F.R_MIX, box, (0.8, 2.0), 1.0, 0.4, clearance,
                                                    float(zc), others=others)
                    break
                except RuntimeError:
                    motion = None
            if motion is not None:
                break
        if motion is None:
            failed.append(zi)
            print(f"arena {zi}: no path", flush=True)
            continue
        dyn.append(dict(id=next_id, shape="sphere", radius=F.R_MIX, height=0.0, speed=round(float(sp.mean()), 4),
                        speed_range=[round(float(sp.min()), 4), round(float(sp.max()), 4)], kind=kind, vertical=False,
                        zone=zi, motion=motion))
        added.append((zi, next_id, kind, float(zc)))
        print(f"arena {zi}: sphere {next_id} {kind} at z {zc}", flush=True)
        next_id += 1
finally:
    F.SG.SEP_MARGIN = sep0

scn["layout"] = dict(scn["layout"], n_dynamic=len(dyn))
scn["edits"].append(f"2026-11-04: one sphere added to arenas {[a[0] for a in added]} (search over kinds and altitudes "
                    f"1..3 m), no room in {failed}")
with open(path, "w") as fh:
    yaml.safe_dump(scn, fh, sort_keys=False)
with open(os.path.join(pkg, "worlds", name + ".sdf"), "w") as fh:
    fh.write(world_sdf(scn))
S = Scenario(scn)
T = np.arange(0, 120, 0.1)
P = [d.motion.pos(T) for d in S.dynamic]
dd = min(float(np.linalg.norm(P[a] - P[b], axis=1).min() - S.dynamic[a].radius - S.dynamic[b].radius)
         for a in range(len(P)) for b in range(a + 1, len(P)))
ds = min(float(S.static_distance(pp).min() - d.radius) for d, pp in zip(S.dynamic, P) if d.id in [a[1] for a in added]) \
    if added else None
print(f"spheres now {len(dyn)}; added {added}; failed {failed}; sphere-sphere min gap {dd:.2f}; "
      f"new spheres min gap to statics {ds}")
