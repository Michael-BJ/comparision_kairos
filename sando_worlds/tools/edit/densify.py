"""A denser copy of a saved dyn_sando world: the same statics and spheres, plus spheres added to every arena until
it holds TARGET (from_sando rules: every MIX kind in the training shares, altitudes 1..3 m, GAP = 1.75 m to the
statics, SEP between the new sphere and every other sphere, at every time). Written for dyn_sando_hard ->
dyn_sando_hard_dense (2026-11-05: the arenas of dyn_long200_smooth_021_clear2 hold 6.1 spheres on average, hard
3.9). With SEP = GAP (1.75 m) hard took only 5 more (44); the user chose SEP 0.3 m, the training generator's
sphere-sphere margin (scenario_gen.SEP_MARGIN, as in long200), and kept GAP to the statics.

  python3 densify.py <pkg dir> <source world> <new world> <target per arena> [tries per sphere] [SEP m]
"""
import copy
import os
import sys

import numpy as np
import yaml

pkg, src, dst, target = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tries = int(sys.argv[5]) if len(sys.argv) > 5 else 40
sep_arg = float(sys.argv[6]) if len(sys.argv) > 6 else None
sys.path.insert(0, os.path.join(pkg, "tools"))
sys.argv = sys.argv[:1]
import from_sando as F  # noqa: E402
from dyn_expert.core.scenario import Scenario  # noqa: E402
from dyn_expert.make_world import world_sdf  # noqa: E402

scn = yaml.safe_load(open(os.path.join(pkg, "scenarios", src + ".yaml")))
scn = copy.deepcopy(scn)
scn["name"] = dst
rng = np.random.default_rng(5000)
zones, dyn = scn["zones"], scn["dynamic"]
sdist = Scenario(dict(static=scn["static"])).static_distance
clearance = lambda P: sdist(P) - (F.GAP - 0.05)
names = list(F.MIX_KINDS)
p = np.asarray([F.MIX_KINDS[k] for k in names], float)
p /= p.sum()
next_id = max(d["id"] for d in dyn) + 1
arenas = [i for i, z in enumerate(zones) if z["kind"] == "arena"]
added, short = [], {}
SEP = F.GAP if sep_arg is None else sep_arg
F.SG.SEP_MARGIN, sep0 = SEP, F.SG.SEP_MARGIN
try:
    for zi in arenas:
        z = zones[zi]
        box = (z["x0"] + 0.1, z["x1"] - 0.1, -F.HALF_WIDTH, F.HALF_WIDTH)
        while sum(d.get("zone") == zi for d in dyn) < target:
            others = [(F.smooth_track(d["motion"]), d["radius"]) for d in dyn if d.get("zone") == zi]
            motion = None
            for _ in range(tries):
                kind = str(rng.choice(names, p=p))       # the training shares, a fresh draw per try
                zc = float(rng.choice(np.round(np.arange(1.0, 3.01, 0.25), 2)))
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
            if motion is None:
                short[zi] = sum(d.get("zone") == zi for d in dyn)
                print(f"arena {zi}: no room after {short[zi]} spheres", flush=True)
                break
            dyn.append(dict(id=next_id, shape="sphere", radius=F.R_MIX, height=0.0, speed=round(float(sp.mean()), 4),
                            speed_range=[round(float(sp.min()), 4), round(float(sp.max()), 4)], kind=kind,
                            vertical=False, zone=zi, motion=motion))
            added.append((zi, next_id, kind, zc))
            print(f"arena {zi}: sphere {next_id} {kind} at z {zc}", flush=True)
            next_id += 1
finally:
    F.SG.SEP_MARGIN = sep0

scn["layout"] = dict(scn["layout"], n_dynamic=len(dyn))
scn["edits"] = list(scn.get("edits", [])) + [
    f"2026-11-05: densify.py from {src}: {len(added)} spheres added, target {target} per arena "
    f"(long200 density), sphere-sphere gap >= {SEP} m for the new spheres, >= {F.GAP} m to the statics; "
    f"no more room in {short}"]
S = Scenario(scn)
T = np.arange(0, 200, 0.1)
P = [d.motion.pos(T) for d in S.dynamic]
new = {a[1] for a in added}
dd = min(float(np.linalg.norm(P[a] - P[b], axis=1).min() - S.dynamic[a].radius - S.dynamic[b].radius)
         for a in range(len(P)) for b in range(a + 1, len(P))
         if S.dynamic[a].id in new or S.dynamic[b].id in new) if added else None
ds = min(float(S.static_distance(pp).min() - d.radius) for d, pp in zip(S.dynamic, P) if d.id in new) if added else None
with open(os.path.join(pkg, "scenarios", dst + ".yaml"), "w") as fh:
    yaml.safe_dump(scn, fh, sort_keys=False)
with open(os.path.join(pkg, "worlds", dst + ".sdf"), "w") as fh:
    fh.write(world_sdf(scn))
per = {zi: sum(d.get("zone") == zi for d in dyn) for zi in arenas}
print(f"{dst}: spheres {len(dyn)} (added {len(added)}), per arena {per}; new spheres: min gap to spheres {dd:.2f} m, "
      f"to statics {ds:.2f} m (rules: spheres {SEP}, statics {F.GAP})")
