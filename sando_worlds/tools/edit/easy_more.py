"""More spheres in dyn_sando_easy (2026-11-06, user: easy harder with more dynamic obstacles, still below medium /
hard): 10 -> N spheres, statics unchanged. Writes the world in place (the old one stays in git history).

  python3 easy_more.py <pkg dir> [N, default 18] [world, default dyn_sando_easy]

The added spheres follow from_sando.py's rules for its spheres: SANDO's next dynamic draws (dyn_obstacles.launch.py,
seed 0, draws 10, 11, ... after the 10 already in the world), each at its SANDO z0 mapped to the training altitudes
1..3 m and placed near its SANDO x (window x0 +- 6 m, then +- 12 m, then the whole course 5..100, y +-6); a kind drawn
with the training shares (MIX_KINDS; a table kind falls back to a line, as tools_smooth); GAP = 1.75 m to the statics
and between spheres at every time (user's choice, the rule of the spheres already there).
"""
import os
import sys

import numpy as np
import yaml

pkg = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 18
name = sys.argv[3] if len(sys.argv) > 3 else "dyn_sando_easy"
sys.path.insert(0, os.path.join(pkg, "tools"))
sys.argv = sys.argv[:1]
import from_sando as F  # noqa: E402
from dyn_expert.core.scenario import Scenario  # noqa: E402
from dyn_expert.make_world import world_sdf  # noqa: E402

path = os.path.join(pkg, "scenarios", name + ".yaml")
scn = yaml.safe_load(open(path))
dyn = scn["dynamic"]
F.set_course(F.COURSE["easy"])
draws = [o for o in F.sando_obstacles(F.CASES["easy"], 0) if o["dynamic"]]
used = {d["id"] for d in dyn}
todo = [o for o in draws if o["i"] not in used]
rng = np.random.default_rng(6000)
names = list(F.MIX_KINDS)
p = np.asarray([F.MIX_KINDS[k] for k in names], float)
p /= p.sum()
sdist = Scenario(dict(static=scn["static"])).static_distance
clearance = lambda P: sdist(P) - (F.GAP - 0.05)
X0, X1, post_r = 5.0, 100.0, 0.1
added, skipped = [], []
sep0, F.SG.SEP_MARGIN = F.SG.SEP_MARGIN, F.GAP
try:
    for o in todo:
        if len(dyn) >= N:
            break
        kind0 = str(rng.choice(names, p=p))
        zc = round(1.0 + (min(max(o["z0"], 0.5), 4.5) - 0.5) / 2.0, 3)
        others = [(F.smooth_track(d["motion"]), d["radius"]) for d in dyn]
        windows = [(max(X0, o["x0"] - w), min(X1, o["x0"] + w)) for w in (6.0, 12.0)] + [(X0, X1)]
        attempts = [(win, kind0) for win in windows] + ([(win, "line") for win in windows] if kind0 in F.TABLE_KINDS else [])
        motion = None
        for (a, b), kind in attempts:
            box = (a + post_r, b - post_r, -F.HALF_WIDTH, F.HALF_WIDTH)
            try:
                if kind in F.TABLE_KINDS:
                    motion, sp = F._table_path(rng, kind, F.R_MIX, box, (0.5, 2.5), 2.0, 1.5, clearance, zc, others=others)
                else:
                    motion, sp = F._smooth_path(rng, kind, F.R_MIX, box, (0.8, 2.0), 1.0, 0.4, clearance, zc, others=others)
                break
            except RuntimeError:
                motion = None
        if motion is None:
            skipped.append(o["i"])
            print(f"draw {o['i']} (x {o['x0']:.1f}): no path", flush=True)
            continue
        dyn.append(dict(id=o["i"], shape="sphere", radius=F.R_MIX, height=0.0, speed=round(float(sp.mean()), 4),
                        speed_range=[round(float(sp.min()), 4), round(float(sp.max()), 4)], kind=kind,
                        vertical=False, motion=motion))
        added.append((o["i"], kind, zc, round(a, 1), round(b, 1)))
        print(f"draw {o['i']}: {kind} at z {zc}, x {a:.1f}..{b:.1f} (SANDO x {o['x0']:.1f})", flush=True)
finally:
    F.SG.SEP_MARGIN = sep0

scn["layout"] = dict(scn["layout"], n_dynamic=len(dyn))
scn["edits"] = list(scn.get("edits", [])) + [
    f"2026-11-06 (user): easy_more.py, {len(added)} spheres added ({len(dyn)} in all) from SANDO's dynamic draws "
    f"{[a[0] for a in added]} (skipped {skipped}), GAP {F.GAP} m to statics and spheres"]
S = Scenario(scn)
T = np.arange(0, 300, 0.05)
P = {d.id: d.motion.pos(T) for d in S.dynamic}
R = {d.id: d.radius for d in S.dynamic}
new = {a[0] for a in added}
ids = list(P)
dd = min(float(np.linalg.norm(P[a] - P[b], axis=1).min() - R[a] - R[b])
         for i, a in enumerate(ids) for b in ids[i + 1:] if a in new or b in new)
ds = min(float(S.static_distance(P[i]).min() - R[i]) for i in new)
with open(path, "w") as fh:
    yaml.safe_dump(scn, fh, sort_keys=False)
with open(os.path.join(pkg, "worlds", name + ".sdf"), "w") as fh:
    fh.write(world_sdf(scn))
print(f"{name}: spheres {len(dyn)} (added {len(added)}); new spheres over 300 s: min gap to spheres {dd:.2f} m, "
      f"to statics {ds:.2f} m (rule {F.GAP})")
