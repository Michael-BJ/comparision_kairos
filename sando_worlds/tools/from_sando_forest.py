"""SANDO's static forests (its "Sim Static" benchmark) as static-only worlds of this package.

  python3 from_sando_forest.py <sando worlds dir> <dyn_gz_worlds dir>
  -> dyn_sando_forest_{easy,medium,hard}: scenarios/<name>.yaml and worlds/<name>.sdf (rebuild to install)

Source: github.com/mit-acl/sando 1a738df, worlds/{easy,medium,hard}_forest.world, the Gazebo worlds of
scripts/run_benchmark.py --mode gazebo (case_environments), made by scripts/generate_random_forest.py
(shape_mode cylinder, min_clearance 1.5 m; footprint density 0.05 / 0.1 / 0.2 over x 3..103, y -20..20; radius
1.0..1.5 m, height 6 m). Every cylinder is copied as it is (x, y, radius, height); nothing is moved or left out.
Mission as SANDO's benchmark (run_benchmark.py --start 0 0 2 --goal 105 0 2): pads A (0, 0) and B (105, 0), route z
2 m with this test's altitude band 1..3 m. Made on 2026-11-05 (user: the static-only worlds like SANDO's paper).
Differences from SANDO's setup: this package's pads, drone, sensors and referee; SANDO's forests keep a 1.5 m gap
between cylinders, below the 1.75 m (2.5 x the drone's size) of the dyn_sando_* worlds.
"""
import os
import re
import sys

import numpy as np
import yaml

src, pkg = sys.argv[1], sys.argv[2]
from dyn_expert.core.scenario import Scenario  # noqa: E402
from dyn_expert.make_world import world_sdf  # noqa: E402

A, B = (0.0, 0.0), (105.0, 0.0)
for case in ("easy", "medium", "hard"):
    text = open(os.path.join(src, f"{case}_forest.world")).read()
    cyl = []
    for name, body in re.findall(r"<model name=['\"](\d+)['\"]>(.*?)</model>", text, re.S):
        if "<cylinder>" not in body:
            continue                    # the <state> block repeats every model's pose (identical, checked)
        pose = [float(v) for v in re.search(r"<pose>([^<]+)</pose>", body).group(1).split()[:3]]
        g = re.search(r"<cylinder>(.*?)</cylinder>", body, re.S).group(1)
        r = float(re.search(r"<radius>([\d.eE+-]+)</radius>", g).group(1))
        h = float(re.search(r"<length>([\d.eE+-]+)</length>", g).group(1))
        cyl.append(dict(type="cylinder", kind="forest", xy=[round(pose[0], 4), round(pose[1], 4)], radius=round(r, 4),
                        height=round(h, 4), group=f"sando_forest_{name}"))
    P = np.array([c["xy"] for c in cyl]); R = np.array([c["radius"] for c in cyl])
    gap = np.linalg.norm(P[:, None] - P[None], axis=2) - R[:, None] - R[None]
    np.fill_diagonal(gap, np.inf)
    pad = min(float(np.min(np.linalg.norm(P - np.asarray(q), axis=1) - R)) for q in (A, B))
    scn = dict(seed=0, generator="sando_forest (from_sando_forest.py)",
               bounds=dict(lo=[-2.0, -22.0, 0.0], hi=[107.0, 22.0, 7.0]), spawn=[0.0, 0.0, 0.0],
               route=dict(a=list(A), b=list(B), jitter=0.0, z=2.0, z_band=1.0),
               layout=dict(source="github.com/mit-acl/sando 1a738df, worlds/%s_forest.world "
                                  "(scripts/generate_random_forest.py, shape_mode cylinder, min_clearance 1.5)" % case,
                           case=case, n_static=len(cyl), course=105.0, static_obstacles=True, n_dynamic=0,
                           footprint_density={"easy": 0.05, "medium": 0.1, "hard": 0.2}[case],
                           min_gap=round(float(gap.min()), 3), pad_clearance=round(pad, 3)),
               static=cyl, name=f"dyn_sando_forest_{case}", dynamic=[],
               edits=[f"2026-11-05: from_sando_forest.py, {len(cyl)} cylinders of SANDO's {case}_forest.world as they are"])
    Scenario(scn)                                  # loads: shapes understood by the referee
    with open(os.path.join(pkg, "scenarios", scn["name"] + ".yaml"), "w") as fh:
        yaml.safe_dump(scn, fh, sort_keys=False)
    with open(os.path.join(pkg, "worlds", scn["name"] + ".sdf"), "w") as fh:
        fh.write(world_sdf(scn))
    print(f"{scn['name']}: {len(cyl)} cylinders, r {R.min():.2f}..{R.max():.2f}, h {min(c['height'] for c in cyl):.1f}, "
          f"x {P[:, 0].min():.1f}..{P[:, 0].max():.1f}, y {P[:, 1].min():.1f}..{P[:, 1].max():.1f}, "
          f"min gap {gap.min():.2f} m, nearest surface to a pad {pad:.2f} m")
