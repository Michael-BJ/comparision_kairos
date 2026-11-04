"""Worlds after SANDO's dynamic benchmark (Kondo et al., T-RO 2026, Tables VI / VII) for this package.

  python3 from_sando.py      -> dyn_sando_<case>, dyn_sando_<case>_static, dyn_sando_<case>_trefoil
  python3 from_sando.py --seed 3 --cases hard                         (case = easy, medium, hard)

Source of the layout: SANDO's own obstacle generator, the same draws in the same order as
sando/launch/dyn_obstacles.launch.py (_spawn_static_block) with the defaults of rviz_only.launch.py and
run_benchmark.py (github.com/mit-acl/sando 1a738df): easy / medium / hard = 50 / 100 / 200 obstacles, dynamic
ratio 0.65, positions uniform in x 5..100, y -6..6, z 0.5..4.5 m, Python random.seed(seed). Mission: SANDO's
start (0, 0) and goal (105, 0) are pads A and B (hard: course 150 m, pads 0 and 150, SANDO's x stretched to 5..145, user
2026-11-04); route z 2 m, z band 1 m (this test's altitude band 1..3 m).

Changes from SANDO, chosen by the user on 2026-11-04 ("not abstract", "the drone must be able to pass"):
- static: SANDO's x, y draws of its static obstacles (35 % pillars, 65 % bars), in SANDO's order, as bricks on the
  ground: pillars -> brick pillars 0.6 x 0.6 x 4 m (the pillars of this package's courses); bars -> brick walls
  4 m long in y (SANDO's bar), 1.5 m high (the low walls of the courses), thin 0.3 m or thick 0.9 m (alternating
  in SANDO's order). Every footprint keeps GAP = 2.5 x the drone's size (0.7 m = 2 x dataset.collision_radius)
  = 1.75 m to every other one and to the pads; one that does not is moved to the nearest free spot within
  SHIFT_MAX (random tries), else left out.
- zones (user's choice 2026-11-04, after medium / hard did not fit with statics and spheres mixed): x 5..100 as
  the training courses, fields (statics) and arenas (spheres) in turn, field first and last: 8 fields of
  FIELD_L = 4.875 m, 7 arenas of ARENA_L = 8 m. A static's SANDO x along the course is put at the same fraction
  of the fields laid end to end, its footprint inside its field.
- dynamic (dyn_sando_<case>): N_DYN = 10 / 20 / 30 spheres of radius R_MIX, spread evenly over the arenas (the
  first N_DYN of SANDO's dynamic draws, sorted by x, one arena after another; an arena without room for a path:
  the least occupied other arena, nearest first; a table kind tries every arena before it falls back to a
  line), each at its SANDO z0 mapped from 0.5..4.5 to the training altitudes
  1..3 m, on a path of the training worlds' motions (KAIROS's head 2 was trained on them): the path generator of
  dyn_expert.tools_smooth --mix (scenario_gen._smooth_path / _table_path, MIX_KINDS, the same speed, acceleration
  and turn limits), with GAP instead of 0.3 m between spheres and GAP instead of 0.05 m to the statics, at every
  time. A sphere without such a path is left out (printed).
- dyn_sando_<case>_trefoil: the same statics with SANDO's own motions, for an ablation (out of the training
  distribution: vertical motion, constant turning); the first N_DYN of SANDO's dynamic draws, radius R_SPHERE,
  no spacing guarantee. SANDO writes x = sx/6*(sin(tt)+2*sin(2*tt))+x0 etc. as strings with tt = "t/slower+offset",
  so "2*tt" is evaluated as 2t/slower + offset (no parentheses): x = sx/6 (sin(w t + o) + 2 sin(2 w t + o)) + x0,
  y = sy/5 (cos(w t + o) - 2 cos(2 w t + o)) + y0, z = sz/2 (-sin(3 w t + o)) + z0, w = 1/slower, o = offset
  (checked against the /trajs strings of SANDO's dynamic_forest_node): dyn_expert's `trefoil` motion with center
  (x0, y0, z0), scale (sx/6, sy/5, sz/2), omega w, phases [o, o, o]. SANDO's 0.8 m cube -> a sphere of the same
  volume (r 0.5).
- dyn_sando_<case>_static: the statics only.

Writes scenarios/<name>.yaml and worlds/<name>.sdf of this package (rebuild to install them).
"""
import argparse
import copy
import math
import os
import random
import sys

import numpy as np
import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "dyn_expert"))
import dyn_expert.core.scenario_gen as SG  # noqa: E402
from dyn_expert.core.motion import make_motion  # noqa: E402
from dyn_expert.core.scenario import Scenario  # noqa: E402
from dyn_expert.core.scenario_gen import MIX_KINDS, TABLE_KINDS, _smooth_path, _table_path, smooth_track  # noqa: E402
from dyn_expert.make_world import world_sdf  # noqa: E402

CASES = {"easy": 50, "medium": 100, "hard": 200}       # SANDO's obstacle counts (its generator draws)
N_DYN = {"easy": 10, "medium": 20, "hard": 30}          # dynamic spheres here
DRONE = 0.7                     # drone size: 2 x dataset.collision_radius
GAP = 2.5 * DRONE               # 1.75 m between obstacles
SHIFT_MAX = 3.0                 # [m] a static that keeps no GAP is moved at most this far
PAD_CLEAR = 2.0                 # [m] statics stay this far (+ GAP) from pads A and B
CUBE = 0.8
R_SPHERE = 0.5                  # _trefoil worlds: the volume of SANDO's cube
R_MIX = 0.45                    # main worlds: the largest radius of the training worlds
COURSE = {"easy": 105.0, "medium": 105.0, "hard": 150.0}    # A -> B [m]; SANDO: 105 (hard 150: user 2026-11-04)
SANDO_X = (5.0, 100.0)          # x range of SANDO's draws; stretched to X0..X1 = 5..course - 5
X0 = 5.0
ARENA_L = 8.0                   # [m] arenas (dynamic spheres); training worlds: median 7.3
FIELD_TARGET = 4.875            # [m] fields (statics): 105 m course -> 7 arenas, 8 fields of 4.875 m
HALF_WIDTH = 6.0                # SANDO's y range
PILLAR = [0.6, 0.6, 4.0]
PILLAR_HS = (3.0, 3.3, 3.6, 3.9)   # pillar heights (user 2026-11-04: tall, not all the same)
WALL_L, WALL_T = 4.0, (0.3, 0.9)
WALL_HS = (1.5, 1.8, 2.1)       # wall heights (training low walls 1.5..2.1)
BRICK_LAYER = 0.3


def set_course(length):
    """Pads, x range and zones (field, arena, ..., field) of a course of this length."""
    global X1, PADS, N_ARENAS, FIELD_L, ZONES, FIELDS
    X1 = length - 5.0
    PADS = [(0.0, 0.0), (length, 0.0)]
    N_ARENAS = int(round((X1 - X0 - FIELD_TARGET) / (ARENA_L + FIELD_TARGET)))
    FIELD_L = (X1 - X0 - N_ARENAS * ARENA_L) / (N_ARENAS + 1)
    ZONES, x = [], X0
    for k in range(2 * N_ARENAS + 1):
        L = FIELD_L if k % 2 == 0 else ARENA_L
        ZONES.append({"kind": "field" if k % 2 == 0 else "arena", "x0": round(x, 4), "x1": round(x + L, 4)})
        x += L
    FIELDS = [z for z in ZONES if z["kind"] == "field"]


set_course(105.0)


def sando_obstacles(num_obstacles, seed, dynamic_ratio=0.65, x_min=5.0, x_max=100.0, y_min=-6.0, y_max=6.0,
                    z_min=0.5, z_max=4.5, slower_min=4.0, slower_max=6.0):
    """dyn_obstacles.launch.py, _spawn_static_block: the same draws in the same order."""
    scale_range = [[2.0, 4.0], [2.0, 4.0], [2.0, 4.0]]
    offset_range = [0.0, 3.0]
    percentage_vert = 0.35
    random.seed(seed)
    num_dynamic = int(num_obstacles * dynamic_ratio)
    num_static = num_obstacles - num_dynamic
    out = []
    for i in range(num_obstacles):
        x = x_min + (x_max - x_min) * random.random()
        y = y_min + (y_max - y_min) * random.random()
        z = z_min + (z_max - z_min) * random.random()
        if i < num_dynamic:
            sx = scale_range[0][0] + (scale_range[0][1] - scale_range[0][0]) * random.random()
            sy = scale_range[1][0] + (scale_range[1][1] - scale_range[1][0]) * random.random()
            sz = scale_range[2][0] + (scale_range[2][1] - scale_range[2][0]) * random.random()
            offset = random.uniform(*offset_range)
            slower = random.uniform(slower_min, slower_max)
            out.append(dict(dynamic=True, i=i, x0=x, y0=y, z0=z, sx=sx, sy=sy, sz=sz, offset=offset, slower=slower))
        else:
            kind = "pillar" if (i - num_dynamic) < num_static * percentage_vert else "bar"
            out.append(dict(dynamic=False, i=i, x=x, y=y, z=z, kind=kind))
    return out


def _rect_gap(a, b):
    """Gap between axis-aligned footprints (x0, y0, x1, y1)."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def statics(draws, seed):
    """SANDO's static draws as brick pillars / walls on the ground, GAP apart (module docstring)."""
    rng = np.random.default_rng(seed + 1000)
    rng_h = np.random.default_rng(seed + 2000)              # heights, apart from the placement draws
    out, rects, moved, dropped, n_bar = [], [], 0, 0, 0
    pads = [(px - PAD_CLEAR, py - PAD_CLEAR, px + PAD_CLEAR, py + PAD_CLEAR) for px, py in PADS]
    for o in draws:
        if o["kind"] == "pillar":
            size = [PILLAR[0], PILLAR[1], float(rng_h.choice(PILLAR_HS))]
        else:
            size = [WALL_T[n_bar % 2], WALL_L, float(rng_h.choice(WALL_HS))]
            n_bar += 1
        rect = lambda x, y: (x - size[0] / 2, y - size[1] / 2, x + size[0] / 2, y + size[1] / 2)
        # footprint inside a field: x in [x0 + half, x1 - half] of one field
        in_field = lambda x: any(f["x0"] + size[0] / 2 <= x <= f["x1"] - size[0] / 2 for f in FIELDS)
        free = lambda r: all(_rect_gap(r, q) >= GAP for q in rects + pads)
        # SANDO's x along the course -> the same fraction of the fields laid end to end
        s = (o["x"] - SANDO_X[0]) / (SANDO_X[1] - SANDO_X[0]) * FIELD_L * len(FIELDS)
        f = FIELDS[min(len(FIELDS) - 1, int(s // FIELD_L))]
        x = min(max(f["x0"] + (s % FIELD_L), f["x0"] + size[0] / 2), f["x1"] - size[0] / 2)
        y = o["y"]
        x_s, y_s = x, y
        if not free(rect(x, y)):
            cand = [(x_s + dx, y_s + dy) for dx, dy in rng.uniform(-SHIFT_MAX, SHIFT_MAX, (800, 2))
                    if math.hypot(dx, dy) <= SHIFT_MAX]
            cand = [c for c in cand if in_field(c[0]) and -HALF_WIDTH <= c[1] <= HALF_WIDTH and free(rect(*c))]
            if not cand:
                dropped += 1
                continue
            x, y = min(cand, key=lambda c: math.hypot(c[0] - x_s, c[1] - y_s))
            moved += 1
        rects.append(rect(x, y))
        r4 = lambda v: [round(float(c), 4) for c in v]
        out.append({"type": "box", "kind": "brick", "center": r4([x, y, size[2] / 2.0]), "size": size, "yaw": 0.0,
                    "layers": max(1, round(size[2] / BRICK_LAYER)), "group": f"sando_{o['kind']}_{o['i']}",
                    "sando": {"kind": o["kind"], "xy": r4([o["x"], o["y"]])}})
    return out, moved, dropped


def base(case, seed, static):
    return {
        "seed": seed, "generator": "sando_dyn_obstacles (from_sando.py)",
        "bounds": {"lo": [-2.0, -9.0, 0.0], "hi": [PADS[1][0] + 2.0, 9.0, 7.0]},
        "spawn": [0.0, 0.0, 0.0],
        "route": {"a": list(PADS[0]), "b": list(PADS[1]), "jitter": 0.0, "z": 2.0, "z_band": 1.0},
        "layout": {"source": "github.com/mit-acl/sando 1a738df, launch/dyn_obstacles.launch.py (rviz_only defaults)",
                   "case": case, "sando_num_obstacles": CASES[case], "gap": GAP, "half_width": HALF_WIDTH,
                   "n_static": len(static),
                   "course": PADS[1][0], "n_arenas": N_ARENAS, "field_length": round(FIELD_L, 4),
                   "static_shape": f"brick pillars 0.6 x 0.6 m, heights {PILLAR_HS}; brick walls {WALL_L} m long, "
                                   f"thickness {WALL_T}, heights {WALL_HS}; "
                                   f"gap >= {GAP} m, at SANDO's x, y", "static_obstacles": True},
        "static": static,
    }


def peak_speed(motion, n=4000):
    m = make_motion(motion)
    t = np.linspace(0.0, 2.0 * math.pi / motion["omega"], n)
    return float(np.linalg.norm(m.vel(t), axis=-1).max())


def trefoil_dynamic(draws, n):
    r4 = lambda v: [round(float(c), 4) for c in v]
    out = []
    for o in draws[:n]:
        scale = r4([o["sx"] / 6.0, o["sy"] / 5.0, o["sz"] / 2.0])
        motion = {"type": "trefoil", "center": r4([X0 + (o["x0"] - SANDO_X[0]) * (X1 - X0) / (SANDO_X[1] - SANDO_X[0]), o["y0"], o["z0"]]),
                  "scale": scale,
                  "omega": round(1.0 / o["slower"], 6), "phase": round(o["offset"], 6),
                  "phases": [round(o["offset"], 6)] * 3}
        out.append({"id": o["i"], "shape": "sphere", "radius": R_SPHERE, "height": 0.0,
                    "speed": round(peak_speed(motion), 4), "kind": "trefoil", "vertical": True, "motion": motion,
                    "sando": {"name": f"obstacle_{o['i']}", "bbox": [CUBE] * 3, "slower": round(o["slower"], 6),
                              "scale_xyz": r4([o["sx"], o["sy"], o["sz"]])}})
    return out


def mix_dynamic(draws, n, static, seed, speed=(0.8, 2.0), acc=1.0, turn=0.4, post_r=0.1, table_speed=(0.5, 2.5),
                table_acc=2.0, table_turn=1.5):
    """tools_smooth.smooth with MIX_KINDS and its defaults, GAP between spheres and to the statics; spheres spread
    evenly over the arenas (module docstring)."""
    rng = np.random.default_rng(seed)
    names = list(MIX_KINDS)
    p = np.asarray([MIX_KINDS[k] for k in names], float)
    p /= p.sum()
    arena_ids = [i for i, z in enumerate(ZONES) if z["kind"] == "arena"]     # index k -> ZONES[arena_ids[k]]
    zones = [ZONES[i] for i in arena_ids]
    sdist = Scenario(dict(static=static)).static_distance
    clearance = lambda P: sdist(P) - (GAP - 0.05)          # the generator checks clearance >= r + 0.05
    chosen = sorted(draws[:n], key=lambda o: o["x0"])
    out, dropped, placed = [], [], {}
    sep0 = SG.SEP_MARGIN
    SG.SEP_MARGIN = GAP                                      # sphere-sphere gap of _smooth_path / _table_path
    try:
        for j, o in enumerate(chosen):
            k0 = min(N_ARENAS - 1, int(j * N_ARENAS / n))
            kind0 = str(rng.choice(names, p=p))
            zc = round(1.0 + (min(max(o["z0"], 0.5), 4.5) - 0.5) / 2.0, 3)
            # its own arena first, then the others, least occupied and nearest first
            order = [k0] + sorted((k for k in range(N_ARENAS) if k != k0),
                                  key=lambda k: (len(placed.get(k, [])), abs(k - k0)))
            # the drawn kind in every arena before a short line (tools_smooth's fallback for a table kind), so the
            # kinds keep the training shares as far as the room allows
            attempts = [(k, kind0) for k in order] + ([(k, "line") for k in order] if kind0 in TABLE_KINDS else [])
            motion = None
            for k, kind in attempts:
                zone = zones[k]
                box = (zone["x0"] + post_r, zone["x1"] - post_r, -HALF_WIDTH, HALF_WIDTH)
                others = placed.get(k, [])
                try:
                    if kind in TABLE_KINDS:
                        motion, sp = _table_path(rng, kind, R_MIX, box, table_speed, table_acc, table_turn,
                                                 clearance, zc, others=others)
                    else:
                        motion, sp = _smooth_path(rng, kind, R_MIX, box, speed, acc, turn, clearance, zc,
                                                  others=others)
                    break
                except RuntimeError:
                    motion = None
            if motion is None:
                dropped.append(o["i"])
                continue
            placed.setdefault(k, []).append((smooth_track(motion), R_MIX))
            out.append(dict(id=o["i"], shape="sphere", radius=R_MIX, height=0.0, speed=round(float(sp.mean()), 4),
                            speed_range=[round(float(sp.min()), 4), round(float(sp.max()), 4)], kind=kind,
                            vertical=False, zone=arena_ids[k], motion=motion))
    finally:
        SG.SEP_MARGIN = sep0
    return out, copy.deepcopy(ZONES), dropped


def write(scn):
    with open(os.path.join(PKG, "scenarios", scn["name"] + ".yaml"), "w") as fh:
        yaml.safe_dump(scn, fh, sort_keys=False)
    with open(os.path.join(PKG, "worlds", scn["name"] + ".sdf"), "w") as fh:
        fh.write(world_sdf(scn))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    a = ap.parse_args()
    for case in a.cases:
        set_course(COURSE[case])
        draws = sando_obstacles(CASES[case], a.seed)
        dyn_draws = [o for o in draws if o["dynamic"]]
        st, moved, st_dropped = statics([o for o in draws if not o["dynamic"]], a.seed)
        name = f"dyn_sando_{case}" + ("" if a.seed == 0 else f"_s{a.seed}")
        n = N_DYN[case]

        mix = dict(base(case, a.seed, st), name=name)
        mix["dynamic"], mix["zones"], dropped = mix_dynamic(dyn_draws, n, st, a.seed)
        mix["layout"] = dict(mix["layout"], n_dynamic=len(mix["dynamic"]), n_arenas=N_ARENAS, boundary="smooth",
                             kinds=MIX_KINDS, dynamic_shape=f"sphere r {R_MIX}",
                             motion="training-world motions (tools_smooth --mix generator), gap >= GAP")
        mix["edits"] = [f"from_sando.py (seed {a.seed}): {len(st)} statics ({moved} moved, {st_dropped} left out for "
                        f"the gap), {len(mix['dynamic'])} of {n} spheres ({len(dropped)} without a path: {dropped})"]
        write(mix)

        tref = dict(base(case, a.seed, st), name=name + "_trefoil", dynamic=trefoil_dynamic(dyn_draws, n))
        tref["layout"] = dict(tref["layout"], n_dynamic=n, dynamic_shape=f"sphere r {R_SPHERE}",
                              motion="SANDO's trefoils, no spacing guarantee")
        write(tref)

        stat = dict(base(case, a.seed, st), name=name + "_static", dynamic=[])
        stat["layout"] = dict(stat["layout"], n_dynamic=0)
        write(stat)

        kinds = [s["sando"]["kind"] for s in st]
        sp = sorted(d["speed_range"][1] for d in mix["dynamic"]) or [0.0]
        print(f"{case}: static {len(st)} ({kinds.count('pillar')} pillars, {kinds.count('bar')} walls; {moved} moved, "
              f"{st_dropped} left out of {len(draws) - len(dyn_draws)}); dynamic {len(mix['dynamic'])} of {n} "
              f"(peak speed {sp[0]:.2f}..{sp[-1]:.2f} m/s; left out {dropped})")


if __name__ == "__main__":
    main()
