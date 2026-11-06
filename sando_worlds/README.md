# SANDO-based test worlds (easy / medium / hard)

Three closed-loop test worlds derived from SANDO's dynamic-obstacle benchmark (Kondo, Tordesillas, How,
"SANDO: Safe Autonomous Trajectory Planning for Dynamic Unknown Environments", T-RO 2026, Tables VI / VII;
github.com/mit-acl/sando, commit 1a738df), plus the tools used to build, edit, measure and fly them.
Built on 2026-11-04.

The worlds take SANDO's obstacle layout (its own generator, seed 0) and change what the test protocol needs
(`../README.md`): obstacles a drone can physically pass, bricks on the ground instead of floating bars, and obstacle
motions from the KAIROS training distribution. **They are not SANDO's benchmark.** Report them as "worlds derived
from SANDO's benchmark layout"; SANDO's own numbers (Tables VI / VII) are not comparable to results here.

## Contents

| Path | What |
|---|---|
| `worlds/history/` | earlier versions that results were flown in |
| `worlds/scenarios/*.yaml`, `worlds/sdf/*.sdf` | copies of the 10 worlds as they are in `~/Dynamic_obstacle/src/dyn_obstacle_nav/dyn_gz_worlds/{scenarios,worlds}` (the package copy is the one Gazebo and the planners load) |
| `tools/from_sando.py` | generator (canonical copy: `dyn_gz_worlds/tools/from_sando.py`) |
| `tools/from_sando_forest.py` | SANDO's static forests (`worlds/*_forest.world`) -> `dyn_sando_forest_*` |
| `tools/edit/` | scripts that saved the Gazebo edits and changed heights / added spheres; `densify.py` (-> `dyn_sando_hard_dense`) |
| `tools/measure/` | lidar / tracker probe for SANDO, its analysis, motion statistics, SANDO layout plot |
| `tools/sando_original/` | SANDO's own rviz-only benchmark on ROS 2 Jazzy (build, demo, benchmark, `/trajs` capture) |
| `tools/run_worlds.sh` | fly a method on the worlds and count collisions |
| `tools/show_world.sh` | open a world in Gazebo only (no PX4, no planner) |
| `patches/` | changes to shared code (`dyn_expert`, `KinematicMover`) and SANDO's Jazzy build patches |
| `results/2ep_20261104/` | first 2-episode runs of KAIROS+MIGHTY and SANDO |

## The worlds

| World | Course A -> B | Layout | Statics | Dynamic spheres |
|---|---|---|---|---|
| `dyn_sando_easy` | (0, 0) -> (105, 0), 105 m | statics and spheres mixed | 18: 7 brick pillars 0.6 x 0.6 x 4.0 m; 11 brick walls 4 m long, 0.3 / 0.9 m thick, 1.5 (2), 1.8 (7), 2.1 (2) m high | 18 (10 + 8 added 2026-11-06) |
| `dyn_sando_medium` | (0, 0) -> (105, 0), 105 m | zones: 8 static fields 4.875 m, 7 dynamic arenas 8 m | 26: 13 pillars 3.0-3.9 m; 13 walls 1.5-2.1 m | 28 (24 + 4 added 2026-11-06) |
| `dyn_sando_hard_dense` | (0, 0) -> (150, 0), 150 m | zones: 11 static fields 5.45 m, 10 dynamic arenas 8 m | 35: 19 pillars 3.0-3.9 m; 16 walls 1.5-2.1 m | 60 (39 + 21 added 2026-11-05, 6 per arena) |
| `dyn_sando_forest_easy` | (0, 0) -> (105, 0), 105 m | SANDO's static forest, x 4-102, y +-18.5 | 41 cylinders r 1.0-1.5 m, 6 m high, 5 % footprint | 0 |
| `dyn_sando_forest_medium` | (0, 0) -> (105, 0), 105 m | SANDO's static forest | 81 cylinders, 10 % footprint | 0 |
| `dyn_sando_forest_hard` | (0, 0) -> (105, 0), 105 m | SANDO's static forest | 162 cylinders, 20 % footprint | 0 |

Common to all: route altitude 2 m, altitude band 1-3 m (`route.z`, `route.z_band`); obstacles within y -6..6 m
(walls up to +-8 m); spheres radius 0.45 m (the largest of the training worlds), horizontal motions at
1-3 m altitude, peak speeds 0.51-2.02 m/s, kinds of the training worlds (line / loop sinusoids, constant / accel /
turn tables); no collision geometry (the referee counts contacts). Time limit per pass with the protocol
(`timeout_factor` 4, v_max 1.5): 290 s (easy, medium), 410 s (hard).

Variants: `<world>_trefoil` = the same statics with SANDO's own 3D trefoil motions (first N of SANDO's dynamic draws, r 0.5 m,
no spacing guarantee), kept for an ablation of motions outside the training distribution.

### How they were made (all steps on 2026-11-04, choices by the user)

1. **SANDO's layout.** `from_sando.sando_obstacles` repeats SANDO's generator draw for draw
   (`sando/launch/dyn_obstacles.launch.py`, rviz-only defaults): 50 / 100 / 200 obstacles, 65 % dynamic, uniform in
   x 5..100, y -6..6, z 0.5..4.5, `random.seed(0)`.
2. **Bricks on the ground.** SANDO's 0.4 m pillars -> brick pillars 0.6 x 0.6 m; its floating 0.4 x 4 x 0.4 m bars ->
   brick walls on the ground, 4 m long, thin (0.3 m) or thick (0.9 m), drawn as 0.3 m brick courses.
3. **Passable.** Every gap >= 2.5 x the drone's size (0.7 m = 2 x `dataset.collision_radius`) = **1.75 m**:
   static to static (footprints), static to pads, sphere to sphere and sphere to static at every time. A static
   that keeps no gap is moved up to 3 m, else left out.
4. **Fewer spheres, training motions.** 10 / 20 / 30 spheres (hard was SANDO's density x 0.25), paths from the
   generator of the `*_mix` training worlds (`dyn_expert.tools_smooth --mix`: `scenario_gen._smooth_path` /
   `_table_path`, the same speed / acceleration / turn limits), with the 1.75 m gaps. KAIROS's head 2 was trained
   on these motions only; SANDO has no learned model, so the change does not favour either method in that respect.
5. **Zones** (medium, hard): with statics and spheres mixed and 1.75 m gaps, medium / hard had no room for the
   spheres (85 % of them reduced to short lines). As in the training courses, the course alternates static fields
   and dynamic arenas; SANDO's static x is mapped to the same fraction of the fields laid end to end.
6. **Hard 150 m**: course 0 -> 150, SANDO's x stretched to 5..145 (10 arenas, 11 fields); all 30 spheres fit.
7. **Edited by hand in Gazebo** and saved from the live poses (`tools/edit/save_edit.py`, `save_statics.py`,
   `edit_medium.py`; every scenario's `edits` field lists it):
   - easy: 11 statics moved (the mixed version of step 4, before the zones);
   - medium: 2 statics deleted, 17 moved;
   - hard: 11 statics deleted, 13 moved.
8. **Heights varied**: walls 1.5 / 1.8 / 2.1 m (training low walls 1.5-2.1); pillars 4.0 m in easy, 3.0-3.9 m in
   medium and hard (`vary_heights.py`, `edit_medium.py`, generator).
9. **One more sphere per arena** (`add_spheres.py`): medium 18 -> 24 (arena 11 had no room), hard 30 -> 39
   (arena 9 had no room).
10. **Hard, wall `static_023` moved** from x 36.7, y -2.2..1.8 (across the route centre; it sent both KAIROS
    passes to the left, and one of them then flew 85 m along the empty edge at y 6-7) to y 2..6. KAIROS's mean
    |y| went from 3.38 to 0.83 m, time at |y| > 4 m from 37 % to 0 %.

### Known deviations from the rules (not fixed)

- Sphere-to-static gaps below 1.75 m after the hand edits: easy spheres 5 (1.03 m), 7 (1.13), 2 (1.32), 6 (1.43);
  medium 12 (1.29), 1 (1.64); hard 3 (1.08), 10 (1.51). Easy also has a sphere-sphere minimum of 1.09 m (its mixed
  generator checked the gap within an arena only).
- **Edge lanes**: no side boundary. Spheres stay within |y| <= 5.55 m, so a planner can fly along y 6-8 m almost
  free of obstacles (KAIROS in medium reached |y| 7.6 m, SANDO in hard 7.9 m). Side walls at y +-7 m would close it.
- Easy's scenario was rebuilt from the SDF Gazebo had loaded: its table motions carry the kind label `table`
  instead of constant / accel / turn (motions identical).
- Walls near the goal still cross the centre in hard (x 129.6, 140.2, 143.2).

### Harder easy and medium worlds, hard_dense as the hard level (2026-11-06, choices by the user)

- **Bounds y +-6.5 m** (were +-9 m) in `dyn_sando_easy`, `_medium`, `_hard_dense` (scenario only; the `.sdf` does
  not carry the bounds and is unchanged). The spheres move within |y| <= 6.0 m, so the wider bounds left an
  empty lane of 3 m on both sides: in `dyn_sando_hard_dense` KAIROS+MIGHTY used it in 9 of 23 passes (up to 64 %
  of a pass), SANDO in 23 of 24 (median 50 %), KAIROS raw in its one (partial) pass 44 %. Both planners get the
  bounds: SANDO `y_min` / `y_max` from the scenario (`SANDO_planner/launch/sando_test.launch.py`); KAIROS keeps its
  local targets >= 1 m inside them (planner config `bounds_margin: 1.0`), its trajectory is not bounded (KAIROS raw
  flew to |y| 7.5-8.6 m in its first passes after the change). No method's episode is ended out of bounds (KAIROS
  `run.sh` `bounds_margin:=1000`; with 1.0 every excursion split a pass into 3 episodes, two of them empty, and
  SANDO's bridge has no such end): two raw runs (with 1000, then 1.0) were stopped and discarded. The referee applies KAIROS's rule to every method
  (`hits.py`: Gazebo pose > 1 m outside the bounds; such a pass is no success). Hidden side walls (visible to
  the sensors, not in the Gazebo window) were tried first: a visual with `visibility_flags` 0x20000000 was hidden
  from the depth cameras too, so the bounds were chosen instead. Some walls reach beyond +-6.5 m (|y| up to
  8.7 / 10.0 / 8.3 m), they stay obstacles. All runs in these three worlds before this change no longer count.

- **`dyn_sando_hard` deleted** (2026-11-06): with medium at 0.27 spheres per m it was no longer harder in spheres
  (0.26 per m). `dyn_sando_hard_dense` (same statics, 60 spheres, 0.40 per m) is the hard level and the default of
  `run_worlds.sh`. `dyn_sando_hard` is in git history before this change and is still the source that
  `densify.py` made `dyn_sando_hard_dense` from; `dyn_sando_hard_trefoil` (its statics, SANDO's trefoils) stays.

- **`dyn_sando_easy`** now has 18 spheres instead of 10 (`tools/edit/easy_more.py <pkg> 18`, in place; the
  10-sphere version is in git history before this change). The user wanted easy harder but still below medium /
  hard: 0.17 spheres per m of route (medium 0.23, hard 0.26). The 8 new spheres are SANDO's next dynamic draws
  (seed 0, draws 10-14 and 16-18; draw 15 found no path), each near its SANDO x and at its SANDO z0 mapped to
  1..3 m, kind drawn with the training shares (7 lines, 1 accel), GAP 1.75 m to the statics and to every sphere
  at every time (checked over 300 s: >= 1.85 m to spheres, >= 1.77 m to statics). Statics unchanged. Runs flown
  in the 10-sphere easy no longer count against this world.
- **`dyn_sando_medium`** now has 28 spheres instead of 24. The user first wanted medium like long200
  (`dyn_long200_smooth_021_clear2`: 0.30 spheres per m, smooth `loop` / `line` motions only, spheres down to 0.3 m
  apart): `tools/edit/densify_even.py <pkg> dyn_sando_medium 32 loop,line 0.3` added 8 spheres (32, 0.30 per m),
  to the least occupied arena first. The user then removed 4 so that medium stays below hard: the last round
  (ids 30-33, arenas 1 / 3 / 5 / 7), which leaves 4 spheres in every arena and is the same as running
  `densify_even.py` with 28. Now 0.27 spheres per m (easy 0.17, hard 0.26 but 150 m with 35 statics). The 4 new
  spheres: 3 loops, 1 line, radius 0.45 m as every sphere here (long200 0.25-0.44 m), altitude 1..3 m in 0.25 m
  steps, the training generator's smooth-path limits, 1.75 m to the statics, >= 0.3 m to the other spheres at
  every time (checked over 300 s: 0.37 m; 7 sphere pairs now come closer than 1.75 m). Statics unchanged; the
  24-sphere version is in git history before this change. Runs flown in it no longer count against this world.

### Denser hard world and SANDO's static forests (2026-11-05, choices by the user)

- **`dyn_sando_hard_dense`** (`tools/edit/densify.py <pkg> dyn_sando_hard dyn_sando_hard_dense 6 40 0.3`):
  `dyn_sando_hard` with 21 more spheres, 6 per arena, 60 in all: the density of `dyn_long200_smooth_021_clear2`
  (61 spheres, 6.1 per occupied arena). Statics, zones, route and the 39 old spheres unchanged. New spheres: the
  training kinds and shares (constant 5, line 5, accel 4, loop 4, turn 3), altitudes 1-3 m, r 0.45 m, >= 1.75 m to
  the statics (measured 1.86 m) but only >= 0.3 m to the other spheres, the training generator's margin
  (`scenario_gen.SEP_MARGIN`), as in long200 (min 0.32 m there). With 1.75 m between spheres only 5 fitted (44).
  76 sphere pairs now come closer than 1.75 m (long200: 105): gaps between spheres can close for a while. Per m of
  route it is denser than long200 (0.40 against 0.30 spheres per m: every arena is occupied). No `stopgo` motion
  (none in the training worlds either).
- **`dyn_sando_forest_easy` / `_medium` / `_hard`** (`tools/from_sando_forest.py ~/sando_orig_ws/src/sando/worlds
  <pkg>`): the static-only worlds, SANDO's own static benchmark ("Sim Static": `run_benchmark.py --mode gazebo`,
  `worlds/{easy,medium,hard}_forest.world`, made by `scripts/generate_random_forest.py`, cylinders,
  `min_clearance` 1.5). Every cylinder copied as it is: 41 / 81 / 162, r 1.0-1.5 m, 6 m high (above the 1-3 m
  band: no way over), x 4-102, y +-18.7, footprint density 5 / 10 / 20 %, min gap 1.71 / 1.50 / 1.50 m (below the
  1.75 m of the other worlds, kept as SANDO's). Pads as SANDO's mission: (0, 0) -> (105, 0), nearest cylinder
  surface 5.8 / 5.8 / 4.3 m from a pad; the straight line A -> B runs 5.8 / 9.2 / 26.2 m inside cylinders.
  They replace the `dyn_sando_{easy,medium,hard}_static` controls (same statics as the dynamic worlds, no spheres),
  deleted on 2026-11-05 before any run in them (recoverable from git history, commit 45b531c).
  `from_sando.py`, `tools/edit/save_edit.py` and `edit_medium.py` still write a `_static` variant when run again:
  delete it, the forests are the static-only worlds.
- **SANDO in the forests** (2026-11-06, the user's choice: the forests are the default static-only test).
  `run_worlds.sh sando` flies them with SANDO's own settings for its static forests: `environment_assumption
  static` and `static_global_mapper.yaml` (what its `run_sim.py` uses for `*_forest`) and `drone_bbox` 0.2 m (its
  simulation `sando.yaml`; the hardware `sando_hw_quadrotor.yaml` has 0.9 m). With `run.sh`'s defaults
  (`environment_assumption dynamic`, `hw_global_mapper.yaml`, bbox 0.9; `SANDO_FOREST=hw`) SANDO deadlocked: its
  map puts the PX4 drone inside the inflated cylinders ("Start is not free") and it has no recovery (24 passes:
  SR 12.5 / 4.2 / 0 %). bbox 0.7 (the referee's contact size) still deadlocked in a 1.51 m gap of medium after 13
  passes. With bbox 0.2: medium 24/24 without contact, hard 24/24 reached with 13 static contacts (12 passes) in
  the narrowest gaps. Contacts are counted with the referee's 0.35 m for every method. Looser forests
  (SANDO's generator with `min_clearance` 2.5 m, `_wide`) were made and flown on 2026-11-06 and deleted the same
  day: the user kept SANDO's original forests.

## Shared-code changes (patches/)

- `dyn_obstacle_nav_trefoil_phases.patch` (`~/Dynamic_obstacle/src/dyn_obstacle_nav`): optional per-harmonic phases
  `phases: [p1, p2, p3]` for the `trefoil` motion in `dyn_expert/core/motion.py`, `KinematicMover.cc` and
  `make_world.py`. Without `phases` the motion is unchanged (`dyn_expert` tests 20/20, identical positions).
  Needed because **SANDO's obstacles are not standard trefoils**: SANDO writes them as expression strings with
  `tt = "t/slower+offset"`, so `2*tt` evaluates as `2t/slower + offset` (no parentheses); every harmonic has the
  phase `offset`. Checked against the strings SANDO's `dynamic_forest_node` publishes (`grab_trajs.py`): 0.5 mm
  over 250 s with the phases, up to 3.7 m with standard trefoils. Gazebo's motion was checked against the scenario
  (0.00 mm).
- `sando_1a738df_jazzy_compat.patch`, `acl-mapping_...`, `DecompROS2_...`: SANDO built on ROS 2 Jazzy in
  `~/sando_orig_ws` (fresh clone): tf2 / cv_bridge `.hpp` headers, Gazebo Classic plugins removed,
  `dynamic_forest_node` kept, `-march=native` removed, VTK/MPI target, `run_sim.py` sources Jazzy,
  `run_benchmark.py` records sqlite3 bags (Humble's default, read by SANDO's analyzer). The behaviour patch of
  `SANDO_planner/patches/sando_jazzy.patch` (`z_floor_thresh`) is **not** applied there.

## Use

```bash
# look at a world (Gazebo only); close the window before opening the next
~/Dynamic_obstacle/show_world.sh dyn_sando_medium        # or tools/show_world.sh

# fly a method, 2 episodes on the three worlds, then collisions from the Gazebo pose
tools/run_worlds.sh kairos 2
tools/run_worlds.sh sando 2 dyn_sando_hard_dense          # sando | sando_d435 | neo | neo_regression | agile
tools/run_worlds.sh kairos 24 dyn_sando_forest_easy       # official runs: 24 passes

# SANDO's lidar / tracking probe on a world
tools/measure/run_sando_probe.sh dyn_sando_easy 2 /tmp/probe_easy

# motion statistics: training worlds vs test worlds (edit the groups in the script)
source ~/inference_kairos/install/setup.bash && python3 tools/measure/motion_stats.py
```

Saving a hand edit: while Gazebo still shows the edited world, dump the live poses
(`gz topic -e -t /world/<world>/pose/info -n 1 > poses.txt`), check that the saved scenario reproduces the SDF Gazebo
loaded (`/tmp/dyn_world_*/<world>.sdf`), then `tools/edit/save_statics.py <scenario> poses.txt <dyn_gz_worlds dir>
<world>`; rebuild `dyn_gz_worlds` (`colcon build --packages-select dyn_gz_worlds`). The package's installed
scenarios are symlinks to `src`: a failed write empties both, so keep a copy before editing.

Regenerating from scratch (`python3 dyn_gz_worlds/tools/from_sando.py`) gives the generator's worlds, **not** the
hand-edited ones above; use the copies in `worlds/`.

## Results so far: 2 episodes per world (2026-11-04)

Standard protocol (`../README.md`): x500 tw2, PX4 odometry, `trajectory_server`, v_max 1.5. KAIROS+MIGHTY with
2 x OAK-D (checkpoint `runs/2nov_exc3/B`), SANDO with its own Mid-360 (`SANDO_planner/run.sh`). Collisions from
the Gazebo pose (`hits.py`). One pass = A -> B or B -> A.

| World | Method | Reached | No collision | Static / dynamic contacts | Time per pass | Mean speed |
|---|---|---|---|---|---|---|
| easy | KAIROS+MIGHTY | 2/2 | 2/2 | 0 / 0 | 76, 75 s | 1.44 m/s |
| easy | SANDO (Mid-360) | 2/2 | 1/2 | 0 / 1 | 89, 84 s | 1.25 m/s |
| medium | KAIROS+MIGHTY | 2/2 | 2/2 | 0 / 0 | 73, 76 s | 1.47 m/s |
| medium | SANDO (Mid-360) | 2/2 | 1/2 | 0 / 1 | 115, 90 s | 1.15 m/s |
| hard | KAIROS+MIGHTY | 2/2 | 2/2 | 0 / 0 | 103, 105 s | 1.48 m/s |
| hard | SANDO (Mid-360) | 2/2 | 0/2 | 1 / 3 | 144, 135 s | 1.14 m/s |

Hard rows after step 10 (`KAIROS_dyn_sando_hard_2ep_v2`); `KAIROS_dyn_sando_hard_2ep` (before, 2/2, 0 / 0) is kept
for the lateral-deviation comparison. It was flown in `worlds/history/dyn_sando_hard_before_move23.yaml`: `hits.py`
against the current hard world reports 2 false static contacts for it (the drone flew where `static_023` stands now).
A run must always be counted against the world it was flown in; after any world edit, re-fly. KAIROS's depth tracker tracked 97-98 % of the seen obstacle events with 0-1
false tracks. **Two episodes only**: a first check, not a result (2/2 has a 95 % interval of about 34-100 %).
Official numbers need 24 passes per world (`../README.md`).

## Findings on SANDO behind these worlds

- **SANDO's paper numbers.** Table VI (100 / 100 / 100 %) is the rviz-only benchmark: ground-truth obstacle
  trajectories (exact future), perfect tracking, v_max 5 m/s, 10 trials. Table VII is point-cloud sensing with a
  **D435 depth camera** (`run_benchmark.py --mode gazebo-dynamic` passes `--d435`), still perfect tracking:
  80 / 60 / 40 % at its best N 5 / P 2 (10 trials, best of 7 (N, P) settings; hard drops to 0 % at N 6, P >= 3).
  `SANDO_planner` uses the hardware config, N 5 / P 2. SANDO's own benchmark ran here on Jazzy
  (`tools/sando_original/`): dynamic easy, seed 0, reached in 21.4 s, 105.4 m (paper: 24.4 s, 107.6 m, mean of 10).
- **SANDO's prediction** (no learned model): EKF track, constant-velocity line for 2 s (speed clamped to 1 m/s), then
  at rest; used only by the global planner's heat map. Its safety corridors inflate each obstacle's current position
  by `obst_max_vel` x t (0.5 m/s), direction-free.
- **Lidar vs tracking** (probe, an earlier easy version with r 0.69 m spheres): the Mid-360 hit a sphere within 10 m
  in 98 % of the scans inside its vertical field of view (tens to hundreds of returns); SANDO's temporal-grid
  tracker had a track on it in 37-55 % of the scans within 10 m, 7 % at 10-15 m (its 20 x 20 x 4 m grid), median
  position error 0.56 m, 12 % tracks on no sphere. Not explained by its 0.8 m/s publish threshold.
- KAIROS's depth tracker accepts spheres of radius 0.12-0.60 m only (`perception.py`): with r 0.69 m it tracked none.
