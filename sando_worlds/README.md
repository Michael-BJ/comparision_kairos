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
| `worlds/scenarios/*.yaml`, `worlds/sdf/*.sdf` | copies of the 9 worlds as they are in `~/Dynamic_obstacle/src/dyn_obstacle_nav/dyn_gz_worlds/{scenarios,worlds}` (the package copy is the one Gazebo and the planners load) |
| `tools/from_sando.py` | generator (canonical copy: `dyn_gz_worlds/tools/from_sando.py`) |
| `tools/edit/` | scripts that saved the Gazebo edits and changed heights / added spheres |
| `tools/measure/` | lidar / tracker probe for SANDO, its analysis, motion statistics, SANDO layout plot |
| `tools/sando_original/` | SANDO's own rviz-only benchmark on ROS 2 Jazzy (build, demo, benchmark, `/trajs` capture) |
| `tools/run_worlds.sh` | fly a method on the worlds and count collisions |
| `tools/show_world.sh` | open a world in Gazebo only (no PX4, no planner) |
| `patches/` | changes to shared code (`dyn_expert`, `KinematicMover`) and SANDO's Jazzy build patches |
| `results/2ep_20261104/` | first 2-episode runs of KAIROS+MIGHTY and SANDO |

## The worlds

| World | Course A -> B | Layout | Statics | Dynamic spheres |
|---|---|---|---|---|
| `dyn_sando_easy` | (0, 0) -> (105, 0), 105 m | statics and spheres mixed | 18: 7 brick pillars 0.6 x 0.6 x 4.0 m; 11 brick walls 4 m long, 0.3 / 0.9 m thick, 1.5 (2), 1.8 (7), 2.1 (2) m high | 10 |
| `dyn_sando_medium` | (0, 0) -> (105, 0), 105 m | zones: 8 static fields 4.875 m, 7 dynamic arenas 8 m | 26: 13 pillars 3.0-3.9 m; 13 walls 1.5-2.1 m | 24 |
| `dyn_sando_hard` | (0, 0) -> (150, 0), 150 m | zones: 11 static fields 5.45 m, 10 dynamic arenas 8 m | 35: 19 pillars 3.0-3.9 m; 16 walls 1.5-2.1 m | 39 |

Common to all: route altitude 2 m, altitude band 1-3 m (`route.z`, `route.z_band`); obstacles within y -6..6 m
(walls up to +-8 m); spheres radius 0.45 m (the largest of the training worlds), horizontal motions at
1-3 m altitude, peak speeds 0.51-2.02 m/s, kinds of the training worlds (line / loop sinusoids, constant / accel /
turn tables); no collision geometry (the referee counts contacts). Time limit per pass with the protocol
(`timeout_factor` 4, v_max 1.5): 290 s (easy, medium), 410 s (hard).

Variants: `<world>_static` = the same statics, no spheres (separates static from dynamic failures);
`<world>_trefoil` = the same statics with SANDO's own 3D trefoil motions (first N of SANDO's dynamic draws, r 0.5 m,
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
tools/run_worlds.sh sando 2 dyn_sando_hard                # sando | sando_d435 | neo | neo_regression | agile
tools/run_worlds.sh kairos 24 dyn_sando_easy_static       # official runs: 24 passes

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
