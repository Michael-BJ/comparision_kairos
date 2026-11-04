# Operating the closed-loop tests

How KAIROS and the comparison methods are flown in Gazebo, and which settings must stay the same so the results can be compared. Anything a method changes from this file is written next to its result.

Last checked: 2026-11-04.

## 1. Principle

Each method plans for the same drone, on the same sensors, in the same worlds, with the same mission and the same referee. The only thing that changes is the planner node. Methods differ in their own internal parameters (map resolution, optimizer weights, safety margins). Those stay at the method's defaults; we do not tune them on the test worlds.

## 2. Fixed for every method

### Simulation and vehicle

| Item | Value |
|---|---|
| Simulator | Gazebo + PX4 SITL, `ros2 launch dyn_gz_worlds sim.launch.py world:=<W> drone_model:=x500_oakd_x2_60_tw2` |
| Drone | `x500_oakd_x2_60_tw2`: x500 with motors for thrust-to-weight 2, two OAK-D depth cameras, 60 deg apart |
| Depth | range 0.4 to 6.0 m (`camera.min_range`, `camera.max_range`) |
| Obstacle tracker range | 12 m (OAK-D) |
| MAVROS | `ros2 launch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14557`, started 25 s after the simulator |
| Planner launch | 8 s after MAVROS |
| Window | `HEADLESS=false` (Gazebo window open) or `true`. Use one setting for all final runs of a table. |

### Inputs the planner may use

| Input | Source |
|---|---|
| Depth images | both cameras, `ros_gz_bridge` |
| Drone state | PX4's estimate, `/mavros/local_position/odom` put into the world frame by `odom_estimate` (`/kairos/odom`, `odom_local_yaw` 0) |
| Goal | the scenario's pads A and B |
| Time | sim time (`use_sim_time`) |

Ground truth (Gazebo pose, obstacle positions, the static map) is never a planner input. It is used only by the referee and the recorder. `perception:=sensor` and `guidance:=none` for KAIROS.

### Execution

All methods send their trajectory to the same `trajectory_server` (`dyn_expert`, Hermite trajectory, `recalib_tau` 1.0), which sends setpoints to PX4. No method sets PX4 parameters itself.

| Item | Value | Config key |
|---|---|---|
| Max speed | 1.5 m/s | `planner.v_max`, launch `v_max:=1.5` |
| Max acceleration | 4.0 m/s^2 | `planner.a_max` |
| Max jerk | 25 m/s^3 | `planner.j_max` |
| Replanning period | 0.2 s | `planner.replan_dt` |
| Plan start | 0.4 s after its event | `planner.foresee_dt` |
| Local target distance | 4.0 m | `planner.target_dist` |
| Navigation altitude | route z 2.0 m, band 1.0 to 3.0 m | scenario `route.z`, `route.z_band` |

A method whose original timing differs (for example NEO: replan 1.0 s, plan 1.0 s ahead, target 5 m) is run with the values above, and this change is written in its README.

### Mission

| Item | Value |
|---|---|
| Route | pad A (x -100.8) to pad B (x +100.8) and back, about 200 m per pass |
| Passes | 24 per world for the final table (12 is used for quick checks and must be stated) |
| Reached | within 0.6 m of the pad (`dataset.goal_reach_tol`) |
| Time limit | 4.0 x distance / 1.5 + 10 s (`timeout_factor:=4.0`) |
| Bounds | no out-of-bounds end (`bounds_margin:=1000`); the planner is not told the bounds |
| After a collision | the pass continues (`continue_on_collision:=true`); the world has no collision geometry |

### Worlds

| World | Use |
|---|---|
| `dyn_long200_smooth_021_clear2` | first test world, smooth (sinusoidal) sphere motions |
| `dyn_long200_mix_021_clear2` | second test world, mixed sphere motions |
| `dyn_long200_smooth_021_clear2_static` | static-only control: `dyn_long200_smooth_021_clear2` with all 61 spheres removed (`tools/drop_kinds.py ... line loop --suffix _static`); the same 100 static elements, route, spawn and bounds. Separates static-avoidance failures from dynamic ones. |
| one long200 world not used before | final numbers, chosen before any method is flown on it |
| `dyn_sando_easy` / `_medium` / `_hard` (+ `_static`) | difficulty ladder derived from SANDO's benchmark layout (105 / 105 / 150 m, bricks, gaps >= 1.75 m, training-world motions, hand-edited); how they were made, tools and first 2-episode runs: `sando_worlds/README.md` |

**All results so far (2026-11-03) were flown in `dyn_long200_smooth_021_clear2` only.** `dyn_long200_mix_021_clear2` and the final world have not been flown by any method yet; every number in this file is for the smooth world. The `dyn_sando_*` worlds have 2-pass checks only (2026-11-04, `sando_worlds/README.md`); how to measure them: section 3.

The worlds are not used for training. The training data is `~/dynamic_39_world_1nov_exc3` (39 `*_mix` worlds).

### Referee and collision count

The referee reads Gazebo's exact pose (`/model/<model>/odometry`), never the planner's state.

- A collision is the drone body, radius 0.35 m (`dataset.collision_radius`), touching an obstacle surface.
- A contact is counted once: samples under 0.35 m that are less than 0.5 s apart belong to one contact.
- Static and dynamic contacts are counted separately.
- Count from the recorded pose: `python3 ~/kairos_test/20261102_compare/hits.py <NAME> [WORLD]`, which reads `odom_<NAME>.npz` written by `~/kairos_test/20261027_odomlag/record.py`.
- `summary.json` of a KAIROS run lists each collision under `episodes[i].collisions`. The title of `trajectories.png` counts episode outcomes, which stay `reached` with `continue_on_collision`; the X marks on the plot are the collisions.

### Metrics

| Metric | Definition |
|---|---|
| Success rate (SR) | passes that end within 0.6 m of the pad before the time limit **and** with no collision, out of all passes. A pass that times out and the short follow-on episode that finishes it count as one failed pass. The drone keeps flying after a collision; SR is computed after the run from the referee's per-episode collisions |
| Collision rate (CR) | passes with at least one collision, out of all passes |
| Reached | passes that end within 0.6 m of the pad, collisions ignored (reported next to SR) |
| Static hits, dynamic hits | contacts as above, also per 100 m flown |
| Distance | path length of the Gazebo pose |
| Mean speed | distance / flight time |
| Closest approach to a static | minimum distance from the Gazebo pose to a static surface |
| Planning time | median and p90 per event, and late plans (plan not ready by its start time) |

### Hardware

All runs on the same machine: Intel i5-12400 (12 threads), NVIDIA RTX 5060 Ti 16 GB. No other Gazebo, training or GPU job runs at the same time.

## 3. Running a test

The scripts stop every running Gazebo/PX4/ROS process first (`~/kairos_test/cam_compare_20261022/stop_stack.sh`), start the stack, fly, record the pose, and stop the stack again.

```bash
# KAIROS+MIGHTY
cd ~/kairos_test/20261102_compare
NAME=KAIROS_MIGHTY EPISODES=24 WORLD=dyn_long200_smooth_021_clear2 ./run.sh
python3 hits.py KAIROS_MIGHTY dyn_long200_smooth_021_clear2

# NEO-Planner (PlannerNet + 2D MINCO)
cd ~/comparision_for_kairos/NEO_planner
MODEL=~/saved_net/neo_kairos/run_<stamp>/best.pth EPISODES=24 ./run.sh
cd ~/kairos_test/20261102_compare && python3 hits.py NEO

# NEO-style regression (KAIROS network trained with train.mode=regression, not NEO)
cd ~/comparision_for_kairos/NEO_planner
EPISODES=24 ./run_regression.sh
cd ~/kairos_test/20261102_compare && python3 hits.py NEO_regression
```

Results land in `~/kairos_test/20261102_compare`: `logs/<NAME>/`, `runs/<NAME>/<stamp>_<world>_<mode>/` (episode `.h5`, `summary.json`, `trajectories.png`) and `odom_<NAME>.npz`.

A run is invalid if `test.log` shows `left OFFBOARD` or a traceback; the script prints a warning.

### Measuring on the SANDO-based worlds (easy / medium / hard)

Six worlds, each flown by every method: the static + dynamic world and its static-only control.

| World | Static-only control | Course | Time limit per pass | Statics | Spheres |
|---|---|---|---|---|---|
| `dyn_sando_easy` | `dyn_sando_easy_static` | (0, 0) -> (105, 0), 105 m | 290 s | 18 | 10 |
| `dyn_sando_medium` | `dyn_sando_medium_static` | (0, 0) -> (105, 0), 105 m | 290 s | 26 | 24 |
| `dyn_sando_hard` | `dyn_sando_hard_static` | (0, 0) -> (150, 0), 150 m | 410 s | 35 | 39 |

Everything in section 2 applies, except the course: pad A at x 0, pad B at x 105 (hard: 150), so a pass is
105 m (hard 150 m) instead of ~200 m. Passes alternate A -> B and B -> A; **24 passes per world** for the final
table (2 for a quick check, stated as such). How the worlds were made, their known deviations (a few sphere paths
closer than 1.75 m to a static after the hand edits; open edge lanes at y 6-8 m) and the tools:
`sando_worlds/README.md`.

**Before a measurement**

1. The worlds are frozen. Check that the installed files are the measured ones:
   `cd ~/Dynamic_obstacle/install/dyn_gz_worlds/share/dyn_gz_worlds && md5sum -c ~/comparision_for_kairos/sando_worlds/worlds/CHECKSUMS.md5`
   (all `OK`). After any change to a world, every run in it is flown again: a run is counted against the world it
   was flown in.
2. No other Gazebo, training or GPU job runs (section 2). The scripts stop any running stack, including a Gazebo
   window opened with `show_world.sh`.
3. To look at a world first: `~/Dynamic_obstacle/show_world.sh dyn_sando_medium` (Gazebo only; close it before a run).

**Flying a method on all six worlds** (`sando_worlds/tools/run_worlds.sh <method> <passes> [worlds]`: runs the
method's own `run.sh` once per world, then `hits.py` for each run):

```bash
cd ~/comparision_for_kairos/sando_worlds/tools
W="dyn_sando_easy_static dyn_sando_medium_static dyn_sando_hard_static dyn_sando_easy dyn_sando_medium dyn_sando_hard"
./run_worlds.sh kairos 24 $W            # KAIROS+MIGHTY          -> KAIROS_<world>_24ep
./run_worlds.sh sando 24 $W             # SANDO (own sensors: Mid-360)
./run_worlds.sh sando_d435 24 $W        # SANDO with its Table VII sensor (D435), optional
./run_worlds.sh neo 24 $W               # NEO-Planner            (NEO_MODEL=... to change the checkpoint)
./run_worlds.sh neo_regression 24 $W    # NEO-style regression
./run_worlds.sh agile 24 $W             # Agile Autonomy         (AGILE_MODEL=... to change the checkpoint)
```

The same run by hand, one world (here KAIROS+MIGHTY in `dyn_sando_hard`):

```bash
cd ~/kairos_test/20261102_compare
NAME=KAIROS_dyn_sando_hard_24ep EPISODES=24 WORLD=dyn_sando_hard ./run.sh
source ~/inference_kairos/install/setup.bash
python3 hits.py KAIROS_dyn_sando_hard_24ep dyn_sando_hard
```

For the other methods replace the first two lines with the method's script from the list above (e.g.
`cd ~/comparision_for_kairos/SANDO_planner && NAME=SANDO_dyn_sando_hard_24ep EPISODES=24 WORLD=dyn_sando_hard ./run.sh`).
SANDO is flown with `run.sh`; `run_restart.sh` (restart after a permanent deadlock) is a variant reported next to it
and only needed if SANDO deadlocks (it did not in the 2-pass checks). MIGHTY, EGO-Planner and NavRL have no run
script yet (section 4).

**Run names** are `<METHOD>_<world>_<passes>ep` (add a suffix such as `_v2` for a repeat: `SUFFIX=_v2 ./run_worlds.sh ...`).
Results: `~/kairos_test/20261102_compare/logs/<NAME>/`, `runs/<NAME>/`, `odom_<NAME>.npz`, as above.

**Duration** (one method, 24 passes): about 35 min per 105 m world for KAIROS (~75 s per pass) and 45 min for SANDO
(~90-115 s), 45-60 min for hard; about 4-5 h for the six worlds.

**Reporting.** Per world and method, the metrics of section 2 (SR, CR, reached, static / dynamic hits, distance, mean
speed, planning time), the static-only control next to the static + dynamic world. First 2-pass checks (KAIROS+MIGHTY
6/6 without a collision; SANDO 2/6, its contacts growing with density): `sando_worlds/README.md`, "Results so far".

## 4. Per-method settings

### KAIROS+MIGHTY (`kairos_planner`, `mode:=kairos_mighty`)

| Item | Value |
|---|---|
| Checkpoint | `~/kairosV3/runs/2nov_exc3/B/best.pt`, `kairos_root:=~/kairosV3` |
| Config | `config:=~/kairos_test/20261102_compare/config/expert_m4.yaml` (= `expert_v3_oakd_x2_60.yaml` with `dataset.memory_time` 4, the event format of the training data). `check_model` reads this value from the yaml; `memory_time:=4.0` on the command line does not satisfy it. |
| Samples | K 16 (`k`), 6 refined (`n_refine`) |
| Selection | the MIGHTY cost; SFC + L-BFGS on the cheapest samples, the cheapest result is flown; no verify, no fallback |
| Seeds | `seed_mode:=network` |
| Obstacle prediction | `prediction:=head2` |
| Obstacle memory | `memory_time:=4.0`, `memory_belief:=-1.0`, `safety_vision:=0.0`, `remembered_pred:=hold`, `memory_inflate:=false` |
| Safety extras | `emergency:=false`; only `static_clearance:=0.55` (the training expert's value; SFC margin 0.60) |
| Head 2 inflation | `sigma_k` 0 (off) |

No other safety parameter is switched on: the test judges the model and its optimizer, not added safeguards.

### NEO-style regression (`run_regression.sh`)

The KAIROS pipeline with one output: `runs/2nov_exc3/NEO/best.pt`, trained with `train.mode=regression` on the same 26 v_max 1.5 worlds and split as B, no head 2. Same settings as KAIROS+MIGHTY except:

- K is 1 (set by `kairos_planner` for a regression checkpoint).
- `prediction:=cv`, the tracker's constant velocity, because there is no head 2.

Report it as "NEO-style regression", not as NEO.

### NEO-Planner (`NEO_planner/run.sh`)

PlannerNet + 2D MINCO on an octomap ESDF, trained on the KAIROS dataset with the same train / val files as B. Its changes from the original NEO (image size, depth panorama, timing, execution through `trajectory_server`, fixed cruise altitude at route z) are listed in `NEO_planner/README.md`. It plans in 2D at 2.0 m and has no obstacle motion model.

### SANDO, original sensors (`SANDO_planner/run.sh`)

The one exception to "Inputs the planner may use" (section 2): SANDO flies with its own sensor and perception pipeline, as in its hardware experiments: a Livox Mid-360 (360 deg lidar, 20 m) to its ACL mapper, whose temporal grid and tracker give the dynamic obstacles. The model (`SANDO_planner/models/x500_mid360_d435_tw2`, same x500 tw2 airframe) also carries the D435 of SANDO's URDF, which SANDO uses only in its perception-only simulation; it is not used here. Further exceptions: SANDO's own timing (replans as fast as it can, from its committed plan), its own yaw (`goal.yaw`, including its spin after repeated failed replans), and its 100 Hz setpoint stream executed through `trajectory_server`. Everything else is as above: PX4 odometry, v_max / a_max / j_max (as norms), mission, worlds, referee. No ground truth (`use_global_pc: false`, no obstacle trajectories). Report it as "SANDO (own sensors: Mid-360)": it sees more than the other methods. The diagnosis of its first runs, the integration fixes and its tuning for this vehicle and world (all without changes to its code) are in `SANDO_planner/README.md`. Its runs before `SANDO_v3` (`SANDO`, `SANDO_Linf`, `SANDO_v2`) are not results: the package's drone model had the Mid-360 inside the airframe (sensor poses taken from the model origin, 0.24 m below the x500's base_link), so SANDO mapped the drone itself as an obstacle; the lidar's returns on the airframe (the x500's propeller tips at its lowest rays) are now dropped as on a real drone.

### Methods still to be added (MIGHTY, EGO-Planner, NavRL)

Each new method:

1. Reads only the inputs of section 2: both depth cameras, `/kairos/odom`, the goal, sim time.
2. Sends its trajectory to `trajectory_server`, at replan 0.2 s and start 0.4 s ahead, with v_max 1.5, a_max 4.0, j_max 25.
3. Keeps its own internal parameters at their published defaults; any change is listed in its README with the reason.
4. Uses the same mission, worlds, number of passes, recorder and `hits.py`.
5. A learned method trains on `~/dynamic_39_world_1nov_exc3` with the split of `runs/2nov_exc3/B/split.json`, or states which data it used.

## 5. Things that make a comparison unfair

- Different number of passes or a different world without saying so.
- Gazebo window open for some methods and closed for others.
- Another job using the CPU or GPU during a run.
- Ground truth reaching a planner, through guidance, perception or state.
- Turning on a safety extra (emergency manoeuvres, radius inflation, larger clearance) for one method only.
- Tuning a method on the test worlds.
- Reading hits from `trajectories.png` titles instead of `hits.py` or `summary.json`.

## 6. Status, 2026-11-04

| Method | World | Passes | Reached | Static | Dynamic | Note |
|---|---|---|---|---|---|---|
| KAIROS+MIGHTY, `runs/2nov_exc3/B` (`KAIROS_MIGHTY_24`) | smooth clear2 | 24 | 24 | 2 | 0 | window open; SR 22/24; 5074 m, 1.45 m/s, plan median 223 ms |
| KAIROS+MIGHTY, `runs/2nov_exc3/B` (`KAIROS_MIGHTY`, superseded) | smooth clear2 | 12 | 12 | 1 | 0 | window open |
| KAIROS+MIGHTY, `runs/1nov_mix/B` (old) | smooth clear2 | 12 | 12 | 0 | 0 | headless |
| NEO-style regression | - | - | - | - | - | not flown |
| NEO-Planner | smooth clear2 | 24 | 22 | 4 | 146 | window open; SR 0/24; 6212 m, 0.67 m/s, plan median 15 ms |
| SANDO (own sensors: Mid-360), `SANDO_v3` | smooth clear2 | 24 | 4 | 6 | 9 | window open; SR 1/24; from pass 5 it hovered without a plan (deadlock, no recovery), passes 6-24 are follow-ons; 1178 m; passes 1-4 1.04 m/s; plan median 56 ms |
| SANDO, `SANDO_v3R` (stack restarted after a deadlock) | smooth clear2 | 24 | 20 | 31 | 91 | variant for SANDO only (`SANDO_planner/run_restart.sh`); SR 0/24; deadlocks in passes 10 and 23; 4868 m; reached passes 1.05 m/s; plan median 56 ms |
| SANDO, `SANDO_v3R_omv1` (as `SANDO_v3R`, `obst_max_vel` 1.0) | smooth clear2 | 24 | 20 | 27 | 122 | sensitivity to SANDO's obstacle speed bound (0.5 m/s, its simulation default; 1.0 = its paper's simulations); SR 0/24; deadlocks in passes 8 and 17; 5082 m; reached passes 0.89 m/s; 5.55 dynamic contacts per reached pass against 4.35 with 0.5 |
| SANDO, `SANDO_v2` (invalid) | smooth clear2 | 24 | 14 | 65 | 126 | the drone model's lidar mapped the drone itself (`SANDO_planner/README.md`) |

Details: `~/kairos_test/20261102_compare/results.md`.

### Result table, `dyn_long200_smooth_021_clear2`, 24 passes

All values are for the smooth world only (see Worlds). Window open, same machine.

| Metric | KAIROS+MIGHTY (`runs/2nov_exc3/B`) | NEO-Planner | SANDO (own sensors: Mid-360), `SANDO_v3` |
|---|---|---|---|
| SR (reached, no collision) | 22/24 | 0/24 | 1/24 |
| CR (passes with a collision) | 2/24 | 24/24 | 23/24 |
| Reached | 24/24 | 22/24 (2 timeouts, passes 1 and 22) | 4/24 (deadlock in pass 5, passes 6-24 its follow-ons) |
| Static hits (per 100 m) | 2 (0.04) | 4 (0.06) | 6 (0.51) |
| Dynamic hits (per 100 m) | 0 | 142 (2.29) | 9 (0.77) |
| Distance | 5071 m | 6205 m | 1174 m |
| Mean speed | 1.45 m/s | 0.66 m/s | 0.43 m/s (passes 1-4: 1.04 m/s) |
| Planning time, median / p90 | 223 / 279 ms | 15 / 58 ms | 56 / 104 ms |
| Late plans | not computed | 827 of 32352 replans | not applicable (replans continuously; 20 % of 43775 replans succeeded) |
| Closest approach to a static | not computed | not computed | -0.15 m |

SANDO with a stack restart after a deadlock (`SANDO_v3R`, a variant for SANDO only: it has no recovery from a hover point outside every corridor, so in `SANDO_v3` one deadlock in pass 5 ended the run): SR 0/24, CR 22/24, reached 20/24, static 31 (0.64 per 100 m), dynamic 90 (1.85 per 100 m), 4857 m, 0.81 m/s (reached passes 1.05 m/s), planning 56 / 104 ms, closest approach to a static -0.03 m. The same with `obst_max_vel` 1.0, the bound of SANDO's paper simulations, instead of 0.5 (`SANDO_v3R_omv1`): SR 0/24, CR 24/24, reached 20/24, static 27 (0.53 per 100 m), dynamic 121 (2.39 per 100 m), 5069 m, 0.74 m/s (reached passes 0.89 m/s), planning 60 / 116 ms, closest approach -0.13 m: the larger bound covers the obstacles' speeds but neither the success rate nor the dynamic contacts improve. One pass with 2.0, above every obstacle's speed: reached, 2 static and 7 dynamic contacts, 4 of them on the hovering drone. Details in `SANDO_planner/README.md`.

How the values are computed:

- **Source.** KAIROS: `runs/KAIROS_MIGHTY_24/<stamp>_.../summary.json` (`episodes[i].collisions`, `travel`, `plan_ms`). NEO: `runs/NEO/<stamp>_.../summary.json` (`episodes[i].hits_static`, `hits_dynamic`, `path`, `duration`, `plan_ms`). SANDO: `runs/SANDO_v3/<stamp>_.../summary.json`, same fields as NEO; `plan_ms` is SANDO's `total_replanning_ms` of its successful replans. The pose-based counts of `hits.py` are the reference; the NEO row of the status table above (146 dynamic hits, 6212 m) comes from it, while the 142 and 6205 m here are summed from `summary.json`. Run `hits.py` and use its numbers for the final table.
- **Reached.** Episodes whose `outcome` is `reached`.
- **SR.** A pass counts as a success if it is reached and has no collision. A timeout and the short follow-on episode that finishes it are one failed pass (NEO passes 1 and 22 with their follow-ons 2 and 23).
- **CR.** Passes with at least one static or dynamic hit, divided by 24.
- **Hits per 100 m.** Total hits divided by total distance flown, times 100. Example: KAIROS static 2 / 5071 m x 100 = 0.04.
- **Distance and mean speed.** Distance is the sum of `path` over the episodes. Mean speed is total distance divided by total flight time (sum of `duration`).
- **Planning time.** Median and 90th percentile of the planner's time per event (`plan_ms`). For KAIROS this is the whole event (target, prediction, network, score, refine); NEO's median includes only its own planner, its network alone takes 3 ms.
- **Late plans.** Replans not ready by their start time. KAIROS does not log this in `summary.json`.
- **Closest approach to a static.** Minimum of `scn.static_distance(p)` over the recorded Gazebo pose `odom_<NAME>.npz`. Not computed yet for either method.
