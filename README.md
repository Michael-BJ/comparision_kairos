# Operating the closed-loop tests

How KAIROS and the comparison methods are flown in Gazebo, and which settings must stay the same so the results can be compared. Anything a method changes from this file is written next to its result.

Last checked: 2026-11-02.

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
| one long200 world not used before | final numbers, chosen before any method is flown on it |

**All results so far (2026-11-03) were flown in `dyn_long200_smooth_021_clear2` only.** `dyn_long200_mix_021_clear2` and the final world have not been flown by any method yet; every number in this file is for the smooth world.

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

The one exception to "Inputs the planner may use" (section 2): SANDO flies with its own sensors, a Livox Mid-360 (360 deg lidar, 20 m, to its ACL mapper) and a RealSense D435 (to its obstacle tracker), on the same x500 tw2 airframe (`SANDO_planner/models/x500_mid360_d435_tw2`). Everything else is as above: PX4 odometry, `trajectory_server`, v_max / a_max / j_max, mission, worlds, referee. No ground truth (`use_global_pc: false`, no obstacle trajectories). Report it as "SANDO (original sensors: Mid-360 + D435)": it sees more than the other methods. Its other changes (Jazzy patches, yaw, obstacle speed bound 2.0 m/s, drone size) are listed in `SANDO_planner/README.md`.

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

## 6. Status, 2026-11-03

| Method | World | Passes | Reached | Static | Dynamic | Note |
|---|---|---|---|---|---|---|
| KAIROS+MIGHTY, `runs/2nov_exc3/B` (`KAIROS_MIGHTY_24`) | smooth clear2 | 24 | 24 | 2 | 0 | window open; SR 22/24; 5074 m, 1.45 m/s, plan median 223 ms |
| KAIROS+MIGHTY, `runs/2nov_exc3/B` (`KAIROS_MIGHTY`, superseded) | smooth clear2 | 12 | 12 | 1 | 0 | window open |
| KAIROS+MIGHTY, `runs/1nov_mix/B` (old) | smooth clear2 | 12 | 12 | 0 | 0 | headless |
| NEO-style regression | - | - | - | - | - | not flown |
| NEO-Planner | smooth clear2 | 24 | 22 | 4 | 146 | window open; SR 0/24; 6212 m, 0.67 m/s, plan median 15 ms |

Details: `~/kairos_test/20261102_compare/results.md`.

### Result table, `dyn_long200_smooth_021_clear2`, 24 passes

All values are for the smooth world only (see Worlds). Window open, same machine.

| Metric | KAIROS+MIGHTY (`runs/2nov_exc3/B`) | NEO-Planner |
|---|---|---|
| SR (reached, no collision) | 22/24 | 0/24 |
| CR (passes with a collision) | 2/24 | 24/24 |
| Reached | 24/24 | 22/24 (2 timeouts, passes 1 and 22) |
| Static hits (per 100 m) | 2 (0.04) | 4 (0.06) |
| Dynamic hits (per 100 m) | 0 | 142 (2.29) |
| Distance | 5071 m | 6205 m |
| Mean speed | 1.45 m/s | 0.66 m/s |
| Planning time, median / p90 | 223 / 279 ms | 15 / 58 ms |
| Late plans | not computed | 827 of 32352 replans |
| Closest approach to a static | not computed | not computed |

How the values are computed:

- **Source.** KAIROS: `runs/KAIROS_MIGHTY_24/<stamp>_.../summary.json` (`episodes[i].collisions`, `travel`, `plan_ms`). NEO: `runs/NEO/<stamp>_.../summary.json` (`episodes[i].hits_static`, `hits_dynamic`, `path`, `duration`, `plan_ms`). The pose-based counts of `hits.py` are the reference; the NEO row of the status table above (146 dynamic hits, 6212 m) comes from it, while the 142 and 6205 m here are summed from `summary.json`. Run `hits.py` and use its numbers for the final table.
- **Reached.** Episodes whose `outcome` is `reached`.
- **SR.** A pass counts as a success if it is reached and has no collision. A timeout and the short follow-on episode that finishes it are one failed pass (NEO passes 1 and 22 with their follow-ons 2 and 23).
- **CR.** Passes with at least one static or dynamic hit, divided by 24.
- **Hits per 100 m.** Total hits divided by total distance flown, times 100. Example: KAIROS static 2 / 5071 m x 100 = 0.04.
- **Distance and mean speed.** Distance is the sum of `path` over the episodes. Mean speed is total distance divided by total flight time (sum of `duration`).
- **Planning time.** Median and 90th percentile of the planner's time per event (`plan_ms`). For KAIROS this is the whole event (target, prediction, network, score, refine); NEO's median includes only its own planner, its network alone takes 3 ms.
- **Late plans.** Replans not ready by their start time. KAIROS does not log this in `summary.json`.
- **Closest approach to a static.** Minimum of `scn.static_distance(p)` over the recorded Gazebo pose `odom_<NAME>.npz`. Not computed yet for either method.
