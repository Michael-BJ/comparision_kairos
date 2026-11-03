# SANDO (original sensors)

Status 2026-11-03: built, bridge / launch / `run.sh` written, **flies in Gazebo**; official 24-pass run `SANDO` running (fixes 1-6, L2 limits). Debugging log:
1. `obst_max_vel` 2.0 (the worlds' bound): no plan, 1-point global paths (unknown space inflated by 2.0 x trajectory time) -> 0.5, SANDO's simulation value.
2. `obstacle_tracker_node` SIGSEGV in `pcl::VoxelGrid::filter` (gdb): `-march=native` breaks the Eigen ABI with the system PCL -> flag removed (also used by `sando`).
3. With +-200 m bounds SANDO flew around the arenas outside the corridor (y -11 m) at z 0.7 m -> scenario x / y bounds and the 1.0 .. 3.0 m band.
4. Tuning: `sando.yaml` -> "Start is not free" in almost every HGP call, no progress; `sando_hw_quadrotor.yaml` (default) flies.
5. First 24-pass run (02:38-03:49, archived as `runs/SANDO_invalid_unknown_sky`, **invalid**): from pass 3 SANDO hovered at (49, 1.8, 1.2) for 35 min without a plan, 22 passes of ~3 m. Cause: Gazebo Harmonic's `gpu_lidar` publishes rays without a return as inf points (~73 % of the Mid-360 rays, the open sky); SANDO's Livox plugin publishes them at range_max (20 m), and the ACL mapper ray-traces free space along them. With inf points the air above the walls stayed unknown, and `findSafeSubGoal` cut every path climbing over a wall to its start point (`global_path.size(): 1` in every replan, debug run 2026-11-03; `obst_max_vel` 0.25 did not help). -> `SANDO_planner/lidar_fill.py` between the gz bridge and the mapper restores the plugin's output (inf -> 20 m along the ray, intensity 0; ray directions checked against the returns, max error 0.03 deg). 3-pass check `SANDO_fill`: no stall, passes cover the corridor (timeouts near the goal).
6. Second 24-pass run (`runs/SANDO_Linf`, 04:33-06:00; episodes 1-14 valid, reference only): SANDO's default `dynamic_constraint_type: Linf` bounds v / a / j **per axis**, so with v_max 1.5 it flew 1.42 m/s median, 2.23 p95, 3.16 max (36 % of the time > 1.6 m/s; acceleration p95 4.3 m/s^2), KAIROS 1.57 p95 / 1.82 max. Static contacts were real collisions (5-12 per full pass, 28 closer than 0.15 m, several inside the obstacle), KAIROS's 2 were grazes at 0.19-0.22 m. -> `dynamic_constraint_type: L2` (SANDO's own option: ||v|| <= v_max etc., the test's rule for every method; launch argument). L2 smoke test: no stall, pass 1 0 static / 7 dynamic hits, slower (793 plans in 548 s vs ~2800).
7. Same run, episodes 15-24: SANDO stopped replanning 1.25 m from goal B (status GOAL_SEEN, its last plan ends within goal_radius 0.35 of the goal, so `needReplan` returns false) while the drone hovered 1.25 m away; the follow-on episodes got 14 s each. Not odometry drift (PX4 estimate vs Gazebo pose 0.3 m). Cause not found yet: the bridge now logs `[diag]` after 5 s without a plan (drone, end of the last forwarded trajectory, SANDO's G, SANDO's own setpoint stream `goal`); in normal hovers all three agree within 0.15 m.
8. Third 24-pass run (`runs/SANDO`, L2, 06:3x-07:0x): all 24 episodes timeouts, one single A -> B attempt. 2 static / 11 dynamic hits, 420 m, 29 min; plan time 60 ms median / 132 p90; 1164 of 40316 replan callbacks succeeded (2.9 %). From episode 6 SANDO hovered at (92.2, -3.1, 2.1), 9 m before goal B, until the end (14 min): HGP succeeds (0.1 ms), the local trajectory step returns at once; `[diag]`: forwarded trajectory = SANDO's setpoint = drone (execution is consistent). Unknown space near the goal: the edge cells of the mapper's rolling 20 x 20 m grid (x 100.2 and 101.7 while the drone is at 92; all three of SANDO's mapper configs use 20 m), 0.9 m from goal B, inside SANDO's unknown inflation (0.45 + obst_max_vel x trajectory time ~ 1.95 m). Probably also the Linf run's goal stall (item 7, 1.5 m from x 101.7).
Planning time: `summary.json` `plan_ms` = SANDO's `total_replanning_ms` of successful replans (median, p90); `stats.replans` counts replan callbacks (incl. "no replan needed"), `replans_ok` the successful ones.
Behaviour before fix 5: ~0.6 m/s in the open, hovers up to ~2 min in front of the first low wall, moves back and forth in the dynamic arenas; ~3 % of replans succeed, most fail with Gurobi "Infeasible or Unbounded" (SANDO then keeps its committed trajectory / hovers, as designed). `sando` uses ~7 cores.

**Role:** Optimization-based planner for dynamic unknown environments (MIQP with hard collision constraints, spatio-temporal safe corridors, tracked obstacles inflated by their reachable sets). Flown with **its own original sensors** (Livox Mid-360 + RealSense D435), not the 2 x OAK-D rig of the other methods: if KAIROS with two depth cameras matches it, SANDO had the better sensing. Report it as "SANDO (original sensors: Mid-360 + D435)".

**Source:** Kondo, Tordesillas, How et al., "SANDO: Safe Autonomous Trajectory Planning for Dynamic Unknown Environments", arXiv 2604.07599 (T-RO). Code: github.com/mit-acl/sando, tag v0.0.3 (93b2eed), in `ws/src/sando`, with the submodules DecompROS2 (a2bc42e), dynus_interfaces (f569702), acl-mapping (61b9cff), gazebo_ros_pkgs (ed1f7b8, only for `gazebo_msgs`), livox_laser_simulation_ros2 (044e83e, read for the Mid-360 model only).

## Sensors: SANDO's own

From SANDO's `urdf/quadrotor.urdf.xacro` and its simulation runner (`scripts/run_sim.py`):

| Sensor | SANDO | Here (`models/x500_mid360_d435_tw2`) |
|---|---|---|
| Livox Mid-360 | base_link (0, 0, 0.2), pitch 0.3 rad; 360 deg x -7.22 .. 55.22 deg; 0.001 .. 20 m; Gazebo Classic Livox plugin along the real scan pattern (`mid360.csv`), 2000 rays per ms | same pose and field of view; Gazebo Harmonic `gpu_lidar`, 720 x 28 rays at 10 Hz (201600 points/s; real Mid-360 200000); 0.1 .. 20 m. No Livox scan-pattern plugin exists for Gazebo Harmonic. Rays without a return: inf from `gpu_lidar`, set to 20 m as the Livox plugin does (`lidar_fill.py`) |
| RealSense D435 | base_link (0.05, 0, 0), level; depth 120 x 90, 85.2 deg, 0.1 .. 10 m, 30 Hz | same, `depth_camera` |
| Who uses what | static forest sim and hardware: Mid-360 -> ACL global mapper (`occupancy_grid`, `unknown_grid`); D435 -> obstacle tracker. Dynamic-obstacle sim with perception (`unknown_dynamic`): D435 for both | Mid-360 -> mapper, D435 -> tracker (SANDO's hardware setup); launch argument to switch the mapper to the D435 |

Everything else is the test's: airframe and motors of `x500_oakd_x2_60_tw2` (thrust-to-weight 2), PX4 SITL, PX4 odometry `/kairos/odom` for the state (no ground truth to the planner), `trajectory_server`, v_max 1.5, a_max 4, j_max 25, worlds, 24 passes, referee and `hits.py`.

## What is not SANDO's original setup

| Item | SANDO | Here, and why |
|---|---|---|
| ROS / simulator | ROS 2 Humble, Gazebo Classic (`gazebo_ros`) | ROS 2 Jazzy, Gazebo Harmonic (the test stack). Gazebo Classic build dependencies removed by a patch; `gazebo_msgs` built from SANDO's own `gazebo_ros_pkgs` submodule |
| Vehicle | `fake_sim` (perfect tracking: the model is set to the planned state) or a real PX4 drone | PX4 SITL x500 tw2 tracking SANDO's committed trajectory (`/trajs`, cubic pieces -> Hermite knots, exact) through `trajectory_server` |
| Timing | replans at 100 Hz (effective ~30-50 Hz) | its own (not forced to replan 0.2 s / start 0.4 s ahead); stated with the result |
| Obstacle knowledge | the paper's dynamic table gives ground-truth obstacle trajectories; `use_global_pc` true gives the known map | `use_global_pc: false`, perception only (mapper + tracker): no ground truth |
| Obstacle speed bound | `obst_max_vel` 0.5 m/s (`sando.yaml`, its dynamic simulations), 0.25 (`sando_hw_quadrotor.yaml`) | 0.5, SANDO's simulation value. The worlds' true bound (2.0 m/s) was tried first: SANDO inflates unknown space by `obst_max_vel` x trajectory time (`findSafeSubGoal`), which with a 20 x 20 m local map leaves no safe sub-goal (1-point global path, no plan in ~10000 replans, 2026-11-03). Its own simulations also have obstacles faster than this bound |
| Tuning | `sando.yaml` (simulation) and `sando_hw_quadrotor.yaml` (its real drone with Mid-360 + D435) | `sando_hw_quadrotor.yaml` and the mapper's `hw_global_mapper.yaml`, as the sensors are its hardware ones; map bounds / altitude / frames / sim mode kept as simulation (`SANDO_CONFIG=sando.yaml` for the other) |
| Limits, size | `sando.yaml`: v_max 5, a_max 20, j_max 100, drone_bbox 0.2 m, limits per axis (`Linf`); hardware drone_bbox 0.9 x 0.9 x 0.5 m | the test's v_max 1.5, a_max 4, j_max 25 as norms (`dynamic_constraint_type: L2`, SANDO's own option; per axis it flew up to 3.2 m/s); drone_bbox 0.9 x 0.9 x 0.7 m: the hardware width, height raised from its 0.5 m to 2 x the referee's 0.35 m. Tried: 0.7 m cube (11 static hits in 2 passes), 0.9 x 0.9 x 0.5 (5 hits, all grazes at 0.33-0.34 m, e.g. 0.25 m over a 1.5 m wall) |
| Yaw | SANDO's yaw solver (`goal.yaw`); a YAWING phase toward the goal before flying | `trajectory_server` yaws along the velocity, as for every method; `skip_initial_yawing: true` (SANDO's own option), else SANDO waits for a yaw nobody executes |
| Map bounds, altitude | `sando.yaml` +-200 m, z 0.5 .. 6 m | the scenario's x / y bounds, as NEO's arena walls: with +-200 m SANDO flew around the arenas outside the corridor (y -11 m, where nothing stands); z = the test's altitude band 1.0 .. 3.0 m (it flew at 0.7 m) |
| Visualization | `visual_level` 2 | 0 (RViz markers only; saves CPU) |

## Setup

| Step | State |
|---|---|
| Gurobi 11.0.3 in `~/opt/gurobi1103`, `libgurobi_c++.a` rebuilt with the system g++ (as SANDO's `setup.sh`) | done |
| Gurobi academic license (ID 2880601, `~/gurobi.lic`, expires 2027-11-02); test MIQP solved by `gurobi_cl` | done |
| nlohmann_json v3.11.3 built in `ws/src` (no sudo for `nlohmann-json3-dev`); OpenMP from gcc (libgomp), `libomp-dev` (clang's) not needed | done |
| `./build.sh`: deps symlinked into `ws/src`, Gazebo Classic packages ignored, `patches/*_jazzy.patch`, decomp_util first, system Python | done: 12 packages incl. `sando`, `obstacle_tracker_node`, `global_mapper_node` |
| Sensor check in Gazebo (`drone_model:=x500_mid360_d435_tw2`, `GZ_SIM_RESOURCE_PATH` with `models/`) | to do |
| `SANDO_planner/sando_bridge.py`: `/kairos/odom` -> `/NX01/state` + TF map -> NX01/base_link + sensor frames; `/trajs` (own id) -> `trajectory_server`; `/NX01/term_goal` per pass; referee and episodes as `../NEO_planner`. `launch/sando_test.launch.py`, `run.sh` | written |
| Smoke tests (`SANDO_smoke`, `SANDO_smoke2`, `SANDO_fill`), then 24 (`SANDO`) | smoke tests done; 24 running |

Jazzy patches (`patches/`): `-march=native` removed (with it Eigen's alignment differs from the system PCL of Ubuntu 24.04 and `pcl::VoxelGrid::filter` frees an invalid pointer: `obstacle_tracker_node` SIGSEGV on its first cloud, gdb 2026-11-03); empty-cloud guard after the tracker's z filter; the planner's floor filter drops voxels below z_min - drone_bbox_z / 2 + res instead of z_min + res (with z_min = the 1.0 m band it removed the lower half of 1.5 m walls, 2026-11-03); Gazebo Classic `find_package(gazebo / gazebo_ros)` and its three plugins removed; `ament_index_cpp` added to `fake_sim`; `dynamic_forest_node`, `analyze_benchmark`, `convert_bag` (SANDO's own simulated obstacles and bag tools, rosbag2 Humble API) not built; `tf2_*/*.h` and `cv_bridge.h` includes -> `.hpp`; global_mapper_ros: empty `MPI::MPI_C` target for Ubuntu 24.04's VTK. The planner, tracker and mapper code is unchanged.

## Fly

```bash
cd SANDO_planner && EPISODES=2 NAME=SANDO_smoke ./run.sh     # then EPISODES=24 NAME=SANDO
cd ~/kairos_test/20261102_compare && python3 hits.py SANDO
```

`MAPPER=d435` uses the D435 for the map too (SANDO's `unknown_dynamic` simulation setup).

Test rules: see `../README.md` and `../operate_collection.md` (the sensor exception of this package must be listed there).
