# SANDO (original sensors)

Status 2026-11-04: diagnosed with 19 instrumented single passes (E0-E18) and fixed without changing SANDO's algorithm (section "Diagnosis"). The main cause of SANDO's poor runs was a bug in this package's drone model: the Mid-360 sat inside the airframe and mapped the drone itself as an obstacle. Every run before that fix (`SANDO`, `SANDO_Linf`, `SANDO_invalid_unknown_sky`, `SANDO_v2`, passes E0-E10) is not a result. Official 24-pass run with the fixed setup: `SANDO_v3`, SR 1/24: SANDO reached B / A in the first 4 passes and then hovered without a feasible corridor for the rest of the run (it has no recovery). With a stack restart after such a deadlock (`SANDO_v3R`, `run_restart.sh`): reached 20/24 at 1.05 m/s, SR 0/24, 4.35 dynamic and 1.35 static contacts per reached pass (section "Results"). With `obst_max_vel` 1.0, the bound of the paper's simulations, instead of 0.5 (`SANDO_v3R_omv1`): reached 20/24 at 0.89 m/s, SR 0/24, 5.55 dynamic contacts per reached pass. The speed bound is not the cause: most remaining contacts come from obstacles that SANDO's lidar tracker saw late or not at all (section "What remains").

**Role:** Optimization-based planner for dynamic unknown environments (MIQP with hard collision constraints, spatio-temporal safe corridors, tracked obstacles inflated by their reachable sets). Flown with **its own sensors and perception pipeline**, not the 2 x OAK-D rig of the other methods: the Livox Mid-360 for the map and the dynamic-obstacle tracking, as in its hardware experiments. The RealSense D435 of its URDF is on the model but unused by default; SANDO uses it only in its perception-only simulation (see "Diagnosis"). If KAIROS with two depth cameras matches it, SANDO had the better sensing. Report it as "SANDO (own sensors: Livox Mid-360)".

**Source:** Kondo, Tordesillas, How et al., "SANDO: Safe Autonomous Trajectory Planning for Dynamic Unknown Environments", arXiv 2604.07599 (T-RO). Code: github.com/mit-acl/sando, tag v0.0.3 (93b2eed), in `ws/src/sando`, with the submodules DecompROS2 (a2bc42e), dynus_interfaces (f569702), acl-mapping (61b9cff), gazebo_ros_pkgs (ed1f7b8, only for `gazebo_msgs`), livox_laser_simulation_ros2 (044e83e, read for the Mid-360 model only).

## Diagnosis (2026-11-04)

Instrumented single passes A -> B (`runs/` and `logs/SANDO_diag_E0` .. `E18` next to the KAIROS run), each with SANDO's `debug_verbose` (per-replan outcome) and a probe recording the Gazebo pose, SANDO's setpoint stream, the reference the trajectory server tracks, the tracker output against the scenario's true obstacles, and the mapper's grids. Live captures of the lidar cloud and the occupancy grid during E10-E12 (`selfpts.py`, `selfvox.py` in the session scratchpad).

### Main cause: the Mid-360 mapped the drone (integration bug, fixed)

`models/x500_mid360_d435_tw2` gave the sensor poses relative to the model origin, but `x500_base` puts its `base_link` 0.24 m above the model origin (its `<pose>0 0 .24 0 0 0</pose>`; the KAIROS model's cameras account for it). The Mid-360 was therefore 0.04 m *below* base_link, inside the frame plate, instead of 0.2 m above it as in SANDO's URDF (the D435 0.24 m too low; unused).

- 2040 of the Mid-360's ~4230 returns per scan were the underside of the frame plate, 0.1-0.3 m away (20 scans, E10). The ACL mapper has no minimum range, so they were inserted as hits: every grid message sampled had 6-9 occupied voxels within 1 m of the drone, none of them near a real obstacle, moving with it.
- SANDO's corridors start at A, inside those voxels after the drone_radius inflation (0.45 m): 94-96 % of the replans were MIQP infeasible in every factor thread (E0-E10), whatever the time factor, the corridor size (E10: 3 polytopes of 3 m, `sando.yaml`) or the norm (E2). The HGP start was occupied ("Start is not free", debugging log 4), and the temporal grid saw the voxels move with the drone: false tracks within 1 m of the drone in 15 % of the replans (E8).
- **Fix.** The sensor poses are now relative to base_link (= SANDO's URDF), and the bridge's TF gives the same offsets from the odometry frame (`BASE_LINK_Z` 0.24). Mounted correctly, the lowest rays (-24 deg ahead) still graze the x500's 13" front propeller tips when the drone pitches forward (up to ~50 returns per scan, 0.4 m away, at base_link (0.31, +-0.2, 0.05)). `lidar_fill.py` drops returns inside the airframe's box (|x|, |y| <= 0.36 m, -0.25 <= z <= 0.12 m in base_link), as real drone lidar pipelines do: ~1 return per scan on average over a pass.
- **Effect** (one pass each, E8 -> E11 mount -> E12 + self filter): replans OK 2.6 -> 23.5 -> 24.1 %, pass time 441 -> 238 -> 225 s, mean speed 0.68 -> 1.0 -> 1.1 m/s. Phantom voxels near the drone 21 -> 9 % of the occupied ones and track messages on a true obstacle 65 -> 80 % (E8 -> E13). Replan success now depends on the obstacles, as designed: 31 % with no track within 3 m of the drone, 13 % with a true track there (E12).

### Tuning, re-checked with the fixed sensors

The tuning of the first diagnosis was made with the bug. Each part was flown again against SANDO's own hardware value (E13-E17, cumulative from E12 unless noted):

| Tuning (first diagnosis) | Reason given then | With the fixed sensors | Now |
|---|---|---|---|
| time-factor window 2.5 / `factor_final` 4.0 | MIQP feasible only at factors 2.2-2.4 | hw window 1.8 / 2.5: 188 s, replans OK 28.6 % (E13) against 225 s, 24.1 % (E12) | **hw value** |
| `clear_radius` 2.0 m | unknown cells below the drone cut the corridors | hw 0.8 m: 197 s, 27.2 % (E14) | **hw value** |
| `sando.yaml`'s heat map | the hw heat map (dynamic tube 3 m, `heat_Hmax` 20, static halo 2 m) flew 2 x the course | hw heat map: 408 m in 440 s against 220 m in 197 s (E15) | kept |
| tracker: `min_cluster_size` 3, velocity clamp 2.0, 3 observations, `miss_inc` -0.1 | the world's spheres are 3-8 voxels (hw minimum 7), faster than the 1.0 m/s clamp, and leave trails | hw tracker (E16): detection 3-6 m ahead 42 against 72 %, 6-10 m ahead 15 against 43 %, to the sides / behind 10-14 against 20-35 %; speed estimate 0.63 against 0.77 m/s (true 0.85) | kept |
| `obst_max_vel` 0.5 (`sando.yaml`; hw 0.25) | the worlds' bound 2.0 leaves no safe sub-goal | 1.0 (E17): 2 dynamic contacts against 4-7, but 306 s, replans OK 12.8 %, a 61 s stall. Over 24 passes (`SANDO_v3R_omv1` against `SANDO_v3R`): 5.55 against 4.35 dynamic contacts per reached pass, 0.89 against 1.05 m/s, reached 20/24 in both. 2.0 (E18): 331 s, 2 / 7 contacts, 85 % of the replans ended before the MIQP, 190 s of hover | 0.5 (SANDO's simulation default) |

### Other integration fixes (in place)

- **Two trackers.** The package started `obstacle_tracker_node` on the D435 next to SANDO's own tracker inside `global_mapper_ros` (temporal occupancy grid -> dynamic voxels -> AEKF, `obstacle_tracker.enabled`, paper Sec. VII). `onboard_sando.launch.py` starts `obstacle_tracker_node` only with `use_obstacle_tracker:=true` (default false). Both publish `predicted_trajs` with ids from 0 and SANDO keeps one trajectory per id (`addTraj`), so they overwrote each other. Now only SANDO's own pipeline runs (`tracker_sensor:=none`).
- **Flown trajectory = SANDO's plan.** The bridge executed `/trajs`, the trajectory SANDO shares with other agents. `appendToPlan()` drops a plan computed after the drone passed its start A (`plan_size < k_value`) but returns true, so the plan is still published: the drone flew a plan SANDO had discarded (E5: 2.6 m from SANDO's own setpoint, then a pole contact). Now the bridge executes SANDO's 100 Hz setpoint stream `goal`, what its hardware path streams to PX4: the tracked reference equals SANDO's setpoint within 3 mm (p95, E7).
- **SANDO's yaw.** The package flew the test's velocity yaw instead of SANDO's `goal.yaw`, which also carries SANDO's recovery (after `yaw_spinning_threshold` failed replans, 10 on hardware, it spins at 1 rad/s and the tilted Mid-360 re-scans around the drone). Now SANDO's yaw through `sando_trajectory_server.py`.
- SANDO pops its plan with a 100 Hz wall timer while its times are ROS (simulation) time; the real-time factor here is 0.97-0.99, so the plan runs at most 3 % fast.

### Paper conditions

SANDO's 100 % rows are simulations with perfect tracking (Assumption 4; the vehicle is set to the plan): the static forest (Table V; cylinders of radius 1.0-1.5 m, 5-20 % coverage, simulated Mid-360) and the dynamic benchmark with **ground-truth obstacle positions** (Table VI). Its perception-only simulation (Table VII; D435, the same dynamic worlds, obstacles up to 0.83 / 0.94 / 1.5 m/s per axis) reaches 80 / 60 / 40 % (easy / medium / hard, N 5, P 2), late detection being the named failure. Hardware (Sec. X): 12 of 12 dynamic flights collision-free, 1-5 obstacles of 0.3 x 2.0 m at ~0.5 m/s, replan failure 0.4-21.5 %, PX4 tracking error 0.21-0.57 m RMS. This world has 61 spheres of 0.5-0.9 m at 0.52-1.88 m/s peak (median 0.92) and 100 static obstacles over a 200 m course, perception only, PX4 tracking: SANDO's hard perception-only case or harder, and each pass is long.

### Platform and motors: not a limit

4 x 11.84 N for 2.17 kg (x500 base 2.06 kg + Mid-360 0.04 + D435 0.07; with the real 265 g / 72 g sensors 2.40 kg, as the KAIROS airframe's 2.41 kg): thrust-to-weight 2.2 (2.0). SANDO's a_max 4 m/s^2 needs a 22 deg tilt, PX4's default limit (45 deg) allows 9.8 m/s^2. PX4 tracks SANDO's setpoints within 0.16 m (p50) / 0.38 m (p95) while moving (E13), as SANDO's own hardware.

### Sensor field of view: blind zones

On base_link (0, 0, 0.2), pitched 0.3 rad nose-down (SANDO's URDF; the paper only says "mounted on top"), the Mid-360's -7.2 .. +55.2 deg vertical field of view becomes -24.4 .. +38 deg ahead, about -7 .. +52 deg to the sides and +10 .. +72 deg behind:
- **Below:** most unknown voxels near the drone are below it (95 % within 3 m in E3, measured with the mount bug). SANDO's corridors treat unknown as occupied (its global A* as free).
- **Behind:** nothing at or below the drone's altitude behind it. 3 of the 9 dynamic contacts of E11-E12 were never tracked in the 3 s before (obstacles from the side, behind or behind-below).

The D435 (0.05, 0, 0 on base_link), level, sees 85 deg ahead to 10 m. SANDO uses it only in its perception-only simulation because "the simulated LiDAR produces too few points on the small dynamic obstacles for reliable detection. This is a simulation artifact; in hardware experiments (Section X), the real LiDAR sensor provides sufficient point density" (IX-D); on hardware "dynamic-obstacle positions, AABBs, and velocities are estimated entirely from the onboard Livox Mid-360 LiDAR point cloud" (X). Here the 720 x 28 ray `gpu_lidar` gives enough points (detection above).

### What remains (SANDO's own limits)

- **Dynamic contacts.** 9 in E11-E12: 5 with obstacles tracked for the full 3 s before, moving 0.84-0.95 m/s against `obst_max_vel` 0.5 (the corridors inflate an obstacle's current position by `obst_max_vel` x time, not its predicted path); 3 never tracked (blind zones above); 1 tracked part of the time. 4 of the 9 happened within 0.2 s of a successful replan. Paper VIII-D: "faster obstacles can leave their inflated regions. Larger bounds shrink corridors and can make optimization infeasible": with 1.0 (E17) the drone was 1.6 x slower, replanned successfully half as often and stalled 61 s.

  Over the 24-pass series, from the Gazebo pose and the bridge's plan count (status line every ~10 s; `dynspeed.py` in the session scratchpad):

  | | 0.5 (`SANDO_v3R`) | 1.0 (`SANDO_v3R_omv1`) |
  |---|---|---|
  | Dynamic contacts | 91 | 122 |
  | Obstacle faster than the bound | 93 % | 15 % |
  | Drone nearly still (< 0.2 m/s) | 22 % | 43 % |
  | Median drone speed at contact | 1.21 m/s | 0.58 m/s |
  | Median closing speed, obstacle / drone | 0.47 / 0.37 m/s | 0.48 / 0.07 m/s |
  | No successful replan in the status interval of the contact | 1 | 17 |

  With 0.5, almost every obstacle outran its inflated region. With 1.0, 85 % of the contacts came from obstacles inside the bound, and most of them happened while SANDO replanned successfully (median 14 plans in the interval). SANDO planned around the obstacles it had; the contact came from one it tracked late or not at all. E17's 2 contacts are of that kind: one obstacle was first tracked 0.75 s before, at 0.43 m/s against a true 0.78 m/s, and the other was never tracked. The larger bound still shrinks the corridors. Passes took longer (264 against 219 s median), the drone hovered more often, and obstacles reached it: more contacts per pass. This is the paper's own perception-only failure ("late detection", Table VII).

  With 2.0 (E18), above every obstacle's speed, SANDO now flies; on 2026-11-03, with the mount bug, it found no plan. It hovered for 190 s of its 331 s pass, in 17 stops of more than 5 s, and 85 % of its replans ended before the MIQP (no global path or no safe sub-goal). 4 of its 7 dynamic contacts hit the hovering drone (setpoint 0). 3 of them came from one obstacle (1.48 m/s), at one hover point within 13 s, with no successful replan for 5-18 s before each; the tracker had not seen it in the 3 s before the first two. No value of the bound removes these contacts: the obstacles were not in SANDO's obstacle set, and with no feasible corridor SANDO hovers rather than evades.
- **Static contacts.** `drone_bbox` 0.9 m (radius 0.45) is 0.1 m over the referee's 0.35 m, and the voxels are 0.3 m: PX4's 0.16 m (p50) tracking error and the voxel offset can use it up, mostly at thin poles. A larger margin would be a safety extra for this method only (`../README.md`, section 5).
- **Deadlocks.** Every local plan ends at rest at its sub-goal E. When the replans before E fail, the drone stops at E and every later corridor must contain that hover point; SANDO has no recovery besides its yaw spin. Its corridors inflate each obstacle voxel by `drone_radius` (0.45 m) *per axis* of the segment's frame (DecompROS / FASTER), up to 0.78 m diagonally, and the frame changes with the new path's direction. Seen with the fixed sensors (`SANDO_v3`, diagnostic run `SANDO_diag_D1`, 8 passes, probe snapshots of the map while stalled):
  - `SANDO_v3` pass 5: hover at (78.9, 4.0, 1.9) after crossing the 1.5 m wall at x 78.3, 0.61 m from its top corner; no plan for 30 min, the rest of the run.
  - D1 pass 2: hover at (10.0, -3.95, 2.63), 0.31 m above and beside a 2.32 m pole top; 3 occupied voxels at (0, -0.55, -0.53) m from the drone (0.76 m away, inside the per-axis inflation), no unknown voxel within 2.3 m, 100 % of 11664 replans MIQP infeasible; freed after 131 s.
  - D1 pass 2, later: hover at (-89.4, 5.6, 2.45), 1.7 m from a wall and a pole, no track within 6 m; 1023 replans in 280 s, 60 % infeasible, 40 % without a solution after 263 ms (p50, against 9 ms); the follow-on did not move (1 m).
  - `SANDO_v3R_omv1` passes 8 and 17: hover at (-90.7, 5.75, 2.0), 0.66 m from the top edge of the 1.5 m wall at x -91.3 near pad A. Then at (53.85, 2.1, 2.59), just after crossing the 2.1 m wall at x 54.4, 0.63 m from its top edge.
  
  With 2 permanent deadlocks in the 14 fixed-sensor passes before the restart series (E11-E17, `SANDO_v3`, D1), and 2 in each 24-pass restart series, a 24-pass run is likely to end in one early on: `SANDO_v3` measures mostly when that happens.

### First diagnosis (E0-E10, with the mount bug)

Kept for the record; the numbers are with the drone mapped as an obstacle.
- 94-96 % of the replans MIQP infeasible; L1 instead of L2 the same (E2); 3 polytopes of 3 m (`sando.yaml`, E10) the same. Cause: the mount bug.
- Time-factor window and `clear_radius` raised (E3, E4): reverted with the fixed sensors (table above).
- Heat map (E5) and tracker (E6): still needed (table above).
- Stop-point deadlocks: 223 s at (44, -2.9, 2.7) in E7, 1041 s at (63, 2.7, 2.3), 0.6 m in front of a 1.8 m wall, from pass 16 of `SANDO_v2` (9 of its 24 passes). While moving, SANDO's start A lies ~0.4 s ahead on its plan, away from the voxels the drone leaves behind; once the drone hovered, A was the hover point among its own voxels, so no corridor could contain it again.

### Diagnostic passes

| Pass | Change | Outcome | Time [s] | Path [m] | x reached | Static / dynamic hits | Replans OK |
|---|---|---|---|---|---|---|---|
| E0 | as run `SANDO` (hw tuning, D435 tracker, `/trajs`, test yaw) | timeout | 548 | 232 | -0.5 | 0 / 4 | 3.4 % |
| E1 | SANDO's own tracking only | timeout | 548 | 266 | 29 | 1 / 7 | 4.5 % |
| E2 | L1 instead of L2 | stopped at 60 s | | | | | 4.0 % |
| E3 | E1 + factor window 2.5 / 4.0 | stopped at 228 s | | | -24 | 2 / 3 | 5.2 % |
| E4 | + `clear_radius` 2.0 | timeout | 548 | 406 | 99 | 5 / 4 | 5.8 % |
| E5 | + heat map of `sando.yaml` | reached | 359 | 275 | | 4 / 9 | 5.7 % |
| E6 | + tracker / map decay tuning | reached | 391 | 264 | | 2 / 6 | 3.5 % |
| E7 | + executes SANDO's setpoint stream | timeout (deadlock 223 s) | 548 | 246 | 44 | 2 / 9 | 1.2 % |
| E8 | + SANDO's yaw (= setup of `SANDO_v2`) | reached | 441 | 299 | | 1 / 10 | 2.6 % |
| E9 | E8 + `obst_max_vel` 1.0 | reached | 512 | 299 | | 6 / 8 | 4.2 % |
| E10 | E8 + `num_P` 3, `sfc_size` 3 m (`sando.yaml`) | reached | 392 | 249 | | 4 / 3 | 3.9 % |
| E11 | E8 + **sensors on base_link** | reached | 238 | 249 | | 1 / 5 | 23.5 % |
| E12 | + **self filter** | reached | 225 | 239 | | 0 / 4 | 24.1 % |
| E13 | + hw factor window 1.8 / 2.5 | reached | 188 | 224 | | 1 / 4 | 28.6 % |
| E14 | + hw `clear_radius` 0.8 (= setup of `SANDO_v3`) | reached | 197 | 220 | | 0 / 7 | 27.2 % |
| E15 | E14 + hw heat map | reached | 440 | 408 | | 1 / 4 | 26.0 % |
| E16 | E14 + hw tracker / map decay | reached | 193 | 219 | | 0 / 5 | 29.3 % |
| E17 | E14 + `obst_max_vel` 1.0 | reached | 306 | 230 | | 2 / 2 | 12.8 % |
| E18 | E14 + `obst_max_vel` 2.0 (the worlds' bound, above every obstacle's speed) | reached | 331 | 228 | | 2 / 7 | 15.0 % |
| D1 | E14 setup, 8 passes in sequence, stall snapshots | pass 1 reached (241 s, 0 / 4); pass 2 deadlocks (above), stopped in its follow-on | | | | | 12.2 % |

Hits of these passes are the referee's (radius 0.35 m); replans OK = successful replan callbacks (`computation_times`) while airborne. One pass each: hit differences of a few contacts are within pass-to-pass variation; the setup follows the causes found, not these counts.

## Results

`dyn_long200_smooth_021_clear2`, 24 passes, window open, values as `../README.md` defines them (`results.py`, `hits.py`).

| Metric | `SANDO_v3` (fixed sensors, `run.sh`) | `SANDO_v3R` (restart after a deadlock, `run_restart.sh`) | `SANDO_v3R_omv1` (as `SANDO_v3R`, `obst_max_vel` 1.0) | `SANDO_v2` (mount bug, superseded) |
|---|---|---|---|---|
| SR (reached, no collision) | 1/24 | 0/24 | 0/24 | 0/24 |
| CR (passes with a collision) | 23/24 (passes 6-24 are the follow-ons of pass 5, with its 2 static hits) | 22/24 | 24/24 | 24/24 |
| Reached | 4/24 (passes 1-4; deadlock in pass 5, 19 follow-ons of 70 s without moving) | 20/24 (deadlocks in passes 10 and 23, episodes 11 and 24 their follow-ons; restarted after 11) | 20/24 (deadlocks in passes 8 and 17, episodes 9 and 18 their follow-ons; restarted after each) | 14/24 (deadlock from pass 16: 9 timeouts) |
| Static hits (per 100 m) | 6 (0.51) | 31 (0.64) | 27 (0.53) | 65 (1.42) |
| Dynamic hits (per 100 m) | 9 (0.77) | 90 (1.85) | 121 (2.39) | 122 (2.67) |
| Distance | 1174 m | 4857 m | 5069 m | 4565 m |
| Mean speed | 0.43 m/s (passes 1-4: 913 m in 878 s, 1.04 m/s) | 0.81 m/s (the 20 reached passes: 4574 m in 4362 s, 1.05 m/s; 219 s median) | 0.74 m/s (the 20 reached passes: 4761 m in 5326 s, 0.89 m/s; 264 s median) | 0.57 m/s |
| Planning time, median / p90 | 56 / 104 ms | 56 / 104 ms | 56-60 / 112-120 ms (per run) | 92 / 164 ms |
| Replans OK | 8759 of 43775 callbacks (20 %) | 36800 of 257697 callbacks (14 %) | 20280 of 124419 callbacks in `_r1` and `_r3` (16 %) | 10219 of 449290 callbacks (2.3 %) |
| Closest approach to a static | -0.15 m | -0.03 m | -0.13 m | -0.24 m |

Pose-based counts (`hits.py`): `SANDO_v3` static 6, dynamic 9, 1178 m; `SANDO_v3R` (`_r1` + `_r2`) static 31, dynamic 91, 4868 m; `SANDO_v3R_omv1` (`_r1` + `_r2` up to the end of its pass 9 + `_r3`) static 27, dynamic 122, 5082 m; `SANDO_v2` static 65, dynamic 126, 4569 m. Planning time = SANDO's `total_replanning_ms` of the successful replans.

`SANDO_v3R`, the 20 reached passes: 1.35 static and 4.35 dynamic contacts per pass; 4 passes without a static contact, 1 without a dynamic one, none without both. The static contacts are grazes (closest pose-to-surface distance per contact p50 0.25 m, 1 of 31 inside an obstacle). Its 4 deadlocks (`SANDO_v3` pass 5, D1 pass 2, `SANDO_v3R` passes 10 and 23) were at two places: the 1.5 m wall at x 78.3 (hover 0.6 m from its top corner, and 0.6 m over its top) and the wall and 3.3 m pole near pad A (x -88 .. -89).

`SANDO_v3R_omv1`, the 20 reached passes: 1.20 static and 5.55 dynamic contacts per pass. 6 passes had no static contact, none had no dynamic contact. The static contacts are again grazes (p50 0.26-0.29 m, 3 of 27 inside an obstacle). Its deadlocks in passes 8 and 17 were in front of low walls again, at the 1.5 m wall near pad A and the 2.1 m wall at x 54.4 ("What remains").

In `_r2` the drone hovered from pass 17 on, staying within 0.3 m. Each follow-on still logged 10 m of hover jitter in 422 s, over the 5 m path limit of the first stop rule, so `_r2` ran 6 more follow-ons in the same deadlock. The rule now measures progress toward the goal (bridge `STUCK_M`). Those 6 episodes are cut (`results.py SANDO_v3R_omv1_r1 SANDO_v3R_omv1_r2:9 SANDO_v3R_omv1_r3`), and the remaining 6 passes were flown from a fresh stack (`DONE=18 K=2 ... run_restart.sh` -> `_r3`), as the rule now does. `SANDO_v3R` is unchanged by the new rule.

`SANDO_v3` passes 1-4: 220, 239, 200, 219 s, hits (static / dynamic) 1 / 3, 3 / 3, 0 / 0, 0 / 3. The restart series is a variant of the protocol for SANDO only (the other methods flew 24 passes in one run): a pass that ends in a permanent deadlock and its follow-on count as failed episodes, as in `run.sh`, but the next pass starts from a fresh stack at pad A instead of from the hover point.

## Sensors: SANDO's own

From SANDO's `urdf/quadrotor.urdf.xacro` and its simulation runner (`scripts/run_sim.py`):

| Sensor | SANDO | Here (`models/x500_mid360_d435_tw2`) |
|---|---|---|
| Livox Mid-360 | base_link (0, 0, 0.2), pitch 0.3 rad; 360 deg x -7.22 .. 55.22 deg; 0.001 .. 20 m; Gazebo Classic Livox plugin along the real scan pattern (`mid360.csv`), 2000 rays per ms | same pose on the x500's base_link (0.24 m above the model origin; until 2026-11-04 the pose was taken from the model origin, see "Diagnosis") and field of view; Gazebo Harmonic `gpu_lidar`, 720 x 28 rays at 10 Hz (201600 points/s; real Mid-360 200000); 0.1 .. 20 m. No Livox scan-pattern plugin exists for Gazebo Harmonic. Rays without a return: inf from `gpu_lidar`, set to 20 m as the Livox plugin does, and returns on the airframe (propeller tips) dropped (`lidar_fill.py`) |
| RealSense D435 | base_link (0.05, 0, 0), level; depth 120 x 90, 85.2 deg, 0.1 .. 10 m, 30 Hz | same, `depth_camera` |
| Who uses what | hardware: Mid-360 -> ACL global mapper (`occupancy_grid`, `unknown_grid`, and its temporal-grid tracker -> `predicted_trajs`). Perception-only simulation (`unknown_dynamic`): the D435 instead | Mid-360 -> mapper and its tracker (hardware). `tracker_sensor:=d435 / mid360` adds `obstacle_tracker_node` on that sensor (not SANDO's default; ids collide with the mapper's tracker), `MAPPER=d435` maps with the D435 |

Everything else is the test's: airframe and motors of `x500_oakd_x2_60_tw2` (thrust-to-weight 2), PX4 SITL, PX4 odometry `/kairos/odom` for the state (no ground truth to the planner), `trajectory_server` (with SANDO's yaw, below), v_max 1.5, a_max 4, j_max 25, worlds, 24 passes, referee and `hits.py`.

## What is not SANDO's original setup

| Item | SANDO | Here, and why |
|---|---|---|
| ROS / simulator | ROS 2 Humble, Gazebo Classic (`gazebo_ros`) | ROS 2 Jazzy, Gazebo Harmonic (the test stack). Gazebo Classic build dependencies removed by a patch; `gazebo_msgs` built from SANDO's own `gazebo_ros_pkgs` submodule |
| Lidar self returns | none (its Livox plugin; on hardware the airframe is outside the field of view) | the x500's 13" propeller tips reach the lowest rays: returns inside the airframe's box dropped (`lidar_fill.py`) |
| Vehicle | `fake_sim` (perfect tracking: the model is set to the planned state) or a real PX4 drone fed with SANDO's setpoint stream `goal` | PX4 SITL x500 tw2 fed with the same stream (`execute:=goal`): every 100 Hz setpoint (p, v, a, j) -> a 0.2 s Hermite trajectory -> `trajectory_server` (p + v + a feed-forward) |
| Yaw | SANDO's `goal.yaw` (velocity direction; spins after repeated failed replans); a YAWING phase before flying | SANDO's `goal.yaw` through `SANDO_planner/sando_trajectory_server.py` (dyn_expert's `trajectory_server` unchanged, only the yaw source); `skip_initial_yawing: true` (SANDO's option). `yaw:=test` = the test's velocity policy of the other methods |
| Timing | replans at 100 Hz (effective ~30-50 Hz) | its own (not forced to replan 0.2 s / start 0.4 s ahead); stated with the result |
| Obstacle knowledge | the paper's dynamic table gives ground-truth obstacle trajectories; `use_global_pc` true gives the known map | `use_global_pc: false`, perception only (mapper + its tracker): no ground truth |
| Obstacle speed bound | `obst_max_vel` 0.5 m/s (`sando.yaml`), 0.25 (`sando_hw_quadrotor.yaml`); paper simulations (1.0, 1.0, 1.5) per axis | 0.5. The worlds' bound (2.0 m/s) left no safe sub-goal on 2026-11-03, with the mount bug (unknown space inflated by `obst_max_vel` x trajectory time). 1.0 (E17, `obst_max_vel:=1.0`) was 1.6 x slower with fewer dynamic contacts in its one pass. Over 24 passes (`SANDO_v3R_omv1`) it was slower (0.89 against 1.05 m/s) with more dynamic contacts (5.55 against 4.35 per reached pass). With the fixed sensors 2.0 flies (E18), but the drone hovered for more than half of the pass |
| Tuning | `sando.yaml` (simulation) and `sando_hw_quadrotor.yaml` / `hw_global_mapper.yaml` (its real drone) | hardware files with `tuning:=test` (`SANDO_TUNING` / `MAPPER_TUNING` in the launch file): `sando.yaml`'s heat map; tracker cluster size / velocity clamp / observations and map decay ("Diagnosis"). `tuning:=hw` = the hardware files unchanged |
| Limits, size | `sando.yaml`: v_max 5, a_max 20, j_max 100, drone_bbox 0.2 m, limits per axis (`Linf`); hardware drone_bbox 0.9 x 0.9 x 0.5 m | the test's v_max 1.5, a_max 4, j_max 25 as norms (`dynamic_constraint_type: L2`, SANDO's own option; per axis it flew up to 3.2 m/s); drone_bbox 0.9 x 0.9 x 0.7 m: the hardware width, height 2 x the referee's 0.35 m |
| Map bounds, altitude | `sando.yaml` +-200 m, z 0.5 .. 6 m | the scenario's x / y bounds, as NEO's arena walls: with +-200 m SANDO flew around the arenas outside the corridor (y -11 m, where nothing stands); z = the test's altitude band 1.0 .. 3.0 m (it flew at 0.7 m) |
| Visualization | `visual_level` 2 | 0 (RViz markers only; saves CPU) |

## Setup

| Step | State |
|---|---|
| Gurobi 11.0.3 in `~/opt/gurobi1103`, `libgurobi_c++.a` rebuilt with the system g++ (as SANDO's `setup.sh`) | done |
| Gurobi academic license (ID 2880601, `~/gurobi.lic`, expires 2027-11-02); test MIQP solved by `gurobi_cl` | done |
| nlohmann_json v3.11.3 built in `ws/src` (no sudo for `nlohmann-json3-dev`); OpenMP from gcc (libgomp), `libomp-dev` (clang's) not needed | done |
| `./build.sh`: deps symlinked into `ws/src`, Gazebo Classic packages ignored, `patches/*_jazzy.patch`, decomp_util first, system Python | done: 12 packages incl. `sando`, `obstacle_tracker_node`, `global_mapper_node` |
| `SANDO_planner/sando_bridge.py`: `/kairos/odom` -> `/NX01/state` + TF map -> NX01/base_link + sensor frames; SANDO's `goal` stream (or `/trajs`) -> `trajectory_server`; `/NX01/term_goal` per pass; referee and episodes as `../NEO_planner`. `sando_trajectory_server.py`, `lidar_fill.py`, `launch/sando_test.launch.py`, `run.sh` | written |
| Diagnostic passes E0-E18, D1 | done (above) |
| 24 passes with the fixed setup (`SANDO_v3`) | done (section "Results") |
| 24 passes with restarts after a deadlock (`SANDO_v3R`, `run_restart.sh`, bridge `stop_if_stuck`) | done (section "Results") |
| The same with `obst_max_vel` 1.0 (`SANDO_v3R_omv1`) | done (section "Results") |

Jazzy patches (`patches/`): `-march=native` removed (with it Eigen's alignment differs from the system PCL of Ubuntu 24.04 and `pcl::VoxelGrid::filter` frees an invalid pointer: `obstacle_tracker_node` SIGSEGV on its first cloud, gdb 2026-11-03); empty-cloud guard after the tracker's z filter; the planner's floor filter drops voxels below z_min - drone_bbox_z / 2 + res instead of z_min + res (with z_min = the 1.0 m band it removed the lower half of 1.5 m walls, 2026-11-03); Gazebo Classic `find_package(gazebo / gazebo_ros)` and its three plugins removed; `ament_index_cpp` added to `fake_sim`; `dynamic_forest_node`, `analyze_benchmark`, `convert_bag` (SANDO's own simulated obstacles and bag tools, rosbag2 Humble API) not built; `tf2_*/*.h` and `cv_bridge.h` includes -> `.hpp`; global_mapper_ros: empty `MPI::MPI_C` target for Ubuntu 24.04's VTK. The planner, tracker and mapper code is unchanged.

## Fly

```bash
cd SANDO_planner && EPISODES=2 NAME=SANDO_smoke ./run.sh     # then EPISODES=24 NAME=SANDO_v3
cd ~/kairos_test/20261102_compare && python3 hits.py SANDO_v3
# variant: restart the stack after a permanent deadlock -> runs SANDO_v3R_r1, _r2, ...
cd SANDO_planner && EPISODES=24 NAME=SANDO_v3R ./run_restart.sh
# resume a series after its run _r2 with 18 episodes kept
cd SANDO_planner && DONE=18 K=2 EPISODES=24 NAME=SANDO_v3R_omv1 EXTRA="obst_max_vel:=1.0" ./run_restart.sh
```

Launch arguments for the variants (via `EXTRA="..."`): `tuning:=hw`, `yaw:=test`, `execute:=trajs`, `tracker_sensor:=d435`, `obst_max_vel:=...`, `sando_set:=name=value,...`, `mapper_set:=name=value,...`; `MAPPER=d435` maps with the D435; `SANDO_DEBUG=true` prints SANDO's per-replan outcome.

Test rules: see `../README.md` (the sensor exception of this package must be listed there).

## Debugging log (2026-11-03, before the diagnosis)

All of these runs had the mount bug of "Diagnosis".

1. `obst_max_vel` 2.0 (the worlds' bound): no plan, 1-point global paths (unknown space inflated by 2.0 x trajectory time) -> 0.5, SANDO's simulation value.
2. `obstacle_tracker_node` SIGSEGV in `pcl::VoxelGrid::filter` (gdb): `-march=native` breaks the Eigen ABI with the system PCL -> flag removed (also used by `sando`).
3. With +-200 m bounds SANDO flew around the arenas outside the corridor (y -11 m) at z 0.7 m -> scenario x / y bounds and the 1.0 .. 3.0 m band.
4. Tuning: `sando.yaml` -> "Start is not free" in almost every HGP call, no progress; `sando_hw_quadrotor.yaml` (default) flies. The occupied start was the drone's own voxels.
5. First 24-pass run (archived as `runs/SANDO_invalid_unknown_sky`, **invalid**): from pass 3 SANDO hovered at (49, 1.8, 1.2) for 35 min without a plan. Cause: Gazebo Harmonic's `gpu_lidar` publishes rays without a return as inf points (~73 % of the Mid-360 rays, the open sky); SANDO's Livox plugin publishes them at range_max (20 m), and the ACL mapper ray-traces free space along them. With inf points the air above the walls stayed unknown, and `findSafeSubGoal` cut every path climbing over a wall to its start point. -> `SANDO_planner/lidar_fill.py` between the gz bridge and the mapper restores the plugin's output (inf -> 20 m along the ray, intensity 0; ray directions checked against the returns, max error 0.03 deg).
6. Second 24-pass run (`runs/SANDO_Linf`; episodes 1-14 valid, reference only): SANDO's default `dynamic_constraint_type: Linf` bounds v / a / j **per axis**, so with v_max 1.5 it flew 1.42 m/s median, 2.23 p95, 3.16 max (KAIROS 1.57 p95 / 1.82 max), 5-12 static contacts per full pass. -> `dynamic_constraint_type: L2` (SANDO's own option, the test's rule for every method).
7. Same run, episodes 15-24: SANDO stopped replanning 1.25 m from goal B (status GOAL_SEEN, its last plan ends within goal_radius 0.35 of the goal, so `needReplan` returns false) while the drone hovered 1.25 m away. Probably the discarded-plan execution of "Diagnosis / Other integration fixes" (the drone flew `/trajs`, SANDO's plan ended at the goal).
8. Third 24-pass run (`runs/SANDO`, L2): all 24 episodes timeouts, one single A -> B attempt, 2 static / 11 dynamic hits, 420 m; 1164 of 40316 replan callbacks succeeded (2.9 %). From episode 6 SANDO hovered at (92.2, -3.1, 2.1), 9 m before goal B, for 14 min (a stop-point deadlock as in "First diagnosis").

Planning time: `summary.json` `plan_ms` = SANDO's `total_replanning_ms` of successful replans (median, p90); `stats.replans` counts replan callbacks (incl. "no replan needed"), `replans_ok` the successful ones.
