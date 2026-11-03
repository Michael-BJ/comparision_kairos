# NEO-Planner

Status: model trained (`~/saved_net/neo_kairos/run_20261102_161626/best.pth`, KAIROS training configuration, best val MSE 0.0870 at epoch 45 of 80); 24-pass test done 2026-11-02, results below.

**Role:** Learning-based baseline with its own planner: one network guess refined by an optimizer, no obstacle motion model.

**Source:** NEO-Planner (IROS 2025, github.com/Amos-Chen98/neo-planner), through our ROS 2 port `~/neo_planner_ws/src/neo_planner` (`min_jerk_planner.py` and `esdf_ros2.py` are unchanged copies) and the trainer port `~/Training_model_FM_planner/nn_trainer.py`.

## What is NEO here, and what is not

| Part | Same as NEO | Changed for the KAIROS test, and why |
|---|---|---|
| Network | `PlannerNet`: ResNet18 (pretrained, frozen, conv1 1-channel, fc 24) + motion MLP + fusion MLP -> 9 | Image 128 x 400 (rig panorama) instead of 640 x 480 |
| Input | depth / max depth x 255 as uint8; 24 motion values (body velocity, R, init and target relative to the drone) | Depth = KAIROS range panorama of both OAK-D cameras (`dyn_expert` `CameraRig.panorama`), as in the dataset |
| Output | 2 waypoints (body frame) + 3 durations of a 3-piece MINCO trajectory | - |
| Training | MSELoss (NEO's loss) | KAIROS dataset (`~/dynamic_39_world_1nov_exc3`), same train / val files as `runs/2nov_exc3/B` (`split.json`). Label = the KAIROS expert's best mode resampled at T/3 and 2T/3, ts = T/3 (NEO's expert stores its own MINCO waypoints). Training configuration read from B's `config.yaml`, see below |
| Planner | NN guess -> `warm_start_plan` (2D MINCO, L-BFGS, ESDF of octomap's projected map), hard clearance check, seed loop, adaptive safe_dis, local target with lateral nudge, escape, collision and look-ahead guards, v_max time scaling | - |
| Map | octomap_server from depth points, band cruise z -0.7 .. +1.0, ESDF of the projected map | Points of both cameras, range 0.4-6.0 m, every `camera.map_stride` (9) pixel, resolution 0.1 (`grid.resolution`; NEO 0.15). Free rays as the KAIROS map (`camera.invalid_is_free`): no return or beyond 6 m clears along the ray; NEO dropped those pixels and moving spheres left trails that closed the arenas |
| Arena | virtual walls at the arena bounds (`nn_inference_node` v2.2 `arena_*`) | the scenario's bounds; the grid is grown to them, else the walls exist only where something was seen (the drone left the corridor to y = -18 m without) |
| Safety margins | ESDF distances from the inflated obstacles: inflation 0.25, hard 0.35, soft safe_dis 0.7, emergency guard 0.4 (centre to surface 0.60 / 0.95 m) | KAIROS's: inflation = drone_radius 0.35, hard = static_clearance - drone_radius = 0.20, soft 0.40, emergency 0.25 (centre to surface 0.55 / 0.75 m). With NEO's values no plan passed the 1.9-2.3 m low-wall gaps (0 of 6 offline on the recorded map, stuck at the first wall in Gazebo) |
| Altitude | fixed cruise z | the scenario's route.z (2.0 m) |
| Sensors, state | - | `x500_oakd_x2_60_tw2` rig, PX4 odometry `/kairos/odom`, sim time |
| Execution | - | `trajectory_server` (HermiteTrajectory), as KAIROS; NEO sent MAVROS setpoints itself and set PX4 parameters |
| Timing | - | replan 0.2 s, plan starts 0.4 s ahead, local target 4 m with zero velocity: the dataset's replan_dt, foresee_dt, target_dist and target velocity (NEO: 1.0 s, 1.0 s, 5 m, 0.8 v_max) |
| Mission, referee | - | as kairos_planner: pad A <-> B, reached within 0.6 m, timeout 4.0 x distance / 1.5 + 10 s, collisions counted (Gazebo pose, never the planner) |

## Files

| File | What |
|---|---|
| `NEO_planner/neo_data.py` | KAIROS `.h5` event -> NEO sample; NEO's depth normalization and motion vector |
| `NEO_planner/planner_net.py` | `PlannerNet` with the panorama size |
| `NEO_planner/train.py` | trainer |
| `NEO_planner/neo_node.py` | inference node |
| `launch/neo_test.launch.py` | bridge + trajectory_server + odom_estimate + octomap_server + neo_node |
| `run.sh` | closed-loop test, like `~/kairos_test/20261102_compare/run.sh` |
| `run_regression.sh` | the other baseline: KAIROS network trained with `train.mode=regression` (`runs/2nov_exc3/NEO`), through kairos_planner. Not NEO |

## Train

```bash
cd NEO_planner && /usr/bin/python3 -u train.py --out_dir ~/saved_net/neo_kairos      # --kairos_run ~/kairosV3/runs/2nov_exc3/B
```

The training configuration is read from the KAIROS run's `config.yaml` (written to `train_config.json` of the NEO run):

| Item | KAIROS B = NEO here | NEO's own trainer |
|---|---|---|
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 | Adam, lr 1e-3 |
| Schedule | per step: 3 epochs linear warm-up, cosine down to 0.05 lr | constant |
| Epochs, batch | 80, 64, new random order every epoch | 5, 2, no shuffle |
| Gradient clip | 1.0 | - |
| Seed | 0 | 42 |
| Augmentation | mirror p 0.5; depth noise 0.004 d^2, random holes up to 3 %, 3 rectangles up to 12 x 20, cut at 8 m (`kairos.data.augment_depth` itself, in meters, before NEO's normalization); drone velocity + N(0, 0.03) | mirror option |
| Validation, best | every 5 epochs and the last; best.pth = lowest validation MSE | test loss once at the end |

KAIROS's obstacle augmentation (`data.obs_aug`) has no counterpart: NEO has no obstacle inputs. The loss stays NEO's MSE on its 9 outputs. The first run caches the samples in `~/saved_net/neo_kairos/cache_B_*_mm.npz` (depth in mm, as stored).

## Fly

```bash
MODEL=~/saved_net/neo_kairos/run_<stamp>/best.pth EPISODES=24 ./run.sh
cd ~/kairos_test/20261102_compare && python3 hits.py NEO
```

Results: `~/kairos_test/20261102_compare/logs/NEO`, `runs/NEO/<stamp>_<world>_neo/summary.json`, `odom_NEO.npz`.

- NEO plans in 2D at 2.0 m. In `dyn_long200_smooth_021_clear2` every static obstacle stands on the ground; low walls with a top above 1.3 m are in the octomap band, so NEO goes around them where the KAIROS expert may climb over.
- NEO has no model of obstacle motion; spheres enter its map as occupied cells.

## Result, 2026-11-02 (24 passes, `dyn_long200_smooth_021_clear2`)

| | NEO (24 passes) | KAIROS+MIGHTY (24 passes, `KAIROS_MIGHTY_24`, 2026-11-02/03) |
|---|---|---|
| Reached | 22 of 24; 2 timeouts (ep 1, 22). Ep 2 and 23 (12-13 m) only finished the timed-out pass, so 20 of 22 full passes | 24 of 24 |
| Success rate (reached, no collision) | 0 of 24 (0 %): every full pass had 2-11 collisions | 22 of 24 (91.7 %): passes 4 and 16 had 1 static hit each |
| Distance, time (`hits.py`, Gazebo pose) | 6212 m, 156.2 min | 5074 m, 59.0 min |
| Hits static / dynamic | 4 / 146 (node referee: 4 / 142) | 2 / 0 |
| Hits per 100 m | 2.41 (static 0.06, dynamic 2.35) | 0.04 (static 0.04, dynamic 0) |
| Path per full pass (straight 201.6 m) | median 282 m | mean 211 m |
| Mean speed | 0.67 m/s (0.68 on full reached passes) | 1.45 m/s |
| Planning time per replan, median / p90 | 15 / 58 ms (network 3.2 ms) | 223 / 280 ms per event |

NEO node counters: 32352 replans, 361 failed, 827 late (2.6 %), 409 escapes, 132 collision-guard and 35 look-ahead stops. Logs `~/kairos_test/20261102_compare/logs/NEO`, run `runs/NEO/20261102_175152_dyn_long200_smooth_021_clear2_neo`. The smoke test (`NEO_smoke`, 2 passes) is not part of the result. KAIROS's earlier 12-pass run (`KAIROS_MIGHTY`: 12/12 reached, 1 static hit) is superseded by the 24-pass run.

Test rules: see `../README.md`.
