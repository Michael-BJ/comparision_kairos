# Agile Autonomy

Status: package written 2026-11-02; model training (`~/saved_net/aa_kairos`); not flown.

**Role:** Learning-based baseline closest to KAIROS's idea: a network that outputs several trajectories with a predicted cost each, trained by imitation of a privileged expert. No obstacle motion model, no map at deployment.

**Source:** Loquercio et al., "Learning High-Speed Flight in the Wild", Science Robotics 2021. Code: github.com/uzh-rpg/agile_autonomy (commit 958c0d2, TensorFlow, ROS 1 + Flightmare). Ported here to PyTorch / ROS 2: `PlaNet` (`planner_learning/.../nets.py`), the losses (`utils.py`), the data layout (`data_loader.py`) and the deployment (`PlannerBase.py`, `agile_autonomy.cpp`, `trajectory_ext.cpp`).

## What is Agile Autonomy here, and what is not

| Part | Same as Agile Autonomy | Changed for the KAIROS test, and why |
|---|---|---|
| Network | `PlaNet`: MobileNet v1 (ImageNet weights of `keras.applications`, trainable) -> Conv1D 128 -> Conv1D 128/64/64/32 -> modes; state Conv1D 64/32/32/32 -> modes; plan Conv1D 64/128/128 -> 31 per mode; 3 modes; LeakyReLU slopes as the original | Image 128 x 400 (rig panorama) instead of 224 x 224: MobileNet gives 4 x 12 cells instead of 7 x 7 |
| Depth input | depth in mm, capped at 20 m, / 80, 3 channels, `mobilenet.preprocess_input` | The KAIROS range panorama of both OAK-D cameras (`CameraRig.panorama`), as in the dataset; no resize |
| State input | attitude R (9), velocity (3), goal direction (3); no position (`inputs.position false`) | No body rates (the dataset has none). Frame: the yaw frame of the KAIROS dataset (Agile Autonomy: full camera attitude). Velocity: the plan start's, see Timing. Goal direction: to the episode goal; Agile Autonomy's points 5 s ahead on a privileged global reference, not available in a test run |
| Output | per mode: alpha (predicted cost) + 10 positions (x_1..x_10, y_1..y_10, z_1..z_10) relative to the start | Points every 0.35 s (3.5 s, the median duration of the expert's best mode) instead of 0.1 s (1 s at 7 m/s); the path is resampled to the flight speed at deployment anyway |
| Labels | top 3 expert trajectories, copies of the best if fewer | The KAIROS expert's modes (`mode_z`, mode 0 = lowest cost) instead of its Metropolis-Hastings sampler |
| Loss | `MixtureSpaceLoss` (relaxed winner-takes-all, eps 0.05, 10 x MSE) + `TrajectoryCostLoss` (2 x MSE of alpha to the collision cost of the predicted trajectory in the privileged world) | The privileged world is the episode's scenario (exact boxes, cylinders, and the dynamic spheres at each point's time) instead of a KD-tree on a static point cloud. Cost size / threshold = the test's drone radius 0.35 / 0.85 (Agile Autonomy 0.3 / 0.8); a point inside an obstacle counts as distance 0 |
| Training | - | KAIROS dataset, same train / val files as `runs/2nov_exc3/B`; KAIROS training configuration (table below) |
| Deployment | modes sorted by \|alpha\|; the best and any with best/alpha > 0.95 are candidates; the one closest to the previous reference wins; first point = start state; 5th-order polynomial with continuity 3 fitted to the points; resampled at constant arc-length speed min(\|v\| + 1, v_max) and fitted again; z clamped to the altitude band | Previous reference compared at the same times. The start-up nudge (+0.2 m/s on world x below 0.1 m/s) points to the goal |
| Safety | none besides the network (no map, no checks) | - |
| Limits | the polynomial is not limited; Agile Autonomy's MPC tracked it within the drone's limits | Tracked by `trajectory_server` + PX4. Peak acceleration of each plan is logged (`plan_acc_max` in `summary.json`); offline it reached ~9 m/s^2 when a plan starts far below the target speed |
| Goal | stops by progress on the reference | Within 2 m of the pad: rest-to-rest quintic to the pad (the network has no goal distance input) |
| Sensors, state | - | `x500_oakd_x2_60_tw2` rig, PX4 odometry `/kairos/odom`, sim time |
| Timing | network at 15 Hz, plan from the current state | replan 0.2 s, plan starts 0.4 s ahead from the state of the trajectory in flight (the dataset's replan_dt and foresee_dt) |
| Execution | MPC on the quadrotor | `trajectory_server` (HermiteTrajectory, exact for the polynomial), as KAIROS |
| Mission, referee | - | as kairos_planner: pad A <-> B, reached within 0.6 m, timeout 4.0 x distance / 1.5 + 10 s, collisions counted from the Gazebo pose |

## Files

| File | What |
|---|---|
| `AgileAutonomy_planner/aa_data.py` | KAIROS `.h5` event -> sample (panorama, state, top-3 labels, privileged geometry) |
| `AgileAutonomy_planner/aa_net.py` | `PlaNet` and MobileNet v1 with the Keras ImageNet weights |
| `AgileAutonomy_planner/aa_loss.py` | `MixtureSpaceLoss`, `TrajectoryCostLoss` |
| `AgileAutonomy_planner/train.py` | trainer |
| `AgileAutonomy_planner/aa_node.py` | inference node |
| `launch/aa_test.launch.py` | bridge + trajectory_server + odom_estimate + aa_node |
| `run.sh` | closed-loop test, like `../NEO_planner/run.sh` |

## Train

```bash
cd AgileAutonomy_planner && /usr/bin/python3 -u train.py --out_dir ~/saved_net/aa_kairos      # --kairos_run ~/kairosV3/runs/2nov_exc3/B
```

Needs `~/saved_net/aa_kairos/mobilenet_1_0_224_tf_no_top.h5` (keras-applications, sha256 9f9cfd8c...). The first run caches the samples in `~/saved_net/aa_kairos/cache_B_*.npz`.

| Item | KAIROS B = Agile Autonomy here | Agile Autonomy's own trainer |
|---|---|---|
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 | Adam, lr 1e-3 |
| Schedule | per step: 3 epochs linear warm-up, cosine down to 0.05 lr | cosine decay with restarts (50000 steps, x1.5, x0.75, min 0.01) |
| Epochs, batch | 80, 64 | 150, 8 |
| Gradient clip | global norm 1.0 | each gradient to norm 1 |
| Seed | 0 | - |
| Augmentation | mirror p 0.5 (image, state, labels, geometry); depth noise 0.004 d^2, holes, cut at 8 m (`kairos.data.augment_depth`); start velocity + N(0, 0.03) | none |
| Validation, best | every 5 epochs and the last; best.pth = lowest validation space + cost loss | every epoch; checkpoint at the lowest validation loss |

## Fly

```bash
MODEL=~/saved_net/aa_kairos/run_<stamp>/best.pth EPISODES=24 ./run.sh
cd ~/kairos_test/20261102_compare && python3 hits.py AGILE
```

Results: `~/kairos_test/20261102_compare/logs/AGILE`, `runs/AGILE/<stamp>_<world>_agile_autonomy/summary.json`, `odom_AGILE.npz`.

Test rules: see `../README.md`.
