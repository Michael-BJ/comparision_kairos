# NavRL

Status: plan only (2026-11-02). Repository read (github.com/Zhefan-Xu/NavRL, commit 3725bcc, 2025-07-03); no node yet.

**Role:** External learning-based method: reinforcement learning (PPO) for static and dynamic obstacles.

**Source:** Paper: https://arxiv.org/abs/2409.15634. Code: https://github.com/Zhefan-Xu/NavRL

## What NavRL is

| Part | NavRL | File |
|---|---|---|
| Policy | PPO actor (Beta distribution), mean action at deployment; output = velocity in the goal frame, per axis in [-vel_limit, vel_limit] | `ros2/navigation_runner/scripts/ppo.py`, `ckpts/navrl_checkpoint.pt` |
| Input 1, static | "lidar" ray cast in its own occupancy map: 36 x 4 rays (10 deg horizontal over 360 deg, vertical -10 .. 20 deg), range 4 m, stored as 4 - distance | `map_manager` (occupancy map from one depth camera or a point cloud, resolution 0.1) |
| Input 2, dynamic | 5 nearest dynamic obstacles within 4 m: direction, 2D distance, z distance, velocity (goal frame), width class, height class | `onboard_detector` (DBSCAN on depth + optional YOLO on colour, Kalman tracking, one camera) |
| Input 3, state | goal direction (goal frame), 2D and z distance to the goal, velocity (goal frame) | `navigation.py` |
| Safety shield | velocity obstacles on the tracked obstacles and the ray points, horizon 2 s, safety_distance 0.3 | `safe_action_node.cpp` |
| Rate | 20 Hz velocity command | `navigation.py` timers 0.05 s |
| Training | Isaac Sim 2023.1.0-hotfix.1 + Orbit + OmniDrones, 40 x 40 m maps of pillars (width 0.4-1.1 m) and moving cuboids / cylinders (0.5-1.5 m/s), robot radius 0.3, action_limit 2 m/s | `isaac-training/` |

The ROS 2 package is set up for a Unitree Go2 (ground robot: `height_control` false, shield height 0.7-1.3 m); the drone setup is in `ros1/`.

## Retrain or not

**Use the published checkpoint, no retraining.**

- The policy never sees images: its inputs are map rays and obstacle states, made by NavRL's own modules. Our two OAK-D cameras only feed those modules, so the camera rig does not require a new policy. NavRL's own claim is zero-shot transfer from its training scene.
- NavRL is trained by RL in Isaac Sim. The KAIROS dataset (expert demonstrations in `.h5`) cannot train it without changing the method; a behaviour-cloned NavRL would no longer be NavRL.
- Retraining in our worlds means rebuilding them in Isaac Sim 2023.1.0-hotfix.1 (pinned version, not installed; its support of the RTX 5060 Ti / Blackwell is doubtful) and days of PPO. That result would be "NavRL retrained by us", not the published method.
- Rule 5 of `../README.md` is met by stating the data: published checkpoint, NavRL's Isaac training scene.

Out of distribution, stated with the result: spheres 0.8-2.0 m/s (training 0.5-1.5), low walls with 1.7-2.3 m gaps and a 12.9 m corridor (training: free-standing pillars).

## What must be adapted (interface, not tuning)

| Item | NavRL default | Here | Why |
|---|---|---|---|
| Speed | vel_limit 1.0 | 1.5, and the norm clipped to 1.5 (NavRL limits each axis) | v_max of the test; within its training limit 2.0 |
| Sensors | one depth camera | both rig cameras into `map_manager` (point cloud input) and the detector | same sensors as KAIROS |
| Detector | DBSCAN + YOLO (people), `constrain_size` to 0.5 x 0.5 x 1.5 m | DBSCAN only (no colour images, YOLO knows no spheres); size constraint off | Go2 / human setting, not a parameter of the policy |
| State | its own odometry | `/kairos/odom`, sim time | test rule |
| Altitude | goal z = current z | goal at route z 2.0, `height_control` on (drone) | as the drone setup |
| Execution | velocity command to the robot | velocity -> short Hermite segment to `trajectory_server`, replan 0.2 s, start 0.4 s ahead | test rule; slower than its 20 Hz, stated |
| Shield | on | on (published method); agent size 0.35 | collision radius of the test; the policy keeps its trained 0.3 |

Internal parameters (map, detector thresholds, shield horizon) stay at NavRL's values.

## Work

1. Build `map_manager`, `onboard_detector`, `navigation_runner` (C++, Humble) for Jazzy; install `torchrl` / `tensordict` (the repository's pinned copies) for `/usr/bin/python3`.
2. Run the policy offline on recorded rays and obstacle states, compare with `quick-demos` outputs.
3. Node + launch + `run.sh` like `../NEO_planner`; smoke test 2 passes, then 24.

Test rules: see `../README.md`.
