# Obstacle prediction: KAIROS head 2 vs Xu et al. (ICRA 2023)

Status 2026-11-03: open-loop comparison done on both test_022 sets. Not a planner and not flown.

**Compared method:** Xu, Zhan, Chen, Xiu, Yang, Shimada, "A real-time dynamic obstacle tracking and mapping system for UAV navigation and collision avoidance with an RGB-D camera", ICRA 2023, arXiv 2209.08258. Code: github.com/Zhefan-Xu/onboard_detector (commit 4b4be99) and github.com/Zhefan-Xu/trajectory_planner (CERLAB-UAV-Autonomy).

**Question:** head 2 is a small MLP on the obstacle history. Does it predict better than the hand-built tracker + predictor of a published depth-camera system?

## What is compared

Every predictor gets the same sightings of the same slots (the J = 4 slots head 2 sees) and is scored against the same GT future `k_pred` over 0 to 4 s after t0, on the times where `k_pred_valid` holds. That is the same scoring as `kairos/metrics.py`.

| Name | Predictor | Source |
|---|---|---|
| `head2` | KAIROS-Net head 2, `runs/2nov_exc3/B/best.pt` | the checkpoint flown in the closed-loop test |
| `markov` | **their predictor**: KF state, then a Markov chain over a path library, environment term from the static map (paper Sec. III-D, Eqs. 10-11), most probable path | written from the paper; **not in their released code** |
| `markov_or` | oracle: the library path closest to the truth | upper bound of any path selection from that library |
| `kf_cv` | their tracker (KF [x, y, vx, vy]), linear extrapolation | what their released planner does: `bsplineTraj.h`, `predPos = pos + vel * t` |
| `ctrv` | constant turn rate and speed fitted to `obs_frames`, no learning | `kairos/metrics.py` |
| `cv_hist` | constant velocity fitted to `obs_hist` | `kairos/metrics.py` |

The `head2`, `ctrv` and `cv_hist` values reproduce `runs/2nov_exc3/B/eval_mix/metrics.json` exactly (ADE 0.853 / 1.484 / 2.348).

### Their tracker

From `onboard_detector`: `kalmanFilterMatrixVel` and `getKalmanObservationVel`, with the noise values of `cfg/detector_param.yaml` (e_p 0.25, e_q_pos 0.01, e_q_vel 0.05, e_r_pos 0.04, e_r_vel 0.3). The measured velocity is (current detection - estimate k frames back) / time. They use k = 10 at 30 Hz, which is 0.33 s. Our sightings come at about 15 Hz, so k = 5 keeps the same 0.33 s. The input is `obs_frames`, the camera-rate sightings.

### Their predictor

The paper gives no parameter values and the released code has no Markov chain, so these are our choices:

| Item | Paper | Here |
|---|---|---|
| Path library | l paths fitted to human walking data, left turn ... straight ... right turn; example l = 5 | l constant-speed arcs, yaw rate evenly spaced in [-w_max, w_max]; speed and heading from the KF |
| P_init, P_trans | discrete Gaussians centred on the straight path / on the diagonal | same, sigma 1.0 / 1.0 (in path indices) |
| P_env | Softmax of the distance to the first collision with the static map | same, over 4 s of path; **exact static geometry from `scenario_yaml`** instead of an onboard map |
| Update | P^{t+1} = (P^t P_trans) * P_env^{t+1}, every frame | same, from the first velocity estimate to the last sighting |
| Default | | l = 5, w_max = 0.4 rad/s (the largest turn rate of the smooth worlds) |

Two choices favour their method: the static map is exact, and the parameters are chosen with knowledge of our worlds. The sweep below goes further and picks the best of 36 settings (l 5 / 9, w_max 0.2 / 0.4 / 0.8, sigma_trans 0.5 / 1 / 2, environment on / off) **on the test data itself**.

## Data

| Set | Folders | Episodes | Events | Slots |
|---|---|---|---|---|
| mix | `~/dynamic_39_world_1nov_exc3/dyn_kairos2{,_fullDyn,_N}_test_022_mix` | 25 | 2999 | 8558 |
| smooth | `~/dynamic_39_world_28oct_smooth/dyn_kairos2{,_fullDyn,_N}_test_022_smooth` | 24 | 3002 | 9201 |

Test worlds, not used in training. All obstacles move at constant altitude, so their 2-D filter loses nothing in z. Smooth has only `sinusoid` obstacles. Mix has `constant`, `accel`, `turn` and `sinusoid`. The sightings are the simulator's, the same input for every method. Detection quality (their U-map / DBSCAN detector) is **not** part of this test.

## Results

ADE (m) over 0 to 4 s. "recent" = slots sighted within the last 1 s; the others are remembered obstacles. CI = 95 % bootstrap over episodes.

| Set, slots | head2 | markov | markov_or | kf_cv | ctrv | cv_hist | markov - head2 [CI] |
|---|---|---|---|---|---|---|---|
| mix, all | **0.85** | 1.90 | 1.41 | 1.91 | 1.48 | 2.35 | +1.04 [+0.92, +1.17] |
| mix, recent | **0.62** | 1.43 | 1.12 | 1.43 | 1.07 | 1.83 | +0.80 [+0.71, +0.90] |
| smooth, all | **0.81** | 2.26 | 1.44 | 2.31 | 1.45 | 2.72 | +1.45 [+1.35, +1.55] |
| smooth, recent | **0.39** | 1.48 | 1.07 | 1.52 | 0.94 | 1.95 | +1.09 [+1.05, +1.13] |

Error at 1 / 2 / 4 s (m):

| Set, slots | head2 | markov | kf_cv | ctrv |
|---|---|---|---|---|
| mix, recent | 0.28 / 0.60 / 1.35 | 0.63 / 1.38 / 3.13 | 0.63 / 1.39 / 3.15 | 0.40 / 1.00 / 2.52 |
| smooth, recent | 0.18 / 0.38 / 0.86 | 0.64 / 1.42 / 3.32 | 0.65 / 1.45 / 3.42 | 0.36 / 0.89 / 2.23 |

Per motion kind, mix (ADE, m):

| Kind | Slots | head2 | markov | markov_or | ctrv | markov - head2 [CI] |
|---|---|---|---|---|---|---|
| constant | 859 | 0.96 | 1.05 | 0.82 | 0.84 | +0.09 [-0.09, +0.25] |
| constant, recent | 705 | 0.73 | 0.90 | 0.77 | **0.70** | +0.17 [+0.04, +0.28] |
| accel | 1025 | 1.15 | 1.25 | 0.90 | 1.08 | +0.11 [-0.12, +0.31] |
| accel, recent | 738 | 0.74 | 0.88 | 0.65 | **0.70** | +0.14 [+0.08, +0.20] |
| turn | 1304 | 1.63 | 2.03 | 1.33 | 1.68 | +0.39 [+0.18, +0.61] |
| turn, recent | 1002 | 1.24 | 1.49 | 1.02 | **1.13** | +0.25 [+0.19, +0.31] |
| sinusoid | 5370 | **0.59** | 2.12 | 1.62 | 1.62 | +1.53 [+1.44, +1.64] |
| sinusoid, recent | 3932 | **0.43** | 1.61 | 1.30 | 1.19 | +1.18 [+1.13, +1.24] |

Sweep, best setting on the test data (optimistic for markov): mix 1.88 (l 9, w_max 0.4, sigma_trans 0.5), smooth 2.24 (l 5, w_max 0.4, sigma_trans 0.5). Both are within 0.03 m of the default. The environment term helps a little: the best setting without it is 1.90 / 2.31.

## Reading

1. **Head 2 beats their predictor in every group**, by 0.8 to 1.5 m ADE overall. The CI excludes zero everywhere except constant and accel when remembered slots are included. Tuning their parameters on the test data changes the result by less than 0.03 m.
2. **Their Markov chain is almost their KF line.** It picks the straight path 89 to 90 % of the time, and its ADE is within 0.05 m of `kf_cv`. In our worlds the obstacles move in open arenas with no walls in their way, so the environment term rarely prefers a turn. The chain has no term for the observed motion: a turning obstacle is predicted to turn only if a wall makes the straight path short.
3. **The library is not the limit, the selection is.** `markov_or`, the best library path chosen with the truth, beats head 2 on constant, accel and turn (mix, all slots). It does not beat head 2 on sinusoid. A better path *choice* from the observed history would close much of the gap. That is what `ctrv` does analytically.
4. **Where head 2 does not win:** on constant, accel and turn obstacles sighted within the last second, the non-learned `ctrv` is slightly better than head 2 (0.70 vs 0.73, 0.70 vs 0.74, 1.13 vs 1.24). Head 2's lead comes mostly from sinusoid obstacles (63 % of the mix slots, all of smooth) and from remembered obstacles. This is the distribution it was trained on: 39 `*_mix` worlds of the same generator. Their path library was built for pedestrians.

## Limits of this test

- Open loop. It says nothing about collisions; the closed-loop table in `../README.md` does.
- Same sightings for all methods. Their detector (U-map, map refinement, continuity filter) is not tested, and neither is the KAIROS tracker against it.
- Their predictor is our implementation of the paper, with our parameters; no reference implementation exists to check it against.
- Head 2 was trained on worlds from the same generator as the test worlds; their method had no training on them.

## Run

```bash
cd ~/comparision_for_kairos/prediction_compare
python3 compare_pred.py --set mix --sweep        # CPU only, a few minutes
python3 compare_pred.py --set smooth --sweep
```

Results: `results/pred_compare_{mix,smooth}.json` (all groups, 1 / 2 / 4 s errors, bootstrap CIs, sweep). Run it when no Gazebo test is running (`../README.md`, section 5).
