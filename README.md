# Comparison methods for KAIROS

One ROS 2 package per method, compared with KAIROS+MIGHTY. NEO is flown, Agile Autonomy is written and training; NavRL has a plan in its README; the others are empty.

| Package | Method | Role | Status |
|---|---|---|---|
| `NEO_planner` | NEO-Planner (PlannerNet + 2D MINCO), trained on the KAIROS dataset | External learning-based method: one network guess refined by an optimizer | 24-pass test done 2026-11-02: SR 0/24 (no pass without collision), reached 22/24 (2 timeouts), 2.41 hits per 100 m (static 0.06, dynamic 2.35), 0.67 m/s, plan median 15 ms |
| `AgileAutonomy_planner` | Agile Autonomy (Loquercio et al. 2021, PlaNet + polynomial), trained on the KAIROS dataset | External learning-based method closest to KAIROS: several trajectories with a predicted cost, imitation of a privileged expert | Package written 2026-11-02, model training, not flown |
| `MIGHTY_planner` | MIGHTY | Foundation of KAIROS: what does the network add over the optimizer it is built on? | Empty |
| `EGO_planner` | EGO-Planner | Classical gradient-based local planner from depth. | Empty |
| `NavRL_planner` | NavRL | External learning-based method: reinforcement learning for static and dynamic obstacles. | Plan: published checkpoint, no retraining; node not written |
| `SANDO_planner` | SANDO (Kondo et al. 2026, MIQP + spatio-temporal corridors), **original sensors** Mid-360 + D435 | Optimization-based planner for dynamic unknown environments; flown with its own (stronger) sensors | Flies in Gazebo after 4 fixes (see its README); slow and conservative; smoke test running |
| `Fast_planner` | Fast-Planner | Optional second classical planner; overlaps with EGO-Planner. | Empty |

## Test rules

Every method runs under the same conditions, or the difference is stated with the result. Full procedure and settings: `operate_collection.md`.

| Item | Value |
|---|---|
| Worlds | `dyn_long200_smooth_021_clear2`, `dyn_long200_mix_021_clear2`, plus one unused long200 world for the final numbers |
| Passes | 24, full A<->B |
| Drone | `x500_oakd_x2_60_tw2`, two OAK-D cameras |
| Inputs | Sensors and PX4 odometry only, no ground truth in the planner |
| Speed | `v_max` 1.5 m/s |
| Referee | The simulation scenario, collisions counted as in the KAIROS tests |
| Metrics | Hits per 100 m, reached passes, mean speed, planning time |
