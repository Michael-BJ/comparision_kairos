"""Agile Autonomy's losses (planner_learning/src/PlannerLearning/models/utils.py) in PyTorch.

MixtureSpaceLoss: relaxed winner-takes-all over the modes. For every expert trajectory the closest mode
    (10 x MSE) takes (1 - eps) of its loss; modes that win no expert trajectory get eps / (modes - 1) of their
    loss to the first expert trajectory; eps 0.05; mean over modes and batch.
TrajectoryCostLoss: 2 x MSE between each mode's alpha and the collision cost of that mode's predicted
    trajectory in the privileged world (no gradient through the cost). Per point, d = distance to the nearest
    obstacle surface: 4 - 2 d^2 / size^2 below the drone size, 2 (size - d) / (thr - size) + 2 up to the
    threshold, 0 beyond; averaged over the 10 points.
Changed: the world is the scenario's exact geometry (boxes, cylinders, and the dynamic spheres at each point's
time) instead of a KD-tree on the static point cloud, and a point inside an obstacle has d = 0 (the point cloud
only has surfaces). Size and threshold follow the test's drone radius (Agile Autonomy: 0.3 and 0.8).
"""
import torch

from aa_data import OUT_SEQ_LEN, STATE_DIM


def mixture_space_loss(y_true, y_pred, eps=0.05):
    """y_true (B, K, 30) expert trajectories, y_pred (B, M, 31) network output."""
    pred = y_pred[:, :, 1:]
    M = pred.shape[1]
    mode_losses = 10.0 * ((y_true[:, :, None, :] - pred[:, None, :, :]) ** 2).mean(-1)    # (B, K, M)
    win = torch.nn.functional.one_hot(mode_losses.argmin(-1), M).float()                # (B, K, M)
    loss_matrix = (win * mode_losses * (1 - eps)).sum(1)                                 # (B, M)
    if M > 1:
        relaxed = (loss_matrix <= 0).float() * mode_losses[:, 0, :] * eps / (M - 1.0)
        loss_matrix = loss_matrix + relaxed
    return loss_matrix.mean()


def trajectories(y_pred):
    """(B, M, 31) -> points (B, M, 10, 3) relative to the plan start."""
    B, M = y_pred.shape[:2]
    return y_pred[:, :, 1:].reshape(B, M, STATE_DIM, OUT_SEQ_LEN).transpose(2, 3)


def obstacle_distance(p, boxes, cyls, spheres, radii):
    """Distance from points p (B, M, T, 3) to the nearest obstacle surface, negative inside.
    boxes (B, NB, 7) [c, half, yaw], cyls (B, NC, 5) [cx, cy, r, z_lo, z_hi], spheres (B, NS, T, 3), radii (B, NS)."""
    q = p[:, :, :, None, :] - boxes[:, None, None, :, 0:3]                               # (B, M, T, NB, 3)
    c, s = torch.cos(boxes[..., 6]), torch.sin(boxes[..., 6])
    c, s = c[:, None, None], s[:, None, None]
    ql = torch.stack([c * q[..., 0] + s * q[..., 1], -s * q[..., 0] + c * q[..., 1], q[..., 2]], -1)
    e = ql.abs() - boxes[:, None, None, :, 3:6]
    d = (e.clamp(min=0).norm(dim=-1) + e.max(-1).values.clamp(max=0)).min(-1).values    # (B, M, T)
    if cyls.shape[1]:
        dxy = (p[:, :, :, None, :2] - cyls[:, None, None, :, :2]).norm(dim=-1) - cyls[:, None, None, :, 2]
        dz = torch.maximum(p[:, :, :, None, 2] - cyls[:, None, None, :, 4],
                           cyls[:, None, None, :, 3] - p[:, :, :, None, 2]).clamp(min=0)
        dc = torch.where((dxy < 0) & (dz == 0), dxy, torch.sqrt(dxy.clamp(min=0) ** 2 + dz ** 2))
        d = torch.minimum(d, dc.min(-1).values)
    if spheres.shape[1]:
        ds = (p[:, :, None] - spheres[:, None]).norm(dim=-1) - radii[:, None, :, None]   # (B, M, NS, T)
        d = torch.minimum(d, ds.min(2).values)
    return d


def collision_cost(points, geo, size=0.35, thr=0.85):
    """Agile Autonomy's _compute_single_traj_cost for every mode: (B, M)."""
    d = obstacle_distance(points, *geo).clamp(min=0.0)
    cost = torch.where(d < size, 4.0 - 2.0 / size ** 2 * d ** 2,
                       torch.where(d < thr, 2.0 * (size - d) / (thr - size) + 2.0, torch.zeros_like(d)))
    return cost.mean(-1)


def trajectory_cost_loss(y_pred, geo, size=0.35, thr=0.85):
    with torch.no_grad():
        cost = collision_cost(trajectories(y_pred), geo, size, thr)
    return 2.0 * ((cost - y_pred[:, :, 0]) ** 2).mean()
