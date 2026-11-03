"""KAIROS episode files (HDF5) -> Agile Autonomy samples (Loquercio et al., Science Robotics 2021,
github.com/uzh-rpg/agile_autonomy, planner_learning/src/PlannerLearning/models/data_loader.py).

Agile Autonomy sample = (depth image, state, top-3 expert trajectories), plus the privileged world it was
recorded in (its pointcloud-unity.ply) for the trajectory cost loss:
    depth   = the KAIROS range panorama of the x500_oakd_x2_60 rig (128 x 400, both cameras) in mm, as stored;
              the trainer applies Agile Autonomy's normalization (min(mm, 20000) / 80, 3 channels)
    state   = [R (9), v (3), goal_dir (3)]: Agile Autonomy's network input with inputs.position false,
              inputs.attitude true, inputs.bodyrates false (the dataset has no body rates)
    label   = (3, 30): the 3 lowest-cost expert trajectories, each [x_1..x_10, y_1..y_10, z_1..z_10]
              (out_seq_len 10, state_dim 3, the column order of its trajectories_bf_*.csv); fewer than 3
              expert modes are filled with copies of the best one, as load_trajectory does
    geometry = static boxes and cylinders and the dynamic spheres at the 10 label times, near the start

Frames and times, changed from Agile Autonomy:
  * Frame: the KAIROS local frame (x along the drone yaw, z up) about the plan start S_init, for the state,
    the label and the geometry. Agile Autonomy uses the full camera attitude ('bf'); here roll and pitch stay
    in the R input only.
  * Start: the plan starts at S_init, foresee_dt (0.4 s) after the image, as every method in this test;
    Agile Autonomy planned from the state at the image. v = S_init's velocity, label = expert positions
    relative to S_init.
  * Points every LABEL_DT = 0.35 s (3.5 s, the median duration of the expert's best mode) instead of 0.1 s
    (1 s at 7 m/s in the paper); at deployment the path is resampled to the flight speed anyway
    (setConstantArcLengthSpeed).
  * goal_dir: unit vector to the episode goal. Agile Autonomy's points to the privileged global reference
    trajectory 5 s ahead, which a test run here does not have.
  * Expert: the KAIROS expert's modes (mode 0 = lowest cost, mode_mask), instead of its Metropolis-Hastings
    sampler's top 3 trajectories.
"""
import glob
import json
import os
import sys

import h5py
import numpy as np
import yaml

try:
    from dyn_expert.core.scenario import Scenario
except ImportError:
    sys.path.insert(0, os.path.expanduser("~/Dynamic_obstacle/install/dyn_expert/lib/python3.12/site-packages"))
    from dyn_expert.core.scenario import Scenario

PANO_H, PANO_W = 128, 400
OUT_SEQ_LEN = 10
STATE_DIM = 3
TOP_TRAJ = 3
STATE_SIZE = 15
LABEL_DT = 0.35
M_KAIROS = 5
GEO_RADIUS = 10.0          # obstacles whose surface is within this distance of S_init enter the cost loss
FAR = 1.0e3                # position of padded / inactive obstacles
# lateral mirror y -> -y: R01, R10, R12, R21, v_y, goal_y; label y_1..y_10
MIRROR_STATE_NEG = [1, 3, 5, 7, 10, 13]
MIRROR_LABEL_NEG = list(range(OUT_SEQ_LEN, 2 * OUT_SEQ_LEN))
STATE_VEL = slice(9, 12)


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_to_rot(q):
    """xyzw quaternion -> rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def state_vector(R_local, v_local, goal_dir):
    return np.concatenate([np.asarray(R_local).reshape(-1), v_local, goal_dir]).astype(np.float32)


def _quintic(p0, v0, a0, p1, v1, a1, T, s):
    """Quintic Hermite segment at s in [0, 1] (dyn_expert.core.hermite.quintic_coeffs)."""
    c0, c1, c2 = p0, T * v0, 0.5 * T * T * a0
    A = p1 - c0 - c1 - c2
    B = T * v1 - c1 - 2.0 * c2
    C = T * T * a1 - 2.0 * c2
    c3 = 10.0 * A - 4.0 * B + 0.5 * C
    c4 = -15.0 * A + 7.0 * B - C
    c5 = 6.0 * A - 3.0 * B + 0.5 * C
    return c0 + s * (c1 + s * (c2 + s * (c3 + s * (c4 + s * c5))))


def hermite_pos(s_init, s_target, z, t):
    """Position at time t of the KAIROS label z (41) between s_init and s_target; held at the end after T."""
    n = 3 * (M_KAIROS - 1)
    P = np.vstack([s_init[0:3], z[0:n].reshape(-1, 3), s_target[0:3]])
    V = np.vstack([s_init[3:6], z[n:2 * n].reshape(-1, 3), s_target[3:6]])
    A = np.vstack([s_init[6:9], z[2 * n:3 * n].reshape(-1, 3), s_target[6:9]])
    T = z[3 * n:3 * n + M_KAIROS]
    tk = np.concatenate([[0.0], np.cumsum(T)])
    i = int(np.clip(np.searchsorted(tk, t, side="right") - 1, 0, M_KAIROS - 1))
    s = np.clip((t - tk[i]) / T[i], 0.0, 1.0)
    return _quintic(P[i], V[i], A[i], P[i + 1], V[i + 1], A[i + 1], T[i], s)


def label_times():
    return LABEL_DT * np.arange(1, OUT_SEQ_LEN + 1)


class WorldGeometry:
    """The privileged world of one episode file (its scenario), expressed about an event's S_init."""

    def __init__(self, scenario_yaml):
        self.scn = Scenario(yaml.safe_load(scenario_yaml))

    def local(self, origin, yaw, t_start):
        """Obstacles near origin in the local frame: boxes (nb, 7) [c, half, yaw], cylinders (nc, 5)
        [cx, cy, r, z_lo, z_hi], spheres (ns, 10, 3) at t_start + label_times(), radii (ns,)."""
        scn, Rt = self.scn, rot_z(yaw).T
        p0 = np.asarray(origin, float)[None]
        boxes = []
        for c, h, byaw in scn.boxes:
            if Scenario._box_distance(p0, c, h, byaw)[0] < GEO_RADIUS:
                boxes.append(np.concatenate([Rt @ (c - origin), h, [byaw - yaw]]))
        cyls = []
        if len(scn.cyl):
            d = scn._cyl_distance(p0)[0]
            for (cx, cy, r, hgt), dk in zip(scn.cyl, d):
                if dk < GEO_RADIUS:
                    q = Rt @ (np.array([cx, cy, 0.0]) - origin)
                    cyls.append([q[0], q[1], r, -origin[2], hgt - origin[2]])
        ts = t_start + label_times()
        sph, rad = [], []
        for ob in scn.dynamic:
            pw = np.asarray(ob.pos(ts), float).reshape(len(ts), 3)
            if np.min(np.linalg.norm(pw - origin, axis=1)) - ob.radius >= GEO_RADIUS:
                continue
            pl = (pw - origin) @ Rt.T
            pl[~np.asarray(ob.active(ts), bool)] = FAR
            sph.append(pl)
            rad.append(ob.radius)
        return (np.asarray(boxes, np.float32).reshape(-1, 7), np.asarray(cyls, np.float32).reshape(-1, 5),
                np.asarray(sph, np.float32).reshape(-1, OUT_SEQ_LEN, 3), np.asarray(rad, np.float32))


def event_sample(ev, i, world):
    """One event -> (depth (128, 400) uint16 mm, state (15), label (3, 30), geometry) or None without a mode."""
    mask = np.asarray(ev["mode_mask"][i], bool)
    if int(ev["n_modes"][i]) < 1 or not mask[0]:
        return None
    yaw = float(ev["frame_yaw"][i])
    origin = np.asarray(ev["frame_origin"][i], float)
    s_i = np.asarray(ev["s_init"][i], float)
    s_t = np.asarray(ev["s_target"][i], float)
    R_local = rot_z(yaw).T @ quat_to_rot(np.asarray(ev["drone_att_world"][i], float))
    g = np.asarray(ev["goal"][i], float) - s_i[0:3]
    state = state_vector(R_local, s_i[3:6], g / max(np.linalg.norm(g), 1e-6))
    modes = [j for j in range(len(mask)) if mask[j]][:TOP_TRAJ]
    label = np.zeros((TOP_TRAJ, STATE_DIM * OUT_SEQ_LEN), np.float32)
    for k in range(TOP_TRAJ):
        z = np.asarray(ev["mode_z"][i, modes[min(k, len(modes) - 1)]], float)
        pts = np.stack([hermite_pos(s_i, s_t, z, t) for t in label_times()]) - s_i[0:3]
        label[k] = pts.T.reshape(-1)
    geo = world.local(origin + rot_z(yaw) @ s_i[0:3], yaw,
                      float(ev["t_event_abs"][i]) + float(ev["foresee_dt"][i]))
    return np.asarray(ev["depth"][i], np.uint16), state, label, geo


def load_files(paths, log=print):
    """All samples of a list of episode files as a dict of stacked arrays (geometry padded to the largest count:
    boxes / cylinders far away and with zero size, spheres at FAR)."""
    depth, state, label, boxes, cyls, sph, rad, skipped = [], [], [], [], [], [], [], 0
    for k, path in enumerate(paths):
        with h5py.File(path, "r") as h:
            world = WorldGeometry(h.attrs["scenario_yaml"])
            ev = h["events"]
            for i in range(ev["n_modes"].shape[0]):
                s = event_sample(ev, i, world)
                if s is None:
                    skipped += 1
                    continue
                depth.append(s[0]); state.append(s[1]); label.append(s[2])
                boxes.append(s[3][0]); cyls.append(s[3][1]); sph.append(s[3][2]); rad.append(s[3][3])
        if (k + 1) % 50 == 0:
            log(f"[aa_data] {k + 1}/{len(paths)} files, {len(depth)} samples")
    log(f"[aa_data] {len(paths)} files: {len(depth)} samples, {skipped} events without an expert mode skipped")
    box_fill = np.array([FAR, FAR, FAR, 0, 0, 0, 0], np.float32)
    cyl_fill = np.array([FAR, FAR, 0, 0, 0], np.float32)
    out = dict(depth=np.stack(depth), state=np.stack(state), label=np.stack(label),
               boxes=_pad_rows(boxes, box_fill), cyls=_pad_rows(cyls, cyl_fill),
               spheres=_pad_rows(sph, np.full((OUT_SEQ_LEN, 3), FAR, np.float32)),
               radii=_pad_rows(rad, np.float32(0.0)))
    log("[aa_data] geometry per sample (max): " + ", ".join(f"{k} {out[k].shape[1]}"
                                                             for k in ("boxes", "cyls", "spheres")))
    return out


def _pad_rows(arrs, fill):
    """Stack variable-length arrays (n_i, ...) into (N, max n_i, ...), padding with `fill`."""
    fill = np.asarray(fill, np.float32)
    n = max([len(a) for a in arrs] + [1])
    out = np.empty((len(arrs), n) + fill.shape, np.float32)
    out[:] = fill
    for k, a in enumerate(arrs):
        if len(a):
            out[k, :len(a)] = a
    return out


def load_split(run_dir, data_root=None):
    """Train / val file lists of a kairos run (split.json + config.yaml data.dirs)."""
    with open(os.path.join(run_dir, "split.json")) as fh:
        split = json.load(fh)
    with open(os.path.join(run_dir, "config.yaml")) as fh:
        dirs = yaml.safe_load(fh)["data"]["dirs"]
    root = data_root or os.path.dirname(dirs[0].rstrip("/"))
    return ({k: [os.path.join(root, f) for f in split[k]] for k in ("train", "val")}, dirs)


def find_files(dirs):
    return sorted(f for d in dirs for f in glob.glob(os.path.join(os.path.expanduser(d), "*.h5")))
