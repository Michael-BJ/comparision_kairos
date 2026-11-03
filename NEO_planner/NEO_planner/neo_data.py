"""KAIROS episode files (HDF5) -> NEO-Planner samples, and the NEO network input at deployment.

NEO sample = (depth_u8, motion_info (24), label (9)), as NEO's expert_planner_node records it:
    depth_u8    = depth / max(valid depth) * 255 as uint8, the formula of NEO's _form_nn_input. Here the image
                  is the KAIROS range panorama of the x500_oakd_x2_60 rig (128 x 400, both cameras), not
                  NEO's single 640 x 480 z-depth image.
    motion_info = [drone_local_vel (3), drone_R (9), init_pos_body, init_vel_body, tgt_pos_body, tgt_vel_body]
    label       = [wpt1_body (3), wpt2_body (3), ts1..ts3]: the interior waypoints of NEO's 3-piece MINCO
                  trajectory, relative to the drone, and the piece durations.
Body frame = NEO's: the drone's yaw frame about its position (neo_planner_node builds the attitude from yaw
only). The KAIROS local frame (origin S_init, x along the drone yaw) has the same axes, so a body vector is
local position minus the drone's local position, and velocities subtract the drone velocity as NEO does.

Label: the KAIROS expert's best mode (mode 0, lowest cost; what the KAIROS regression baseline learns), a
5-segment quintic Hermite trajectory S_init -> S_target, resampled at 1/3 and 2/3 of its duration with
ts = T/3 each. NEO's own expert labels are the waypoints of its optimized 3-piece MINCO trajectory.
"""
import glob
import json
import os

import h5py
import numpy as np

PANO_H, PANO_W = 128, 400
MOTION_INPUT_SIZE = 24
OUTPUT_SIZE = 9
M_KAIROS = 5
# lateral mirror y -> -y (NEO's nn_trainer MirrorAugmentSet): y components of every vector, R01/R10/R12/R21
MIRROR_MOTION_NEG = [1, 4, 6, 8, 10, 13, 16, 19, 22]
MIRROR_OUTPUT_NEG = [1, 4]
# motion slots that hold the drone velocity: + v_drone (drone_local_vel), - v_drone (init_vel_body, tgt_vel_body)
MOTION_VEL, MOTION_INIT_VEL, MOTION_TGT_VEL = slice(0, 3), slice(15, 18), slice(21, 24)


def depth_to_u8(depth_m):
    """NEO's depth normalization: depth / max valid depth * 255, uint8 (0 = no return)."""
    valid = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    if valid.size == 0:
        return np.zeros(depth_m.shape, np.uint8)
    return np.clip(depth_m / (float(valid.max()) + 1e-6) * 255, 0, 255).astype(np.uint8)


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def motion_info(yaw, drone_vel_body, init_pos_body, init_vel_body, tgt_pos_body, tgt_vel_body):
    """NEO's 24 motion inputs; vectors already in the yaw frame about the drone."""
    return np.concatenate([drone_vel_body, rot_z(yaw).reshape(-1), init_pos_body, init_vel_body,
                           tgt_pos_body, tgt_vel_body]).astype(np.float32)


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
    """Position at time t of the KAIROS label z (41) between s_init and s_target (9 each)."""
    n = 3 * (M_KAIROS - 1)
    P = np.vstack([s_init[0:3], z[0:n].reshape(-1, 3), s_target[0:3]])
    V = np.vstack([s_init[3:6], z[n:2 * n].reshape(-1, 3), s_target[3:6]])
    A = np.vstack([s_init[6:9], z[2 * n:3 * n].reshape(-1, 3), s_target[6:9]])
    T = z[3 * n:3 * n + M_KAIROS]
    tk = np.concatenate([[0.0], np.cumsum(T)])
    i = int(np.clip(np.searchsorted(tk, t, side="right") - 1, 0, M_KAIROS - 1))
    s = np.clip((t - tk[i]) / T[i], 0.0, 1.0)
    return _quintic(P[i], V[i], A[i], P[i + 1], V[i + 1], A[i + 1], T[i], s)


def event_sample(ev, i):
    """One event of an open `events` group -> (depth (128, 400) uint16 mm as stored, motion (24), label (9)) or
    None without a mode. The trainer augments the depth in meters (as KAIROS) and then applies depth_to_u8."""
    if int(ev["n_modes"][i]) < 1 or not bool(ev["mode_mask"][i, 0]):
        return None
    yaw = float(ev["frame_yaw"][i])
    origin = np.asarray(ev["frame_origin"][i], float)
    drone_loc = rot_z(yaw).T @ (np.asarray(ev["drone_pos_world"][i], float) - origin)
    v_d = np.asarray(ev["drone_vel"][i], float)
    s_i = np.asarray(ev["s_init"][i], float)
    s_t = np.asarray(ev["s_target"][i], float)
    mot = motion_info(yaw, v_d, s_i[0:3] - drone_loc, s_i[3:6] - v_d, s_t[0:3] - drone_loc, s_t[3:6] - v_d)
    z = np.asarray(ev["mode_z"][i, 0], float)
    T = float(z[-M_KAIROS:].sum())
    w1 = hermite_pos(s_i, s_t, z, T / 3.0) - drone_loc
    w2 = hermite_pos(s_i, s_t, z, 2.0 * T / 3.0) - drone_loc
    label = np.concatenate([w1, w2, np.full(3, T / 3.0)]).astype(np.float32)
    return np.asarray(ev["depth"][i], np.uint16), mot, label


def load_split(run_dir, data_root=None):
    """Train / val file lists of a kairos run (split.json + config.yaml data.dirs), so NEO trains on the same files."""
    import yaml
    with open(os.path.join(run_dir, "split.json")) as fh:
        split = json.load(fh)
    with open(os.path.join(run_dir, "config.yaml")) as fh:
        dirs = (yaml.safe_load(fh)["data"]["dirs"])
    root = data_root or os.path.dirname(dirs[0].rstrip("/"))
    return ({k: [os.path.join(root, f) for f in split[k]] for k in ("train", "val")}, dirs)


def load_files(paths, log=print):
    """All samples of a list of episode files: depth (N, 128, 400) uint16 mm, motion (N, 24), label (N, 9)."""
    depths, motions, labels, skipped = [], [], [], 0
    for k, path in enumerate(paths):
        with h5py.File(path, "r") as h:
            ev = h["events"]
            for i in range(ev["n_modes"].shape[0]):
                s = event_sample(ev, i)
                if s is None:
                    skipped += 1
                    continue
                depths.append(s[0])
                motions.append(s[1])
                labels.append(s[2])
        if (k + 1) % 50 == 0:
            log(f"[neo_data] {k + 1}/{len(paths)} files, {len(depths)} samples")
    log(f"[neo_data] {len(paths)} files: {len(depths)} samples, {skipped} events without an expert mode skipped")
    return np.stack(depths), np.stack(motions), np.stack(labels)


def find_files(dirs):
    return sorted(f for d in dirs for f in glob.glob(os.path.join(os.path.expanduser(d), "*.h5")))
