"""Probe for a SANDO run: lidar returns on each true dynamic obstacle per scan, SANDO's tracks, the true drone pose.

  python3 sando_probe.py <scenario.yaml> <out.npz> [model]

Start it after the stack is up (SANDO_planner/run.sh), stop it with SIGTERM / Ctrl+C; it saves every 15 s and on exit.
Needs the ROS environment of the run (inference_kairos + SANDO_planner/ws: dynus_interfaces).

Lidar points (NX01_livox frame) -> world with the Gazebo pose of the model origin, base_link +0.24 m, mount
(0, 0, 0.2) pitch 0.3 (SANDO_planner/models/x500_mid360_d435_tw2). No-return rays (lidar_fill sets them to range_max
20 m) are skipped. A return belongs to sphere k when it lies within radius + 0.15 m of its true centre (scenario
motion at the scan's sim time).
Saved arrays: scans (t, drone xyz, then per obstacle: returns, distance, elevation in the sensor frame [deg]),
tracks (t_recv, id, x, y, z, vx, vy, vz, bx, by, bz of /NX01/predicted_trajs), odom (t, x, y, z), ids, radius.
"""
import signal
import sys

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from dynus_interfaces.msg import DynTraj

from dyn_expert.core.scenario import Scenario

scn_path, out = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "x500_mid360_d435_tw2_0"
scn = Scenario.load(scn_path)
DYN = scn.dynamic
RAD = np.array([d.radius for d in DYN])
R_MOUNT = Rotation.from_euler("y", 0.3).as_matrix()
T_MOUNT = np.array([0.0, 0.0, 0.24 + 0.2])


def centres(t):
    return np.array([d.motion.pos(t)[0] for d in DYN])


class Probe(Node):
    def __init__(self):
        super().__init__("sando_probe", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.pose = None
        self.scans, self.tracks, self.odom = [], [], []
        self.create_subscription(Odometry, f"/model/{model}/odometry", self.on_odom, 50)
        self.create_subscription(PointCloud2, "/NX01/mid360_PointCloud2", self.on_cloud, qos_profile_sensor_data)
        self.create_subscription(DynTraj, "/NX01/predicted_trajs", self.on_track, 100)
        self.create_timer(15.0, self.save)

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        self.pose = (t, np.array([p.x, p.y, p.z]), Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix())
        self.odom.append((t, p.x, p.y, p.z))

    def on_cloud(self, m):
        if self.pose is None:
            return
        t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        _, P, R = self.pose
        pts = point_cloud2.read_points_numpy(m, field_names=("x", "y", "z"))
        pts = pts[np.isfinite(pts).all(axis=1)]
        pts = pts[np.linalg.norm(pts, axis=1) < 19.9]
        sensor_pos = P + R @ T_MOUNT
        Rs = R @ R_MOUNT
        w = pts @ Rs.T + sensor_pos
        C = centres(t)
        rel = C - sensor_pos
        dist = np.linalg.norm(rel, axis=1)
        rel_s = rel @ Rs
        elev = np.degrees(np.arctan2(rel_s[:, 2], np.hypot(rel_s[:, 0], rel_s[:, 1])))
        n = np.zeros(len(DYN))
        near = np.flatnonzero(dist < 25.0)
        if len(w) and len(near):
            d = np.linalg.norm(w[None, :, :] - C[near, None, :], axis=2)
            n[near] = (d < RAD[near, None] + 0.15).sum(axis=1)
        self.scans.append(np.concatenate([[t], P, n, dist, elev]))

    def on_track(self, m):
        t = self.get_clock().now().nanoseconds * 1e-9
        v = list(m.poly_coeffs_x[:1]) + list(m.poly_coeffs_y[:1]) + list(m.poly_coeffs_z[:1])
        v = v if len(v) == 3 else [np.nan] * 3
        b = list(m.bbox) + [np.nan] * (3 - len(m.bbox))
        self.tracks.append([t, m.id, m.pos.x, m.pos.y, m.pos.z] + v + b[:3])

    def save(self):
        np.savez(out, scans=np.array(self.scans), tracks=np.array(self.tracks), odom=np.array(self.odom),
                 ids=np.array([d.id for d in DYN]), radius=RAD)


rclpy.init()
node = Probe()


def stop(*_):
    node.save()
    raise SystemExit(0)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
try:
    rclpy.spin(node)
finally:
    node.save()
