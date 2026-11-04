#!/usr/bin/env python3
"""lidar_fill: the Mid-360 cloud as SANDO's own Livox plugin gives it to the ACL global mapper.

SANDO's livox_points_plugin.cpp (livox_laser_simulation_ros2) clamps every ray to [range_min, range_max] and
publishes it, also when nothing was hit (range = max, retro 0). The mapper ray-traces free space along every
point and marks only returns within depth_max (15 m) as occupied, so rays into the open sky clear it.
Gazebo Harmonic's gpu_lidar publishes such rays as inf points instead (~73 % of the rays in the test world,
2026-11-03); the mapper skips them, the air above the walls stays unknown, and SANDO's unknown-space check
(findSafeSubGoal) cuts every path that climbs over a wall to its start point (SANDO hovered 35 min at x 49).
This node restores the plugin's output: non-finite points of the organized cloud (height = vertical samples,
width = horizontal samples, as the model's gpu_lidar) -> range_max along the ray's direction, intensity 0.

Self filter: returns inside the airframe's box (base_link frame, via the sensor's mount) are dropped, as on any real
drone lidar. Mounted as SANDO's URDF, the lowest rays (-24 deg ahead) graze the x500's 13" front propeller tips,
~50 returns per scan 0.4 m away (2026-11-04), which the mapper made occupied voxels at the drone. Dropped, not set
to range_max: what lies behind the propeller is unknown. The cloud is then unorganized (height 1).
"""
import sys

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


class LidarFill(Node):
    def __init__(self):
        super().__init__("lidar_fill")
        p = self.declare_parameter
        # models/x500_mid360_d435_tw2/model.sdf, sensor mid360
        p("h_min", -3.14159265)
        p("h_max", 3.13286601)
        p("v_min", -0.12601277)
        p("v_max", 0.96377081)
        p("range_max", 20.0)
        # mount on base_link (SANDO's URDF) and the x500's extent there: rotors at (+-0.174, +-0.174, 0.06) with
        # 0.165 m props, frame plate, landing gear down to -0.22
        p("mount_z", 0.2)
        p("mount_pitch", 0.3)
        p("self_xy", 0.36)
        p("self_z_min", -0.25)
        p("self_z_max", 0.12)
        g = lambda n: float(self.get_parameter(n).value)
        self.h = (g("h_min"), g("h_max"))
        self.v = (g("v_min"), g("v_max"))
        self.r_max = g("range_max")
        c, s = np.cos(g("mount_pitch")), np.sin(g("mount_pitch"))
        self.R_bs = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])    # sensor -> base_link (pitch about y)
        self.t_bs = np.array([0.0, 0.0, g("mount_z")])
        self.self_box = (g("self_xy"), g("self_z_min"), g("self_z_max"))
        self.n_self = 0
        self.dirs = None
        self.n = 0
        self.pub = self.create_publisher(PointCloud2, "points_out", qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "points_in", self.cb, qos_profile_sensor_data)

    def directions(self, rows, cols):
        el = np.linspace(self.v[0], self.v[1], rows) if rows > 1 else np.zeros(1)
        az = np.linspace(self.h[0], self.h[1], cols) if cols > 1 else np.zeros(1)
        E, A = np.meshgrid(el, az, indexing="ij")
        return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)

    def cb(self, m):
        off = {f.name: f.offset for f in m.fields}
        if not {"x", "y", "z"} <= off.keys() or m.point_step % 4:
            self.pub.publish(m)
            return
        rows, cols = m.height, m.width
        if self.dirs is None or len(self.dirs) != rows * cols:
            self.dirs = self.directions(rows, cols)
        buf = np.frombuffer(bytes(m.data), np.float32).reshape(rows * cols, m.point_step // 4).copy()
        xyz = buf[:, [off["x"] // 4, off["y"] // 4, off["z"] // 4]]
        bad = ~np.isfinite(xyz).all(1)
        if self.n == 0:                     # once: the assumed ray order must match the returns
            good = ~bad
            u = xyz[good] / np.linalg.norm(xyz[good], axis=1, keepdims=True)
            err = np.degrees(np.arccos(np.clip((u * self.dirs[good]).sum(1), -1.0, 1.0)))
            self.get_logger().info(f"[lidar_fill] {rows} x {cols} rays, {int(bad.sum())} without return; "
                                   f"direction check on returns: max {err.max():.3f} deg")
        self.n += 1
        b = np.where(bad[:, None], 0.0, xyz) @ self.R_bs.T + self.t_bs
        xy, z0, z1 = self.self_box
        own = ~bad & (np.abs(b[:, 0]) <= xy) & (np.abs(b[:, 1]) <= xy) & (b[:, 2] >= z0) & (b[:, 2] <= z1)
        self.n_self += int(own.sum())
        if self.n % 600 == 0:
            self.get_logger().info(f"[lidar_fill] self returns dropped: {self.n_self / self.n:.1f} per scan")
        fill = self.dirs[bad] * self.r_max
        for k, c in enumerate("xyz"):
            buf[bad, off[c] // 4] = fill[:, k]
        if "intensity" in off:
            buf[bad, off["intensity"] // 4] = 0.0
        if own.any():
            buf = buf[~own]
            m.height, m.width = 1, len(buf)
            m.row_step = m.point_step * len(buf)
        m.data = buf.tobytes()
        m.is_dense = True
        self.pub.publish(m)


def main():
    rclpy.init(args=sys.argv)
    node = LidarFill()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
