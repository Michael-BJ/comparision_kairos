#!/usr/bin/env python3
"""sando_trajectory_server: the test's trajectory_server (dyn_expert, unchanged) with SANDO's own yaw.

Everything is dyn_expert's TrajectoryServer: calibration of the world <-> PX4 local frame, takeoff, OFFBOARD,
tracking of the HermiteTrajectory (position + velocity + acceleration feed-forward). Only the yaw of the READY
phase comes from SANDO's setpoint stream <ns>/goal (goal.yaw, world frame) while that stream is fresh, instead of
the test's YawPolicy. SANDO's yaw faces the velocity like the policy, and after yaw_spinning_threshold failed
replans in a row it spins at yaw_spinning_dyaw (sando_hw_quadrotor.yaml: 10, 1 rad/s) so the tilted Mid-360
sweeps the space around the drone; without it SANDO stayed in a failing hover (223 s, E7 2026-11-04).
"""
import sys

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from dynus_interfaces.msg import Goal as SandoGoal
from dyn_expert.nodes.trajectory_server_node import TrajectoryServer

FRESH_S = 0.2      # [s] SANDO's yaw is used while its last setpoint is younger than this (it streams at 100 Hz)


class SandoYaw:
    """Stands in for the server's YawPolicy: SANDO's yaw when fresh, else the policy."""

    def __init__(self, node, policy):
        self.node, self.policy = node, policy
        self.sando = None                       # (receive time [s], yaw)

    @property
    def yaw(self):
        return self.policy.yaw

    @yaw.setter
    def yaw(self, value):                       # the server sets the hold yaw before takeoff
        self.policy.yaw = value

    def update(self, vel, dt, goal_dir=None, t=None):
        y = self.policy.update(vel, dt, goal_dir, t)     # keeps the policy's state continuous
        now = self.node.get_clock().now().nanoseconds * 1e-9
        if self.sando is not None and now - self.sando[0] < FRESH_S:
            self.policy.yaw = self.sando[1]
            return self.sando[1]
        return y


class SandoTrajectoryServer(TrajectoryServer):
    def __init__(self):
        super().__init__()
        self.declare_parameter("sando_goal_topic", "/NX01/goal")
        self.yaw = SandoYaw(self, self.yaw)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(SandoGoal, self.get_parameter("sando_goal_topic").value, self.on_sando_goal, qos)
        self.get_logger().info("yaw: SANDO's goal.yaw while its setpoint stream is fresh, else the test's policy")

    def on_sando_goal(self, msg):
        self.yaw.sando = (self.get_clock().now().nanoseconds * 1e-9, float(msg.yaw))


def main():
    rclpy.init(args=sys.argv)
    node = SandoTrajectoryServer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):     # SIGINT of the launch at the end of a run
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
