"""Record the obstacle expression strings SANDO's dynamic_forest_node publishes on /trajs (rviz-only benchmark), to
check from_sando.py against SANDO's own obstacles.

  ROS_DOMAIN_ID=20 python3 grab_trajs.py <out.json>     (while SANDO's `dynamic` demo runs: sim_dbg.sh -m dynamic ...)

Writes {id: {"f": [x(t), y(t), z(t)], "bbox": [...]}} for every obstacle seen in 4 s. Comparison (2026-11-04, medium,
seed 0): with the per-harmonic phases of from_sando.py the positions agree within 0.5 mm over 250 s; with the
standard trefoil phases (k x offset) they differ by up to 3.7 m.
"""
import json
import sys
import time

import rclpy
from dynus_interfaces.msg import DynTraj
from rclpy.node import Node

rclpy.init()
n = Node("grab_trajs")
got = {}


def cb(m):
    if not m.is_agent:
        got[m.id] = dict(f=list(m.function), bbox=list(m.bbox))


n.create_subscription(DynTraj, "/trajs", cb, 100)
t0 = time.time()
while time.time() - t0 < 4:
    rclpy.spin_once(n, timeout_sec=0.1)
json.dump(got, open(sys.argv[1], "w"))
print(len(got), "obstacles")
