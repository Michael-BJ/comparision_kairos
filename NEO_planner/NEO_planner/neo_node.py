#!/usr/bin/env python3
"""neo_node: NEO-Planner inference in the KAIROS test stack (x500_oakd_x2_60_tw2, dyn_long200 worlds).

The planner is the one of ~/neo_planner_ws/src/neo_planner/neo_planner/neo_planner_node.py: PlannerNet initial
guess (2 waypoints + 3 durations) -> MinJerkPlanner.warm_start_plan (2D MINCO, ESDF of the octomap's projected
map) -> hard clearance check, with NEO's local target (lateral nudge on the ESDF), seed loop, adaptive safe_dis,
stuck / no-progress escape, collision guard, look-ahead guard and v_max time scaling. min_jerk_planner.py and
esdf_ros2.py are unchanged copies. NEO flies at a fixed altitude (the scenario's route.z).

What differs from neo_planner_node, so that NEO runs on the stack KAIROS is tested on:
  * sensor: the rig's two depth images (camera.rig of the dyn_expert config), paired by stamp as kairos_planner
    does. The network gets their range panorama (dyn_expert CameraRig.panorama, = the dataset's depth); octomap
    gets both images as point clouds in their camera frames (TF world -> neo_cam_<i> at the image stamp).
  * state: odom_topic (/kairos/odom = PX4's estimate in the world frame), interpolated at the depth stamp; sim time.
  * output: HermiteTrajectory to dyn_expert's trajectory_server (takeoff, OFFBOARD, tracking, yaw), as KAIROS.
    A MINCO piece is a quintic, so its knots (p, v, a) give the same curve. NEO's escape velocity command is
    sent as a 1 s constant-velocity trajectory. The node never sets PX4 parameters.
  * timing: a new plan starts planning_time_ahead after the planning started (t0); neo_planner_node switched
    to it at once. replan_period / planning_time_ahead / longitu_step default to the dataset's replan_dt 0.2,
    foresee_dt 0.4 and target_dist 4.0, and the local target's velocity to the dataset's (zero): the network
    was trained on those events (neo_planner_node: 1.0 s, 1.0 s, 5.0 m, 0.8 v_max).
  * mission and referee as kairos_planner: episodes pad A <-> B of the scenario (route_goal), reached within
    dataset.goal_reach_tol of the goal, timeout after timeout_factor * distance / v_nominal + 10 s, collisions
    counted and the episode continued. The referee uses Gazebo's pose (referee_odom_topic); the planner never.
"""
import copy
import json
import math
import os
import sys
import threading
import time
import types
from collections import deque

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster

from dyn_nav_interfaces.msg import HermiteTrajectory as HermiteTrajectoryMsg
from dyn_expert.core.camera import CameraRig
from dyn_expert.core.config import load_config
from dyn_expert.core.hermite import HermiteTrajectory
from dyn_expert.core.scenario import Scenario
from dyn_expert.nodes.common import image_to_depth_m, odom_to_state, to_sec, to_stamp, traj_to_msg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import neo_data as nd
from esdf_ros2 import ESDF
from min_jerk_planner import MinJerkPlanner, PlannerConfig
from planner_net import PlannerNet

# constants of neo_planner_node.py
DISCONTINUITY_THRESHOLD = 3.0
EMERGENCY_STOP_THRESHOLD = 3.0
STUCK_SPEED_THRESHOLD = 0.15
STUCK_TIME_THRESHOLD = 5.0
EMERGENCY_STOP_PERSIST_TIME = 2.0
NO_PROGRESS_TIME = 8.0
NO_PROGRESS_DELTA = 0.8
ESCAPE_VELOCITY = 0.4
ESCAPE_PROBE_DIST = 1.5
ESCAPE_EXIT_DIST = 1.2
ESCAPE_MAX_TIME = 15.0
DEPTH_MAX_LAG_S = 0.3
LOOKAHEAD_HORIZON_S = 1.5
LOOKAHEAD_STEP_S = 0.3
REPLAN_BUDGET_S = 2.5
GUARD_HZ = 10.0               # collision / look-ahead guard (neo_planner_node: every 10th tick of 100 Hz)


def _rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _yaw(R):
    return math.atan2(R[1, 0], R[0, 0])


class State:
    """NEO's DroneState: position, velocity (world), yaw; the body frame is the yaw frame."""

    def __init__(self, p, v, yaw):
        self.global_pos = np.asarray(p, float).copy()
        self.global_vel = np.asarray(v, float).copy()
        self.yaw = float(yaw)

    def body(self, x):
        return _rot_z(self.yaw).T @ x


class NeoNode(Node):
    def __init__(self):
        super().__init__("neo_planner")
        p = self.declare_parameter
        p("config_file", "")
        p("scenario_file", "")
        p("model_path", "")
        p("out_dir", os.path.expanduser("~/kairos_test"))
        p("run_name", "neo")
        p("seed", 0)
        p("episodes", 0)                   # stop after this many episodes (0 = unlimited)
        p("odom_topic", "/kairos/odom")
        p("odom_twist_in_body", True)
        p("referee_odom_topic", "")
        p("trajectory_topic", "/dyn_nav/trajectory")
        p("ready_topic", "/dyn_nav/ready")
        p("cloud_topic", "/neo/cloud_in")
        p("map_topic", "/projected_map")
        p("world_frame", "world")
        p("cloud_stride", 0)               # pixel stride of the octomap clouds, 0 = camera.map_stride (the KAIROS map's rays)
        p("v_max", 0.0)                    # 0 = planner.v_max of the config
        p("timeout_factor", 0.0)           # 0 = dataset.timeout_factor of the config
        p("replan_period", 0.0)            # 0 = planner.replan_dt
        p("planning_time_ahead", 0.0)      # 0 = planner.foresee_dt (dTf)
        p("longitu_step", 0.0)             # 0 = planner.target_dist
        p("lateral_step", 1.0)
        p("target_vel_frac", 0.0)          # local target velocity = goal_dir * this * v_max (neo_planner: 0.8)
        # safety margins, ESDF distances from the drone body (obstacles inflated by inflation_radius). Defaults (< 0)
        # follow the KAIROS test: inflation = planner.drone_radius 0.35, hard = planner.static_clearance - drone_radius
        # = 0.20 (centre to surface 0.55, KAIROS's verify), soft safe_dis = hard + 0.20, emergency guard = hard + 0.05
        # (NEO's 0.4 vs 0.35). neo_planner's values (inflation 0.25, safe_dis 0.7, hard 0.35, emergency 0.4: centre
        # to surface 0.60 hard / 0.95 soft) close the 1.9-2.3 m low-wall gaps of the dyn_long200 worlds (2026-11-02)
        p("safe_dis", -1.0)
        p("safe_dis_emergency", -1.0)
        p("collision_cost_tol", 1.0)
        p("hard_clearance", -1.0)
        p("inflation_radius", -1.0)
        p("device", "auto")
        gp = lambda n: self.get_parameter(n).value

        self.cfg = load_config(os.path.expanduser(gp("config_file")))
        P, D = self.cfg.planner, self.cfg.dataset
        self.scn = Scenario.load(os.path.expanduser(gp("scenario_file")))
        self.rig = CameraRig(self.cfg.camera)
        self.rng = np.random.default_rng(int(gp("seed")))
        np.random.seed(int(gp("seed")))
        self.v_max = float(gp("v_max")) or float(P.v_max)
        self.timeout_factor = float(gp("timeout_factor")) or float(D.timeout_factor)
        self.replan_period = float(gp("replan_period")) or float(P.replan_dt)
        self.dt_ahead = float(gp("planning_time_ahead")) or float(P.foresee_dt)
        self.longitu_step = float(gp("longitu_step")) or float(P.target_dist)
        self.lateral_step = float(gp("lateral_step"))
        self.target_vel_frac = float(gp("target_vel_frac"))
        inflation = float(gp("inflation_radius"))
        inflation = inflation if inflation >= 0.0 else float(P.drone_radius)
        self.hard_clearance = float(gp("hard_clearance"))
        if self.hard_clearance < 0.0:
            self.hard_clearance = max(0.05, float(P.static_clearance) - inflation)
        self.safe_dis_norm = float(gp("safe_dis"))
        if self.safe_dis_norm < 0.0:
            self.safe_dis_norm = self.hard_clearance + 0.20
        self.safe_dis_emer = float(gp("safe_dis_emergency"))
        if self.safe_dis_emer < 0.0:
            self.safe_dis_emer = self.hard_clearance + 0.05
        self.cruise_z = float(self.scn.flight_z if self.scn.flight_z is not None else 2.0)
        self.goal_tol = float(D.goal_reach_tol)
        self.coll_radius = float(D.collision_radius)
        self.max_episodes = int(gp("episodes"))
        self.cloud_stride = int(gp("cloud_stride")) or int(self.cfg.camera.map_stride)
        self.world_frame = gp("world_frame")
        self.twist_in_body = bool(gp("odom_twist_in_body"))

        # network (neo_planner_node: .pth state dict of PlannerNet)
        dev = gp("device")
        self.device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if dev == "auto" else dev)
        self.net = PlannerNet(pretrained=False).to(self.device)
        self.net.load_state_dict(torch.load(os.path.expanduser(gp("model_path")), map_location=self.device))
        self.net.eval()
        with torch.no_grad():                       # warm-up (neo_planner_node._warm_up)
            for _ in range(20):
                self.net(torch.zeros(1, nd.PANO_H * nd.PANO_W + nd.MOTION_INPUT_SIZE, device=self.device))

        # planner config of neo_planner_node (SAME as expert v25), v_max of the test
        cfg = PlannerConfig()
        cfg.v_max = self.v_max
        cfg.T_min, cfg.T_max = 0.5, 5.0
        cfg.safe_dis = self.safe_dis_norm
        cfg.hard_dis = self.hard_clearance       # the optimizer's hard barrier (neo_planner: 0.30 with hard check 0.35)
        cfg.init_wpts_num = 2
        cfg.init_T = 2.5
        cfg.delta_t = 0.1
        cfg.collision_cost_tol = float(gp("collision_cost_tol"))
        cfg.opt_tol = 1e-2
        cfg.weights = [1.0, 1.0, 1.0, 10000.0]
        self.planner = MinJerkPlanner(cfg)
        # arena walls of the scenario bounds, as neo_planner's nn_inference_node v2.2 (arena_x/y_min/max): without
        # them escape and the optimizer leave the corridor, where nothing is mapped (seen 2026-11-02, y -18 m)
        lo, hi = np.asarray(self.scn.lo, float), np.asarray(self.scn.hi, float)
        self.arena = (lo[0], hi[0], lo[1], hi[1])
        self.esdf = ESDF(inflation_radius=inflation, arena_bounds=self.arena)

        self.lock = threading.Lock()                # pose buffers, latest depth, trajectory
        self.poses = deque(maxlen=400)
        self.ref_poses = deque(maxlen=400)
        self.rig_latest = [None] * len(self.rig.cams)
        self.rig_info = [False] * len(self.rig.cams)
        self.depth_wait = deque(maxlen=8)
        self.last_depth_t = -1.0
        self.last_map_t = -np.inf
        self.latest = None                         # (t, pano_u8, State at t)
        self.traj = None                            # last published HermiteTrajectory (time-scaled)
        self.ready = False
        self.done = False
        self.plan_id = 0

        # NEO mission / guard state
        self.global_target = None
        self.target_state = np.zeros((2, 2))
        self.replan_fail_count = 0
        self.stuck_since = None
        self.last_pos_for_stuck = None
        self.escape_active = False
        self.escape_vel = np.zeros(2)
        self.escape_start = None
        self.progress_ref = None
        self.emergency_first_t = None

        # episodes (referee)
        self.episode = None
        self.episodes = []
        self.last_goal, self.last_outcome = None, None
        self.contact = dict(static=None, dynamic=None)
        self.stats = dict(replans=0, fails=0, escapes=0, late=0, plan_ms=[], nn_ms=[], guard=0, lookahead=0,
                          lag_skip=0, frames=0)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(os.path.expanduser(gp("out_dir")), f"{stamp}_{self.scn.name}_neo")
        os.makedirs(self.out_dir, exist_ok=True)
        self.run_name = gp("run_name")
        self.config_record = dict(model_path=os.path.expanduser(gp("model_path")), v_max=self.v_max,
                                  replan_period=self.replan_period, planning_time_ahead=self.dt_ahead,
                                  longitu_step=self.longitu_step, lateral_step=self.lateral_step,
                                  target_vel_frac=self.target_vel_frac, safe_dis=self.safe_dis_norm,
                                  safe_dis_emergency=self.safe_dis_emer, hard_clearance=self.hard_clearance,
                                  collision_cost_tol=cfg.collision_cost_tol, inflation_radius=self.esdf.inflation_radius,
                                  cruise_z=self.cruise_z, timeout_factor=self.timeout_factor,
                                  goal_reach_tol=self.goal_tol, collision_radius=self.coll_radius,
                                  config_file=gp("config_file"), scenario=self.scn.name)

        odom_qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Odometry, gp("odom_topic"), self.on_odom, odom_qos)
        ref_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Odometry, gp("referee_odom_topic") or gp("odom_topic"), self.on_referee, ref_qos)
        latest = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        for i, topic in enumerate(self.rig.topics):
            self.create_subscription(Image, topic, lambda m, i=i: self.on_rig_depth(i, m), latest)
            self.create_subscription(CameraInfo, topic + "/camera_info", lambda m, i=i: self.on_rig_info(i, m),
                                     qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, gp("map_topic"), self.on_map, 10)
        self.create_subscription(Bool, gp("ready_topic"), self.on_ready, 10)
        rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.pub_traj = self.create_publisher(HermiteTrajectoryMsg, gp("trajectory_topic"), rel)
        self.pub_cloud = self.create_publisher(PointCloud2, gp("cloud_topic"), 10)
        self.tf = TransformBroadcaster(self)
        self.create_timer(1.0 / GUARD_HZ, self.on_guard)
        self.create_timer(10.0, self.on_status)
        self._plan_thread = threading.Thread(target=self._plan_loop, name="plan", daemon=True)
        self.get_logger().info(
            f"neo_planner ready: scenario {self.scn.name}, cruise z {self.cruise_z:.2f} m, v_max {self.v_max}, "
            f"replan {self.replan_period} s, dTf {self.dt_ahead} s, step {self.longitu_step} m, margins: inflation "
            f"{inflation:.2f}, hard {self.hard_clearance:.2f}, soft {self.safe_dis_norm:.2f}, emergency "
            f"{self.safe_dis_emer:.2f} m, rig {self.rig.topics}, model {gp('model_path')} on {self.device}, results in {self.out_dir}")

    # ------------------------------------------------------------------ helpers
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def pose_at(self, t, max_dt=0.05, buf=None):
        """(t, p, R, v) interpolated at t (kairos_planner_node.pose_at)."""
        with self.lock:
            poses = list(self.poses if buf is None else buf)
        if not poses:
            return None
        ts = np.array([x[0] for x in poses])
        j = int(np.searchsorted(ts, t))
        if j == 0 or j == len(ts):
            near = poses[min(j, len(ts) - 1)]
            return near if abs(near[0] - t) <= max_dt else None
        a, b = poses[j - 1], poses[j]
        if b[0] - a[0] > 2.0 * max_dt:
            near = a if t - a[0] <= b[0] - t else b
            return near if abs(near[0] - t) <= max_dt else None
        w = (t - a[0]) / max(b[0] - a[0], 1e-9)
        rot = Rotation.from_matrix(a[2]) * Rotation.from_rotvec(w * Rotation.from_matrix(a[2].T @ b[2]).as_rotvec())
        return t, (1.0 - w) * a[1] + w * b[1], rot.as_matrix(), (1.0 - w) * a[3] + w * b[3]

    def drone(self):
        """Newest planner state (odometry), NEO's _drone_state."""
        with self.lock:
            if not self.poses:
                return None
            t, p, R, v = self.poses[-1]
        return State(p, v, _yaw(R))

    # ------------------------------------------------------------------ callbacks
    def on_ready(self, msg):
        if msg.data and not self.ready:
            self.get_logger().info("trajectory server reports READY: test starts")
            self._plan_thread.start()
        self.ready = self.ready or bool(msg.data)

    def on_map(self, msg, pad=1.0, min_dt=0.1):
        """octomap's projected map, grown (unknown cells) to the arena bounds + pad before the ESDF: the ESDF marks its
        arena walls only on cells of the grid, and the grid covers only what has been seen. octomap publishes one map
        per cloud (2 x 15 Hz) and a rebuild of the 205 m arena takes ~10 ms: at most one every min_dt (the guards'
        rate; the planner replans every 0.2 s)."""
        t = self.now_s()
        if t - self.last_map_t < min_dt:
            return
        self.last_map_t = t
        r = msg.info.resolution
        ox, oy, W, H = msg.info.origin.position.x, msg.info.origin.position.y, msg.info.width, msg.info.height
        x0, x1, y0, y1 = self.arena[0] - pad, self.arena[1] + pad, self.arena[2] - pad, self.arena[3] + pad
        cl = max(0, int(np.ceil((ox - x0) / r)))           # cells added left / below / right / above
        cb = max(0, int(np.ceil((oy - y0) / r)))
        cr = max(0, int(np.ceil((x1 - (ox + W * r)) / r)))
        ct = max(0, int(np.ceil((y1 - (oy + H * r)) / r)))
        if cl or cb or cr or ct:
            a = np.full((H + cb + ct, W + cl + cr), -1, np.int8)
            a[cb:cb + H, cl:cl + W] = np.asarray(msg.data, np.int8).reshape(H, W)
            info = copy.deepcopy(msg.info)
            info.width, info.height = a.shape[1], a.shape[0]
            info.origin.position.x, info.origin.position.y = ox - cl * r, oy - cb * r
            msg = types.SimpleNamespace(info=info, data=a.ravel())     # what ESDF.occupancy_map_cb reads
        self.esdf.occupancy_map_cb(msg)

    def on_rig_info(self, i, msg):
        cam = self.rig.cams[i]
        if self.rig_info[i] and cam.W == msg.width and abs(cam.fx - msg.k[0]) < 1e-6:
            return
        cam.set_intrinsics(msg.k[0], msg.k[4], msg.k[2], msg.k[5], msg.width, msg.height)
        self.rig_info[i] = True
        self.get_logger().info(f"camera {self.rig.topics[i]} fx={msg.k[0]:.1f} {msg.width}x{msg.height}")

    def on_rig_depth(self, i, msg, max_skew=0.01):
        t = to_sec(msg.header.stamp)
        self.rig_latest[i] = (t, msg)
        if any(x is None or abs(x[0] - t) > max_skew for x in self.rig_latest):
            return
        frame = [x[1] for x in self.rig_latest]
        self.rig_latest = [None] * len(self.rig_latest)
        if not all(self.rig_info) or t <= self.last_depth_t:
            return
        with self.lock:
            newest = self.poses[-1][0] if self.poses else -np.inf
        if newest < t:                      # wait for the odometry sample after the frame (kairos_planner)
            self.depth_wait.append(frame)
            return
        self.process_depth(frame)

    def on_odom(self, msg):
        t, p, R, v = odom_to_state(msg, self.twist_in_body)
        with self.lock:
            self.poses.append((t, p, R, v))
        ready = None
        while self.depth_wait and to_sec(self.depth_wait[0][0].header.stamp) <= t:
            ready = self.depth_wait.popleft()
        if ready is not None:
            self.process_depth(ready)

    def process_depth(self, frame):
        t = to_sec(frame[0].header.stamp)
        if t <= self.last_depth_t:
            return
        pose = self.pose_at(t)
        if pose is None:
            return
        _, p, R, v = pose
        depths = [image_to_depth_m(m) for m in frame]
        with self.lock:      # the panorama is made by the planning thread when it uses the frame (network_depth)
            self.latest = [t, depths, State(p, v, _yaw(R)), None]
        self.last_depth_t = t
        self.stats["frames"] += 1
        self.publish_clouds(t, depths, p, R)

    def publish_clouds(self, t, depths, p, R):
        """Both images as point clouds for octomap, in the camera's optical frame; TF world -> neo_cam_<i> at the
        image stamp. Hits as neo_planner's depth_to_pointcloud_node (valid depth within [min_range, max_range]).
        Free rays as the KAIROS map (dyn_expert DepthCamera.backproject): a pixel beyond max_range, or without a
        return when camera.invalid_is_free (true in simulation), is a point 0.5 m beyond max_range, which octomap
        (sensor_model.max_range) only clears along its ray. neo_planner dropped them, and the cells a moving sphere
        had filled were never cleared where nothing stands behind it: the arenas filled with sphere trails
        (2026-11-02)."""
        c = self.cfg.camera
        s = self.cloud_stride
        for i, (cam, depth) in enumerate(zip(self.rig.cams, depths)):
            o_w, R_wo = cam.camera_pose(p, R)
            tf = TransformStamped()
            tf.header.stamp = to_stamp(t)
            tf.header.frame_id = self.world_frame
            tf.child_frame_id = f"neo_cam_{i}"
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = [float(x) for x in o_w]
            q = Rotation.from_matrix(R_wo).as_quat()
            tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = \
                [float(x) for x in q]
            self.tf.sendTransform(tf)
            H, W = depth.shape
            vs, us = np.mgrid[0:H:s, 0:W:s]
            z = depth[vs, us].astype(np.float32)
            valid = np.isfinite(z) & (z > c.min_range)
            hit = valid & (z <= c.max_range)
            far = (valid & (z > c.max_range)) | (~valid & bool(c.invalid_is_free))
            u = us * (cam.W / W)
            v_ = vs * (cam.H / H)
            dirs = np.stack([(u - cam.cx) / cam.fx, (v_ - cam.cy) / cam.fy, np.ones_like(u)], -1).astype(np.float32)
            far_scale = (c.max_range + 0.5) / np.linalg.norm(dirs[far], axis=1)
            pts = np.concatenate([dirs[hit] * z[hit, None], dirs[far] * far_scale[:, None]]).astype(np.float32)
            if len(pts) == 0:
                continue
            msg = PointCloud2()
            msg.header.stamp = to_stamp(t)
            msg.header.frame_id = f"neo_cam_{i}"
            msg.height, msg.width = 1, len(pts)
            msg.is_dense, msg.is_bigendian = False, False
            msg.fields = [PointField(name=n, offset=4 * k, datatype=PointField.FLOAT32, count=1)
                          for k, n in enumerate("xyz")]
            msg.point_step, msg.row_step = 12, 12 * len(pts)
            msg.data = pts.tobytes()
            self.pub_cloud.publish(msg)

    # ------------------------------------------------------------------ referee / episodes
    def on_referee(self, msg):
        t, p, R, v = odom_to_state(msg, self.twist_in_body)
        with self.lock:
            self.ref_poses.append((t, p, R, v))
            ep = self.episode
        if ep is None or self.done:
            return
        ep["path"] += float(np.linalg.norm(p - ep["last_p"])) if ep["last_p"] is not None else 0.0
        ep["last_p"] = p.copy()
        for kind, d in (("static", float(self.scn.static_distance(p[None])[0])),
                        ("dynamic", float(self.scn.dynamic_distance(p, t).min()) if self.scn.dynamic else np.inf)):
            if d < self.coll_radius:
                if self.contact[kind] is None or t - self.contact[kind] > 0.5:     # a new contact (hits.py rule)
                    ep["hits_" + kind] += 1
                    self.get_logger().warn(f"[referee] {kind} collision at {np.round(p, 2)}")
                self.contact[kind] = t
        if np.linalg.norm(p - ep["goal"]) < self.goal_tol:
            self.end_episode(t, "reached", p)
        elif t > ep["t_limit"]:
            self.end_episode(t, "timeout", p)

    def start_episode(self, t, p):
        if (self.last_goal is not None and self.last_outcome != "reached"
                and np.linalg.norm(p - self.last_goal) > 2.0 * self.goal_tol):
            goal = self.last_goal.copy()            # dataset.keep_goal_after_failure
        else:
            goal = self.scn.route_goal(self.rng, p)
        dist = float(np.linalg.norm(goal - p))
        ep = dict(n=len(self.episodes) + 1, goal=goal, start=p.copy(), t_start=t, path=0.0, last_p=None,
                  hits_static=0, hits_dynamic=0,
                  t_limit=t + self.timeout_factor * dist / float(self.cfg.planner.v_nominal) + 10.0)
        with self.lock:
            self.episode = ep
        self.global_target = goal[:2].copy()
        self.reset_progress()
        self.get_logger().info(f"[neo] episode {ep['n']} start, goal {np.round(goal, 2)}")

    def end_episode(self, t, outcome, p):
        with self.lock:
            ep, self.episode = self.episode, None
        if ep is None:
            return
        dur = t - ep["t_start"]
        rec = dict(episode=ep["n"], outcome=outcome, duration=dur, path=ep["path"], mean_speed=ep["path"] / max(dur, 1e-6),
                   hits_static=ep["hits_static"], hits_dynamic=ep["hits_dynamic"], start=ep["start"].tolist(),
                   goal=ep["goal"].tolist(), end=p.tolist())
        self.episodes.append(rec)
        self.last_goal, self.last_outcome = ep["goal"], outcome
        self.get_logger().info(f"[neo] episode {ep['n']} end: {outcome}, {dur:.1f} s, {ep['path']:.0f} m, "
                               f"hits static {ep['hits_static']} dynamic {ep['hits_dynamic']}")
        self.write_summary()
        if self.max_episodes and len(self.episodes) >= self.max_episodes:
            self.done = True
            oc = {}
            for e in self.episodes:
                oc[e["outcome"]] = oc.get(e["outcome"], 0) + 1
            n = len(self.episodes)
            self.get_logger().info(
                f"real episodes {n}: reached {oc.get('reached', 0)} | success {100.0 * oc.get('reached', 0) / n:.0f}% | "
                f"timeout {oc.get('timeout', 0)} | hits static {sum(e['hits_static'] for e in self.episodes)} "
                f"dynamic {sum(e['hits_dynamic'] for e in self.episodes)}")

    def write_summary(self):
        pm = np.array(self.stats["plan_ms"]) if self.stats["plan_ms"] else np.zeros(1)
        s = dict(run_name=self.run_name, config=self.config_record, episodes=self.episodes,
                 stats={k: v for k, v in self.stats.items() if k not in ("plan_ms", "nn_ms")},
                 plan_ms=dict(median=float(np.median(pm)), p90=float(np.percentile(pm, 90)), n=len(self.stats["plan_ms"])),
                 nn_ms_median=float(np.median(self.stats["nn_ms"])) if self.stats["nn_ms"] else 0.0)
        with open(os.path.join(self.out_dir, "summary.json"), "w") as fh:
            json.dump(s, fh, indent=1)

    # ------------------------------------------------------------------ trajectories
    def publish(self, traj, mode_index=0):
        with self.lock:
            self.traj = traj
            goal = None if self.episode is None else self.episode["goal"]
        self.plan_id += 1
        self.pub_traj.publish(traj_to_msg(traj, HermiteTrajectoryMsg, self.world_frame, self.plan_id, mode_index, goal))

    def minco_to_hermite(self, t0):
        """The optimized 2D MINCO trajectory as quintic Hermite knots at cruise z, slowed so that its peak speed
        is <= v_max (neo_planner_node._install_trajectory time scaling)."""
        pl = self.planner
        pl.get_coeffs(pl.int_wpts, pl.ts)
        total = float(np.sum(pl.ts))
        peak = max(float(np.linalg.norm(pl.get_vel(t))) for t in np.arange(0.0, total, 0.02))
        k = peak / self.v_max if self.v_max > 1e-3 and peak > self.v_max else 1.0
        tk = np.concatenate([[0.0], np.cumsum(pl.ts)])
        Pk = np.array([[*pl.get_pos(t), self.cruise_z] for t in tk])
        Vk = np.array([[*(pl.get_vel(t) / k), 0.0] for t in tk])
        Ak = np.array([[*(pl.get_acc(t) / (k * k)), 0.0] for t in tk])
        return HermiteTrajectory(Pk, Vk, Ak, np.asarray(pl.ts, float) * k, t0)

    def escape_traj(self, t):
        st = self.drone()
        p0 = np.array([*st.global_pos[:2], self.cruise_z])
        v = np.array([*self.escape_vel, 0.0])
        T = 1.0
        return HermiteTrajectory(np.array([p0, p0 + v * T]), np.array([v, v]), np.zeros((2, 3)), [T], t)

    def hold_traj(self, t):
        st = self.drone()
        p0 = np.array([*st.global_pos[:2], self.cruise_z])
        return HermiteTrajectory(np.array([p0, p0]), np.zeros((2, 3)), np.zeros((2, 3)), [0.5], t)

    def traj_state(self, t):
        with self.lock:
            tr = self.traj
        if tr is None:
            return None, None, 0.0
        p, v, _ = tr.state(t)
        return p, v, tr.t_end - t

    # ------------------------------------------------------------------ guards (neo_planner_node._publish_cmd)
    def on_guard(self):
        if not self.ready or self.done or self.episode is None or not self.esdf.is_ready():
            return
        t = self.now_s()
        st = self.drone()
        if st is None:
            return
        if self.escape_active:
            self.publish(self.escape_traj(t), mode_index=-1)
            return
        pos = st.global_pos[:2]
        if self.esdf.get_edt_dis(pos) < self.safe_dis_emer:
            self.get_logger().warn("[NEO] Collision guard! ESDF too close - cancel traj", throttle_duration_sec=0.5)
            self.stats["guard"] += 1
            self.enter_escape(t)
            return
        with self.lock:
            tr = self.traj
        if tr is not None and t < tr.t_end:
            for dt in np.arange(LOOKAHEAD_STEP_S, LOOKAHEAD_HORIZON_S + 1e-6, LOOKAHEAD_STEP_S):
                if t + dt > tr.t_end:
                    break
                d = self.esdf.get_edt_dis(tr.state(t + dt)[0][:2])
                if d < self.safe_dis_emer:
                    self.get_logger().warn(f"[NEO] Look-ahead guard! clearance {d:.2f}m - cancel traj",
                                           throttle_duration_sec=0.5)
                    self.stats["lookahead"] += 1
                    self.enter_escape(t)
                    return
            gap = float(np.linalg.norm(tr.state(t)[0][:2] - pos))
            if gap > EMERGENCY_STOP_THRESHOLD:
                self.emergency_first_t = self.emergency_first_t or t
                if t - self.emergency_first_t > EMERGENCY_STOP_PERSIST_TIME:
                    self.emergency_first_t = None
                    self.enter_escape(t)
            else:
                self.emergency_first_t = None

    # ------------------------------------------------------------------ NEO escape / stuck (unchanged logic)
    def find_escape_direction(self):
        pos = self.drone().global_pos[:2]
        to_goal = self.global_target - pos
        n = np.linalg.norm(to_goal)
        to_goal_dir = to_goal / n if n > 1e-3 else np.array([1.0, 0.0])
        best_score, best_dir = -np.inf, np.array([1.0, 0.0])
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            d = np.array([np.cos(angle), np.sin(angle)])
            score = min(self.esdf.get_edt_dis(pos + d * ESCAPE_PROBE_DIST),
                        self.esdf.get_edt_dis(pos + d * ESCAPE_PROBE_DIST * 0.5)) + 0.2 * float(d @ to_goal_dir)
            if score > best_score:
                best_score, best_dir = score, d
        return best_dir

    def enter_escape(self, t):
        if self.escape_active or not self.esdf.is_ready():
            return
        d = self.find_escape_direction()
        self.escape_vel = d * ESCAPE_VELOCITY
        self.escape_active = True
        self.escape_start = t
        self.replan_fail_count = 0
        self.reset_progress()
        self.emergency_first_t = None
        self.stats["escapes"] += 1
        self.publish(self.escape_traj(t), mode_index=-1)
        self.get_logger().warn(f"[NEO] >>> ESCAPE: vel=({d[0]:.2f},{d[1]:.2f})")

    def should_exit_escape(self, t):
        if t - self.escape_start > ESCAPE_MAX_TIME:
            self.get_logger().warn("[NEO] Escape timeout - exit")
            return True
        pos = self.drone().global_pos[:2]
        dist = self.esdf.get_edt_dis(pos)
        if dist > ESCAPE_EXIT_DIST:
            return True
        if self.esdf.get_edt_dis(pos + self.escape_vel * 1.0) < dist * 0.8:
            self.escape_vel = self.find_escape_direction() * ESCAPE_VELOCITY
        return False

    def exit_escape(self, t):
        self.get_logger().info("[NEO] <<< Exit escape mode")
        self.escape_active = False
        self.stuck_since = None
        self.publish(self.hold_traj(t))

    def check_stuck(self, t):
        st = self.drone()
        speed = float(np.linalg.norm(st.global_vel[:2]))
        pos = st.global_pos[:2].copy()
        if speed > STUCK_SPEED_THRESHOLD:
            self.stuck_since, self.last_pos_for_stuck = None, None
            return False
        if self.stuck_since is None:
            self.stuck_since, self.last_pos_for_stuck = t, pos
            return False
        if np.linalg.norm(pos - self.last_pos_for_stuck) > 0.5:
            self.stuck_since, self.last_pos_for_stuck = t, pos
            return False
        return t - self.stuck_since > STUCK_TIME_THRESHOLD

    def reset_progress(self):
        self.progress_ref = None

    def check_no_progress(self, t):
        d = float(np.linalg.norm(self.drone().global_pos[:2] - self.global_target))
        if self.progress_ref is None or d < self.progress_ref[0] - NO_PROGRESS_DELTA:
            self.progress_ref = (d, t)
            return False
        return t - self.progress_ref[1] > NO_PROGRESS_TIME

    # ------------------------------------------------------------------ NEO local target / state / NN
    def set_local_target(self, seed=0, shorter=False, ref_pos=None):
        curr = ref_pos if ref_pos is not None else self.drone().global_pos[:2]
        step = 2.0 if shorter else self.longitu_step
        dist = np.linalg.norm(self.global_target - curr)
        if dist < step:
            self.target_state[0] = self.global_target
            self.target_state[1] = np.zeros(2)
            return
        fwd = (self.global_target - curr) / dist
        lat = np.array([[fwd[1], -fwd[0]], [-fwd[1], fwd[0]]])
        local = curr + step * fwd + (np.random.normal(0, 1, 2) if seed > 0 else 0.0)
        lat_flag, lat_move, n_iter = 0, self.lateral_step, 0
        while self.esdf.is_ready() and self.esdf.has_collision(local) and n_iter < 8:
            local = local + lat_move * lat[lat_flag]
            lat_flag = 1 - lat_flag
            lat_move += self.lateral_step
            n_iter += 1
        g = self.global_target - local
        gn = np.linalg.norm(g)
        self.target_state[0] = local
        self.target_state[1] = (g / gn if gn > 1e-3 else g) * self.v_max * self.target_vel_frac

    def state_at(self, t_future_abs):
        """neo_planner_node._get_drone_state_at: the active trajectory's state dTf ahead if it is continuous with
        the drone, else the drone's position extrapolated with its (capped) velocity."""
        st = self.drone()
        out = State(st.global_pos, st.global_vel, st.yaw)
        p, v, remaining = self.traj_state(t_future_abs)
        if p is not None and remaining > 0.3 and np.linalg.norm(p[:2] - st.global_pos[:2]) < DISCONTINUITY_THRESHOLD:
            out.global_pos = np.array([p[0], p[1], self.cruise_z])
            out.global_vel = np.array([v[0], v[1], 0.0])
            return out
        speed = float(np.linalg.norm(out.global_vel[:2]))
        if speed > self.v_max:
            out.global_vel[:2] *= self.v_max / speed
        if speed < 0.1:
            out.global_vel[:2] = 0.0
        out.global_pos[:2] += out.global_vel[:2] * (t_future_abs - self.now_s())
        return out

    def nn_initial_guess(self, depth_u8, sync, init, target):
        """neo_planner_node._form_nn_input + _nn_initial_guess with 3D init / target at cruise z."""
        dp = sync.global_pos
        init_p = np.array([init.global_pos[0], init.global_pos[1], self.cruise_z])
        init_v = np.array([init.global_vel[0], init.global_vel[1], 0.0])
        tgt_p = np.array([target[0, 0], target[0, 1], self.cruise_z])
        tgt_v = np.array([target[1, 0], target[1, 1], 0.0])
        mot = nd.motion_info(sync.yaw, sync.body(sync.global_vel), sync.body(init_p - dp), sync.body(init_v - sync.global_vel),
                             sync.body(tgt_p - dp), sync.body(tgt_v - sync.global_vel))
        x = np.concatenate([depth_u8.reshape(-1).astype(np.float32), mot])
        tic = time.perf_counter()
        with torch.no_grad():
            out = self.net(torch.from_numpy(x)[None].to(self.device)).cpu().numpy().ravel()
        self.stats["nn_ms"].append(1e3 * (time.perf_counter() - tic))
        Rz = _rot_z(sync.yaw)
        w1 = Rz @ out[0:3] + dp
        w2 = Rz @ out[3:6] + dp
        int_wpts = np.array([[w1[0], w2[0]], [w1[1], w2[1]]])
        ts = np.clip(out[6:9], self.planner.T_min + 1e-3, self.planner.T_max - 1e-3)
        return int_wpts, ts

    def validate_planned_traj(self):
        cur = float(self.esdf.get_edt_dis(self.drone().global_pos[:2]))
        thr = min(self.hard_clearance, max(cur - 0.05, 0.1))
        for p in self.planner.get_pos_array():
            d = self.esdf.get_edt_dis(p[:2])
            if d < thr:
                raise ValueError(f"Trajectory grazing: clearance {d:.2f}m < hard {thr:.2f}m")

    # ------------------------------------------------------------------ planning loop (sim time)
    def _plan_loop(self):
        nxt = self.now_s()
        while rclpy.ok() and not self.done:
            t = self.now_s()
            if t < nxt:
                time.sleep(0.005)
                continue
            nxt = max(nxt + self.replan_period, t)
            try:
                if self.episode is None:
                    st = self.drone()
                    if st is not None:
                        self.start_episode(t, st.global_pos)
                    continue
                self.replan(t)
            except Exception as e:  # keep flying on a bug in one event, but say so
                self.get_logger().error(f"[NEO] replan error: {type(e).__name__}: {e}", throttle_duration_sec=2.0)

    def replan(self, t_now):
        """neo_planner_node._replan with sim time; the result starts dTf after t_now."""
        if not self.esdf.is_ready():
            self.get_logger().warn("[NEO] ESDF not ready yet - /projected_map not received", throttle_duration_sec=2.0)
            return
        if self.escape_active:
            if self.should_exit_escape(t_now):
                self.exit_escape(t_now)
            else:
                return
        if self.check_stuck(t_now):
            self.enter_escape(t_now)
            return
        if self.check_no_progress(t_now):
            self.get_logger().warn("[NEO] No progress to the goal - escape (local minimum)")
            self.enter_escape(t_now)
            return
        with self.lock:
            latest = self.latest
        if latest is None:
            return
        t_d, depths, sync, depth_u8 = latest
        if depth_u8 is None:     # the network's image of this frame, once (the rig panorama, as in the dataset)
            pano = self.rig.panorama(depths)
            pano[pano > self.cfg.camera.store_max_range] = 0.0      # collector._build_event
            depth_u8 = latest[3] = nd.depth_to_u8(pano)
        if t_now - t_d > DEPTH_MAX_LAG_S:
            self.stats["lag_skip"] += 1
            self.get_logger().warn(f"[NEO] Skip replan, depth lag {(t_now - t_d) * 1000:.0f}ms",
                                   throttle_duration_sec=2.0)
            return
        if np.linalg.norm(sync.global_pos[:2] - self.global_target) < 0.8:
            return                                          # near the goal: the last plan ends there
        cur_clear = float(self.esdf.get_edt_dis(sync.global_pos[:2]))
        self.planner.safe_dis = float(np.clip(cur_clear - 0.05, self.hard_clearance + 0.05, self.safe_dis_norm))
        t0 = t_now + self.dt_ahead
        tic = time.perf_counter()
        ok, head = False, None
        for shorter in (False, True):
            for seed in range(11):
                if time.perf_counter() - tic > REPLAN_BUDGET_S:
                    break
                self.set_local_target(seed, shorter=shorter, ref_pos=sync.global_pos[:2])
                init = self.state_at(t0)
                head = np.zeros((3, 2))
                head[0], head[1] = init.global_pos[:2], init.global_vel[:2]
                tail = np.zeros((3, 2))
                tail[0], tail[1] = self.target_state[0], self.target_state[1]
                try:
                    wpts, ts = self.nn_initial_guess(depth_u8, sync, init, self.target_state)
                    self.planner.warm_start_plan(self.esdf, head, tail, wpts, ts)
                    self.validate_planned_traj()
                    ok = True
                    break
                except Exception:
                    pass
            if ok or time.perf_counter() - tic > REPLAN_BUDGET_S:
                break
        ms = 1e3 * (time.perf_counter() - tic)
        if not ok:
            self.replan_fail_count += 1
            self.stats["fails"] += 1
            self.get_logger().warn(f"[NEO] Replan failed ({self.replan_fail_count}x in a row)")
            if self.replan_fail_count >= 2:
                self.enter_escape(self.now_s())
            return
        self.replan_fail_count = 0
        st = self.drone()
        if np.linalg.norm(head[0] - st.global_pos[:2]) > DISCONTINUITY_THRESHOLD:
            self.get_logger().warn("[NEO] Discontinuity - replan from actual")
            self.set_local_target(0)
            tail = np.zeros((3, 2))
            tail[0], tail[1] = self.target_state[0], self.target_state[1]
            head = np.zeros((3, 2))
            head[0] = st.global_pos[:2]
            v = st.global_vel[:2].copy()
            sp = float(np.linalg.norm(v))
            head[1] = v * (self.v_max / sp) if sp > self.v_max else v
            for _ in range(5):
                try:
                    wpts, ts = self.nn_initial_guess(depth_u8, sync, init, self.target_state)
                    self.planner.warm_start_plan(self.esdf, head, tail, wpts, ts)
                    self.validate_planned_traj()
                    break
                except Exception:
                    pass
            else:
                return
            t0 = self.now_s()
        self.stats["replans"] += 1
        self.stats["plan_ms"].append(ms)
        if self.now_s() > t0:
            self.stats["late"] += 1
        self.publish(self.minco_to_hermite(t0))
        self.emergency_first_t = None

    def on_status(self):
        if not self.ready:
            return
        pm = self.stats["plan_ms"][-50:]
        st = self.drone()
        self.get_logger().info(
            f"[neo] at {np.round(st.global_pos, 1) if st is not None else '-'}, episodes {len(self.episodes)}, replans {self.stats['replans']} (fail {self.stats['fails']}, late "
            f"{self.stats['late']}), escapes {self.stats['escapes']}, plan median "
            f"{np.median(pm) if pm else 0:.0f} ms, depth frames {self.stats['frames']}, map "
            f"{'ready' if self.esdf.is_ready() else 'NOT ready'}")


def main(args=None):
    rclpy.init(args=args)
    node = NeoNode()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    try:
        while rclpy.ok() and not node.done:
            ex.spin_once(timeout_sec=0.1)
        node.write_summary()
        while rclpy.ok():                           # keep the last plan / hold until the launch is stopped
            ex.spin_once(timeout_sec=0.5)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.write_summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
