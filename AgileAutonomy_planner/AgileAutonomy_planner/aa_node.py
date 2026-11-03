#!/usr/bin/env python3
"""aa_node: Agile Autonomy inference in the KAIROS test stack (x500_oakd_x2_60_tw2, dyn_long200 worlds).

The deployment pipeline of github.com/uzh-rpg/agile_autonomy, with no map and no other safety layer:
  PlannerBase._generate_plan / trajectory_decision: network on the newest depth image and state; modes sorted by
      |alpha| (predicted cost); the best mode, and any other with best_alpha / alpha > accept_thresh (0.95), are
      candidates
  AgileAutonomy::selectBestNetworkPrediction: body-frame points to the world, z clamped to the altitude band,
      the candidate closest to the previous reference wins
  AgileAutonomy::trajectoryCallback: first point replaced by the start state, 5th-order polynomial with
      continuity 3 (position, velocity, acceleration, jerk) fitted to the points (TrajectoryExt::
      fitPolynomialCoeffs), resampled at constant arc-length speed min(|v| + 1, v_max)
      (setConstantArcLengthSpeed) and fitted again; the MPC then tracks it
What differs, so that it runs on the stack every method is tested on:
  * sensor: the rig's two depth images, paired by stamp as kairos_planner; the network gets their range
    panorama (dyn_expert CameraRig.panorama, = the dataset's depth). State: odom_topic (/kairos/odom, PX4's
    estimate), interpolated at the depth stamp; sim time.
  * timing: a plan every replan_period (planner.replan_dt 0.2 s; Agile Autonomy 15 Hz), starting dt_ahead
    (planner.foresee_dt 0.4 s) later from the state of the trajectory in flight, as the training samples
    (aa_data); points every aa_data.LABEL_DT. The previous reference is compared at the same times.
  * output: the polynomial as HermiteTrajectory knots (exact: one quintic) to dyn_expert's trajectory_server;
    Agile Autonomy's MPC is replaced by it, as for every method.
  * the start-up nudge of fitPolynomialCoeffs (+0.2 m/s on world x below 0.1 m/s) points to the goal.
  * goal: within goal_approach_dist (2 m) of the pad, a rest-to-rest quintic to the pad (Agile Autonomy stops
    by reference progress; the network has no goal distance input).
  * mission and referee as kairos_planner / neo_node: pads A <-> B, reached within dataset.goal_reach_tol,
    timeout timeout_factor * distance / v_nominal + 10 s, collisions counted (Gazebo pose), episode continued.
"""
import json
import math
import os
import sys
import threading
import time
from collections import deque

import numpy as np
import rclpy
import torch
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool

from dyn_nav_interfaces.msg import HermiteTrajectory as HermiteTrajectoryMsg
from dyn_expert.core.camera import CameraRig
from dyn_expert.core.config import load_config
from dyn_expert.core.hermite import HermiteTrajectory
from dyn_expert.core.scenario import Scenario
from dyn_expert.nodes.common import image_to_depth_m, odom_to_state, to_sec, traj_to_msg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aa_data as ad
from aa_net import PlaNet, depth_input

POLY_ORDER, CONTINUITY = 5, 3            # traj_world_ext.fitPolynomialCoeffs(5, 3)
DEPTH_MAX_LAG_S = 0.3


def _rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _yaw(R):
    return math.atan2(R[1, 0], R[0, 0])


class Poly:
    """TrajectoryExt's polynomial: coefficients c (POLY_ORDER + 1, 3), p(t) = sum c_i t^i, from absolute time t0."""

    def __init__(self, c, t0, t_end):
        self.c, self.t0, self.t_end = np.asarray(c, float), float(t0), float(t_end)

    def eval(self, t, der=0):
        """Derivative der at time from start t (scalar or (N,)) -> (3,) or (N, 3)."""
        t = np.atleast_1d(np.asarray(t, float))
        out = np.zeros((len(t), 3))
        for i in range(der, len(self.c)):
            k = math.factorial(i) / math.factorial(i - der)
            out += k * t[:, None] ** (i - der) * self.c[i]
        return out

    def state(self, t_abs):
        """p, v, a, j at absolute time, held at the end."""
        t = min(max(t_abs - self.t0, 0.0), self.t_end)
        p, v, a, j = (self.eval(t, d)[0] for d in range(4))
        if t_abs - self.t0 > self.t_end:
            v, a, j = np.zeros(3), np.zeros(3), np.zeros(3)
        return p, v, a, j


def fit_poly(start, times, points, goal_dir):
    """TrajectoryExt::fitPolynomialCoeffs: c0..c3 from the start (p, v, a, j); c4, c5 least squares to the points
    at times (s from start). Start-up nudge: +0.2 m/s along the goal direction below 0.1 m/s."""
    p, v, a, j = (np.asarray(x, float) for x in start)
    v = v.copy()
    if np.linalg.norm(v) < 0.1:
        v = v + 0.2 * np.asarray(goal_dir, float)
    c = np.zeros((POLY_ORDER + 1, 3))
    c[0], c[1], c[2], c[3] = p, v, 0.5 * a, j / 6.0
    t = np.asarray(times, float)
    A = np.stack([t ** k for k in range(CONTINUITY + 1, POLY_ORDER + 1)], 1)
    b = np.asarray(points, float) - sum(c[k] * t[:, None] ** k for k in range(CONTINUITY + 1))
    c[CONTINUITY + 1:] = np.linalg.lstsq(A, b, rcond=None)[0]
    return c


def constant_arc_length(c, start, speed, n, dt, goal_dir):
    """TrajectoryExt::setConstantArcLengthSpeed: points every speed * dt of arc length along the polynomial (100
    samples over its duration), extrapolated linearly if too short, then fitted again. None if < 2 points."""
    poly = Poly(c, 0.0, n * dt)
    end = n * dt
    step = end / 100
    pts = [np.asarray(start[0], float)]
    jj, t, acc = 1, step, 0.0
    prev = poly.eval(t)[0]
    while t <= end:
        cur = poly.eval(t)[0]
        acc += float(np.linalg.norm(cur - prev))
        if acc >= speed * jj * dt:
            pts.append(cur)
            jj += 1
        if len(pts) >= n + 1:
            break
        prev = cur
        t += step
    if jj < 2:
        return None
    while len(pts) < n + 1:
        pts.append(pts[-1] + (pts[-1] - pts[-2]))
    return fit_poly(start, dt * np.arange(1, n + 1), np.array(pts[1:]), goal_dir)


def rest_to_rest(start, goal, T):
    """Quintic from (p, v, a) to (goal, 0, 0) in T, as polynomial coefficients from the start."""
    p0, v0, a0 = (np.asarray(x, float) for x in start[:3])
    A = goal - p0 - v0 * T - 0.5 * a0 * T * T
    B = -v0 - a0 * T
    C = -a0
    c3 = (10 * A - 4 * B * T + 0.5 * C * T * T) / T ** 3
    c4 = (-15 * A + 7 * B * T - C * T * T) / T ** 4
    c5 = (6 * A - 3 * B * T + 0.5 * C * T * T) / T ** 5
    return np.stack([p0, v0, 0.5 * a0, c3, c4, c5])


class AANode(Node):
    def __init__(self):
        super().__init__("aa_planner")
        p = self.declare_parameter
        p("config_file", "")
        p("scenario_file", "")
        p("model_path", "")
        p("out_dir", os.path.expanduser("~/kairos_test"))
        p("run_name", "agile_autonomy")
        p("seed", 0)
        p("episodes", 0)
        p("odom_topic", "/kairos/odom")
        p("odom_twist_in_body", True)
        p("referee_odom_topic", "")
        p("trajectory_topic", "/dyn_nav/trajectory")
        p("ready_topic", "/dyn_nav/ready")
        p("world_frame", "world")
        p("v_max", 0.0)                    # 0 = planner.v_max (Agile Autonomy: test_time_velocity)
        p("timeout_factor", 0.0)
        p("replan_period", 0.0)            # 0 = planner.replan_dt
        p("planning_time_ahead", 0.0)      # 0 = planner.foresee_dt
        p("accept_thresh", 0.95)           # deploy_settings.yaml test_time.accept_thresh
        p("goal_approach_dist", 2.0)
        p("device", "auto")
        gp = lambda n: self.get_parameter(n).value

        self.cfg = load_config(os.path.expanduser(gp("config_file")))
        P, D = self.cfg.planner, self.cfg.dataset
        self.scn = Scenario.load(os.path.expanduser(gp("scenario_file")))
        self.rig = CameraRig(self.cfg.camera)
        self.rng = np.random.default_rng(int(gp("seed")))
        self.v_max = float(gp("v_max")) or float(P.v_max)
        self.timeout_factor = float(gp("timeout_factor")) or float(D.timeout_factor)
        self.replan_period = float(gp("replan_period")) or float(P.replan_dt)
        self.dt_ahead = float(gp("planning_time_ahead")) or float(P.foresee_dt)
        self.accept_thresh = float(gp("accept_thresh"))
        self.approach_dist = float(gp("goal_approach_dist"))
        z = float(self.scn.flight_z if self.scn.flight_z is not None else 2.0)
        zb = float(self.scn.z_band if self.scn.z_band is not None else 1.0)
        self.z_lo, self.z_hi = z - zb, z + zb             # test_time_min_z / max_z
        self.goal_z = z
        self.goal_tol = float(D.goal_reach_tol)
        self.coll_radius = float(D.collision_radius)
        self.max_episodes = int(gp("episodes"))
        self.world_frame = gp("world_frame")
        self.twist_in_body = bool(gp("odom_twist_in_body"))

        dev = gp("device")
        self.device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if dev == "auto" else dev)
        self.net = PlaNet(pretrained=False).to(self.device)
        self.net.load_state_dict(torch.load(os.path.expanduser(gp("model_path")), map_location=self.device))
        self.net.eval()
        with torch.no_grad():
            for _ in range(20):
                self.net(torch.zeros(1, 3, ad.PANO_H, ad.PANO_W, device=self.device),
                         torch.zeros(1, ad.STATE_SIZE, device=self.device))

        self.lock = threading.Lock()
        self.poses = deque(maxlen=400)
        self.rig_latest = [None] * len(self.rig.cams)
        self.rig_info = [False] * len(self.rig.cams)
        self.depth_wait = deque(maxlen=8)
        self.last_depth_t = -1.0
        self.latest = None                          # [t, depths, (p, R, v) at t, depth tensor or None]
        self.poly = None                            # Poly in flight
        self.ready = False
        self.done = False
        self.plan_id = 0
        self.goal = None

        self.episode = None
        self.episodes = []
        self.last_goal, self.last_outcome = None, None
        self.contact = dict(static=None, dynamic=None)
        self.stats = dict(replans=0, late=0, lag_skip=0, frames=0, approach=0, multi=0, no_arc=0,
                          plan_ms=[], nn_ms=[], acc_max=[])
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(os.path.expanduser(gp("out_dir")), f"{stamp}_{self.scn.name}_agile_autonomy")
        os.makedirs(self.out_dir, exist_ok=True)
        self.run_name = gp("run_name")
        self.config_record = dict(model_path=os.path.expanduser(gp("model_path")), v_max=self.v_max,
                                  replan_period=self.replan_period, planning_time_ahead=self.dt_ahead,
                                  accept_thresh=self.accept_thresh, label_dt=ad.LABEL_DT,
                                  out_seq_len=ad.OUT_SEQ_LEN, goal_approach_dist=self.approach_dist,
                                  z_band=[self.z_lo, self.z_hi], timeout_factor=self.timeout_factor,
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
        self.create_subscription(Bool, gp("ready_topic"), self.on_ready, 10)
        rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.pub_traj = self.create_publisher(HermiteTrajectoryMsg, gp("trajectory_topic"), rel)
        self.create_timer(10.0, self.on_status)
        self._plan_thread = threading.Thread(target=self._plan_loop, name="plan", daemon=True)
        self.get_logger().info(
            f"agile_autonomy ready: scenario {self.scn.name}, v_max {self.v_max}, replan {self.replan_period} s, "
            f"dTf {self.dt_ahead} s, points {ad.OUT_SEQ_LEN} x {ad.LABEL_DT} s, z band [{self.z_lo}, {self.z_hi}], "
            f"rig {self.rig.topics}, model {gp('model_path')} on {self.device}, results in {self.out_dir}")

    # ------------------------------------------------------------------ helpers (as neo_node)
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def pose_at(self, t, max_dt=0.05):
        with self.lock:
            poses = list(self.poses)
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
        with self.lock:
            return self.poses[-1] if self.poses else None

    # ------------------------------------------------------------------ callbacks
    def on_ready(self, msg):
        if msg.data and not self.ready:
            self.get_logger().info("trajectory server reports READY: test starts")
            self._plan_thread.start()
        self.ready = self.ready or bool(msg.data)

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
        if newest < t:
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
        depths = [image_to_depth_m(m) for m in frame]
        with self.lock:      # the panorama is made by the planning thread when it uses the frame
            self.latest = [t, depths, pose[1:], None]
        self.last_depth_t = t
        self.stats["frames"] += 1

    # ------------------------------------------------------------------ referee / episodes (as neo_node)
    def on_referee(self, msg):
        t, p, R, v = odom_to_state(msg, self.twist_in_body)
        with self.lock:
            ep = self.episode
        if ep is None or self.done:
            return
        ep["path"] += float(np.linalg.norm(p - ep["last_p"])) if ep["last_p"] is not None else 0.0
        ep["last_p"] = p.copy()
        for kind, d in (("static", float(self.scn.static_distance(p[None])[0])),
                        ("dynamic", float(self.scn.dynamic_distance(p, t).min()) if self.scn.dynamic else np.inf)):
            if d < self.coll_radius:
                if self.contact[kind] is None or t - self.contact[kind] > 0.5:
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
            goal = self.last_goal.copy()
        else:
            goal = self.scn.route_goal(self.rng, p)
        dist = float(np.linalg.norm(goal - p))
        ep = dict(n=len(self.episodes) + 1, goal=goal, start=p.copy(), t_start=t, path=0.0, last_p=None,
                  hits_static=0, hits_dynamic=0,
                  t_limit=t + self.timeout_factor * dist / float(self.cfg.planner.v_nominal) + 10.0)
        with self.lock:
            self.episode = ep
        self.goal = goal.copy()
        self.get_logger().info(f"[aa] episode {ep['n']} start, goal {np.round(goal, 2)}")

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
        self.get_logger().info(f"[aa] episode {ep['n']} end: {outcome}, {dur:.1f} s, {ep['path']:.0f} m, "
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
        am = np.array(self.stats["acc_max"]) if self.stats["acc_max"] else np.zeros(1)
        s = dict(run_name=self.run_name, config=self.config_record, episodes=self.episodes,
                 stats={k: v for k, v in self.stats.items() if k not in ("plan_ms", "nn_ms", "acc_max")},
                 plan_ms=dict(median=float(np.median(pm)), p90=float(np.percentile(pm, 90)), n=len(self.stats["plan_ms"])),
                 nn_ms_median=float(np.median(self.stats["nn_ms"])) if self.stats["nn_ms"] else 0.0,
                 plan_acc_max=dict(median=float(np.median(am)), p90=float(np.percentile(am, 90))))
        with open(os.path.join(self.out_dir, "summary.json"), "w") as fh:
            json.dump(s, fh, indent=1)

    # ------------------------------------------------------------------ trajectories
    def publish(self, poly):
        """The polynomial as quintic Hermite knots every LABEL_DT (each piece is the polynomial itself)."""
        n = int(round(poly.t_end / ad.LABEL_DT))
        tk = ad.LABEL_DT * np.arange(n + 1)
        traj = HermiteTrajectory(poly.eval(tk, 0), poly.eval(tk, 1), poly.eval(tk, 2), np.full(n, ad.LABEL_DT),
                                 poly.t0)
        with self.lock:
            self.poly = poly
            goal = None if self.episode is None else self.episode["goal"]
        self.plan_id += 1
        self.pub_traj.publish(traj_to_msg(traj, HermiteTrajectoryMsg, self.world_frame, self.plan_id, 0, goal))
        self.stats["acc_max"].append(float(np.linalg.norm(poly.eval(tk, 2), axis=1).max()))

    def start_state(self, t0):
        """Reference state at t0 (p, v, a, j); the drone's state before the first plan."""
        with self.lock:
            poly = self.poly
        if poly is not None:
            return poly.state(t0)
        d = self.drone()
        return d[1].copy(), np.zeros(3), np.zeros(3), np.zeros(3)

    # ------------------------------------------------------------------ planning loop
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
                    d = self.drone()
                    if d is not None:
                        self.start_episode(t, d[1])
                    continue
                self.replan(t)
            except Exception as e:
                self.get_logger().error(f"[aa] replan error: {type(e).__name__}: {e}", throttle_duration_sec=2.0)

    def replan(self, t_now):
        with self.lock:
            latest = self.latest
        if latest is None:
            return
        t_d, depths, (p_d, R_d, v_d), x = latest
        if t_now - t_d > DEPTH_MAX_LAG_S:
            self.stats["lag_skip"] += 1
            self.get_logger().warn(f"[aa] skip replan, depth lag {(t_now - t_d) * 1000:.0f} ms", throttle_duration_sec=2.0)
            return
        tic = time.perf_counter()
        t0 = t_now + self.dt_ahead
        start = self.start_state(t0)
        goal = np.array([self.goal[0], self.goal[1], self.goal_z])
        to_goal = goal - start[0]
        dist = float(np.linalg.norm(to_goal))
        gdir = to_goal / max(dist, 1e-6)
        if dist < self.approach_dist:
            T = max(2.0, 1.875 * dist / self.v_max)
            self.stats["approach"] += 1
            self.finish(Poly(rest_to_rest(start, goal, T), t0, T), tic, t0)
            return
        if x is None:                       # the network's image of this frame, once (the dataset's panorama)
            pano = self.rig.panorama(depths)
            pano[~np.isfinite(pano)] = 0.0
            pano[pano > self.cfg.camera.store_max_range] = 0.0      # collector._build_event
            x = latest[3] = depth_input(torch.as_tensor(pano, dtype=torch.float32, device=self.device)[None])
        yaw = _yaw(R_d)                     # the local frame of the image (dataset frame_yaw)
        Rz = _rot_z(yaw)
        state = ad.state_vector(Rz.T @ R_d, Rz.T @ start[1], Rz.T @ gdir)
        tn = time.perf_counter()
        with torch.no_grad():
            out = self.net(x, torch.as_tensor(state, device=self.device)[None])[0].cpu().numpy()
        self.stats["nn_ms"].append(1e3 * (time.perf_counter() - tn))
        # PlanLearner.inference + trajectory_decision: sort by |alpha|, keep near-equal modes
        out = out[np.abs(out[:, 0]).argsort()]
        alphas = np.abs(out[:, 0])
        cand = [i for i in range(len(out)) if i == 0 or alphas[0] / max(alphas[i], 1e-12) > self.accept_thresh]
        times = ad.label_times()
        worlds = []
        for i in cand:
            pts = out[i, 1:].reshape(ad.STATE_DIM, ad.OUT_SEQ_LEN).T @ Rz.T + start[0]
            pts[:, 2] = np.clip(pts[:, 2], self.z_lo, self.z_hi)
            worlds.append(pts)
        best = 0
        with self.lock:
            prev = self.poly
        if prev is not None and len(worlds) > 1:
            ref = np.stack([prev.state(t0 + tk)[0] for tk in times])
            best = int(np.argmin([np.linalg.norm(w - ref, axis=1).sum() for w in worlds]))
            self.stats["multi"] += 1
        c = fit_poly(start, times, worlds[best], gdir)
        d = self.drone()
        speed = min(float(np.linalg.norm(d[3])) + 1.0, self.v_max)
        c2 = constant_arc_length(c, start, speed, ad.OUT_SEQ_LEN, ad.LABEL_DT, gdir)
        if c2 is None:
            self.stats["no_arc"] += 1
            c2 = c
        self.finish(Poly(c2, t0, ad.OUT_SEQ_LEN * ad.LABEL_DT), tic, t0)

    def finish(self, poly, tic, t0):
        self.stats["replans"] += 1
        self.stats["plan_ms"].append(1e3 * (time.perf_counter() - tic))
        if self.now_s() > t0:
            self.stats["late"] += 1
        self.publish(poly)

    def on_status(self):
        if not self.ready:
            return
        pm = self.stats["plan_ms"][-50:]
        d = self.drone()
        self.get_logger().info(
            f"[aa] at {np.round(d[1], 1) if d is not None else '-'}, episodes {len(self.episodes)}, replans "
            f"{self.stats['replans']} (late {self.stats['late']}, lag skip {self.stats['lag_skip']}, approach "
            f"{self.stats['approach']}), plan median {np.median(pm) if pm else 0:.0f} ms, depth frames {self.stats['frames']}")


def main(args=None):
    rclpy.init(args=args)
    node = AANode()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    try:
        while rclpy.ok() and not node.done:
            ex.spin_once(timeout_sec=0.1)
        node.write_summary()
        while rclpy.ok():
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
