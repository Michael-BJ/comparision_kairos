#!/usr/bin/env python3
"""sando_bridge: SANDO (ws/src/sando, v0.0.3) in the KAIROS test stack, with its original sensors.

SANDO's onboard stack (sando_node, obstacle_tracker_node, ACL global_mapper_node) is started unchanged by
launch/sando_test.launch.py. This node gives it what SANDO's simulator / hardware drivers gave it, from the test
stack, and executes its plan the way every method in this test is executed:
  * state: odom_topic (/kairos/odom = PX4's estimate in the world frame) -> <ns>/state (dynus_interfaces/State,
    world velocity), and TF map -> <ns>/base_link at the odometry stamp. SANDO's simulation frame "map" is the
    world frame of the scenario. Sensor frames as SANDO's quadrotor.urdf.xacro, on the x500's base_link (0.24 m
    above the odometry frame = the model origin): <ns>/<ns>_livox (Mid-360, 0 0 0.2, pitch 0.3),
    <ns>/<ns>_d435_link (D435, 0.05 0 0; the frame Gazebo's depth points are in) and
    <ns>/<ns>_d435_depth_optical_frame.
  * plan: execute "goal" (default) = SANDO's own setpoint stream <ns>/goal (p, v, a, j at 100 Hz, popped from its
    committed plan, what its hardware path streams to PX4) -> every setpoint a 0.2 s HermiteTrajectory (the
    setpoint extrapolated with its jerk: exact for SANDO's cubic pieces) -> trajectory_server, whose newest
    t0 <= now is tracked. execute "trajs" = the trajectory SANDO shares with other agents (/trajs, own id, cubic
    pieces in absolute time -> Hermite knots). /trajs is not always what SANDO flies: appendToPlan() drops a new
    plan computed after the drone passed its start A but still reports success, so it is published while the
    plan keeps the old trajectory (E5 2026-11-04: drone 2.6 m from SANDO's own setpoint, then a pole contact).
  * mission and referee as kairos_planner / neo_node: pads A <-> B (term_goal at the route altitude), reached
    within dataset.goal_reach_tol, timeout timeout_factor * distance / v_nominal + 10 s, collisions counted from
    the Gazebo pose (referee_odom_topic), episode continued.
"""
import json
import math
import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from dynus_interfaces.msg import ComputationTimes, DynTraj, Goal as SandoGoal, State as SandoState
from dyn_nav_interfaces.msg import HermiteTrajectory as HermiteTrajectoryMsg
from dyn_expert.core.config import load_config
from dyn_expert.core.hermite import HermiteTrajectory
from dyn_expert.core.scenario import Scenario
from dyn_expert.nodes.common import odom_to_state, to_stamp, traj_to_msg

# sensor mounts of SANDO's urdf/quadrotor.urdf.xacro on base_link (= models/x500_mid360_d435_tw2), as offsets from
# the odometry frame <ns>/base_link here, the model origin: x500_base puts its base_link 0.24 m above it
BASE_LINK_Z = 0.24
LIVOX_XYZ, LIVOX_RPY = (0.0, 0.0, BASE_LINK_Z + 0.2), (0.0, 0.3, 0.0)
D435_XYZ = (0.05, 0.0, BASE_LINK_Z)
SETPOINT_HORIZON = 0.2        # [s] execute goal: each 100 Hz setpoint is sent as a trajectory this long
STUCK_M = 5.0                 # [m] stop_if_stuck: a timed-out follow-on that got less than this closer to its goal


def pwp_to_hermite(pwp):
    """dynus PWPTraj (n cubic pieces, coefficients [a b c d] of u^3 u^2 u 1, u = t - times[i]) -> HermiteTrajectory."""
    times = np.asarray(pwp.times, float)
    n = len(times) - 1
    if n < 1 or len(pwp.coeff_x) < n:
        return None
    C = np.array([[[c.a, c.b, c.c, c.d] for c in coeffs[:n]] for coeffs in (pwp.coeff_x, pwp.coeff_y, pwp.coeff_z)])
    C = C.transpose(1, 0, 2)                                   # (n, 3 axes, 4)
    T = np.diff(times)
    P, V, A = [], [], []
    for i in range(n):
        a, b, c, d = C[i, :, 0], C[i, :, 1], C[i, :, 2], C[i, :, 3]
        P.append(d), V.append(c), A.append(2.0 * b)
    a, b, c, d = C[-1, :, 0], C[-1, :, 1], C[-1, :, 2], C[-1, :, 3]
    u = T[-1]
    P.append(a * u ** 3 + b * u ** 2 + c * u + d)
    V.append(3 * a * u ** 2 + 2 * b * u + c)
    A.append(6 * a * u + 2 * b)
    keep = T > 1e-6
    if not keep.all():                                         # drop zero-length pieces
        idx = np.r_[np.nonzero(keep)[0], n]
        return HermiteTrajectory(np.array(P)[idx], np.array(V)[idx], np.array(A)[idx], T[keep], times[0])
    return HermiteTrajectory(np.array(P), np.array(V), np.array(A), T, times[0])


class SandoBridge(Node):
    def __init__(self):
        super().__init__("sando_bridge")
        p = self.declare_parameter
        p("config_file", "")
        p("scenario_file", "")
        p("out_dir", os.path.expanduser("~/kairos_test"))
        p("run_name", "sando")
        p("seed", 0)
        p("episodes", 0)
        p("namespace", "NX01")
        p("odom_topic", "/kairos/odom")
        p("odom_twist_in_body", True)
        p("referee_odom_topic", "")
        p("trajectory_topic", "/dyn_nav/trajectory")
        p("ready_topic", "/dyn_nav/ready")
        p("world_frame", "world")
        p("map_frame", "map")
        p("timeout_factor", 0.0)
        p("execute", "goal")                      # goal: SANDO's setpoint stream; trajs: its shared /trajs
        p("sensors", "")                          # recorded in summary.json (set by the launch file)
        p("stop_if_stuck", False)                 # end the run when a follow-on times out without progress (run_restart.sh)
        gp = lambda n: self.get_parameter(n).value

        self.cfg = load_config(os.path.expanduser(gp("config_file")))
        D = self.cfg.dataset
        self.scn = Scenario.load(os.path.expanduser(gp("scenario_file")))
        self.rng = np.random.default_rng(int(gp("seed")))
        self.ns = gp("namespace")
        self.sando_id = int(self.ns[-2:]) if self.ns[-2:].isdigit() else 0
        self.map_frame = gp("map_frame")
        self.world_frame = gp("world_frame")
        self.twist_in_body = bool(gp("odom_twist_in_body"))
        self.timeout_factor = float(gp("timeout_factor")) or float(D.timeout_factor)
        self.goal_z = float(self.scn.flight_z if self.scn.flight_z is not None else 2.0)
        self.goal_tol = float(D.goal_reach_tol)
        self.coll_radius = float(D.collision_radius)
        self.max_episodes = int(gp("episodes"))
        self.stop_if_stuck = bool(gp("stop_if_stuck"))
        self.execute = gp("execute")
        if self.execute not in ("goal", "trajs"):
            raise ValueError(f"execute '{self.execute}': goal or trajs")

        self.lock = threading.Lock()
        self.pose = None                          # (t, p, R, v) newest odometry
        self.ready = False
        self.done = False
        self.plan_id = 0
        self.last_traj_stamp = None
        self.episode = None
        self.episodes = []
        self.last_goal, self.last_outcome = None, None
        self.contact = dict(static=None, dynamic=None)
        self.stats = dict(plans=0, plan_gap_s=[], traj_horizon_s=[], replan_ms=[], replans=0, replans_ok=0)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(os.path.expanduser(gp("out_dir")), f"{stamp}_{self.scn.name}_sando")
        os.makedirs(self.out_dir, exist_ok=True)
        self.run_name = gp("run_name")
        self.config_record = dict(namespace=self.ns, timeout_factor=self.timeout_factor, goal_reach_tol=self.goal_tol,
                                  collision_radius=self.coll_radius, config_file=gp("config_file"),
                                  scenario=self.scn.name, execute=self.execute, sensors=gp("sensors"),
                                  stop_if_stuck=self.stop_if_stuck)

        odom_qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Odometry, gp("odom_topic"), self.on_odom, odom_qos)
        ref_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Odometry, gp("referee_odom_topic") or gp("odom_topic"), self.on_referee, ref_qos)
        self.create_subscription(Bool, gp("ready_topic"), self.on_ready, 10)
        rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(DynTraj, "/trajs", self.on_trajs, rel)
        # diagnostics only: SANDO's own setpoint stream (what its hardware path sends the autopilot)
        self.diag = dict(plan_t=None, end=None, G=None, sp=None, logged=False)
        sp_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(SandoGoal, f"/{self.ns}/goal", self.on_sando_setpoint, sp_qos)
        # planning time: SANDO's own total replanning wall-clock time of successful replans
        self.create_subscription(ComputationTimes, f"/{self.ns}/computation_times", self.on_comp_times, 100)
        self.pub_traj = self.create_publisher(HermiteTrajectoryMsg, gp("trajectory_topic"), rel)
        self.pub_state = self.create_publisher(SandoState, f"/{self.ns}/state", rel)
        self.pub_goal = self.create_publisher(PoseStamped, f"/{self.ns}/term_goal", rel)
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self.publish_static_tf()
        self.create_timer(0.5, self.on_mission)
        self.create_timer(10.0, self.on_status)
        self.get_logger().info(f"sando_bridge ready: scenario {self.scn.name}, namespace {self.ns} (id {self.sando_id}), "
                               f"route z {self.goal_z}, results in {self.out_dir}")

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ frames / state
    def _tf(self, parent, child, xyz, quat_xyzw, stamp=None):
        t = TransformStamped()
        if stamp is not None:
            t.header.stamp = stamp
        t.header.frame_id, t.child_frame_id = parent, child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = [float(x) for x in xyz]
        (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z,
         t.transform.rotation.w) = [float(x) for x in quat_xyzw]
        return t

    def publish_static_tf(self):
        base = f"{self.ns}/base_link"
        q_livox = Rotation.from_euler("xyz", LIVOX_RPY).as_quat()
        q_opt = Rotation.from_euler("xyz", (-math.pi / 2, 0.0, -math.pi / 2)).as_quat()   # x right, y down, z forward
        self.static_tf.sendTransform([
            self._tf(self.world_frame, self.map_frame, (0, 0, 0), (0, 0, 0, 1)),
            self._tf(base, f"{self.ns}/{self.ns}_livox", LIVOX_XYZ, q_livox),
            self._tf(base, f"{self.ns}/{self.ns}_d435_link", D435_XYZ, (0, 0, 0, 1)),
            self._tf(base, f"{self.ns}/{self.ns}_d435_depth_optical_frame", D435_XYZ, q_opt),
        ])

    def on_odom(self, msg):
        t, p, R, v = odom_to_state(msg, self.twist_in_body)
        with self.lock:
            self.pose = (t, p, R, v)
        q = Rotation.from_matrix(R).as_quat()
        st = SandoState()
        st.header.stamp = msg.header.stamp
        st.header.frame_id = self.map_frame
        st.pos.x, st.pos.y, st.pos.z = [float(x) for x in p]
        st.vel.x, st.vel.y, st.vel.z = [float(x) for x in v]
        st.quat.x, st.quat.y, st.quat.z, st.quat.w = [float(x) for x in q]
        self.pub_state.publish(st)
        self.tf.sendTransform(self._tf(self.map_frame, f"{self.ns}/base_link", p, q, msg.header.stamp))

    # ------------------------------------------------------------------ plan -> trajectory_server
    def on_trajs(self, msg):
        if not msg.is_agent or msg.id != self.sando_id or msg.mode != "pwp" or self.done:
            return
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == self.last_traj_stamp:
            return
        self.last_traj_stamp = stamp
        traj = pwp_to_hermite(msg.pwp)
        if traj is None:
            return
        if self.execute == "trajs":
            self.forward(traj)
        t = self.now_s()
        if self.stats.get("_last_t") is not None:
            self.stats["plan_gap_s"].append(t - self.stats["_last_t"])
        self.stats["_last_t"] = t
        self.stats["traj_horizon_s"].append(float(traj.t_end - t))
        self.stats["plans"] += 1
        self.diag.update(plan_t=t, end=np.asarray(traj.evaluate(traj.t_end)).ravel()[:3],
                         G=np.array(msg.goal[:3]) if len(msg.goal) >= 3 else None, logged=False)

    def on_comp_times(self, msg):
        if self.episode is None or self.done:
            return
        self.stats["replans"] += 1
        if msg.result:
            self.stats["replans_ok"] += 1
            self.stats["replan_ms"].append(float(msg.total_replanning_ms))

    def forward(self, traj):
        with self.lock:
            goal = None if self.episode is None else self.episode["goal"]
        self.plan_id += 1
        self.pub_traj.publish(traj_to_msg(traj, HermiteTrajectoryMsg, self.world_frame, self.plan_id, 0, goal))

    def on_sando_setpoint(self, msg):
        p = np.array([msg.p.x, msg.p.y, msg.p.z])
        self.diag["sp"] = p
        if self.execute != "goal" or self.done or not self.ready:
            return
        # the setpoint and its constant-jerk continuation over SETPOINT_HORIZON (SANDO's pieces are cubic)
        v, a, j = (np.array([m.x, m.y, m.z]) for m in (msg.v, msg.a, msg.j))
        u = SETPOINT_HORIZON
        P = np.array([p, p + v * u + a * u ** 2 / 2 + j * u ** 3 / 6])
        V = np.array([v, v + a * u + j * u ** 2 / 2])
        A = np.array([a, a + j * u])
        t0 = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.forward(HermiteTrajectory(P, V, A, np.array([u]), t0))

    def check_idle(self, p):
        """Log once per idle period: SANDO sent no plan for 5 s during an episode (diagnostics)."""
        d = self.diag
        if d["plan_t"] is None or d["logged"] or self.now_s() - d["plan_t"] < 5.0:
            return
        d["logged"] = True
        r = lambda x: None if x is None else np.round(x, 2).tolist()
        self.get_logger().info(f"[diag] no plan for 5 s: drone {r(p)}, last forwarded trajectory ends {r(d['end'])}, "
                               f"SANDO G {r(d['G'])}, SANDO setpoint {r(d['sp'])}")

    # ------------------------------------------------------------------ mission / referee (as neo_node)
    def on_ready(self, msg):
        if msg.data and not self.ready:
            self.get_logger().info("trajectory server reports READY: test starts")
        self.ready = self.ready or bool(msg.data)

    def on_mission(self):
        if not self.ready or self.done:
            return
        with self.lock:
            pose, ep = self.pose, self.episode
        if pose is None:
            return
        if ep is None:
            self.start_episode(pose[0], pose[1])
        else:
            self.send_goal(ep["goal"])        # SANDO keeps the newest; repeated in case it was dropped
            self.check_idle(pose[1])

    def send_goal(self, goal):
        g = PoseStamped()
        g.header.stamp = to_stamp(self.now_s())
        g.header.frame_id = self.map_frame
        g.pose.position.x, g.pose.position.y, g.pose.position.z = float(goal[0]), float(goal[1]), self.goal_z
        g.pose.orientation.w = 1.0
        self.pub_goal.publish(g)

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
        follow_on = (self.last_goal is not None and self.last_outcome != "reached"
                     and np.linalg.norm(p - self.last_goal) > 2.0 * self.goal_tol)
        goal = self.last_goal.copy() if follow_on else self.scn.route_goal(self.rng, p)
        goal = np.array([goal[0], goal[1], self.goal_z])
        dist = float(np.linalg.norm(goal - p))
        ep = dict(n=len(self.episodes) + 1, goal=goal, start=p.copy(), t_start=t, path=0.0, last_p=None,
                  hits_static=0, hits_dynamic=0, follow_on=follow_on,
                  t_limit=t + self.timeout_factor * dist / float(self.cfg.planner.v_nominal) + 10.0)
        with self.lock:
            self.episode = ep
        self.send_goal(goal)
        self.get_logger().info(f"[sando] episode {ep['n']} start, goal {np.round(goal, 2)}")

    def end_episode(self, t, outcome, p):
        with self.lock:
            ep, self.episode = self.episode, None
        if ep is None:
            return
        dur = t - ep["t_start"]
        self.episodes.append(dict(episode=ep["n"], outcome=outcome, duration=dur, path=ep["path"],
                                  mean_speed=ep["path"] / max(dur, 1e-6), hits_static=ep["hits_static"],
                                  hits_dynamic=ep["hits_dynamic"], start=ep["start"].tolist(),
                                  goal=ep["goal"].tolist(), end=p.tolist(), follow_on=bool(ep["follow_on"])))
        self.last_goal, self.last_outcome = ep["goal"], outcome
        self.get_logger().info(f"[sando] episode {ep['n']} end: {outcome}, {dur:.1f} s, {ep['path']:.0f} m, "
                               f"hits static {ep['hits_static']} dynamic {ep['hits_dynamic']}")
        self.write_summary()
        # a follow-on that timed out less than STUCK_M closer to its goal: SANDO hovers where no corridor contains its
        # start and has no recovery; with stop_if_stuck the run ends here and run_restart.sh restarts the stack.
        # Progress, not path: the hover jitter adds ~2 cm/s of path (SANDO_v3R_omv1 pass 18: 10 m, 0.3 m off its start)
        gain = float(np.linalg.norm(ep["start"] - ep["goal"]) - np.linalg.norm(p - ep["goal"]))
        stuck = self.stop_if_stuck and outcome == "timeout" and ep["follow_on"] and gain < STUCK_M
        if stuck:
            self.get_logger().warn(f"[sando] stuck: follow-on episode {ep['n']} got {gain:.1f} m closer to its goal, "
                                   f"run ended")
        if stuck or (self.max_episodes and len(self.episodes) >= self.max_episodes):
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
        g = np.array(self.stats["plan_gap_s"]) if self.stats["plan_gap_s"] else np.zeros(1)
        h = np.array(self.stats["traj_horizon_s"]) if self.stats["traj_horizon_s"] else np.zeros(1)
        r = np.array(self.stats["replan_ms"]) if self.stats["replan_ms"] else np.zeros(1)
        s = dict(run_name=self.run_name, config=self.config_record, episodes=self.episodes,
                 stats=dict(plans=self.stats["plans"], replans=self.stats["replans"], replans_ok=self.stats["replans_ok"]),
                 plan_ms=dict(median=float(np.median(r)), p90=float(np.percentile(r, 90))),
                 plan_gap_s=dict(median=float(np.median(g)), p90=float(np.percentile(g, 90))),
                 traj_horizon_s=dict(median=float(np.median(h)), p10=float(np.percentile(h, 10))))
        with open(os.path.join(self.out_dir, "summary.json"), "w") as fh:
            json.dump(s, fh, indent=1)

    def on_status(self):
        if not self.ready:
            return
        with self.lock:
            pose = self.pose
        g = self.stats["plan_gap_s"][-50:]
        self.get_logger().info(
            f"[sando] at {np.round(pose[1], 1) if pose else '-'}, episodes {len(self.episodes)}, plans "
            f"{self.stats['plans']}, plan gap median {np.median(g) if g else 0:.3f} s")


def main(args=None):
    rclpy.init(args=args)
    node = SandoBridge()
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
