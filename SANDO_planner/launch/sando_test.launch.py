"""SANDO closed-loop test on the KAIROS stack, with SANDO's original sensors (models/x500_mid360_d435_tw2):
ros_gz_bridge + lidar_fill + trajectory_server + odom_estimate + sando_bridge + SANDO's onboard nodes (ACL global_mapper_node,
obstacle_tracker_node, sando_node) as its onboard_sando.launch.py / run_sim.py start them.
Prerequisite (other terminals): Gazebo world with drone_model:=x500_mid360_d435_tw2 (GZ_SIM_RESOURCE_PATH including
../models) + PX4 SITL and MAVROS, as for KAIROS (see ../run.sh). Needs ~/inference_kairos/install/setup.bash and
ws/install/setup.bash sourced, GUROBI_HOME and the Gurobi license.
  ros2 launch ~/comparision_for_kairos/SANDO_planner/launch/sando_test.launch.py episodes:=24
"""
import os
import re
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
BRIDGE = os.path.join(os.path.dirname(HERE), "SANDO_planner", "sando_bridge.py")
SANDO_TS = os.path.join(os.path.dirname(HERE), "SANDO_planner", "sando_trajectory_server.py")
LIDAR_FILL = os.path.join(os.path.dirname(HERE), "SANDO_planner", "lidar_fill.py")
NS = "NX01"
LIDAR_TOPIC = f"/{NS}/mid360_PointCloud2"            # SANDO's Gazebo Classic Livox topic
LIDAR_RAW = f"/{NS}/mid360_gz"                       # gpu_lidar as Gazebo Harmonic gives it (no-return rays inf)
D435_TOPIC = f"/{NS}/d435/depth/color/points"        # SANDO's realsense plugin topic

# tuning "test" (default): SANDO's own parameters set for this vehicle / world from the diagnostic runs of
# 2026-11-04 (README "Diagnosis"); tuning "hw": sando_hw_quadrotor.yaml / hw_global_mapper.yaml as they are
SANDO_TUNING = {
    # (the time-factor window 2.5 / 4.0 of 2026-11-04 is gone: it answered the infeasibility caused by the lidar
    # seeing the drone; with the sensors fixed the hw window 1.8 / 2.5 flew faster, E13)
    # heat map of sando.yaml (its dense simulated worlds); the hw values (tube 3 m, Hmax 20, alpha0 1, p 1,
    # halo 2 m) are for its 1-5 obstacle arena and made it fly 2 x the course length (also with the fixed
    # sensors: 408 m in 440 s against 220 m in 197 s, E15 / E14)
    "heat_alpha0": 0.2, "heat_p": 2, "heat_Hmax": 2.0, "dyn_base_inflation_m": 0.1, "dyn_heat_tube_radius_m": 0.5,
    "static_heat_rmax_m": 1.0, "static_heat_default_radius_m": 0.5,
}
MAPPER_TUNING = {
    # (clear_radius 2.0 of 2026-11-04 is gone: with the sensors fixed the hw 0.8 m flew as well, E14)
    # moving obstacles left occupied trails for seconds with -0.01 (36 % of the phantom voxels); sim value -0.4
    "occupancy_grid.miss_inc": -0.1,
    # the world's obstacles are spheres of 0.5 .. 0.9 m (3 .. 8 voxels of 0.3 m), its tuning is for 0.8 m cubes /
    # 0.3 x 2 m cylinders; speeds up to 1.88 m/s (clamp 1.0); observations before a prediction as global_mapper.yaml.
    # With the fixed sensors the hw values detect about half as much (3-6 m ahead 42 against 72 %, E16 / E13)
    "obstacle_tracker.min_cluster_size": 3, "obstacle_tracker.max_obstacle_velocity": 2.0,
    "obstacle_tracker.min_observations_for_prediction": 3,
}


def _overrides(s):
    """'a=1,b=[1.0,2.0],c=true' -> {a: 1, b: [1.0, 2.0], c: True} (commas inside [...] belong to the value)"""
    out = {}
    for kv in filter(None, re.split(r",(?![^\[]*\])", s)):
        k, v = kv.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def _nodes(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)
    world = lc("world")
    scenario = os.path.expanduser(lc("scenario")) or os.path.join(
        get_package_share_directory("dyn_gz_worlds"), "scenarios", world + ".yaml")
    with open(scenario) as fh:
        scn = yaml.safe_load(fh) or {}
    route_z = float((scn.get("route") or {}).get("z", 1.3))
    model = lc("drone_model") + "_0"
    config = os.path.expanduser(lc("config"))
    gz_odom = f"/model/{model}/odometry"
    odom_topic = "/kairos/odom"
    common = {"use_sim_time": True}
    bridge_args = ["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                   "/sando/mid360/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
                   "/sando/d435/depth/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
                   f"{gz_odom}@nav_msgs/msg/Odometry[gz.msgs.Odometry"]
    remaps = [("/sando/mid360/points", LIDAR_RAW), ("/sando/d435/depth/points", D435_TOPIC)]

    # SANDO's tuning: sando_hw_quadrotor.yaml (default; its real drone with the Mid-360 + D435 perception this
    # package flies) or sando.yaml (its simulation). Overridden only for the environment and the test's rules:
    # simulation mode (no hardware init-pose TF), perception only, the route altitude, v / a / j and drone size of
    # the test, the obstacle speed bound, no benchmark logging (onboard_sando.launch.py sets use_benchmark false too).
    # Map bounds: the scenario's x / y bounds (as NEO's arena walls; with +-200 m SANDO flew around the arenas
    # outside the corridor, y -11 m, 2026-11-03) and the test's altitude band route z +- z_band (it flew at 0.7 m).
    cdir = os.path.join(get_package_share_directory("sando"), "config")
    with open(os.path.join(cdir, lc("sando_config"))) as fh:
        sp = list(yaml.safe_load(fh).values())[0]["ros__parameters"]
    v_max, a_max, j_max = float(lc("v_max")), float(lc("a_max")), float(lc("j_max"))
    # drone_bbox: width of SANDO's hardware tuning (0.9 m), height raised from its 0.5 m to the referee's 2 x 0.35 m
    # (with 0.5 it flew 0.25 m over 1.5 m walls and grazed them; with a 0.7 m cube 11 static hits, 2026-11-03)
    sp["drone_bbox"] = [float(x) for x in lc("drone_bbox").split(",")]
    lo, hi = scn["bounds"]["lo"], scn["bounds"]["hi"]
    zb = float((scn.get("route") or {}).get("z_band", 1.0))
    sp.update(x_min=float(lo[0]), x_max=float(hi[0]), y_min=float(lo[1]), y_max=float(hi[1]),
              z_min=route_z - zb, z_max=route_z + zb)
    sp.update(common, sim_env="gazebo", use_hardware=False, provide_goal_in_global_frame=False,
              state_already_in_global_frame=False, use_global_pc=False, use_benchmark=False,
              skip_initial_yawing=True, force_goal_z=True, default_goal_z=route_z,
              visual_level=int(lc("visual_level")), debug_verbose=lc("sando_debug") == "true",
              v_max=v_max, a_max=a_max, j_max=j_max, dynamic_constraint_type=lc("dynamic_constraint_type"),
              obst_max_vel=float(lc("obst_max_vel")))
    if lc("tuning") == "test":
        sp.update(SANDO_TUNING)
    sp.update(_overrides(lc("sando_set")))                        # debugging only: name=yaml_value,...
    sfile = tempfile.NamedTemporaryFile("w", prefix="sando_params_", suffix=".yaml", delete=False)
    yaml.safe_dump({"/**": {"ros__parameters": sp}}, sfile)
    sfile.close()

    mapper_cfg = os.path.join(get_package_share_directory("global_mapper_ros"), "cfg", lc("mapper_param_file"))
    mapper_in = LIDAR_TOPIC if lc("mapper_sensor") == "mid360" else D435_TOPIC
    sensor_frame = f"{NS}/{NS}_livox" if lc("mapper_sensor") == "mid360" else f"{NS}/{NS}_d435_link"
    # SANDO's own dynamic-obstacle tracking runs inside global_mapper_ros (temporal grid -> dynamic voxels -> AEKF,
    # obstacle_tracker.enabled in its cfg, publishes predicted_trajs); onboard_sando.launch.py starts the separate
    # obstacle_tracker_node only with use_obstacle_tracker:=true (default false). tracker_sensor none = that pipeline;
    # d435 / mid360 add obstacle_tracker_node on that sensor (it publishes predicted_trajs too, ids from 0 like the
    # mapper's tracker: SANDO keeps one trajectory per id, so the two overwrite each other)
    tracker_in = {"mid360": LIDAR_RAW, "d435": D435_TOPIC}.get(lc("tracker_sensor"))
    mapper_over = {"pose_type": "state", "global_frame": "map", "drone_frame": sensor_frame,
                   "world_dimensions": [float(x) for x in lc("mapper_world_dim").split(",")]}
    if lc("tuning") == "test":
        mapper_over.update(MAPPER_TUNING)
    mapper_over.update(_overrides(lc("mapper_set")))              # debugging / tuning: name=yaml_value,...
    tracker_node = [] if tracker_in is None else [
        Node(package="sando", executable="obstacle_tracker_node", name="obstacle_tracker_node", namespace=NS,
             output="screen", parameters=[sfile.name], remappings=[("point_cloud", tracker_in)],
             prefix=lc("tracker_prefix") or None)]

    params = dict(common, config_file=config, scenario_file=scenario, out_dir=os.path.expanduser(lc("out_dir")),
                  run_name=lc("run_name"), seed=int(lc("seed")), episodes=int(lc("episodes")), namespace=NS,
                  odom_topic=odom_topic, referee_odom_topic=gz_odom, timeout_factor=float(lc("timeout_factor")),
                  execute=lc("execute"), stop_if_stuck=lc("stop_if_stuck") == "true",
                  sensors=f"mapper {lc('mapper_sensor')}, extra tracker {lc('tracker_sensor')}")
    pfile = tempfile.NamedTemporaryFile("w", prefix="sando_bridge_params_", suffix=".yaml", delete=False)
    yaml.safe_dump({"sando_bridge": {"ros__parameters": params}}, pfile)
    pfile.close()
    spawn = [float(x) for x in scn.get("spawn", [0.0, 0.0, 0.0])]
    ts_params = dict(common, gt_odom_topic=odom_topic, takeoff_height=route_z, recalib_tau=float(lc("recalib_tau")),
                     config_file=config)
    if lc("yaw") == "sando":
        # dyn_expert's trajectory_server with SANDO's goal.yaw (incl. its spin after repeated failed replans)
        tfile = tempfile.NamedTemporaryFile("w", prefix="sando_ts_params_", suffix=".yaml", delete=False)
        yaml.safe_dump({"trajectory_server": {"ros__parameters": dict(ts_params, sando_goal_topic=f"/{NS}/goal")}}, tfile)
        tfile.close()
        traj_server = ExecuteProcess(cmd=["/usr/bin/python3", SANDO_TS, "--ros-args", "-r", "__node:=trajectory_server",
                                          "--params-file", tfile.name], output="screen", name="trajectory_server")
    else:
        traj_server = Node(package="dyn_expert", executable="trajectory_server", name="trajectory_server",
                           output="screen", parameters=[ts_params])
    return [
        Node(package="ros_gz_bridge", executable="parameter_bridge", name="gz_bridge",
             arguments=bridge_args, remappings=remaps, parameters=[common], output="screen"),
        traj_server,
        Node(package="kairos_planner", executable="odom_estimate", name="odom_estimate", output="screen",
             parameters=[common, {"out_topic": odom_topic, "start_xyz": spawn, "start_yaw": 0.0, "local_yaw": 0.0}]),
        # no-return rays -> range_max, as SANDO's Livox plugin publishes them (see lidar_fill.py)
        ExecuteProcess(cmd=["/usr/bin/python3", LIDAR_FILL, "--ros-args", "-r", "__node:=lidar_fill",
                            "-r", f"points_in:={LIDAR_RAW}", "-r", f"points_out:={LIDAR_TOPIC}"],
                       output="screen", name="lidar_fill"),
        ExecuteProcess(cmd=["/usr/bin/python3", BRIDGE, "--ros-args", "-r", "__node:=sando_bridge",
                            "--params-file", pfile.name], output="screen", name="sando_bridge"),
        Node(package="global_mapper_ros", executable="global_mapper_node", name="global_mapper_ros", namespace=NS,
             output="screen",
             parameters=[mapper_cfg, common, mapper_over],
             remappings=[("depth_pointcloud_topic", mapper_in), ("pose_topic", "state"),
                         ("goal_topic", "/move_base_simple/goal"), ("odom_topic", "odometry/filtered_no"),
                         ("occupancy_grid_topic", "occupancy_grid"), ("unknown_grid_topic", "unknown_grid"),
                         ("frontier_grid_topic", "frontier_grid"), ("distance_grid_topic", "distance_grid"),
                         ("cost_grid_topic", "cost_grid"), ("path_topic", "path"),
                         ("dynamic_grid_topic", "dynamic_grid"), ("sparse_path_topic", "sparse_path")]),
        *tracker_node,
        Node(package="sando", executable="sando", name="sando_node", namespace=NS, output="screen",
             parameters=[sfile.name], arguments=["--ros-args", "--log-level", lc("sando_log_level")]),
    ]


def generate_launch_description():
    home = os.path.expanduser("~")
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="dyn_long200_smooth_021_clear2"),
        DeclareLaunchArgument("scenario", default_value="", description="'' = dyn_gz_worlds/scenarios/<world>.yaml"),
        DeclareLaunchArgument("config", default_value=f"{home}/kairos_test/20261102_compare/config/expert_m4.yaml",
                              description="dyn_expert config of the KAIROS test (referee, trajectory_server)"),
        DeclareLaunchArgument("drone_model", default_value="x500_mid360_d435_tw2"),
        DeclareLaunchArgument("out_dir", default_value=f"{home}/kairos_test"),
        DeclareLaunchArgument("run_name", default_value="sando"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("episodes", default_value="0"),
        DeclareLaunchArgument("v_max", default_value="1.5"),
        DeclareLaunchArgument("a_max", default_value="4.0"),
        DeclareLaunchArgument("j_max", default_value="25.0"),
        DeclareLaunchArgument("dynamic_constraint_type", default_value="L2",
                              description="SANDO's limit type: L2 (||v||, ||a||, ||j|| <= max, the test's rule for every method) "
                                          "or Linf (its default, per axis: up to sqrt(3) x v_max; flew 2.2 m/s p95, 3.2 max "
                                          "with v_max 1.5, 2026-11-03)"),
        DeclareLaunchArgument("drone_bbox", default_value="0.9,0.9,0.7",
                              description="SANDO drone_bbox x,y,z: hardware width 0.9, height 2 x the referee radius 0.35"),
        DeclareLaunchArgument("obst_max_vel", default_value="0.5",
                              description="SANDO's obstacle speed bound: 0.5 = sando.yaml, its dynamic-obstacle simulations "
                                          "(hardware 0.25). The worlds' true bound 2.0 inflates unknown space by "
                                          "2.0 x trajectory time and leaves no safe path (2026-11-03)"),
        DeclareLaunchArgument("timeout_factor", default_value="4.0"),
        DeclareLaunchArgument("yaw", default_value="sando",
                              description="sando: SANDO's goal.yaw (faces the velocity; spins after 10 failed replans, its "
                                          "hw recovery); test: the trajectory_server's policy of the other methods"),
        DeclareLaunchArgument("execute", default_value="goal",
                              description="goal: SANDO's 100 Hz setpoint stream (what its hardware path streams to PX4); "
                                          "trajs: the trajectory it shares on /trajs (also published for plans it drops)"),
        DeclareLaunchArgument("stop_if_stuck", default_value="false",
                              description="true (run_restart.sh): end the run when a follow-on episode times out having "
                                          "moved < 5 m, i.e. SANDO hovers without a feasible corridor (no recovery)"),
        DeclareLaunchArgument("mapper_sensor", default_value="mid360",
                              description="mid360 (SANDO hardware / static sim) or d435 (SANDO unknown_dynamic sim)"),
        DeclareLaunchArgument("tuning", default_value="test",
                              description="test: SANDO_TUNING / MAPPER_TUNING above (diagnosed for this vehicle and world); "
                                          "hw: sando_hw_quadrotor.yaml / hw_global_mapper.yaml unchanged"),
        DeclareLaunchArgument("tracker_sensor", default_value="none",
                              description="none: SANDO's own tracking only (global_mapper_ros temporal grid + AEKF on the mapper's "
                                          "sensor; onboard_sando.launch.py default). d435 / mid360: also obstacle_tracker_node on "
                                          "that sensor (the paper's hardware tracks 'entirely from the onboard Livox Mid-360 LiDAR "
                                          "point cloud', Sec. X)"),
        DeclareLaunchArgument("mapper_set", default_value="",
                              description="extra global_mapper_ros params 'occupancy_grid.miss_inc=-0.4,...'"),
        DeclareLaunchArgument("mapper_param_file", default_value="hw_global_mapper.yaml",
                              description="cfg of global_mapper_ros: hw_global_mapper.yaml (its real drone, as sando_config) "
                                          "or global_mapper.yaml (run_sim.py's dynamic worlds); global_frame set to map"),
        DeclareLaunchArgument("mapper_world_dim", default_value="20.0,20.0,4.0",
                              description="rolling grid of global_mapper_ros x,y,z [m]: 20 x 20 x 4 as SANDO's cfgs. 32 x 32 was "
                                          "tried against the goal-B hover of 2026-11-03 (edge cells inside SANDO's unknown "
                                          "inflation), not completed; with the fixes of 2026-11-04 SANDO reaches B"),
        DeclareLaunchArgument("visual_level", default_value="0", description="SANDO's RViz markers (0: none)"),
        DeclareLaunchArgument("sando_log_level", default_value="warn"),
        DeclareLaunchArgument("sando_debug", default_value="false", description="sando.yaml debug_verbose"),
        DeclareLaunchArgument("sando_set", default_value="", description="debugging: extra SANDO params 'a=1,b=true'"),
        DeclareLaunchArgument("sando_config", default_value="sando_hw_quadrotor.yaml",
                              description="SANDO tuning: sando_hw_quadrotor.yaml (real drone, Mid-360 + D435) or sando.yaml (sim)"),
        DeclareLaunchArgument("tracker_prefix", default_value="", description="e.g. 'gdb -batch -ex run -ex bt --args'"),
        DeclareLaunchArgument("recalib_tau", default_value="1.0"),
        OpaqueFunction(function=_nodes),
    ])
