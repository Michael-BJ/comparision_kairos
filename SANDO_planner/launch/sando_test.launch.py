"""SANDO closed-loop test on the KAIROS stack, with SANDO's original sensors (models/x500_mid360_d435_tw2):
ros_gz_bridge + lidar_fill + trajectory_server + odom_estimate + sando_bridge + SANDO's onboard nodes (ACL global_mapper_node,
obstacle_tracker_node, sando_node) as its onboard_sando.launch.py / run_sim.py start them.
Prerequisite (other terminals): Gazebo world with drone_model:=x500_mid360_d435_tw2 (GZ_SIM_RESOURCE_PATH including
../models) + PX4 SITL and MAVROS, as for KAIROS (see ../run.sh). Needs ~/inference_kairos/install/setup.bash and
ws/install/setup.bash sourced, GUROBI_HOME and the Gurobi license.
  ros2 launch ~/comparision_for_kairos/SANDO_planner/launch/sando_test.launch.py episodes:=24
"""
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
BRIDGE = os.path.join(os.path.dirname(HERE), "SANDO_planner", "sando_bridge.py")
LIDAR_FILL = os.path.join(os.path.dirname(HERE), "SANDO_planner", "lidar_fill.py")
NS = "NX01"
LIDAR_TOPIC = f"/{NS}/mid360_PointCloud2"            # SANDO's Gazebo Classic Livox topic
LIDAR_RAW = f"/{NS}/mid360_gz"                       # gpu_lidar as Gazebo Harmonic gives it (no-return rays inf)
D435_TOPIC = f"/{NS}/d435/depth/color/points"        # SANDO's realsense plugin topic


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
    for kv in filter(None, lc("sando_set").split(",")):          # debugging only: name=yaml_value,...
        k, v = kv.split("=", 1)
        sp[k.strip()] = yaml.safe_load(v)
    sfile = tempfile.NamedTemporaryFile("w", prefix="sando_params_", suffix=".yaml", delete=False)
    yaml.safe_dump({"/**": {"ros__parameters": sp}}, sfile)
    sfile.close()

    mapper_cfg = os.path.join(get_package_share_directory("global_mapper_ros"), "cfg", lc("mapper_param_file"))
    mapper_in = LIDAR_TOPIC if lc("mapper_sensor") == "mid360" else D435_TOPIC
    sensor_frame = f"{NS}/{NS}_livox" if lc("mapper_sensor") == "mid360" else f"{NS}/{NS}_d435_link"

    params = dict(common, config_file=config, scenario_file=scenario, out_dir=os.path.expanduser(lc("out_dir")),
                  run_name=lc("run_name"), seed=int(lc("seed")), episodes=int(lc("episodes")), namespace=NS,
                  odom_topic=odom_topic, referee_odom_topic=gz_odom, timeout_factor=float(lc("timeout_factor")))
    pfile = tempfile.NamedTemporaryFile("w", prefix="sando_bridge_params_", suffix=".yaml", delete=False)
    yaml.safe_dump({"sando_bridge": {"ros__parameters": params}}, pfile)
    pfile.close()
    spawn = [float(x) for x in scn.get("spawn", [0.0, 0.0, 0.0])]
    return [
        Node(package="ros_gz_bridge", executable="parameter_bridge", name="gz_bridge",
             arguments=bridge_args, remappings=remaps, parameters=[common], output="screen"),
        Node(package="dyn_expert", executable="trajectory_server", name="trajectory_server", output="screen",
             parameters=[common, {"gt_odom_topic": odom_topic, "takeoff_height": route_z,
                                  "recalib_tau": float(lc("recalib_tau")), "config_file": config}]),
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
             parameters=[mapper_cfg, common, {"pose_type": "state", "global_frame": "map", "drone_frame": sensor_frame,
                                              "world_dimensions": [float(x) for x in lc("mapper_world_dim").split(",")]}],
             remappings=[("depth_pointcloud_topic", mapper_in), ("pose_topic", "state"),
                         ("goal_topic", "/move_base_simple/goal"), ("odom_topic", "odometry/filtered_no"),
                         ("occupancy_grid_topic", "occupancy_grid"), ("unknown_grid_topic", "unknown_grid"),
                         ("frontier_grid_topic", "frontier_grid"), ("distance_grid_topic", "distance_grid"),
                         ("cost_grid_topic", "cost_grid"), ("path_topic", "path"),
                         ("dynamic_grid_topic", "dynamic_grid"), ("sparse_path_topic", "sparse_path")]),
        Node(package="sando", executable="obstacle_tracker_node", name="obstacle_tracker_node", namespace=NS,
             output="screen", parameters=[sfile.name], remappings=[("point_cloud", D435_TOPIC)],
             prefix=lc("tracker_prefix") or None),
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
        DeclareLaunchArgument("mapper_sensor", default_value="mid360",
                              description="mid360 (SANDO hardware / static sim) or d435 (SANDO unknown_dynamic sim)"),
        DeclareLaunchArgument("mapper_param_file", default_value="hw_global_mapper.yaml",
                              description="cfg of global_mapper_ros: hw_global_mapper.yaml (its real drone, as sando_config) "
                                          "or global_mapper.yaml (run_sim.py's dynamic worlds); global_frame set to map"),
        DeclareLaunchArgument("mapper_world_dim", default_value="32.0,32.0,4.0",
                              description="rolling grid of global_mapper_ros x,y,z [m]; SANDO's cfgs 20 x 20: its never-scanned edge "
                                          "cells, 10 m from the drone, are inside the 15 m (r2) it publishes as unknown, and "
                                          "near goal B they put the goal inside SANDO's inflated unknown space (hovered 14 min "
                                          "9 m before it, 2026-11-03). 32 m: the edge is 16 m away, outside r2"),
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
