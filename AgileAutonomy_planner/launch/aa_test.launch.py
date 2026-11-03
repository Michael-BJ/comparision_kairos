"""Agile Autonomy closed-loop test on the KAIROS stack: ros_gz_bridge + trajectory_server + odom_estimate + aa_node.
kairos_planner's test.launch.py with kairos_planner replaced by Agile Autonomy (no map: it plans from the network only).
Prerequisite (other terminals): Gazebo world + PX4 SITL and MAVROS, as for KAIROS (see ../run.sh).
  ros2 launch ~/comparision_for_kairos/AgileAutonomy_planner/launch/aa_test.launch.py world:=dyn_long200_smooth_021_clear2 \
       model_path:=~/saved_net/aa_kairos/run_XXXX/best.pth episodes:=24
Needs ~/inference_kairos/install/setup.bash sourced (dyn_expert, dyn_gz_worlds, kairos_planner's odom_estimate).
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
NODE = os.path.join(os.path.dirname(HERE), "AgileAutonomy_planner", "aa_node.py")


def _nodes(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)
    world = lc("world")
    scenario = os.path.expanduser(lc("scenario")) or os.path.join(
        get_package_share_directory("dyn_gz_worlds"), "scenarios", world + ".yaml")
    with open(scenario) as fh:
        scn = yaml.safe_load(fh) or {}
    route_z = float((scn.get("route") or {}).get("z", 1.3))
    drone = lc("drone_model")
    model = drone + "_0"
    config = os.path.expanduser(lc("config"))
    with open(config) as fh:
        cfg = yaml.safe_load(fh) or {}
    rig = (cfg.get("camera") or {}).get("rig") or []
    gz_odom = f"/model/{model}/odometry"
    odom_topic = "/kairos/odom"
    bridge_args, remaps = ["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"], []
    for d in [c["topic"] for c in rig]:
        i_gz = d.rsplit("/", 1)[0] + "/camera_info"
        bridge_args += [f"{d}@sensor_msgs/msg/Image[gz.msgs.Image", f"{i_gz}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"]
        remaps.append((i_gz, d + "/camera_info"))
    bridge_args.append(f"{gz_odom}@nav_msgs/msg/Odometry[gz.msgs.Odometry")
    common = {"use_sim_time": True}
    params = dict(common, config_file=config, scenario_file=scenario, model_path=os.path.expanduser(lc("model_path")),
                  out_dir=os.path.expanduser(lc("out_dir")), run_name=lc("run_name"), seed=int(lc("seed")),
                  episodes=int(lc("episodes")), odom_topic=odom_topic, referee_odom_topic=gz_odom,
                  v_max=float(lc("v_max")), timeout_factor=float(lc("timeout_factor")),
                  replan_period=float(lc("replan_period")), planning_time_ahead=float(lc("planning_time_ahead")),
                  accept_thresh=float(lc("accept_thresh")), goal_approach_dist=float(lc("goal_approach_dist")))
    pfile = tempfile.NamedTemporaryFile("w", prefix="aa_params_", suffix=".yaml", delete=False)
    yaml.safe_dump({"aa_planner": {"ros__parameters": params}}, pfile)
    pfile.close()
    one_thread = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    spawn = [float(x) for x in scn.get("spawn", [0.0, 0.0, 0.0])]
    return [
        Node(package="ros_gz_bridge", executable="parameter_bridge", name="gz_bridge",
             arguments=bridge_args, remappings=remaps, parameters=[common], output="screen"),
        Node(package="dyn_expert", executable="trajectory_server", name="trajectory_server", output="screen",
             parameters=[common, {"gt_odom_topic": odom_topic, "takeoff_height": route_z,
                                  "recalib_tau": float(lc("recalib_tau")), "config_file": config}]),
        Node(package="kairos_planner", executable="odom_estimate", name="odom_estimate", output="screen",
             parameters=[common, {"out_topic": odom_topic, "start_xyz": spawn, "start_yaw": 0.0, "local_yaw": 0.0}]),
        ExecuteProcess(cmd=["/usr/bin/python3", NODE, "--ros-args", "-r", "__node:=aa_planner",
                            "--params-file", pfile.name], output="screen", additional_env=one_thread,
                       name="aa_planner"),
    ]


def generate_launch_description():
    home = os.path.expanduser("~")
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="dyn_long200_smooth_021_clear2"),
        DeclareLaunchArgument("scenario", default_value="", description="'' = dyn_gz_worlds/scenarios/<world>.yaml"),
        DeclareLaunchArgument("config", default_value=f"{home}/kairos_test/20261102_compare/config/expert_m4.yaml",
                              description="dyn_expert config of the KAIROS test (camera rig, planner, dataset)"),
        DeclareLaunchArgument("drone_model", default_value="x500_oakd_x2_60_tw2"),
        DeclareLaunchArgument("model_path", description="PlaNet .pth (train.py)"),
        DeclareLaunchArgument("out_dir", default_value=f"{home}/kairos_test"),
        DeclareLaunchArgument("run_name", default_value="agile_autonomy"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("episodes", default_value="0"),
        DeclareLaunchArgument("v_max", default_value="1.5"),
        DeclareLaunchArgument("timeout_factor", default_value="4.0"),
        DeclareLaunchArgument("replan_period", default_value="0.0", description="0 = planner.replan_dt"),
        DeclareLaunchArgument("planning_time_ahead", default_value="0.0", description="0 = planner.foresee_dt"),
        DeclareLaunchArgument("accept_thresh", default_value="0.95",
                              description="other modes are candidates when best_alpha / alpha > this (deploy_settings.yaml)"),
        DeclareLaunchArgument("goal_approach_dist", default_value="2.0",
                              description="rest-to-rest quintic to the pad within this distance"),
        DeclareLaunchArgument("recalib_tau", default_value="1.0"),
        OpaqueFunction(function=_nodes),
    ])
