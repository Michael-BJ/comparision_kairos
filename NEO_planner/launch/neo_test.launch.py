"""NEO-Planner closed-loop test on the KAIROS stack: ros_gz_bridge + trajectory_server + odom_estimate + octomap_server
+ neo_node. kairos_planner's test.launch.py with kairos_planner replaced by NEO (and NEO's octomap).
Prerequisite (other terminals): Gazebo world + PX4 SITL and MAVROS, as for KAIROS (see ../run_neo.sh).
  ros2 launch ~/comparision_for_kairos/NEO_planner/launch/neo_test.launch.py world:=dyn_long200_smooth_021_clear2 \
       model_path:=~/saved_net/neo_kairos/run_XXXX/best.pth episodes:=24
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
NODE = os.path.join(os.path.dirname(HERE), "NEO_planner", "neo_node.py")


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
    max_range = float(cfg["camera"]["max_range"])
    gz_odom = f"/model/{model}/odometry"
    odom_topic = "/kairos/odom"
    bridge_args, remaps = ["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"], []
    for d in [c["topic"] for c in rig]:
        i_gz = d.rsplit("/", 1)[0] + "/camera_info"
        bridge_args += [f"{d}@sensor_msgs/msg/Image[gz.msgs.Image", f"{i_gz}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"]
        remaps.append((i_gz, d + "/camera_info"))
    bridge_args.append(f"{gz_odom}@nav_msgs/msg/Odometry[gz.msgs.Odometry")
    common = {"use_sim_time": True}
    # octomap band of neo_planner_node._configure_octomap_band: [cruise_z - 0.7, cruise_z + 1.0], >= ground + 0.35
    occ_min, occ_max = max(0.35, route_z - 0.7), route_z + 1.0
    params = dict(common, config_file=config, scenario_file=scenario, model_path=os.path.expanduser(lc("model_path")),
                  out_dir=os.path.expanduser(lc("out_dir")), run_name=lc("run_name"), seed=int(lc("seed")),
                  episodes=int(lc("episodes")), odom_topic=odom_topic, referee_odom_topic=gz_odom,
                  v_max=float(lc("v_max")), timeout_factor=float(lc("timeout_factor")),
                  replan_period=float(lc("replan_period")), planning_time_ahead=float(lc("planning_time_ahead")),
                  longitu_step=float(lc("longitu_step")), target_vel_frac=float(lc("target_vel_frac")),
                  cloud_stride=int(lc("cloud_stride")), safe_dis=float(lc("safe_dis")),
                  hard_clearance=float(lc("hard_clearance")), safe_dis_emergency=float(lc("safe_dis_emergency")),
                  inflation_radius=float(lc("inflation_radius")))
    pfile = tempfile.NamedTemporaryFile("w", prefix="neo_params_", suffix=".yaml", delete=False)
    yaml.safe_dump({"neo_planner": {"ros__parameters": params}}, pfile)
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
        Node(package="octomap_server", executable="octomap_server_node", name="octomap_server_neo", output="screen",
             parameters=[common, {"frame_id": "world", "resolution": float(lc("resolution")),
                                  "sensor_model.max_range": max_range, "publish_2d_map": True,
                                  "publish_3d_map": False, "queue_size": 50, "occupancy_min_z": occ_min,
                                  "occupancy_max_z": occ_max, "filter_ground_plane": False,
                                  "use_height_map": False, "latch": False}],
             remappings=[("cloud_in", "/neo/cloud_in")]),
        ExecuteProcess(cmd=["/usr/bin/python3", NODE, "--ros-args", "-r", "__node:=neo_planner",
                            "--params-file", pfile.name], output="screen", additional_env=one_thread,
                       name="neo_planner"),
    ]


def generate_launch_description():
    home = os.path.expanduser("~")
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="dyn_long200_smooth_021_clear2"),
        DeclareLaunchArgument("scenario", default_value="", description="'' = dyn_gz_worlds/scenarios/<world>.yaml"),
        DeclareLaunchArgument("config", default_value=f"{home}/kairos_test/20261102_compare/config/expert_m4.yaml",
                              description="dyn_expert config of the KAIROS test (camera rig, planner, dataset)"),
        DeclareLaunchArgument("drone_model", default_value="x500_oakd_x2_60_tw2"),
        DeclareLaunchArgument("model_path", description="PlannerNet .pth (train.py)"),
        DeclareLaunchArgument("out_dir", default_value=f"{home}/kairos_test"),
        DeclareLaunchArgument("run_name", default_value="neo"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("episodes", default_value="0"),
        DeclareLaunchArgument("v_max", default_value="1.5"),
        DeclareLaunchArgument("timeout_factor", default_value="4.0"),
        DeclareLaunchArgument("replan_period", default_value="0.0", description="0 = planner.replan_dt"),
        DeclareLaunchArgument("planning_time_ahead", default_value="0.0", description="0 = planner.foresee_dt"),
        DeclareLaunchArgument("longitu_step", default_value="0.0", description="0 = planner.target_dist"),
        DeclareLaunchArgument("target_vel_frac", default_value="0.0", description="neo_planner: 0.8"),
        DeclareLaunchArgument("cloud_stride", default_value="0", description="0 = camera.map_stride (neo_planner: 2 on 640 x 480)"),
        DeclareLaunchArgument("resolution", default_value="0.1",
                              description="octomap resolution: grid.resolution of the KAIROS config (neo_planner: 0.15)"),
        # margins (ESDF distances from the drone body), -1 = from the KAIROS config (neo_node); neo_planner's values:
        # inflation 0.25, hard 0.35, safe_dis 0.7, emergency 0.4
        DeclareLaunchArgument("inflation_radius", default_value="-1.0", description="-1 = planner.drone_radius"),
        DeclareLaunchArgument("hard_clearance", default_value="-1.0",
                              description="-1 = planner.static_clearance - inflation"),
        DeclareLaunchArgument("safe_dis", default_value="-1.0", description="-1 = hard_clearance + 0.2"),
        DeclareLaunchArgument("safe_dis_emergency", default_value="-1.0", description="-1 = hard_clearance + 0.05"),
        DeclareLaunchArgument("recalib_tau", default_value="1.0"),
        OpaqueFunction(function=_nodes),
    ])
