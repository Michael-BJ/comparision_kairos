#!/bin/bash
# Open a dyn_gz_worlds world in Gazebo (no PX4), e.g. show_world.sh dyn_sando_easy
source /opt/ros/jazzy/setup.bash; source ~/Dynamic_obstacle/install/setup.bash
exec ros2 launch dyn_gz_worlds world.launch.py world:=${1:-dyn_sando_easy} drone_model:=x500_oakd_x2_60_tw2
