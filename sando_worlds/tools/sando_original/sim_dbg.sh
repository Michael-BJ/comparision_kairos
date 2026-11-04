#!/bin/bash
cd ~/sando_orig_ws
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH
export PATH=$HOME/.local/bin:/usr/bin:$PATH
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export GUROBI_HOME=$HOME/opt/gurobi1103/linux64
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
export ROS_DOMAIN_ID=20
python3 src/sando/scripts/run_sim.py "$@"
