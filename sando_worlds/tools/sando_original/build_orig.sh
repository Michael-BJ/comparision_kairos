#!/bin/bash
set -e
cd ~/sando_orig_ws
export PATH=/usr/bin:$PATH
source /opt/ros/jazzy/setup.bash
export GUROBI_HOME=$HOME/opt/gurobi1103/linux64
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
A="--cmake-args -DCMAKE_BUILD_TYPE=Release -DJSON_BuildTests=OFF -DPython3_EXECUTABLE=/usr/bin/python3"
colcon build --packages-select decomp_util nlohmann_json $A
source install/setup.bash
MAKEFLAGS=-j8 colcon build --parallel-workers 6 $A
