#!/bin/bash
# SANDO upstream (github.com/mit-acl/sando @1a738df), its own dynamic-obstacle benchmark
# (rviz-only, procedural trefoil obstacles, ground-truth obstacle trajectories = paper Table VI), on ROS 2 Jazzy.
# Usage: bench_dyn.sh <num_trials> <case...>   e.g. bench_dyn.sh 10 easy medium hard
cd ~/sando_orig_ws
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH
export PATH=$HOME/.local/bin:/usr/bin:$PATH
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export GUROBI_HOME=$HOME/opt/gurobi1103/linux64
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
export ROS_DOMAIN_ID=20
N=${1:-10}; shift
CASES=${@:-easy medium hard}
OUT=$HOME/sando_orig_ws/benchmark_data/dynamic
python3 src/sando/scripts/run_benchmark.py -s install/setup.bash --mode rviz-only \
  --cases $CASES --config-name dynamic --num-trials $N --output-dir $OUT/run_$(date +%Y%m%d_%H%M%S)
