#!/bin/bash
# Build SANDO v0.0.3 (ws/src/sando) on ROS 2 Jazzy. Reproduces what was done on 2026-11-02:
#   deps symlinked into ws/src as SANDO's setup.sh does; Gazebo Classic packages and the Livox Classic plugin ignored;
#   patches/*.patch applied (Gazebo Classic build deps and plugins removed, tf2 / cv_bridge .h -> .hpp, VTK MPI
#   target, sim-only tools not built); decomp_util first; system Python for ament; Gurobi from GUROBI_HOME.
# Needs: Gurobi 11.0.3 in ~/opt/gurobi1103 (libgurobi_c++.a rebuilt), nlohmann_json in ws/src (v3.11.3).
set -e
HERE=$(cd "$(dirname "$0")" && pwd); WS=$HERE/ws; S=$WS/src
cd $S
for d in DecompROS2 dynus_interfaces acl-mapping gazebo_ros_pkgs; do [ -e $d ] || ln -s sando/deps/$d $d; done
for d in gazebo_dev gazebo_plugins gazebo_ros gazebo_ros_control gazebo_ros_pkgs; do touch sando/deps/gazebo_ros_pkgs/$d/COLCON_IGNORE; done
touch sando/deps/livox_laser_simulation_ros2/COLCON_IGNORE
for p in sando acl-mapping DecompROS2; do
  (cd $S/$p && git apply --check $HERE/patches/${p}_jazzy.patch 2>/dev/null && git apply $HERE/patches/${p}_jazzy.patch) || true
done
cd $WS
export PATH=/usr/bin:$PATH
source /opt/ros/jazzy/setup.bash
export GUROBI_HOME=${GUROBI_HOME:-$HOME/opt/gurobi1103/linux64}
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
A="--cmake-args -DCMAKE_BUILD_TYPE=Release -DJSON_BuildTests=OFF -DPython3_EXECUTABLE=/usr/bin/python3"
colcon build --packages-select decomp_util nlohmann_json $A
source install/setup.bash
MAKEFLAGS=-j8 colcon build --parallel-workers 6 $A
