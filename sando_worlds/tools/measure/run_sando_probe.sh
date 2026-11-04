#!/bin/bash
# SANDO (Mid-360) on one world with sando_probe.py recording lidar returns on the spheres, SANDO's tracks and the
# true pose; then analyze_probe.py. Used on 2026-11-04 to check whether SANDO's lidar sees the spheres (it does,
# 98 % of the scans within 10 m inside its field of view) and where its tracking fails.
#
#   run_sando_probe.sh <world> [episodes] [out dir]
W=${1:?world}; EP=${2:-2}; OUT=${3:-$PWD/probe_$W}
HERE=$(cd "$(dirname "$0")" && pwd)
SCN=$HOME/Dynamic_obstacle/install/dyn_gz_worlds/share/dyn_gz_worlds/scenarios/$W.yaml
mkdir -p $OUT
cd $HOME/comparision_for_kairos/SANDO_planner
NAME=SANDO_${W}_probe EPISODES=$EP WORLD=$W ./run.sh > $OUT/run.out 2>&1 &
RUN=$!
sleep 40                                   # stack up (Gazebo 25 s, MAVROS 8 s, SANDO)
( source ~/inference_kairos/install/setup.bash; source $HOME/comparision_for_kairos/SANDO_planner/ws/install/setup.bash
  exec /usr/bin/python3 $HERE/sando_probe.py $SCN $OUT/probe.npz ) > $OUT/probe.log 2>&1 &
PR=$!
wait $RUN
kill -TERM $PR; sleep 3
source ~/inference_kairos/install/setup.bash
python3 $HERE/analyze_probe.py $SCN $OUT/probe.npz | tee $OUT/analysis.txt
