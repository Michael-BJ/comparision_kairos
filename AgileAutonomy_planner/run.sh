#!/bin/bash
# 2026-11-02 Closed-loop test of Agile Autonomy (PlaNet + polynomial fit, AgileAutonomy_planner/aa_node.py) on the
# KAIROS stack: same world, drone x500_oakd_x2_60_tw2, PX4 odometry, trajectory_server and v_max 1.5 as
# ~/kairos_test/20261102_compare/run.sh (KAIROS+MIGHTY). The network was trained by AgileAutonomy_planner/train.py
# on the KAIROS dataset (same train / val files as runs/2nov_exc3/B) with the rig panorama as its depth image.
# Results go next to the KAIROS run, so hits.py there reads them: python3 hits.py AGILE [WORLD].
# Env: MODEL (PlaNet .pth, required), NAME, WORLD, EPISODES, HEADLESS.
[ -n "$MODEL" ] || { echo "set MODEL=<path to best.pth of AgileAutonomy_planner/train.py>"; exit 1; }
O=$HOME/kairos_test/20261102_compare; S=$HOME/kairos_test/cam_compare_20261022/stop_stack.sh
HERE=$(cd "$(dirname "$0")" && pwd)
W=${WORLD:-dyn_long200_smooth_021_clear2}; EPISODES=${EPISODES:-24}; NAME=${NAME:-AGILE}; VMAX=1.5; TF=4.0
CFG=$O/config/expert_m4.yaml; D=x500_oakd_x2_60_tw2
stop_all() { $S; pkill -INT -f 'AgileAutonomy_planner/[a]a_node.py'; sleep 3; pkill -KILL -f 'AgileAutonomy_planner/[a]a_node.py'; }
source ~/inference_kairos/install/setup.bash
L=$O/logs/$NAME; mkdir -p $L
stop_all > $L/stop_before.log 2>&1
grep -q "STILL RUNNING" $L/stop_before.log && { echo "ABORT: stack survived"; exit 1; }
echo "$(date +%T) start $NAME ($MODEL)"
ros2 launch dyn_gz_worlds sim.launch.py world:=$W headless:=${HEADLESS:-false} drone_model:=$D > $L/sim.log 2>&1 &
sleep 25
ros2 launch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14557 > $L/mavros.log 2>&1 &
sleep 8
ros2 launch $HERE/launch/aa_test.launch.py world:=$W drone_model:=$D config:=$CFG model_path:=$MODEL v_max:=$VMAX \
     timeout_factor:=$TF episodes:=$EPISODES out_dir:=$O/runs/$NAME run_name:=$NAME > $L/test.log 2>&1 &
/usr/bin/python3 $HOME/kairos_test/20261027_odomlag/record.py ${D}_0 14400 $O/odom_$NAME.npz > $L/record.log 2>&1 &
REC=$!
for i in $(seq 1 1800); do
  sleep 10
  grep -a -q "real episodes [0-9]*:\|Traceback (most recent call last):\|left OFFBOARD" $L/test.log && break
done
sleep 10
kill $REC 2>/dev/null
stop_all > $L/stop_after.log 2>&1
echo "$(date +%T) done $NAME: $(grep -a -o 'real episodes [0-9]*: .*' $L/test.log | tail -1)"
grep -a -q "left OFFBOARD" $L/test.log && echo "WARNING: $NAME INVALID (left OFFBOARD)"
grep -a -q "aa_planner-[0-9]*\] Traceback" $L/test.log && echo "WARNING: $NAME traceback of aa_planner in test.log"
echo "ALL DONE"
