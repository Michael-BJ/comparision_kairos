#!/bin/bash
# 2026-11-02 Closed-loop test of SANDO (ws/src/sando v0.0.3, built by build.sh) with its own sensor pipeline (Livox
# Mid-360 -> ACL mapper and its temporal-grid tracker, as SANDO's hardware; models/x500_mid360_d435_tw2 also carries
# the URDF's D435, unused by default) on the KAIROS stack: same world, airframe (x500 tw2), PX4 odometry,
# trajectory_server (with SANDO's yaw), v_max 1.5 as ~/kairos_test/20261102_compare/run.sh. Defaults of
# launch/sando_test.launch.py = the setup of README "Diagnosis" (2026-11-04).
# Results go next to the KAIROS run, so hits.py there reads them: python3 hits.py SANDO [WORLD] (hits.py reads
# odom_<NAME>.npz recorded for model <drone>_0).
# Env: NAME, WORLD, EPISODES, HEADLESS, MAPPER (mid360 | d435), SANDO_LOG (log level of sando_node, default warn), SANDO_CONFIG (sando_hw_quadrotor.yaml | sando.yaml),
# EXTRA (further launch arguments, e.g. "tuning:=hw yaw:=test execute:=trajs tracker_sensor:=d435").
O=$HOME/kairos_test/20261102_compare; S=$HOME/kairos_test/cam_compare_20261022/stop_stack.sh
HERE=$(cd "$(dirname "$0")" && pwd)
W=${WORLD:-dyn_long200_smooth_021_clear2}; EPISODES=${EPISODES:-24}; NAME=${NAME:-SANDO}; VMAX=1.5; TF=4.0
CFG=$O/config/expert_m4.yaml; D=x500_mid360_d435_tw2
P='[s]ando_bridge.py|[s]ando_trajectory_server.py|[l]idar_fill.py|[g]lobal_mapper_node|lib/sando/[s]ando |[o]bstacle_tracker_node'
stop_all() { $S; pkill -INT -f "$P"; sleep 3; pkill -KILL -f "$P"; }
source ~/inference_kairos/install/setup.bash
source $HERE/ws/install/setup.bash
export GUROBI_HOME=$HOME/opt/gurobi1103/linux64 LD_LIBRARY_PATH=$HOME/opt/gurobi1103/linux64/lib:$LD_LIBRARY_PATH
export GRB_LICENSE_FILE=$HOME/gurobi.lic
export GZ_SIM_RESOURCE_PATH=$HERE/models:$GZ_SIM_RESOURCE_PATH
L=$O/logs/$NAME; mkdir -p $L
stop_all > $L/stop_before.log 2>&1
grep -q "STILL RUNNING" $L/stop_before.log && { echo "ABORT: stack survived"; exit 1; }
echo "$(date +%T) start $NAME (SANDO, original sensors, mapper ${MAPPER:-mid360})"
ros2 launch dyn_gz_worlds sim.launch.py world:=$W headless:=${HEADLESS:-false} drone_model:=$D > $L/sim.log 2>&1 &
sleep 25
ros2 launch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14557 > $L/mavros.log 2>&1 &
sleep 8
ros2 launch $HERE/launch/sando_test.launch.py world:=$W drone_model:=$D config:=$CFG v_max:=$VMAX timeout_factor:=$TF \
     mapper_sensor:=${MAPPER:-mid360} sando_log_level:=${SANDO_LOG:-warn} sando_debug:=${SANDO_DEBUG:-false} sando_config:=${SANDO_CONFIG:-sando_hw_quadrotor.yaml} ${TRACKER_PREFIX:+tracker_prefix:="$TRACKER_PREFIX"} ${SANDO_SET:+sando_set:="$SANDO_SET"} $EXTRA episodes:=$EPISODES out_dir:=$O/runs/$NAME run_name:=$NAME > $L/test.log 2>&1 &
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
grep -a -q "sando_bridge-[0-9]*\] Traceback" $L/test.log && echo "WARNING: $NAME traceback of sando_bridge in test.log"
echo "ALL DONE"
