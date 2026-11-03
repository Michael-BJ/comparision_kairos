#!/bin/bash
# 2026-11-02 Closed-loop test of the NEO-style regression baseline (KAIROS network, not NEO; NEO itself: run.sh), runs/2nov_exc3/NEO: one network trajectory per event
# (K = 1, set by kairos_planner for a regression checkpoint), refined by SFC + L-BFGS, as NEO refines its NN guess.
# Same stack and settings as ~/kairos_test/20261102_compare/run.sh (KAIROS+MIGHTY): x500_oakd_x2_60_tw2, PX4 odometry,
# v_max 1.5, memory 4 s, hold, no inflation, no emergency, static_clearance 0.55. One difference: the checkpoint has no
# head 2, so prediction is cv (the tracker's constant velocity) instead of head2.
# Results go next to the KAIROS run, so hits.py there reads them: python3 hits.py NEO_regression [WORLD].
# Env: CK, NAME, WORLD, EPISODES, HEADLESS.
O=$HOME/kairos_test/20261102_compare; S=$HOME/kairos_test/cam_compare_20261022/stop_stack.sh
W=${WORLD:-dyn_long200_smooth_021_clear2}; EPISODES=${EPISODES:-24}; NAME=${NAME:-NEO_regression}; VMAX=1.5; TF=4.0; SC=0.55
CFG=$O/config/expert_m4.yaml
CK=${CK:-$HOME/kairosV3/runs/2nov_exc3/NEO/best.pt}; KR=$HOME/kairosV3; D=x500_oakd_x2_60_tw2
source ~/inference_kairos/install/setup.bash
L=$O/logs/$NAME; mkdir -p $L
$S > $L/stop_before.log 2>&1
grep -q "STILL RUNNING" $L/stop_before.log && { echo "ABORT: stack survived"; exit 1; }
echo "$(date +%T) start $NAME"
ros2 launch dyn_gz_worlds sim.launch.py world:=$W headless:=${HEADLESS:-false} drone_model:=$D > $L/sim.log 2>&1 &
sleep 25
ros2 launch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14557 > $L/mavros.log 2>&1 &
sleep 8
ros2 launch kairos_planner test.launch.py world:=$W drone_model:=$D start_mavros:=false config:=$CFG mode:=kairos_mighty seed_mode:=network prediction:=cv v_max:=$VMAX timeout_factor:=$TF static_clearance:=$SC \
     memory_belief:=-1.0 safety_vision:=0.0 remembered_pred:=hold memory_time:=4.0 memory_inflate:=false emergency:=false \
     continue_on_collision:=true bounds_margin:=1000 ckpt:=$CK kairos_root:=$KR \
     episodes:=$EPISODES out_dir:=$O/runs/$NAME run_name:=$NAME > $L/test.log 2>&1 &
/usr/bin/python3 $HOME/kairos_test/20261027_odomlag/record.py ${D}_0 14400 $O/odom_$NAME.npz > $L/record.log 2>&1 &
REC=$!
for i in $(seq 1 1800); do
  sleep 10
  grep -a -q "real episodes [0-9]*:\|Traceback (most recent call last):\|left OFFBOARD" $L/test.log && break
done
sleep 10
kill $REC 2>/dev/null
$S > $L/stop_after.log 2>&1
echo "$(date +%T) done $NAME: $(grep -a -o 'real episodes [0-9]*: .*' $L/test.log | tail -1)"
grep -a -q "left OFFBOARD" $L/test.log && echo "WARNING: $NAME INVALID (left OFFBOARD)"
grep -a -q "kairos_planner-3\] Traceback" $L/test.log && echo "WARNING: $NAME traceback in test.log"
grep -a -q "regression (single output" $L/test.log || echo "WARNING: $NAME did not load a regression checkpoint"
echo "ALL DONE"
