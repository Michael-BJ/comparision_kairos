#!/bin/bash
# Fly one method on the SANDO-based worlds, one world after another (Gazebo window open unless HEADLESS=true),
# then count the collisions from the Gazebo pose (hits.py). Each run.sh stops any running Gazebo / PX4 / ROS stack.
#
#   run_worlds.sh <method> [episodes] [world ...]
#   method: kairos | sando | sando_d435 | neo | neo_regression | agile
#   default: 2 episodes on the static-only forests (dyn_sando_forest_easy/medium/hard, SANDO's own static benchmark)
#            and the static + dynamic worlds dyn_sando_easy dyn_sando_medium dyn_sando_hard_dense (the hard level;
#            dyn_sando_hard was deleted 2026-11-06)
#   sando in a dyn_sando_forest_* world gets SANDO's own settings for its static forests (2026-11-06): its run_sim.py
#   uses environment_assumption static and static_global_mapper.yaml there, and sando.yaml's simulation drone_bbox
#   0.2 m. SANDO_FOREST=hw keeps the run.sh defaults (environment_assumption dynamic, hw_global_mapper.yaml, bbox 0.9).
#   SANDO_BBOX=0.7 sets that forest bbox to the x500's size instead (2026-11-07: every method plans with the drone
#   radius the referee uses, ~0.35 m; KAIROS and NEO plan with 0.35). SANDO_RESTART=true flies sando through
#   SANDO_planner/run_restart.sh: the stack is restarted when SANDO is stuck for good, runs <NAME>_r1, _r2, ...
#   SUFFIX=_v2 run_worlds.sh kairos 2 dyn_sando_hard_dense -> run name KAIROS_dyn_sando_hard_dense_2ep_v2
#   Bounds: both planners keep their targets / trajectories inside the scenario bounds (KAIROS: local targets >= 1 m
#   inside, planner config bounds_margin 1.0; SANDO: y_min / y_max). No method's run is ended out of bounds
#   (KAIROS run.sh bounds_margin:=1000, SANDO's bridge has no such end); hits.py counts excursions for every method
#
# Results: ~/kairos_test/20261102_compare/{logs,runs}/<NAME>/ and odom_<NAME>.npz, NAME = <METHOD>_<world>_<n>ep<SUFFIX>
M=${1:?method}; EP=${2:-2}
shift; [ $# -gt 0 ] && shift
WORLDS=${*:-dyn_sando_forest_easy dyn_sando_forest_medium dyn_sando_forest_hard dyn_sando_easy dyn_sando_medium dyn_sando_hard_dense}
CMP=$HOME/kairos_test/20261102_compare; REPO=$HOME/comparision_for_kairos
NEO_MODEL=${NEO_MODEL:-$HOME/saved_net/neo_kairos/run_20261102_161626/best.pth}
AGILE_MODEL=${AGILE_MODEL:-$HOME/saved_net/aa_kairos/run_20261102_221529/best.pth}
names=()
SR=run.sh; [ "${SANDO_RESTART:-false}" = true ] && SR=run_restart.sh
for W in $WORLDS; do
  FS=""; FE=""
  if [[ $W == dyn_sando_forest_* && ${SANDO_FOREST:-sim} != hw ]]; then
    FS="environment_assumption=static"; B=${SANDO_BBOX:-0.2}; FE="mapper_param_file:=static_global_mapper.yaml drone_bbox:=$B,$B,$B"
  fi
  case $M in
    kairos)         N=KAIROS_${W}_${EP}ep$SUFFIX; (cd $CMP && NAME=$N EPISODES=$EP WORLD=$W ./run.sh) ;;
    sando)          N=SANDO_${W}_${EP}ep$SUFFIX; (cd $REPO/SANDO_planner && SANDO_SET="$FS${SANDO_SET:+${FS:+,}$SANDO_SET}" EXTRA="$FE $EXTRA" NAME=$N EPISODES=$EP WORLD=$W ./$SR) ;;
    sando_d435)     N=SANDO_d435_${W}_${EP}ep$SUFFIX; (cd $REPO/SANDO_planner && NAME=$N EPISODES=$EP WORLD=$W MAPPER=d435 ./run.sh) ;;
    neo)            N=NEO_${W}_${EP}ep$SUFFIX; (cd $REPO/NEO_planner && MODEL=$NEO_MODEL NAME=$N EPISODES=$EP WORLD=$W ./run.sh) ;;
    neo_regression) N=NEO_regression_${W}_${EP}ep$SUFFIX; (cd $REPO/NEO_planner && NAME=$N EPISODES=$EP WORLD=$W ./run_regression.sh) ;;
    agile)          N=AGILE_${W}_${EP}ep$SUFFIX; (cd $REPO/AgileAutonomy_planner && MODEL=$AGILE_MODEL NAME=$N EPISODES=$EP WORLD=$W ./run.sh) ;;
    *) echo "unknown method $M"; exit 1 ;;
  esac
  names+=("$N $W")
done
source ~/inference_kairos/install/setup.bash
cd $CMP
for nw in "${names[@]}"; do
  set -- $nw
  if [ -d runs/$1 ]; then python3 hits.py $nw; else for r in $(ls -d runs/${1}_r* 2>/dev/null | sort -V); do python3 hits.py $(basename $r) $2; done; fi
done
