#!/bin/bash
# Fly one method on the SANDO-based worlds, one world after another (Gazebo window open unless HEADLESS=true),
# then count the collisions from the Gazebo pose (hits.py). Each run.sh stops any running Gazebo / PX4 / ROS stack.
#
#   run_worlds.sh <method> [episodes] [world ...]
#   method: kairos | sando | sando_d435 | neo | neo_regression | agile
#   default: 2 episodes on dyn_sando_easy dyn_sando_medium dyn_sando_hard
#   SUFFIX=_v2 run_worlds.sh kairos 2 dyn_sando_hard       -> run name KAIROS_dyn_sando_hard_2ep_v2
#
# Results: ~/kairos_test/20261102_compare/{logs,runs}/<NAME>/ and odom_<NAME>.npz, NAME = <METHOD>_<world>_<n>ep<SUFFIX>
M=${1:?method}; EP=${2:-2}
shift; [ $# -gt 0 ] && shift
WORLDS=${*:-dyn_sando_easy dyn_sando_medium dyn_sando_hard}
CMP=$HOME/kairos_test/20261102_compare; REPO=$HOME/comparision_for_kairos
NEO_MODEL=${NEO_MODEL:-$HOME/saved_net/neo_kairos/run_20261102_161626/best.pth}
AGILE_MODEL=${AGILE_MODEL:-$HOME/saved_net/aa_kairos/run_20261102_221529/best.pth}
names=()
for W in $WORLDS; do
  case $M in
    kairos)         N=KAIROS_${W}_${EP}ep$SUFFIX; (cd $CMP && NAME=$N EPISODES=$EP WORLD=$W ./run.sh) ;;
    sando)          N=SANDO_${W}_${EP}ep$SUFFIX; (cd $REPO/SANDO_planner && NAME=$N EPISODES=$EP WORLD=$W ./run.sh) ;;
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
for nw in "${names[@]}"; do python3 hits.py $nw; done
