#!/bin/bash
# 2026-11-04 SANDO, EPISODES passes as run.sh, but the whole stack is restarted when SANDO is stuck for good: a pass
# that timed out and whose follow-on also timed out having got < 5 m closer to its goal (SANDO hovers where no corridor
# contains its start and has no recovery; README "Diagnosis"). The bridge then ends the run (stop_if_stuck:=true) and
# this script starts a new run <NAME>_r<k> from pad A with the remaining episodes. The stuck pass and its follow-on stay
# in the count as failed episodes. A variant of the test protocol for SANDO only: report it next to the plain run.sh
# result. Results: results.py <NAME>_r1 <NAME>_r2 ... (episodes concatenated in order); hits.py per run.
# Env as run.sh (NAME, EPISODES, EXTRA, ...); DONE=<n> K=<k> resume a series after its run <NAME>_r<k>, n episodes kept.
HERE=$(cd "$(dirname "$0")" && pwd)
O=$HOME/kairos_test/20261102_compare
TOTAL=${EPISODES:-24}; BASE=${NAME:-SANDO_restart}; DONE=${DONE:-0}; K=${K:-0}
while [ $DONE -lt $TOTAL ]; do
  K=$((K + 1))
  N=${BASE}_r$K
  EPISODES=$((TOTAL - DONE)) NAME=$N EXTRA="$EXTRA stop_if_stuck:=true" $HERE/run.sh || exit 1
  S=$(ls -t $O/runs/$N/*/summary.json 2>/dev/null | head -1)
  [ -n "$S" ] || { echo "ABORT: no summary for $N"; exit 1; }
  n=$(/usr/bin/python3 -c "import json, sys; print(len(json.load(open(sys.argv[1]))['episodes']))" $S)
  [ "$n" -gt 0 ] || { echo "ABORT: $N flew no episode"; exit 1; }
  DONE=$((DONE + n))
  echo "$(date +%T) $N: $n episodes, $DONE of $TOTAL"
done
echo "RESTART SERIES DONE: ${BASE}_r1 .. ${BASE}_r$K"
