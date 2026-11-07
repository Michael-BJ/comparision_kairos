"""Static and dynamic contacts from the recorded Gazebo pose, the referee's rule: python3 hits.py NAME [WORLD]. Prints the distance flown too."""
import numpy as np, os, sys
from dyn_expert.core.scenario import Scenario
n = sys.argv[1]; w = sys.argv[2] if len(sys.argv) > 2 else "dyn_long200_smooth_021_clear2"
scn = Scenario.load(os.path.expanduser("~/Dynamic_obstacle/src/dyn_obstacle_nav/dyn_gz_worlds/scenarios/%s.yaml" % w))
g = np.load(os.path.expanduser("~/kairos_test/20261102_compare/odom_%s.npz" % n))["gt"]; t = g[:, 0]; p = g[:, 3:6]
d = scn.static_distance(p); x = t[d < .35]; st = int(len(x) > 0) + int(np.sum(np.diff(x) > .5))
dy = 0
for ob in scn.dynamic:
    q = np.asarray(ob.pos(t)).reshape(len(t), 3); x = t[np.linalg.norm(p - q, axis=1) - ob.radius < .35]
    if len(x): dy += int(np.sum(np.r_[True, np.diff(x) > .5]))
m = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
print("%s: %.1f min, %.0f m, static %d, dynamic %d" % (n, (t[-1] - t[0]) / 60, m, st, dy))
# 2026-11-06: out of bounds, the same rule for every method (KAIROS's own: > 1 m outside the scenario bounds);
# a pass with an excursion does not count as a success. Second line, so parsers of the first line keep working.
lo, hi = np.asarray(scn.lo, float), np.asarray(scn.hi, float)
ex = np.maximum(lo - 1.0 - p, p - hi - 1.0).max(axis=1); x = t[ex > 0]
nb = int(len(x) > 0) + int(np.sum(np.diff(x) > .5))
print("%s: out of bounds %d%s" % (n, nb, (" (max %.2f m beyond bounds + 1 m, at t %s)" % (ex.max(), np.round(
    x[np.r_[True, np.diff(x) > .5]] - t[0], 1).tolist())) if nb else ""))
