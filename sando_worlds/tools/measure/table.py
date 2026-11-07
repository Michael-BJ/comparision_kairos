"""Results table of the SANDO-based worlds, every pass judged by the referee's rule of hits.py from the Gazebo pose.

  WORLDS="dyn_sando_easy dyn_sando_medium" python3 table.py KAIROS:24ep_v4 SANDO:24ep_v4R [METHOD:SUFFIX[:LABEL] ...]
  run name <METHOD>_<world>_<SUFFIX>; a restart series of SANDO_planner/run_restart.sh (<name>_r1, _r2, ...) is
  joined in order. Needs ~/inference_kairos (dyn_expert) sourced; poses ~/kairos_test/20261102_compare/odom_<run>.npz.

Rules (2026-11-07, the same for every method):
  contact      the drone, a sphere of radius 0.35 m around its Gazebo pose, touches a static obstacle surface or a
               moving sphere; samples closer than 0.5 s are one contact (hits.py)
  out of bounds  the pose more than 1 m outside the scenario bounds (hits.py)
  follow-on    a pass whose previous pass did not reach the same goal: that goal again from where the drone was left
               (both bridges fly it). One deadlock is one failed pass; its follow-ons are listed, not scored
  Passes       passes scored (not follow-ons) / passes in the run
  Reached, CR, SR over the scored passes: CR = passes with >= 1 contact, SR = reached, no contact, not out of bounds
  Hits         contacts of the whole recording (hits.py totals, between passes too)
  Min static   the closest the pose came to a static surface in a scored pass (< 0: centre inside the obstacle)
  Avg vel      path / duration of the scored passes
Each pass's time window comes from its summary.json duration and end position, matched on the pose.
"""
import glob
import json
import os
import sys

import numpy as np

from dyn_expert.core.scenario import Scenario

CMP = os.path.expanduser(os.environ.get("CMP", "~/kairos_test/20261102_compare"))
SC = os.path.expanduser("~/Dynamic_obstacle/src/dyn_obstacle_nav/dyn_gz_worlds/scenarios/%s.yaml")
R, MERGE, OOB = 0.35, 0.5, 1.0


def events(t):
    return int(len(t) > 0) + int(np.sum(np.diff(t) > MERGE)) if len(t) else 0


def runs(name):
    if os.path.isdir("%s/runs/%s" % (CMP, name)):
        return [name]
    r = sorted(glob.glob("%s/runs/%s_r[0-9]*" % (CMP, name)), key=lambda x: int(x.rsplit("_r", 1)[1]))
    return [os.path.basename(x) for x in r]


def windows(E, t, p):
    """(start, end) time of every pass: the end is the first pose sample at the pass's end position after
    previous end + duration (the summary's end is a pose of the same Gazebo odometry)."""
    out, prev = [], t[0]
    for e in E:
        d, pe = e["duration"], np.asarray(e["end"], float)
        lo, hi = prev + d - 1.0, prev + d + (300.0 if not out else 30.0)
        m = (t >= lo) & (t <= hi)
        if m.any():
            dist = np.linalg.norm(p[m] - pe, axis=1); i = np.where(dist <= dist.min() + 0.05)[0][0]
            te = t[m][i] if dist.min() < 0.5 else prev + d
        else:
            te = prev + d
        out.append((te - d, te)); prev = te
    return out


def score(name, w):
    scn = Scenario.load(SC % w); lo, hi = np.asarray(scn.lo, float), np.asarray(scn.hi, float)
    rows, tot = [], dict(st=0, dy=0)
    for n in runs(name):
        E = json.load(open(sorted(glob.glob("%s/runs/%s/*/summary.json" % (CMP, n)))[-1]))["episodes"]
        g = np.load("%s/odom_%s.npz" % (CMP, n))["gt"]; t, p = g[:, 0], g[:, 3:6]
        d = scn.static_distance(p)
        tot["st"] += events(t[d < R])
        q = [np.asarray(ob.pos(t)).reshape(len(t), 3) for ob in scn.dynamic]
        dd = [np.linalg.norm(p - x, axis=1) - ob.radius for x, ob in zip(q, scn.dynamic)]
        tot["dy"] += sum(events(t[x < R]) for x in dd)
        ex = np.maximum(lo - OOB - p, p - hi - OOB).max(axis=1)
        for k, (e, (ts, te)) in enumerate(zip(E, windows(E, t, p))):
            m = (t >= ts) & (t <= te)
            fo = (k > 0 and E[k - 1]["outcome"] != "reached"
                  and np.allclose(E[k - 1].get("goal", 0), e.get("goal", 0), atol=0.5))
            rows.append(dict(run=n, ep=e["episode"], outcome=e["outcome"], follow_on=fo, path=e["path"],
                             duration=e["duration"], st=events(t[m & (d < R)]),
                             dy=sum(events(t[m & (x < R)]) for x in dd), oob=events(t[m & (ex > 0)]),
                             dmin=float(d[m].min()) if m.any() else float("nan")))
    return rows, tot


def main():
    worlds = os.environ.get("WORLDS", "dyn_sando_easy dyn_sando_medium dyn_sando_hard_dense").split()
    specs = [(a.split(":") + [""])[:3] for a in sys.argv[1:]]
    print("| World | Method | Passes | Reached | CR (%) | SR (%) | Hits static | Hits dynamic | Out of bounds "
          "(passes) | Min static (m) | Avg vel (m/s) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    notes = []
    for w in worlds:
        for m, suf, lab in specs:
            name = "%s_%s_%s" % (m, w, suf)
            if not runs(name):
                print("| %s | %s | missing |" % (w.replace("dyn_sando_", ""), lab or m)); continue
            rows, tot = score(name, w); S = [r for r in rows if not r["follow_on"]]; n = len(S)
            reached = sum(r["outcome"] == "reached" for r in S)
            coll = sum(r["st"] + r["dy"] > 0 for r in S); oob = sum(r["oob"] > 0 for r in S)
            sr = sum(r["outcome"] == "reached" and r["st"] + r["dy"] + r["oob"] == 0 for r in S)
            vel = sum(r["path"] for r in S) / max(sum(r["duration"] for r in S), 1e-9)
            print("| %s | %s | %d/%d | %d/%d | %.1f | %.1f | %d | %d | %d | %.2f | %.2f |" % (
                w.replace("dyn_sando_", ""), lab or m, n, len(rows), reached, n, 100 * coll / n, 100 * sr / n, tot["st"],
                tot["dy"], oob, min(r["dmin"] for r in S), vel))
            fo = [r for r in rows if r["follow_on"]]; nr = len(runs(name))
            if fo or nr > 1:
                notes.append("%s %s: %d run(s), follow-ons %s, failed scored passes %s" % (
                    w, m, nr, [r["ep"] for r in fo] if nr == 1 else ["%s:%d" % (r["run"][-3:], r["ep"]) for r in fo],
                    [(r["ep"], r["outcome"]) for r in S if r["outcome"] != "reached"]))
    for x in notes:
        print("  " + x)


if __name__ == "__main__":
    main()
