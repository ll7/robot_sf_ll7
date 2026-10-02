"""Closed-matrix validation and descriptive Wilson summaries, no admission claims."""

import argparse
import collections
import csv
import gzip
import json
import math
import pathlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=pathlib.Path, required=True, help="Completed probe directory")
parser.add_argument(
    "--reachability", type=pathlib.Path, required=True, help="Physical reachability witness JSON"
)
args = parser.parse_args()
out = args.input.resolve()
manifest = json.loads((out / "manifest.json").read_text())
rows = [json.loads(p.read_text()) for p in (out / "results").glob("*.json")]
by = {(r["planner"], r["arm"], r["probe"], r["scenario"], r["seed"]): r for r in rows}
assert len(rows) == len(by) == 4620, len(rows)
roster = [p["key"] for p in manifest["roster"]]
assert len(roster) == 14
assert all(1001 <= r["seed"] <= 1010 for r in rows)
assert all(r["status"] != "error" and r.get("outcome") is not None for r in rows)
for r in rows:
    assert r["probe"] == 2 or r["seed"] <= 1005
    if r["planner"] in {"ppo", "guarded_ppo", "sacadrl"}:
        ck = r["metadata"]["planner_runtime"]["checkpoint_provenance"]
        assert ck["load_succeeded"] and not ck["fallback_triggered"], r["cell"]
    for key in ["route_complete", "collision_event", "timeout_event"]:
        assert key in r["outcome"], r["cell"]
    # Success, contact and budget expiry are separately defined runner outcomes.
    r.update(
        collision=bool(r["outcome"]["collision_event"]), timeout=bool(r["outcome"]["timeout_event"])
    )
    assert r["success"] == bool(r["outcome"]["route_complete"]), r["cell"]
for p in roster:
    for a in ["DD", "T60-30", "T60-45"]:
        for probe, n in [(1, 25), (2, 60)]:
            rr = [r for r in rows if r["planner"] == p and r["arm"] == a and r["probe"] == probe]
            assert len(rr) == n, (p, a, probe, len(rr))
    for a in ["DD-legacy", "BI-base", "BI-fixed"]:
        assert len([r for r in rows if r["planner"] == p and r["arm"] == a]) == 25
paired = []
for r in rows:
    if r["arm"] in {"T60-30", "T60-45"}:
        dd = by[(r["planner"], "DD", r["probe"], r["scenario"], r["seed"])]
        if r["reset"] != dd["reset"]:
            paired.append(r["cell"])
        c = r["robot_config"]
        assert c["radius"] == 0.64 and c["wheelbase"] == 0.9 and c["max_velocity"] == 1.34
        assert c["max_steer"] == (0.52 if r["arm"] == "T60-30" else 0.79)
        assert c["max_accel"] == c["max_decel"] == 1.0 and not c["allow_backwards"]
assert not paired, paired
# Independent forward-only physical trajectories certify reachability in the empty map.
reach = json.loads(args.reachability.read_text())
assert len(reach) == 10 and all(r["reached"] and r["steps"] < 600 for r in reach)
failures = []
for r in rows:
    if r["arm"] not in {"T60-30", "T60-45"} or r["success"]:
        continue
    dd = by[(r["planner"], "DD", r["probe"], r["scenario"], r["seed"])]
    if not dd["success"]:
        continue
    trace = json.load(gzip.open(out / "traces" / f"{r['cell']}.json.gz", "rt"))
    c = r["robot_config"]
    errors = [
        (t["step"], abs(t["yaw"] - t["v"] * math.tan(t["action"][1]) / c["wheelbase"]))
        for t in trace
    ]
    worst = max(errors, key=lambda e: e[1]) if errors else (None, math.inf)
    # A physical law mismatch is not enough to attribute an adapter defect: collision
    # handling, multiple simulator substeps, and observation timing can also cause it.
    if r["probe"] == 1 and worst[1] < 1e-5:
        label = "planner error"
        why = "Empty matched map is reachable in <8.3s by the same forward-only bicycle physics; executed yaw matches v*tan(steer)/L. Failure is the planner/controller strategy on the changed plant, not a geometric impossibility."
        witness = next(
            x for x in reach if x["steer"] == c["max_steer"] and str(x["bearing"]) == r["scenario"]
        )
    else:
        label = "unclear"
        why = "Paired DD succeeds, but obstacle/crowd response and steering-constrained planning are confounded. No reverse/space or feasible obstacle-path counterfactual certifies a kinematic limit or planner defect."
        witness = None
    row = {
        "cell": r["cell"],
        "planner": r["planner"],
        "arm": r["arm"],
        "probe": r["probe"],
        "scenario": r["scenario"],
        "seed": r["seed"],
        "classification": label,
        "reason": why,
        "stuck": r["stuck"],
        "yaw_law_max_error": worst,
        "trace": f"traces/{r['cell']}.json.gz",
        "dd_record": f"records/{dd['cell']}.json.gz",
        "t60_record": f"records/{r['cell']}.json.gz",
        "empty_witness_steps": witness["steps"] if witness else None,
    }
    if trace:
        row["step_witness"] = next(
            (t for t in trace if abs(t["cmd"][1]) > 1e-5 and abs(t["yaw"]) < 1e-5), trace[-1]
        )
    failures.append(row)
(out / "t60_only_failures.json").write_text(json.dumps(failures, indent=2))


def wilson(k, n):
    """Return the descriptive 95 percent Wilson score interval."""
    z = 1.959963984540054
    d = 1 + z * z / n
    c = (k / n + z * z / (2 * n)) / d
    h = z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def stats(rr):
    """Count independent outcome flags and stuck steps for one group."""
    n = len(rr)
    s = {"n": n, "stuck": sum(r["stuck"] for r in rr)}
    for k in ["success", "collision", "timeout"]:
        count = sum(bool(r[k]) for r in rr)
        lo, hi = wilson(count, n)
        s.update({k: count, k + "_rate": count / n, k + "_lo": lo, k + "_hi": hi})
    return s


summary = []
for p in ["ALL", *roster]:
    for probe in [1, 2]:
        for arm in ["DD", "T60-30", "T60-45"]:
            rr = [
                r
                for r in rows
                if (p == "ALL" or r["planner"] == p) and r["arm"] == arm and r["probe"] == probe
            ]
            summary.append(dict(planner=p, probe=probe, arm=arm, **stats(rr)))
with (out / "comparison.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)
with (out / "episode_summary.csv").open("w") as f:
    fields = [
        "cell",
        "planner",
        "arm",
        "probe",
        "scenario",
        "seed",
        "source_sha",
        "status",
        "success",
        "collision",
        "timeout",
        "steps",
        "stuck",
        "zero_turn",
        "true_infeasible",
        "underreported",
    ]
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: r["cell"]))
legacy = {
    a: stats([r for r in rows if r["arm"] == a]) for a in ["DD-legacy", "BI-base", "BI-fixed"]
}
result = {
    "rows": len(rows),
    "legacy": legacy,
    "summary": summary,
    "t60_only_failure_counts": dict(collections.Counter(r["classification"] for r in failures)),
    "t60_only_by_arm_probe": dict(
        collections.Counter(f"{r['arm']}/P{r['probe']}" for r in failures)
    ),
    "paired_reset_mismatches": paired,
    "claim_boundary": "Diagnostic dev measurements only; Wilson intervals are descriptive episode proportions, not independent-seed ranking tests. No publication, release admission, or trained-bicycle support claim.",
}
(out / "summary.json").write_text(json.dumps(result, indent=2))
print(
    json.dumps(
        {
            "rows": len(rows),
            "legacy": legacy,
            "t60_only_failure_counts": result["t60_only_failure_counts"],
            "aggregate": [r for r in summary if r["planner"] == "ALL"],
        },
        indent=2,
    )
)
