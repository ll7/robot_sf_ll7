"""Check (e): parity of the rebuilt execution against the replication's compact rows."""
import gzip, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import exsel_screen as ex

run_root, compact, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
seeds = [1031, 1032, 1033, 1034, 1035]
ref = {}
with gzip.open(compact, "rt") as fh:
    for line in fh:
        r = json.loads(line)
        if r["scenario_id"] == ex.CELL and r["planner_key"] in ex.ARMS and r["seed"] in seeds:
            ref[(r["planner_key"], r["seed"])] = r
new = ex.load_run(run_root)
assert sorted(ref) == sorted(new), (sorted(ref), sorted(new))
rows, ok = [], True
for key in sorted(new):
    a, b = ref[key], new[key]
    cmp = {
        "outcome_class": (ex.classify(a["metrics"], a["outcome"])[0], b["outcome_class"]),
        "collision_kind": (ex.classify(a["metrics"], a["outcome"])[1], b["collision_kind"]),
        "metrics.success": (a["metrics"]["success"], b["metrics_success"]),
        "outcome.collision_event": (a["outcome"]["collision_event"], b["collision_event"]),
        "outcome.timeout_event": (a["outcome"]["timeout_event"], b["timeout_event"]),
        "steps": (a["steps"], b["steps"]),
        "termination_reason": (a["termination_reason"], b["termination_reason"]),
        "ped_collision_count": (a["metrics"]["ped_collision_count"], b["ped_collision_count"]),
        "obstacle_collision_count": (a["metrics"]["obstacle_collision_count"], b["obstacle_collision_count"]),
        "ablation_disable_hard_guard": (a["factors"]["ablation_disable_hard_guard"], b["algo_ablation_disable_hard_guard"]),
        "mode": (a["mode"], b["mode"]),
        "fallback_count": (a["fallback_count"], b["fallback_count"]),
    }
    match = all(x == y for x, y in cmp.values())
    ok &= match
    rows.append({"arm": key[0], "seed": key[1], "match": match,
                 "replication": {k: v[0] for k, v in cmp.items()},
                 "rebuild": {k: v[1] for k, v in cmp.items()},
                 "replication_execution_commit": a["git_hash"], "rebuild_execution_commit": b["execution_commit"],
                 "replication_source_line_sha256": a["source_line_sha256"], "rebuild_row_sha256": b["episode_row_sha256"]})
res = {"check": "e_parity", "cell": ex.CELL, "seeds": seeds, "arms": list(ex.ARMS),
       "compared_fields": list(cmp), "n_pairs": len(rows), "all_match": ok, "rows": rows}
out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
for r in rows:
    print(r["arm"], r["seed"], r["match"], r["rebuild"]["outcome_class"], r["rebuild"]["steps"], r["rebuild"]["termination_reason"], "| rep:", r["replication"]["outcome_class"], r["replication"]["steps"], r["replication"]["termination_reason"])
print("ALL_MATCH", ok)
sys.exit(0 if ok else 1)
