"""Step 3: compare the trace-capable rerun rows with the screening rows (exact fields)."""
import csv, json, sys
from typing import Any
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import exsel_screen as ex

rerun_root, ledger_csv, seed, out = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
FIELDS = ["outcome_class", "termination_reason", "steps", "ped_collision_count",
          "obstacle_collision_count", "near_misses", "min_ped_separation_m"]
screen = {}
with ledger_csv.open() as fh:
    for r in csv.DictReader(fh):
        if int(r["seed"]) == seed:
            screen[r["arm"]] = r
rerun = ex.load_run(rerun_root)
def norm(v):
    return "" if v is None else str(v)
arms: dict[str, Any] = {}
res: dict[str, Any] = {"seed": seed, "fields": FIELDS, "arms": arms}
ok = True
for arm in ex.ARMS:
    s, t = screen[arm], rerun[(arm, seed)]
    diffs = {f: {"screening": s[f], "rerun": norm(t[f])} for f in FIELDS if s[f] != norm(t[f])}
    arms[arm] = {"match": not diffs, "diffs": diffs, "rerun_row_sha256": t["episode_row_sha256"],
                        "screening_row_sha256": s["episode_row_sha256"],
                        "full_row_identical": t["episode_row_sha256"] == s["episode_row_sha256"],
                        "mode": t["mode"], "fallback_count": t["fallback_count"],
                        "fallback_or_degraded_marker": t["fallback_or_degraded_marker"]}
    ok &= not diffs
res["trace_accepted"] = ok
out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
print(json.dumps(res, indent=2, sort_keys=True))
