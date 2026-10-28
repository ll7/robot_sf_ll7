"""Post-hoc, descriptive: guard pedestrian-cost contrast on (cell, seed) pairs where BOTH hybrid
arms completed (parking-rule invariant: both robots leave the scene). Selection on outcome."""

import csv
import gzip
import json
import sys

import numpy as np

rows = list(csv.DictReader(gzip.open(sys.argv[1], "rt")))
idx = {(r["arm"], r["cell"], r["condition"], r["seed"]): r for r in rows}
out = {}
for cell in sorted({r["cell"] for r in rows}):
    d, b, n = [], [], 0
    for r in rows:
        if r["arm"] != "G1E1" or r["cell"] != cell or r["condition"] != "traverse":
            continue
        g0 = idx[("G0E1", cell, "traverse", r["seed"])]
        if (
            r["success"] == "1"
            and g0["success"] == "1"
            and r["mean_delay_censored_s"]
            and g0["mean_delay_censored_s"]
        ):
            n += 1
            d.append(float(r["mean_delay_censored_s"]) - float(g0["mean_delay_censored_s"]))
            p = int(r["nr_passers"])
            b.append(100 * (int(r["nr_passers_blocked"]) - int(g0["nr_passers_blocked"])) / p)
    out[cell] = {
        "pairs": n,
        "delay_diff_s_mean": round(float(np.mean(d)), 3) if d else None,
        "blocked_diff_pp_mean": round(float(np.mean(b)), 3) if b else None,
    }
print(json.dumps(out, indent=1))
