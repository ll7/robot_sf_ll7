"""Compact and analyze the bounded bicycle comparison.

    python analyze.py compact --results <run>/results --output evidence/episodes_<stage>.csv
    python analyze.py analyze --episodes evidence/episodes_main.csv \
        --contract contract.yaml --output evidence/

Paired by (planner, cell, seed); BI minus DD. Uncertainty: 10,000 bootstrap draws
that resample complete seed blocks (every planner, cell and plant of a seed stays
together), numpy default_rng(1001), 2.5th/97.5th percentiles. Intervals are
pointwise and condition on the fixed six-cell roster.
"""

# ruff: noqa: C901, D103, PLR0912, PLR0915

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

FIELDS = [
    "stage",
    "probe",
    "planner",
    "plant",
    "scenario",
    "scenario_name",
    "seed",
    "mode",
    "command_path",
    "status",
    "termination",
    "success",
    "collision",
    "timeout",
    "ped_collisions",
    "obstacle_collisions",
    "steps",
    "time_s",
    "path_length_m",
    "stuck",
    "creep_steps",
    "max_abs_yaw_rate",
    "min_clearance_m",
    "final_speed",
    "source_sha",
    "reset_sha256",
]
DRAWS = 10_000
ENDPOINTS = ["success", "collision", "timeout"]


def compact(results, output):
    import hashlib

    rows = []
    for path in sorted(Path(results).glob("*.json")):
        r = json.loads(path.read_text())
        reset = r.get("reset")
        row = {k: r.get(k) for k in FIELDS}
        row["reset_sha256"] = (
            hashlib.sha256(json.dumps(reset, sort_keys=True, default=float).encode()).hexdigest()
            if reset
            else None
        )
        for k in ["success", "collision", "timeout"]:
            row[k] = int(bool(row[k]))
        for k in ["time_s", "path_length_m", "max_abs_yaw_rate", "min_clearance_m", "final_speed"]:
            if row[k] is not None:
                row[k] = f"{float(row[k]):.6f}"
        rows.append(row)
    rows.sort(
        key=lambda r: (int(r["probe"]), r["planner"], r["plant"], r["scenario"], int(r["seed"]))
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {output}")


def ci(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def analyze(episodes, contract, outdir):
    c = yaml.safe_load(Path(contract).read_text())
    rows = list(csv.DictReader(open(episodes)))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    planners = c["planners"]
    cells = [x["id"] for x in c["cells"]]
    scen = [r for r in rows if r["probe"] == "2"]
    seeds = sorted({int(r["seed"]) for r in scen})
    mode_counts = {}
    for r in rows:
        mode_counts[r["mode"]] = mode_counts.get(r["mode"], 0) + 1
    # arrays [planner, cell, seed] per plant and endpoint
    idx = {(r["planner"], r["plant"], r["scenario"], int(r["seed"])): r for r in scen}
    missing = [
        k
        for k in [
            (p, pl, cc, s) for p in planners for pl in ("DD", "BI") for cc in cells for s in seeds
        ]
        if k not in idx
    ]
    arr = {}
    for e in ENDPOINTS + ["time_s"]:
        for pl in ("DD", "BI"):
            a = np.full((len(planners), len(cells), len(seeds)), np.nan)
            for i, p in enumerate(planners):
                for j, cc in enumerate(cells):
                    for k, s in enumerate(seeds):
                        r = idx.get((p, pl, cc, s))
                        if r is not None:
                            a[i, j, k] = float(r[e])
            arr[(e, pl)] = a
    rng = np.random.default_rng(1001)
    boot = rng.integers(0, len(seeds), size=(DRAWS, len(seeds)))
    effects = []

    def add(level, planner, cell, endpoint, dd, bi, n, draws):
        effects.append(
            {
                "level": level,
                "planner": planner,
                "cell": cell,
                "endpoint": endpoint,
                "n_pairs": n,
                "dd_rate": round(dd, 6),
                "bi_rate": round(bi, 6),
                "bi_minus_dd_pp": round(100 * (bi - dd), 4),
                "ci_low_pp": round(100 * draws[0], 4),
                "ci_high_pp": round(100 * draws[1], 4),
            }
        )

    for e in ENDPOINTS:
        d = arr[(e, "BI")] - arr[(e, "DD")]  # planner, cell, seed
        for i, p in enumerate(planners):
            for j, cc in enumerate(cells):
                v = d[i, j]
                bs = np.nanmean(v[boot], axis=1)
                add(
                    "cell",
                    p,
                    cc,
                    e,
                    np.nanmean(arr[(e, "DD")][i, j]),
                    np.nanmean(arr[(e, "BI")][i, j]),
                    int(np.sum(~np.isnan(v))),
                    ci(bs),
                )
            pooled = np.nanmean(d[i], axis=0)  # equal-cell mean per seed
            bs = np.nanmean(pooled[boot], axis=1)
            add(
                "planner",
                p,
                "all_six",
                e,
                np.nanmean(arr[(e, "DD")][i]),
                np.nanmean(arr[(e, "BI")][i]),
                int(np.sum(~np.isnan(d[i]))),
                ci(bs),
            )
    # ordering: pairwise planner differences under each plant (success, equal-cell)
    ordering = []
    reversed_planners = set()
    s_dd, s_bi = arr[("success", "DD")], arr[("success", "BI")]
    for a in range(len(planners)):
        for b in range(a + 1, len(planners)):
            rec: dict[str, Any] = {"pair": f"{planners[a]}-minus-{planners[b]}"}
            for pl, s in (("DD", s_dd), ("BI", s_bi)):
                per_seed = np.nanmean(s[a] - s[b], axis=0)
                rec[pl] = {
                    "pp": round(100 * float(np.nanmean(per_seed)), 4),
                    "ci_pp": [round(100 * x, 4) for x in ci(np.nanmean(per_seed[boot], axis=1))],
                }
            dd_ci, bi_ci = rec["DD"]["ci_pp"], rec["BI"]["ci_pp"]
            dd_sign = 1 if dd_ci[0] > 0 else (-1 if dd_ci[1] < 0 else 0)
            bi_sign = 1 if bi_ci[0] > 0 else (-1 if bi_ci[1] < 0 else 0)
            rec["reversal"] = bool(dd_sign and bi_sign and dd_sign != bi_sign)
            if rec["reversal"]:
                reversed_planners.update((planners[a], planners[b]))
            rec["dd_resolved_bi_unresolved"] = bool(dd_sign and not bi_sign)
            ordering.append(rec)
    # time to goal on paired successes
    ttg = []
    for i, p in enumerate(planners):
        both = (s_dd[i] == 1) & (s_bi[i] == 1)
        diff = (arr[("time_s", "BI")][i] - arr[("time_s", "DD")][i])[both]
        ttg.append(
            {
                "planner": p,
                "paired_successes": int(both.sum()),
                "mean_bi_minus_dd_s": round(float(diff.mean()), 4) if diff.size else None,
            }
        )
    # empty-world probe
    emp = [r for r in rows if r["probe"] == "1"]
    probe = {}
    for r in emp:
        key = f"{r['planner']}|{r['plant']}|{r['scenario']}"
        d = probe.setdefault(
            key,
            {"n": 0, "success": 0, "timeout": 0, "collision": 0, "stuck_windows": 0, "time_s": []},
        )
        d["n"] += 1
        d["success"] += int(r["success"])
        d["timeout"] += int(r["timeout"])
        d["collision"] += int(r["collision"])
        d["stuck_windows"] += int(r["stuck"] or 0)
        if r["success"] == "1":
            d["time_s"].append(float(r["time_s"]))
    for d in probe.values():
        d["mean_success_time_s"] = round(float(np.mean(d["time_s"])), 3) if d["time_s"] else None
        del d["time_s"]
    # predeclared outcome classification (contract.outcome_rule)
    rule = c["outcome_rule"]
    plan = {}
    eff = {(x["level"], x["planner"], x["cell"], x["endpoint"]): x for x in effects}
    for p in planners:
        s = eff[("planner", p, "all_six", "success")]
        col = eff[("planner", p, "all_six", "collision")]
        cell_deficit = [
            cc
            for cc in cells
            if eff[("cell", p, cc, "success")]["ci_high_pp"] < -rule["cell_deficit_pp"]
        ]
        comparable = (
            s["ci_low_pp"] >= -rule["success_margin_pp"]
            and s["ci_high_pp"] <= rule["success_margin_pp"]
            and col["ci_low_pp"] >= -rule["collision_margin_pp"]
            and col["ci_high_pp"] <= rule["collision_margin_pp"]
            and not cell_deficit
        )
        no_transfer = (
            s["ci_high_pp"] < -rule["success_margin_pp"]
            or col["ci_low_pp"] > rule["collision_margin_pp"]
            or p in reversed_planners
        )
        incompatible = [
            b
            for b in c["empty_probe"]["bearings"]
            if probe[f"{p}|DD|{b}"]["success"] >= rule["probe_dd_min"]
            and probe[f"{p}|BI|{b}"]["success"] <= rule["probe_bi_max"]
        ]
        plan[p] = {
            "O1_comparable": comparable,
            "O2_no_transfer": no_transfer,
            "O2_cells_with_bicycle_deficit": cell_deficit,
            "O3_empty_world_incompatible_bearings": incompatible,
            "O5_open": not comparable and not no_transfer,
        }
    feas = []
    for j, cc in enumerate(cells):
        dd_any = max(float(np.nanmean(s_dd[i, j])) for i in range(len(planners)))
        bi_any = float(np.nansum(s_bi[:, j]))
        witness_seeds = int(np.sum(np.nanmax(s_bi[:, j], axis=0) == 1))
        feas.append(
            {
                "cell": cc,
                "best_dd_success": round(dd_any, 4),
                "bi_successes_all_planners": int(bi_any),
                "seeds_with_bicycle_witness": witness_seeds,
                "O4_candidate_feasibility_problem": bool(
                    dd_any >= rule["feasibility_dd_min"] and bi_any == 0
                ),
            }
        )
    reversal = any(o["reversal"] for o in ordering)
    counts = {}
    for r in scen:
        key = f"{r['planner']}|{r['plant']}|{r['scenario']}"
        d = counts.setdefault(key, {"n": 0, "success": 0, "collision": 0, "timeout": 0, "other": 0})
        d["n"] += 1
        d["success"] += int(r["success"])
        d["collision"] += int(r["collision"])
        d["timeout"] += int(r["timeout"])
        d["other"] += int(not (int(r["success"]) or int(r["collision"]) or int(r["timeout"])))
    summary = {
        "episodes": len(rows),
        "scenario_episodes": len(scen),
        "probe_episodes": len(emp),
        "seeds": seeds,
        "probe_seeds": sorted({int(r["seed"]) for r in emp}),
        "missing_pairs": len(missing),
        "mode_counts": mode_counts,
        "source_shas": sorted({r["source_sha"] for r in rows}),
        "creep_steps_total": sum(int(r["creep_steps"] or 0) for r in rows),
        "bootstrap": {"draws": DRAWS, "rng": "numpy default_rng(1001)", "unit": "seed block"},
        "ordering": ordering,
        "ordering_reversal": reversal,
        "time_to_goal_paired_successes": ttg,
        "empty_probe": probe,
        "per_planner_outcome": plan,
        "cell_feasibility": feas,
        "counts": counts,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with open(out / "paired_effects.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(effects[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(effects)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "episodes",
                    "missing_pairs",
                    "mode_counts",
                    "ordering_reversal",
                    "per_planner_outcome",
                ]
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("compact")
    a1.add_argument("--results", required=True)
    a1.add_argument("--output", required=True)
    a2 = sub.add_parser("analyze")
    a2.add_argument("--episodes", required=True)
    a2.add_argument("--contract", required=True)
    a2.add_argument("--output", required=True)
    a = ap.parse_args()
    if a.cmd == "compact":
        compact(a.results, a.output)
    else:
        analyze(a.episodes, a.contract, a.output)
