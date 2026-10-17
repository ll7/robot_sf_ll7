"""Summarize matched dev diagnostics without promoting incomplete rankings."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/context/robsweep/evidence"
CONFIG = ROOT / "docs/context/robsweep/inputs"


def row_exclusion(row: dict) -> str | None:
    """Keep execution failures and invalid resets out of outcome denominators."""
    if "outcome" not in row:
        return "execution_failed"
    if row.get("spawn_validity", {}).get("invalid_run"):
        return "invalid_spawn"
    if row.get("integrity", {}).get("contradictions"):
        return "integrity_contradiction"
    if row.get("integrity", {}).get("effective_view", {}).get("degraded"):
        return "degraded_view"

    def fallback(value: object) -> bool:
        if isinstance(value, dict):
            return any(
                (
                    key in {"fallback", "degraded", "fallback_triggered", "fallback_or_degraded"}
                    and item is True
                )
                or (
                    key in {"status", "execution_mode", "readiness_status"}
                    and item in ("fallback", "degraded")
                )
                or fallback(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(fallback(item) for item in value)
        return False

    if fallback(row.get("algorithm_metadata", {})):
        return "fallback_or_degraded"
    return None


def tau(a: list[float], b: list[float]) -> float | None:
    """Return Kendall tau-b with ties, or null for an undefined ordering."""
    value = float(kendalltau(a, b, variant="b").statistic)
    return value if np.isfinite(value) else None


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a portable table retaining missing values as empty fields."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_rows(study: dict, jobs: dict) -> tuple[dict, list, set]:
    """Read completed identities while checking seeds, duplicates and exclusions."""
    names = list(study["families"].values())
    by_cell = defaultdict(dict)
    exclusions = []
    sources = set()
    for job in jobs["jobs"]:
        stem = f"{job['treatment']}_{job['planner']}"
        path = EVIDENCE / "episodes" / f"{stem}.jsonl.gz"
        if not path.exists():
            path = path.with_suffix("")
        if not path.exists():
            continue
        opener = gzip.open if path.suffix == ".gz" else Path.open
        with opener(path, "rt") as handle:
            for line in handle:
                if not line.endswith("\n"):
                    continue
                row = json.loads(line)
                scenario = row.get("scenario_params", {}).get("name", row.get("scenario_id"))
                if scenario not in names or row["seed"] not in study["seeds"]:
                    raise ValueError("unexpected scenario or seed in acquired episode")
                key = job["treatment"], job["planner"], scenario
                if row["seed"] in by_cell[key]:
                    raise ValueError("duplicate matched identity")
                reason = row_exclusion(row)
                by_cell[key][row["seed"]] = row
                sources.add(row.get("git_hash"))
                if reason:
                    exclusions.append(
                        {
                            "treatment": key[0],
                            "planner": key[1],
                            "scenario": key[2],
                            "seed": row["seed"],
                            "reason": reason,
                        }
                    )
    return by_cell, exclusions, sources


def cell_table(study: dict, jobs: dict, by_cell: dict) -> list[dict]:
    """Count every requested cell, with missing and excluded rows visible."""
    rows = []
    for job in jobs["jobs"]:
        for family, scenario in study["families"].items():
            group = by_cell[job["treatment"], job["planner"], scenario]
            eligible = [r for r in group.values() if row_exclusion(r) is None]
            rows.append(
                {
                    "treatment": job["treatment"],
                    "planner": job["planner"],
                    "family": family,
                    "scenario": scenario,
                    "requested": jobs["seed_count"],
                    "recorded": len(group),
                    "eligible": len(eligible),
                    "missing": jobs["seed_count"] - len(group),
                    "excluded": len(group) - len(eligible),
                    "successes": sum(bool(r["outcome"]["route_complete"]) for r in eligible),
                    "collision_episodes": sum(
                        bool(r["outcome"]["collision_event"]) for r in eligible
                    ),
                    "pedestrian_collision_episodes": sum(
                        r.get("metrics", {}).get("ped_collision_count", 0) > 0 for r in eligible
                    ),
                    "obstacle_collision_episodes": sum(
                        r.get("metrics", {}).get("obstacle_collision_count", 0) > 0
                        for r in eligible
                    ),
                    "success_rate": sum(bool(r["outcome"]["route_complete"]) for r in eligible)
                    / len(eligible)
                    if eligible
                    else None,
                    "collision_rate": sum(bool(r["outcome"]["collision_event"]) for r in eligible)
                    / len(eligible)
                    if eligible
                    else None,
                }
            )
    return rows


def matched_matrix(study: dict, planners: list[str], arm: str, by_cell: dict) -> np.ndarray:
    """Build seed blocks only when all ten selected families have eligible rows."""
    names = list(study["families"].values())
    matrix = np.full((len(planners), len(study["seeds"]), 2), np.nan)
    for i, planner in enumerate(planners):
        for j, seed in enumerate(study["seeds"]):
            matched = [by_cell[arm, planner, scenario].get(seed) for scenario in names]
            if all(r is not None and row_exclusion(r) is None for r in matched):
                matrix[i, j] = [
                    sum(r["outcome"][field] for r in matched)
                    for field in ("route_complete", "collision_event")
                ]
    return matrix


def bootstrap_tau(matrices: list[np.ndarray], seed_count: int, rng: np.random.Generator) -> list:
    """Resample paired seeds and explicitly count defined tau replicates."""
    intervals: list[list[float]] = [[], []]
    for _ in range(1000):
        indices = rng.integers(0, seed_count, seed_count)
        a, b = (m[:, indices].mean(axis=1) for m in matrices)
        for metric in range(2):
            correlation = tau(a[:, metric].tolist(), b[:, metric].tolist())
            if correlation is not None:
                intervals[metric].append(correlation)
    return [
        {
            "ci95": np.quantile(values, [0.025, 0.975]).tolist() if values else None,
            "defined": len(values),
            "requested": 1000,
        }
        for values in intervals
    ]


def rank_table(study: dict, jobs: dict, by_cell: dict) -> list[dict]:
    """Compute correlations only for complete paired planner rosters."""
    rankings = []
    rng = np.random.default_rng(1061)
    for treatment in dict.fromkeys(job["treatment"] for job in jobs["jobs"]):
        planners = [job["planner"] for job in jobs["jobs"] if job["treatment"] == treatment]
        matrices = [matched_matrix(study, planners, arm, by_cell) for arm in ("legacy", treatment)]
        complete = all(np.isfinite(m).all() for m in matrices)
        record = {"treatment": treatment, "planners": planners, "complete_matched_roster": complete}
        if complete:
            # Aggregate integer episode counts before normalization. Averaging
            # per-seed fractions first can turn exact ties into false ranks.
            control, changed = (m.mean(axis=1) / len(study["families"]) for m in matrices)
            record.update(
                success_tau_b=tau(control[:, 0].tolist(), changed[:, 0].tolist()),
                collision_tau_b=tau(control[:, 1].tolist(), changed[:, 1].tolist()),
            )
            success_bootstrap, collision_bootstrap = bootstrap_tau(
                matrices, len(study["seeds"]), rng
            )
            record["success_tau_ci95"] = success_bootstrap["ci95"]
            record["collision_tau_ci95"] = collision_bootstrap["ci95"]
            record["success_tau_bootstrap"] = success_bootstrap
            record["collision_tau_bootstrap"] = collision_bootstrap
            record["planner_outcomes"] = [
                {
                    "planner": p,
                    "baseline_success": float(control[i, 0]),
                    "success": float(changed[i, 0]),
                    "baseline_collision": float(control[i, 1]),
                    "collision": float(changed[i, 1]),
                }
                for i, p in enumerate(planners)
            ]
            record["family_outcomes"] = []
            for family, scenario in study["families"].items():
                values = []
                for arm in ("legacy", treatment):
                    matched = [
                        by_cell[arm, p, scenario][seed] for p in planners for seed in study["seeds"]
                    ]
                    values.append(
                        [
                            sum(r["outcome"][field] for r in matched) / len(matched)
                            for field in ("route_complete", "collision_event")
                        ]
                    )
                record["family_outcomes"].append(
                    {
                        "family": family,
                        "baseline_success": values[0][0],
                        "success": values[1][0],
                        "baseline_collision": values[0][1],
                        "collision": values[1][1],
                        "planner_success_tau_b": tau(
                            [
                                sum(
                                    by_cell["legacy", p, scenario][seed]["outcome"][
                                        "route_complete"
                                    ]
                                    for seed in study["seeds"]
                                )
                                for p in planners
                            ],
                            [
                                sum(
                                    by_cell[treatment, p, scenario][seed]["outcome"][
                                        "route_complete"
                                    ]
                                    for seed in study["seeds"]
                                )
                                for p in planners
                            ],
                        ),
                        "planner_collision_tau_b": tau(
                            [
                                sum(
                                    by_cell["legacy", p, scenario][seed]["outcome"][
                                        "collision_event"
                                    ]
                                    for seed in study["seeds"]
                                )
                                for p in planners
                            ],
                            [
                                sum(
                                    by_cell[treatment, p, scenario][seed]["outcome"][
                                        "collision_event"
                                    ]
                                    for seed in study["seeds"]
                                )
                                for p in planners
                            ],
                        ),
                    }
                )
            record["family_success_tau_b"] = tau(
                [r["baseline_success"] for r in record["family_outcomes"]],
                [r["success"] for r in record["family_outcomes"]],
            )
        else:
            record.update(
                success_tau_b=None,
                collision_tau_b=None,
                reason="Incomplete matched roster: ranking non-identifiable",
            )
        rankings.append(record)
    return rankings


def analyze() -> None:
    """Reconcile every declared cell and produce paired ranking diagnostics."""
    study = json.loads((CONFIG / "study.json").read_text())
    jobs = json.loads((EVIDENCE / "jobs.json").read_text())
    by_cell, exclusions, sources = load_rows(study, jobs)
    rows = cell_table(study, jobs, by_cell)
    write_csv(EVIDENCE / "cells.csv", rows)
    summary = {
        "evidence_tier": "diagnostic-only",
        "requested": sum(job["requested"] for job in jobs["jobs"]),
        "recorded": sum(r["recorded"] for r in rows),
        "eligible": sum(r["eligible"] for r in rows),
        "missing": sum(r["missing"] for r in rows),
        "excluded": exclusions,
        "record_git_hashes": sorted(s for s in sources if s),
        "rankings": rank_table(study, jobs, by_cell),
        "ranking_definition": "Success and collision incidence ranked separately; Kendall tau-b retains ties; paired resampling of 30 seeds shared across all cells and planners.",
    }
    (EVIDENCE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    family_ranks = [
        {"treatment": r["treatment"], **f}
        for r in summary["rankings"]
        for f in r.get("family_outcomes", [])
    ]
    if family_ranks:
        write_csv(EVIDENCE / "family_ranks.csv", family_ranks)
    print(json.dumps({k: summary[k] for k in ("requested", "recorded", "eligible", "missing")}))


def manifest() -> None:
    """Hash all portable study inputs and outputs using repository-relative paths."""
    bundle = ROOT / "docs/context/robsweep"
    roots = [bundle] if EVIDENCE == bundle / "evidence" else [EVIDENCE, CONFIG]
    paths = sorted(
        {
            p
            for folder in roots
            for p in folder.rglob("*")
            if p.is_file()
            and p.name != "SHA256SUMS"
            and "__pycache__" not in p.parts
            and p.suffix != ".pyc"
        }
    )
    paths += sorted((ROOT / "scripts/validation").glob("*examiner*20261012.py"))
    paths += [ROOT / "tests/validation/test_examiner_robustness_analysis.py"]
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}" for p in paths]
    (EVIDENCE / "SHA256SUMS").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    parser.add_argument("--config-dir", type=Path, default=CONFIG)
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args()
    EVIDENCE = args.evidence_dir.resolve()
    CONFIG = args.config_dir.resolve()
    analyze()
    if args.manifest:
        manifest()
