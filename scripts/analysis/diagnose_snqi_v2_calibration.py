"""Derive unfrozen SNQI-v2 diagnostics from a complete dev-seed campaign.

This command never remaps seeds, rewrites rows, pins anchors, or admits release evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.snqi.v2_calibration import (
    CalibrationGrid,
    _candidate_calibration_horizons,
    _compact_calibration_record,
    derive_calibration_anchors,
)
from robot_sf.benchmark.snqi.v2_reports import read_episode_files, score_episode
from robot_sf.benchmark.snqi.v2_spec import WEIGHTS, SnqiV2Spec


def _rho(left: list[float], right: list[float]) -> float | None:
    """Return a rank correlation, explicitly absent for constant data."""
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return float(spearmanr(left, right).statistic)


def _ranking(records: list[dict], spec: SnqiV2Spec, algorithms: dict[str, str]) -> list[dict]:
    """Score development rows under fixed anchors and retain the independent outcomes.

    Returns:
        Ordered arm summaries.
    """
    grouped = defaultdict(list)
    for row in records:
        grouped[row["planner_key"]].append(
            score_episode(row, spec, expected_algorithm=algorithms.get(row["planner_key"]))
        )
    result = []
    for arm, rows in grouped.items():
        result.append(
            {
                "arm": arm,
                "episodes": len(rows),
                "snqi_v2_mean": float(np.mean([r["metrics"]["snqi_v2"] for r in rows])),
                "success_rate": float(np.mean([r["metrics"]["success"] for r in rows])),
                "collision_rate": float(
                    np.mean([r["metrics"]["total_collision_count"] > 0 for r in rows])
                ),
            }
        )
    result.sort(key=lambda row: (-row["snqi_v2_mean"], row["arm"]))
    for rank, row in enumerate(result, 1):
        row["rank"] = rank
    return result


def _diagnostic_spec(document: dict) -> SnqiV2Spec:
    """Construct an unfrozen diagnostic specification, never an admitted release asset.

    Returns:
        Immutable calibration-only scoring inputs.
    """
    return SnqiV2Spec(
        weights=WEIGHTS,
        upper_anchors={term: entry["upper"] for term, entry in document["anchors"].items()},
        force_source=document["force_decision"]["source"],
        calibration_rho=document["force_decision"]["spearman_rho_F_N"],
        calibration_split_id=document["calibration"]["split_id"],
        calibration_seeds=tuple(document["calibration"]["seeds"]),
        paths={},
        hashes={},
        metric_schema_version=document["metric_schema_version"],
        diagnostic=True,
    )


def anchor_sensitivity(
    records: list[dict], spec: SnqiV2Spec, algorithms: dict[str, str] | None = None
) -> dict:
    """Report anchor dependence using calibration rows only, without admission gates.

    Returns:
        Full and exclusion p95s, relative changes, tail ownership and rankings.
    """
    if not records or any(row["seed"] not in (1001, 1002) for row in records):
        raise ValueError("Anchor sensitivity uses calibration seeds 1001/1002 only")
    sources = {
        "T": "time_to_goal_ideal_ratio",
        "F": spec.force_source,
        "J": "jerk_mean",
        "K": "curvature_mean",
    }

    def values(rows, term):
        if term == "N":
            return [row["metrics"]["near_misses"] / row["steps"] for row in rows]
        if term == "T":
            return [
                row["metrics"][sources[term]] if row["metrics"]["success"] else 0 for row in rows
            ]
        return [row["metrics"][sources[term]] for row in rows]

    def p95s(rows):
        return {
            term: float(np.percentile(values(rows, term), 95, method="linear"))
            for term in ("T", "N", "F", "J", "K")
        }

    full = p95s(records)
    baseline = _ranking(records, spec, algorithms or {})

    def variant(rows):
        p95 = p95s(rows)
        relative = {
            term: (p95[term] - full[term]) / full[term] if full[term] else None for term in full
        }
        return {
            "p95": p95,
            "relative_change": relative,
            "sets_anchor": {
                term: change is not None and abs(change) > 0.25 for term, change in relative.items()
            },
        }

    by_arm = {}
    for arm in sorted({row["planner_key"] for row in records}):
        entry = variant([row for row in records if row["planner_key"] != arm])
        upper = {**spec.upper_anchors, **{term: entry["p95"][term] for term in ("F", "J", "K")}}
        entry["ranking"] = _ranking(records, replace(spec, upper_anchors=upper), algorithms or {})
        by_arm[arm] = entry
    scenarios = sorted({row["scenario_id"] for row in records})
    seeds = sorted({row["seed"] for row in records})
    return {
        "status": "report_only_no_gate",
        "row_scope": "calibration seeds 1001/1002 only",
        "full_p95": full,
        "fixed_normative_anchors": {"T": 3, "N": 0.25},
        "full_ranking": baseline,
        "leave_one_arm_out": by_arm,
        "leave_one_scenario_out": {
            scenario: variant([row for row in records if row["scenario_id"] != scenario])
            for scenario in scenarios
        },
        "per_seed": {
            str(seed): variant([row for row in records if row["seed"] == seed]) for seed in seeds
        },
        "per_seed_pair": {
            f"{left}-{right}": variant([row for row in records if row["seed"] in (left, right)])
            for left, right in combinations(seeds, 2)
        },
        "per_arm_share_above_anchor": {
            arm: {
                term: float(
                    np.mean(
                        np.array(
                            values([row for row in records if row["planner_key"] == arm], term)
                        )
                        > spec.upper_anchors[term]
                    )
                )
                for term in full
            }
            for arm in by_arm
        },
    }


def force_saturation_diagnostics(records: list[dict], spec: SnqiV2Spec) -> dict:
    """Report clipping and force/near-miss redundancy without changing the switch rule.

    Returns:
        Three correlations and per-arm T/N/F/J/K saturation fractions.
    """
    terms = [score_episode(row, spec)["metrics"]["snqi_v2_terms"] for row in records]
    raw_f = [row["metrics"][spec.force_source] for row in records]
    fraction = [row["metrics"]["near_misses"] / row["steps"] for row in records]
    return {
        "force_source": spec.force_source,
        "rho_raw_F_clipped_N": _rho(raw_f, [row["N"] for row in terms]),
        "rho_normalised_F_clipped_N": _rho(
            [row["F"] for row in terms], [row["N"] for row in terms]
        ),
        "rho_raw_F_raw_N_fraction": _rho(raw_f, fraction),
        "per_arm_clipped_at_one_fraction": {
            arm: {
                term: float(
                    np.mean(
                        [
                            values[term] >= 1
                            for row, values in zip(records, terms, strict=True)
                            if row["planner_key"] == arm
                        ]
                    )
                )
                for term in ("T", "N", "F", "J", "K")
            }
            for arm in sorted({row["planner_key"] for row in records})
        },
    }


def diagnose(campaign_root: Path) -> dict:
    """Validate original dev rows and report diagnostic anchors, scores and alignment.

    Returns:
        Unfrozen anchors and a ranking over the same development rows.
    """
    manifest_path = campaign_root / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    source = manifest["git"]["commit"]
    paths = sorted(campaign_root.glob("runs/*/episodes.jsonl"))
    hashes = {str(path.relative_to(campaign_root)): sha256_file(path) for path in paths}
    digest = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    horizons = _candidate_calibration_horizons()
    arms = [path.parent.name.removesuffix("__differential_drive") for path in paths]
    algorithms = {entry["key"]: entry["algo"] for entry in manifest["planners"] if entry["enabled"]}
    records, seeds, budgets = [], set(), Counter()
    for arm, path in zip(arms, paths, strict=True):
        for row in read_episode_files([path]):
            seed = row.get("seed")
            if type(seed) is not int or not 1001 <= seed <= 1030:
                raise ValueError("diagnostics require original dev seeds 1001..1030")
            if row.get("git_hash") != source or row.get("scenario_id") not in horizons:
                raise ValueError("diagnostic row source/scenario differs from manifest/schedule")
            seeds.add(seed)
            budgets[row["horizon"]] += 1
            records.append(
                _compact_calibration_record(
                    row,
                    arm,
                    expected_horizon=horizons[row["scenario_id"]],
                    expected_algorithm=algorithms[arm],
                )
            )
    calibration = [row for row in records if row["seed"] in (1001, 1002)]
    held_apart = [row for row in records if row["seed"] == 1003]
    if seeds != {1001, 1002, 1003}:
        raise ValueError("Protocol diagnostic requires calibration 1001/1002 and held-apart 1003")
    document = derive_calibration_anchors(
        calibration,
        arms=arms,
        scenarios=sorted(horizons),
        run_id=manifest["campaign_id"],
        source_commit=source,
        episodes_sha256=digest,
        grid=CalibrationGrid(horizons, (1001, 1002), diagnostic=True),
        expected_algorithms=algorithms,
        allow_historical_unbound=True,
    )
    document["calibration"]["episode_files_sha256"] = hashes
    document["calibration"]["episode_file_scope"] = (
        "all original files; fitting filters seeds 1001/1002"
    )
    spec = _diagnostic_spec(document)
    ranking = _ranking(calibration, spec, algorithms)
    held_apart_ranking = _ranking(held_apart, spec, algorithms)
    alignment = {
        "success_minus_collision": _rho(
            [row["snqi_v2_mean"] for row in held_apart_ranking],
            [row["success_rate"] - row["collision_rate"] for row in held_apart_ranking],
        ),
        "success_rate": _rho(
            [row["snqi_v2_mean"] for row in held_apart_ranking],
            [row["success_rate"] for row in held_apart_ranking],
        ),
    }
    if hashes != {str(path.relative_to(campaign_root)): sha256_file(path) for path in paths}:
        raise ValueError("diagnostic source rows changed during analysis")
    return {
        "status": "diagnostic_only",
        "anchors": document,
        "ranking": ranking,
        "ranking_label": "fitted and scored on the same data (calibration seeds 1001/1002)",
        "held_apart_development": {
            "seeds": [1003],
            "episode_count": len(held_apart),
            "label": "held-apart development data; not sealed evaluation",
            "ranking": held_apart_ranking,
        },
        "anchor_sensitivity": anchor_sensitivity(calibration, spec, algorithms),
        "force_saturation": force_saturation_diagnostics(calibration, spec),
        "rank_alignment_spearman": alignment,
        "budget_counts": dict(budgets),
        "manifest_sha256": sha256_file(manifest_path),
        "claim_boundary": "Anchors fitted on development 1001/1002; 1003 held apart; nothing pinned; not sealed evaluation or release evidence.",
    }


def main() -> None:
    """Write a standalone diagnostic without mutating campaign inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.campaign_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "episodes": result["anchors"]["calibration"]["episode_count"],
                "anchors": result["anchors"]["anchors"],
                "rank_alignment_spearman": result["rank_alignment_spearman"],
            }
        )
    )


if __name__ == "__main__":
    main()
