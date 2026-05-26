#!/usr/bin/env python3
"""Build issue #1462 h500 scenario and seed failure-mode tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_EVIDENCE_DIR = Path("docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23")
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_1462_s10_h500_failure_modes_2026-05-24")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--raw-campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-delta-min", type=float, default=0.25)
    parser.add_argument("--unsolved-success-max", type=float, default=0.20)
    parser.add_argument("--broad-success-min", type=float, default=0.65)
    parser.add_argument("--seed-range-min", type=float, default=0.40)
    parser.add_argument(
        "--check-output-dir",
        type=Path,
        help=(
            "After writing --output-dir, compare the generated evidence files with this "
            "committed evidence directory and exit non-zero on drift."
        ),
    )
    return parser.parse_args()


def _float(value: Any, default: float = 0.0) -> float:
    """Parse stable numeric CSV/JSON values."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return default if math.isnan(float(value)) else float(value)
    text = str(value).strip().strip("'")
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _round(value: float, digits: int = 6) -> float:
    """Round numeric outputs for reviewable artifacts."""
    return round(float(value), digits)


def _load_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV file into dictionaries."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _generated_filenames() -> tuple[str, ...]:
    """Return the derived evidence files that should stay fresh together."""
    return (
        "README.md",
        "candidate_vs_core_matrix.csv",
        "planner_scenario_seed_variability.csv",
        "scenario_difficulty_table.csv",
        "seed_difficulty_table.csv",
        "summary.json",
    )


def _output_drift(generated_dir: Path, expected_dir: Path) -> list[str]:
    """Return evidence files whose regenerated content differs from the expected bundle."""
    drift: list[str] = []
    for filename in _generated_filenames():
        generated_path = generated_dir / filename
        expected_path = expected_dir / filename
        if not generated_path.exists():
            drift.append(f"{filename}: generated file is missing")
        elif not expected_path.exists():
            drift.append(f"{filename}: expected file is missing")
        elif not _generated_file_matches(generated_path, expected_path, filename):
            drift.append(f"{filename}: content differs")
    return drift


def _generated_file_matches(generated_path: Path, expected_path: Path, filename: str) -> bool:
    """Compare generated evidence files while ignoring location-only metadata."""
    if filename != "summary.json":
        return generated_path.read_bytes() == expected_path.read_bytes()

    try:
        generated = json.loads(generated_path.read_text(encoding="utf-8"))
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    generated.pop("derived_output_dir", None)
    expected.pop("derived_output_dir", None)
    return generated == expected


def _mean(values: list[float]) -> float:
    """Return a mean, or zero for empty lists."""
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    """Return population standard deviation, or zero for empty/singleton lists."""
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def _planner_groups(campaign_table: list[dict[str, str]]) -> dict[str, str]:
    """Map planner keys to the issue #1462 core/candidate comparison groups."""
    groups: dict[str, str] = {}
    for row in campaign_table:
        planner_key = row["planner_key"]
        if planner_key.startswith("hybrid_rule_v3") or planner_key.startswith("scenario_adaptive"):
            groups[planner_key] = "candidate"
        else:
            groups[planner_key] = "core"
    return groups


def _episode_rollups(raw_campaign_dir: Path | None) -> dict[tuple[str, str], Counter[str]]:
    """Count raw termination/status outcomes by scenario and planner."""
    if raw_campaign_dir is None:
        return {}
    runs_dir = raw_campaign_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Raw campaign runs directory does not exist: {runs_dir}")

    rollups: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for jsonl_path in sorted(runs_dir.glob("*/episodes.jsonl")):
        planner_key = jsonl_path.parent.name.rsplit("__", maxsplit=1)[0]
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                scenario_id = str(payload["scenario_id"])
                key = (scenario_id, planner_key)
                status = str(payload.get("status", "unknown"))
                termination = str(payload.get("termination_reason", "unknown"))
                metrics = payload.get("metrics") or {}
                rollups[key]["episodes"] += 1
                rollups[key][f"status_{status}"] += 1
                rollups[key][f"termination_{termination}"] += 1
                rollups[key]["ped_collisions"] += int(_float(metrics.get("ped_collision_count")))
                rollups[key]["obstacle_collisions"] += int(
                    _float(metrics.get("obstacle_collision_count"))
                )
    return rollups


def _scenario_rows(  # noqa: C901
    scenario_breakdown: list[dict[str, str]],
    groups: dict[str, str],
    raw_rollups: dict[tuple[str, str], Counter[str]],
    scenario_seed_ranges: dict[str, float],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build scenario difficulty and candidate-vs-core rows."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in scenario_breakdown:
        grouped[row["scenario_id"]].append(row)

    difficulty_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for scenario_id, rows in sorted(grouped.items()):
        scenario_family = rows[0].get("scenario_family", "unknown")
        by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_group[groups.get(row["planner_key"], "unknown")].append(row)

        all_success = [_float(row["success_mean"]) for row in rows]
        core_success = [_float(row["success_mean"]) for row in by_group.get("core", [])]
        cand_success = [_float(row["success_mean"]) for row in by_group.get("candidate", [])]
        core_collision = [_float(row["collisions_mean"]) for row in by_group.get("core", [])]
        cand_collision = [_float(row["collisions_mean"]) for row in by_group.get("candidate", [])]
        core_near = [_float(row["near_misses_mean"]) for row in by_group.get("core", [])]
        cand_near = [_float(row["near_misses_mean"]) for row in by_group.get("candidate", [])]
        all_time_to_goal_norm = [_float(row["time_to_goal_norm_mean"]) for row in rows]
        all_comfort_exposure = [_float(row["comfort_exposure_mean"]) for row in rows]
        all_snqi = [_float(row["snqi_mean"]) for row in rows]
        core_time_to_goal_norm = [
            _float(row["time_to_goal_norm_mean"]) for row in by_group.get("core", [])
        ]
        cand_time_to_goal_norm = [
            _float(row["time_to_goal_norm_mean"]) for row in by_group.get("candidate", [])
        ]
        core_comfort_exposure = [
            _float(row["comfort_exposure_mean"]) for row in by_group.get("core", [])
        ]
        cand_comfort_exposure = [
            _float(row["comfort_exposure_mean"]) for row in by_group.get("candidate", [])
        ]

        raw_counts = Counter()
        for row in rows:
            raw_counts.update(raw_rollups.get((scenario_id, row["planner_key"]), Counter()))
        raw_episodes = raw_counts.get("episodes", 0)
        timeout_rate = (
            (
                raw_counts.get("termination_max_steps", 0)
                + raw_counts.get("termination_terminated", 0)
            )
            / raw_episodes
            if raw_episodes
            else 0.0
        )
        collision_rate = (
            raw_counts.get("status_collision", 0) / raw_episodes if raw_episodes else 0.0
        )

        core_mean = _mean(core_success)
        candidate_mean = _mean(cand_success)
        delta = candidate_mean - core_mean
        all_mean = _mean(all_success)
        seed_range = scenario_seed_ranges.get(scenario_id, 0.0)
        taxonomy = "mixed_partial"
        if candidate_mean <= args.unsolved_success_max or all_mean <= args.unsolved_success_max:
            taxonomy = "consistently_unsolved"
        elif delta >= args.candidate_delta_min and candidate_mean >= args.broad_success_min:
            taxonomy = "candidate_specific_improvement"
        elif core_mean >= args.broad_success_min and candidate_mean >= args.broad_success_min:
            taxonomy = "broadly_solvable"
        elif collision_rate >= 0.25:
            taxonomy = "collision_heavy"
        elif timeout_rate >= 0.35:
            taxonomy = "timeout_or_unfinished_heavy"
        elif seed_range >= args.seed_range_min:
            taxonomy = "seed_sensitive"

        difficulty_rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_family": scenario_family,
                "planner_count": len(rows),
                "episodes": sum(int(row["episodes"]) for row in rows),
                "success_mean_all": _round(all_mean),
                "success_mean_core": _round(core_mean),
                "success_mean_candidates": _round(candidate_mean),
                "candidate_success_delta": _round(delta),
                "collision_mean_all": _round(
                    _mean([_float(row["collisions_mean"]) for row in rows])
                ),
                "ped_collision_mean_all": _round(
                    _mean([_float(row["ped_collision_count_mean"]) for row in rows])
                ),
                "obstacle_collision_mean_all": _round(
                    _mean([_float(row["obstacle_collision_count_mean"]) for row in rows])
                ),
                "near_misses_mean_all": _round(
                    _mean([_float(row["near_misses_mean"]) for row in rows])
                ),
                "time_to_goal_norm_mean_all": _round(_mean(all_time_to_goal_norm)),
                "comfort_exposure_mean_all": _round(_mean(all_comfort_exposure)),
                "snqi_mean_all": _round(_mean(all_snqi)),
                "raw_timeout_or_unfinished_rate": _round(timeout_rate),
                "raw_collision_rate": _round(collision_rate),
                "taxonomy": taxonomy,
                "seed_success_range_all": _round(seed_range),
            }
        )
        matrix_rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_family": scenario_family,
                "core_success_mean": _round(core_mean),
                "candidate_success_mean": _round(candidate_mean),
                "candidate_success_delta": _round(delta),
                "core_collision_mean": _round(_mean(core_collision)),
                "candidate_collision_mean": _round(_mean(cand_collision)),
                "candidate_collision_delta": _round(_mean(cand_collision) - _mean(core_collision)),
                "core_near_misses_mean": _round(_mean(core_near)),
                "candidate_near_misses_mean": _round(_mean(cand_near)),
                "candidate_near_miss_delta": _round(_mean(cand_near) - _mean(core_near)),
                "core_time_to_goal_norm_mean": _round(_mean(core_time_to_goal_norm)),
                "candidate_time_to_goal_norm_mean": _round(_mean(cand_time_to_goal_norm)),
                "candidate_time_to_goal_norm_delta": _round(
                    _mean(cand_time_to_goal_norm) - _mean(core_time_to_goal_norm)
                ),
                "core_comfort_exposure_mean": _round(_mean(core_comfort_exposure)),
                "candidate_comfort_exposure_mean": _round(_mean(cand_comfort_exposure)),
                "candidate_comfort_exposure_delta": _round(
                    _mean(cand_comfort_exposure) - _mean(core_comfort_exposure)
                ),
                "taxonomy": taxonomy,
            }
        )
    difficulty_rows.sort(
        key=lambda row: (
            row["success_mean_all"],
            -row["raw_collision_rate"],
            -row["raw_timeout_or_unfinished_rate"],
            row["scenario_id"],
        )
    )
    matrix_rows.sort(key=lambda row: (-row["candidate_success_delta"], row["scenario_id"]))
    return difficulty_rows, matrix_rows


def _seed_tables(
    seed_rows: list[dict[str, str]],
    groups: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build seed difficulty and planner-scenario variability tables."""
    by_seed: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_cell: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in seed_rows:
        by_seed[row["seed"]].append(row)
        by_cell[(row["scenario_id"], row["planner_key"])].append(row)

    seed_table: list[dict[str, Any]] = []
    for seed, rows in sorted(by_seed.items(), key=lambda item: int(item[0])):
        successes = [_float(row["success_per_seed_mean"]) for row in rows]
        collisions = [_float(row["collisions_per_seed_mean"]) for row in rows]
        near_misses = [_float(row["near_misses_per_seed_mean"]) for row in rows]
        time_to_goal_norms = [
            _float(row.get("time_to_goal_norm_per_seed_mean", 0.0)) for row in rows
        ]
        snqis = [_float(row.get("snqi_per_seed_mean", 0.0)) for row in rows]
        seed_table.append(
            {
                "seed": int(seed),
                "cell_count": len(rows),
                "success_mean": _round(_mean(successes)),
                "collision_mean": _round(_mean(collisions)),
                "near_misses_mean": _round(_mean(near_misses)),
                "time_to_goal_norm_mean": _round(_mean(time_to_goal_norms)),
                "snqi_mean": _round(_mean(snqis)),
            }
        )
    seed_table.sort(key=lambda row: (row["success_mean"], -row["collision_mean"], row["seed"]))

    variability_rows: list[dict[str, Any]] = []
    for (scenario_id, planner_key), rows in sorted(by_cell.items()):
        successes = [_float(row["success_per_seed_mean"]) for row in rows]
        collisions = [_float(row["collisions_per_seed_mean"]) for row in rows]
        near_misses = [_float(row["near_misses_per_seed_mean"]) for row in rows]
        time_to_goal_norms = [
            _float(row.get("time_to_goal_norm_per_seed_mean", 0.0)) for row in rows
        ]
        snqis = [_float(row.get("snqi_per_seed_mean", 0.0)) for row in rows]
        variability_rows.append(
            {
                "scenario_id": scenario_id,
                "planner_key": planner_key,
                "planner_group": groups.get(planner_key, "unknown"),
                "seed_count": len(rows),
                "success_mean": _round(_mean(successes)),
                "success_std": _round(_std(successes)),
                "success_range": _round(max(successes) - min(successes) if successes else 0.0),
                "collision_mean": _round(_mean(collisions)),
                "collision_std": _round(_std(collisions)),
                "near_misses_mean": _round(_mean(near_misses)),
                "near_misses_std": _round(_std(near_misses)),
                "time_to_goal_norm_mean": _round(_mean(time_to_goal_norms)),
                "time_to_goal_norm_std": _round(_std(time_to_goal_norms)),
                "snqi_mean": _round(_mean(snqis)),
                "snqi_std": _round(_std(snqis)),
            }
        )
    variability_rows.sort(
        key=lambda row: (
            -row["success_range"],
            -row["success_std"],
            -row["collision_std"],
            row["scenario_id"],
            row["planner_key"],
        )
    )
    return seed_table, variability_rows


def _scenario_seed_ranges(seed_rows: list[dict[str, str]]) -> dict[str, float]:
    """Compute per-scenario success range across seeds after averaging planner cells."""
    by_scenario_seed: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in seed_rows:
        by_scenario_seed[(row["scenario_id"], row["seed"])].append(
            _float(row["success_per_seed_mean"])
        )
    by_scenario: dict[str, list[float]] = defaultdict(list)
    for (scenario_id, _seed), values in by_scenario_seed.items():
        by_scenario[scenario_id].append(_mean(values))
    return {
        scenario_id: (max(values) - min(values) if values else 0.0)
        for scenario_id, values in by_scenario.items()
    }


def _copy_manifest(evidence_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Load source metadata used in the derived evidence bundle."""
    run_meta = json.loads((evidence_dir / "run_meta.json").read_text(encoding="utf-8"))
    campaign_summary = json.loads(
        (evidence_dir / "reports" / "campaign_summary.json").read_text(encoding="utf-8")
    )
    campaign = campaign_summary.get("campaign", {})
    return {
        "source_evidence_dir": str(evidence_dir),
        "source_run_meta": {
            "campaign_id": run_meta.get("campaign_id"),
            "runtime_sec": run_meta.get("runtime_sec"),
            "episodes_per_second": run_meta.get("episodes_per_second"),
            "scenario_matrix_hash": run_meta.get("scenario_matrix_hash"),
            "seed_policy": run_meta.get("seed_policy"),
        },
        "source_campaign": {
            "campaign_id": campaign.get("campaign_id"),
            "git_hash": campaign.get("git_hash"),
            "scenario_matrix": campaign.get("scenario_matrix"),
            "scenario_matrix_hash": campaign.get("scenario_matrix_hash"),
            "config": "configs/benchmarks/issue_1454_s10_scenario_horizons_h500_candidates.yaml",
            "runtime_sec": campaign.get("runtime_sec"),
            "total_episodes": campaign.get("total_episodes"),
            "successful_runs": campaign.get("successful_runs"),
            "total_runs": campaign.get("total_runs"),
            "benchmark_success": campaign.get("benchmark_success"),
            "snqi_contract_status": campaign.get("snqi_contract_status"),
        },
        "derived_output_dir": str(output_dir),
    }


def _top(rows: list[dict[str, Any]], key: str, count: int = 5, reverse: bool = True) -> list[str]:
    """Format top scenario IDs for README prose."""
    selected = sorted(rows, key=lambda row: row[key], reverse=reverse)[:count]
    return [f"`{row['scenario_id']}` ({row[key]:.3f})" for row in selected]


def _write_readme(
    output_dir: Path,
    difficulty_rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
    seed_table: list[dict[str, Any]],
    variability_rows: list[dict[str, Any]],
) -> None:
    """Write a compact evidence-bundle README."""
    hard = _top(difficulty_rows, "success_mean_all", reverse=False)
    wins = _top(matrix_rows, "candidate_success_delta")
    weak = _top(matrix_rows, "candidate_success_mean", reverse=False)
    easy = _top(difficulty_rows, "success_mean_all", reverse=True)
    variable = [
        f"`{row['planner_key']}` on `{row['scenario_id']}` ({row['success_range']:.3f})"
        for row in variability_rows[:5]
    ]
    hard_seeds = [
        f"`{row['seed']}` (success {row['success_mean']:.3f}, collision {row['collision_mean']:.3f})"
        for row in seed_table[:3]
    ]

    readme = f"""# Issue #1462 S10 H500 Failure-Mode Evidence

Date: 2026-05-24

This bundle derives compact scenario, candidate-vs-core, and seed tables from the issue #1454
S10/h500 candidate evidence. It does not rerun the benchmark.

## Source Inputs

- Compact source: `docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/`
- Raw archive: https://github.com/ll7/robot_sf_ll7/releases/tag/artifact/issue1454-s10-h500-candidates-2026-05-23
- Source campaign: `issue1454-s10-h500-candidates`
- Episode count: 5,760

## Headline Tables

- `scenario_difficulty_table.csv`: all S10/h500 scenarios ranked by aggregate success, collision,
  timeout/unfinished rate, time-to-goal, comfort-exposure, diagnostic SNQI, candidate-vs-core delta,
  and taxonomy.
- `candidate_vs_core_matrix.csv`: per-scenario candidate group versus core runnable planner group,
  with time-to-goal and comfort-exposure deltas.
- `seed_difficulty_table.csv`: seed-level aggregate success/collision/near-miss difficulty with
  time-to-goal and diagnostic SNQI.
- `planner_scenario_seed_variability.csv`: planner-scenario cells sorted by seed success range, with
  time-to-goal and diagnostic SNQI.
- `summary.json`: compact machine-readable manifest and highlights.

## Highlights

- Hardest aggregate scenarios: {", ".join(hard)}.
- Easiest aggregate scenarios: {", ".join(easy)}.
- Largest candidate-vs-core success gains: {", ".join(wins)}.
- Candidate weak spots: {", ".join(weak)}.
- Highest seed-sensitive planner/scenario cells: {", ".join(variable)}.
- Hardest seeds by aggregate success: {", ".join(hard_seeds)}.

## Easiest Scenarios

The most broadly solvable S10/h500 scenarios (highest aggregate success mean across all 12
planners) are: {
        ", ".join(
            f"`{row['scenario_id']}` ({row['success_mean_all']:.3f})"
            for row in sorted(difficulty_rows, key=lambda r: r["success_mean_all"], reverse=True)[
                :10
            ]
        )
    }.

These scenarios are useful as high-probability anchor points and for isolating the hardest
seeds/planners in otherwise solvable geometry.

## Relation to H500 Solvability Mechanisms

This taxonomy extends the issue #1045 solvability-mechanism classification
(`docs/context/issue_1045_h500_solvability_mechanisms.md`) to the full S10 candidate surface.
The #1045 categories (clean budget relief, exposure-enabled completion, partial timeout relief,
and safety-regressed completion) were built on ORCA-only trace evidence on the reduced
continuous-h500 matrix. The scenario-level failure-mode taxonomy here is coarser but covers
all 12 planners and 48 scenarios. It cannot confirm or reject individual #1045 mechanism labels
for specific planner-scenario cells without per-step trace or video evidence.

## SNQI Field

SNQI (`snqi_mean`) fields are included in the CSV tables and summary for diagnostic inspection
purposes only. The source campaign `snqi_contract_status` is `fail` (rank alignment:
-0.207, outcome separation: 0.266, dominant component: success_reward), so SNQI is **not**
treated as a decisive quality signal.

## Interpretation Boundary

The taxonomy is aggregate evidence. It can distinguish broad difficulty, candidate-specific gains,
collision-heavy cells, and timeout/unfinished-heavy cells. It does not prove behavioral mechanisms
such as waiting, yielding, or hesitation without trace/video review.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    """Run the issue #1462 analysis."""
    args = parse_args()
    evidence_dir = args.evidence_dir
    output_dir = args.output_dir
    reports_dir = evidence_dir / "reports"
    campaign_table = _load_csv(reports_dir / "campaign_table.csv")
    scenario_breakdown = _load_csv(reports_dir / "scenario_breakdown.csv")
    seed_rows = _load_csv(reports_dir / "seed_variability_by_scenario.csv")
    groups = _planner_groups(campaign_table)
    raw_rollups = _episode_rollups(args.raw_campaign_dir)
    scenario_seed_ranges = _scenario_seed_ranges(seed_rows)

    difficulty_rows, matrix_rows = _scenario_rows(
        scenario_breakdown, groups, raw_rollups, scenario_seed_ranges, args
    )
    seed_table, variability_rows = _seed_tables(seed_rows, groups)
    manifest = _copy_manifest(evidence_dir, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "scenario_difficulty_table.csv", difficulty_rows)
    _write_csv(output_dir / "candidate_vs_core_matrix.csv", matrix_rows)
    _write_csv(output_dir / "seed_difficulty_table.csv", seed_table)
    _write_csv(output_dir / "planner_scenario_seed_variability.csv", variability_rows)
    easiest_scenarios = sorted(
        difficulty_rows, key=lambda row: row["success_mean_all"], reverse=True
    )[:10]
    summary = {
        **manifest,
        "hardest_scenarios": difficulty_rows[:5],
        "easiest_scenarios": easiest_scenarios,
        "largest_candidate_wins": matrix_rows[:5],
        "candidate_weak_spots": sorted(
            matrix_rows, key=lambda row: (row["candidate_success_mean"], row["scenario_id"])
        )[:5],
        "hardest_seeds": seed_table[:5],
        "most_seed_variable_cells": variability_rows[:10],
        "snqi_contract_status": manifest.get("source_campaign", {}).get("snqi_contract_status")
        or "unknown",
    }
    _write_json(output_dir / "summary.json", summary)
    _write_readme(output_dir, difficulty_rows, matrix_rows, seed_table, variability_rows)
    if args.check_output_dir is not None:
        drift = _output_drift(output_dir, args.check_output_dir)
        if drift:
            print(
                "Generated evidence differs from committed evidence:\n"
                + "\n".join(f"- {item}" for item in drift),
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
