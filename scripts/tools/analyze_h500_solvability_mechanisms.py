#!/usr/bin/env python3
"""Classify aggregate fixed-vs-h500 solvability mechanisms."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from robot_sf.benchmark.identity.hash_utils import load_json as _load_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_json", type=Path)
    parser.add_argument("--success-delta-min", type=float, default=0.25)
    parser.add_argument("--base-unfinished-min", type=float, default=0.50)
    parser.add_argument("--candidate-success-min", type=float, default=0.25)
    parser.add_argument("--near-miss-clean-delta-max", type=float, default=0.25)
    parser.add_argument("--collision-clean-delta-max", type=float, default=0.01)
    parser.add_argument("--fast-time-norm-max", type=float, default=0.75)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/analysis/h500_solvability_mechanisms"),
    )
    return parser.parse_args()


def _metric(row: dict[str, Any], metric: str, field: str) -> float:
    """Read one nested comparison metric field.

    Returns:
        Metric field as a float, or ``0.0`` when unavailable.
    """
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return 0.0
    payload = metrics.get(metric)
    if not isinstance(payload, dict):
        return 0.0
    value = payload.get(field, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _round(value: float, digits: int = 6) -> float:
    """Round numeric output values for stable artifacts.

    Returns:
        Rounded float.
    """
    return round(float(value), digits)


def _mechanism(row: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    """Classify the aggregate fixed-vs-h500 solvability mechanism.

    Returns:
        Mechanism label and interpretation text.
    """
    collision_delta = _metric(row, "collisions_mean", "delta")
    near_delta = _metric(row, "near_misses_mean", "delta")
    candidate_time = _metric(row, "time_to_goal_norm_mean", "candidate")
    success_delta = _metric(row, "success_mean", "delta")
    candidate_success = _metric(row, "success_mean", "candidate")

    if collision_delta > float(args.collision_clean_delta_max):
        return (
            "safety_regressed_completion",
            "h500 converts some timeouts, but the aggregate collision rate also increases.",
        )
    if near_delta > float(args.near_miss_clean_delta_max):
        return (
            "exposure_enabled_completion",
            "h500 converts timeouts while increasing near-miss exposure in the aggregate.",
        )
    if candidate_time <= float(args.fast_time_norm_max):
        return (
            "budget_limited_clean_completion",
            "h500 converts timeouts without aggregate safety cost and completes well inside h500.",
        )
    if candidate_success < 0.75 or success_delta < 0.75:
        return (
            "partial_timeout_relief",
            "h500 improves success but leaves a material unresolved timeout tail.",
        )
    return (
        "late_clean_completion",
        "h500 converts timeouts without aggregate safety cost, but completion uses much of the longer budget.",
    )


def _eligible(row: dict[str, Any], args: argparse.Namespace) -> bool:
    """Determine whether a comparison row is a timeout-to-success candidate.

    Returns:
        True when the row meets configured success and unfinished thresholds.
    """
    return (
        _metric(row, "success_mean", "delta") >= float(args.success_delta_min)
        and _metric(row, "unfinished_mean", "base") >= float(args.base_unfinished_min)
        and _metric(row, "success_mean", "candidate") >= float(args.candidate_success_min)
    )


def _case_record(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Build a normalized mechanism case record from one scenario delta row.

    Returns:
        JSON-serializable case payload with rounded evidence metrics.
    """
    mechanism, interpretation = _mechanism(row, args)
    return {
        "planner_key": str(row.get("planner_key", "unknown")),
        "scenario_id": str(row.get("scenario_id", "unknown")),
        "scenario_family": str(row.get("scenario_family", "unknown")),
        "mechanism": mechanism,
        "interpretation": interpretation,
        "evidence_level": "aggregate_comparison",
        "trace_evidence_required_for_waiting_claim": True,
        "base_success": _round(_metric(row, "success_mean", "base")),
        "candidate_success": _round(_metric(row, "success_mean", "candidate")),
        "success_delta": _round(_metric(row, "success_mean", "delta")),
        "base_unfinished": _round(_metric(row, "unfinished_mean", "base")),
        "candidate_unfinished": _round(_metric(row, "unfinished_mean", "candidate")),
        "unfinished_delta": _round(_metric(row, "unfinished_mean", "delta")),
        "collision_delta": _round(_metric(row, "collisions_mean", "delta")),
        "near_miss_delta": _round(_metric(row, "near_misses_mean", "delta")),
        "snqi_delta": _round(_metric(row, "snqi_mean", "delta")),
        "candidate_time_to_goal_norm": _round(_metric(row, "time_to_goal_norm_mean", "candidate")),
        "base_episodes": int(row.get("base_episodes", 0) or 0),
        "candidate_episodes": int(row.get("candidate_episodes", 0) or 0),
    }


def _sort_key(row: dict[str, Any]) -> tuple[float, float, float, str, str]:
    """Build the stable sort key for case prioritization.

    Returns:
        Tuple prioritizing larger success gain, lower safety deltas, and identity.
    """
    return (
        -float(row["success_delta"]),
        float(row["collision_delta"]),
        float(row["near_miss_delta"]),
        str(row["planner_key"]),
        str(row["scenario_id"]),
    )


def _family_rollup(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate mechanism cases by scenario family.

    Returns:
        Scenario-family summary rows.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["scenario_family"])].append(case)

    rows: list[dict[str, Any]] = []
    for family, family_cases in sorted(grouped.items()):
        mechanisms = Counter(str(case["mechanism"]) for case in family_cases)
        rows.append(
            {
                "scenario_family": family,
                "case_count": len(family_cases),
                "mean_success_delta": _round(
                    sum(float(case["success_delta"]) for case in family_cases) / len(family_cases)
                ),
                "mean_collision_delta": _round(
                    sum(float(case["collision_delta"]) for case in family_cases) / len(family_cases)
                ),
                "mean_near_miss_delta": _round(
                    sum(float(case["near_miss_delta"]) for case in family_cases) / len(family_cases)
                ),
                "mechanisms": dict(sorted(mechanisms.items())),
            }
        )
    return rows


def _representatives(cases: list[dict[str, Any]], per_mechanism: int = 3) -> list[dict[str, Any]]:
    """Select representative cases for each mechanism bucket.

    Returns:
        Sorted representative case records, capped per mechanism.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in sorted(cases, key=_sort_key):
        grouped[str(case["mechanism"])].append(case)
    reps: list[dict[str, Any]] = []
    for mechanism in sorted(grouped):
        reps.extend(grouped[mechanism][:per_mechanism])
    return reps


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a CSV artifact, preserving empty outputs as empty files."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_delta(value: Any) -> str:
    """Format a numeric delta for Markdown tables.

    Returns:
        Fixed-width four-decimal string.
    """
    return f"{float(value):.4f}"


def _write_markdown(
    path: Path,
    *,
    payload: dict[str, Any],
    cases: list[dict[str, Any]],
    reps: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
) -> None:
    """Write the h500 solvability mechanism Markdown report."""
    mechanism_counts = Counter(str(case["mechanism"]) for case in cases)
    lines = [
        "# H500 Solvability Mechanism Analysis",
        "",
        "This analysis uses the aggregate fixed-vs-h500 comparison. It can identify which "
        "planner-scenario cells convert fixed-horizon unfinished runs into h500 successes, "
        "but it cannot prove a wait-then-go causal mechanism without per-step traces or video.",
        "",
        "## Summary",
        "",
        f"- Base campaign: `{payload.get('base_campaign_id', 'unknown')}`",
        f"- Candidate campaign: `{payload.get('candidate_campaign_id', 'unknown')}`",
        f"- Timeout-to-success candidate cells: `{len(cases)}`",
        "- Evidence level: `aggregate_comparison`; waiting claims remain `trace_required`.",
        "",
        "## Mechanism Counts",
        "",
        "| Mechanism | Count |",
        "|---|---:|",
    ]
    for mechanism, count in sorted(mechanism_counts.items()):
        lines.append(f"| `{mechanism}` | {count} |")

    lines.extend(
        [
            "",
            "## Representative Cases",
            "",
            "| Mechanism | Planner | Scenario | Success Delta | Collision Delta | Near-Miss Delta | Candidate Time Norm |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in reps:
        lines.append(
            f"| `{row['mechanism']}` | `{row['planner_key']}` | `{row['scenario_id']}` | "
            f"{_format_delta(row['success_delta'])} | {_format_delta(row['collision_delta'])} | "
            f"{_format_delta(row['near_miss_delta'])} | "
            f"{_format_delta(row['candidate_time_to_goal_norm'])} |"
        )

    lines.extend(
        [
            "",
            "## Scenario-Family Rollup",
            "",
            "| Family | Cases | Mean Success Delta | Mean Collision Delta | Mean Near-Miss Delta | Mechanisms |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in family_rows:
        lines.append(
            f"| `{row['scenario_family']}` | {row['case_count']} | "
            f"{_format_delta(row['mean_success_delta'])} | "
            f"{_format_delta(row['mean_collision_delta'])} | "
            f"{_format_delta(row['mean_near_miss_delta'])} | "
            f"`{row['mechanisms']}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- `budget_limited_clean_completion` and `late_clean_completion` support the idea that "
            "some fixed-horizon failures were budget artifacts.",
            "- `exposure_enabled_completion` means the longer horizon opened a path to success but "
            "also increased near-miss exposure; this can be waiting, delayed progress, or simply "
            "more time spent in dense pedestrian flow.",
            "- `safety_regressed_completion` should be treated as a caveat, not benchmark-strengthening "
            "evidence.",
            "- The aggregate comparison does not contain enough state history to determine whether "
            "success came from waiting until dynamic obstacles passed. Use step diagnostics or video "
            "for that causal claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the aggregate h500 mechanism analysis."""
    args = parse_args()
    payload = _load_json(args.comparison_json)
    rows = payload.get("scenario_deltas")
    if not isinstance(rows, list):
        raise ValueError(f"comparison_json is missing scenario_deltas: {args.comparison_json}")

    cases = [
        _case_record(row, args) for row in rows if isinstance(row, dict) and _eligible(row, args)
    ]
    cases = sorted(cases, key=_sort_key)
    reps = _representatives(cases)
    family_rows = _family_rollup(cases)

    output = {
        "version": 1,
        "comparison_json": str(args.comparison_json),
        "base_campaign_id": payload.get("base_campaign_id"),
        "candidate_campaign_id": payload.get("candidate_campaign_id"),
        "thresholds": {
            "success_delta_min": _round(float(args.success_delta_min)),
            "base_unfinished_min": _round(float(args.base_unfinished_min)),
            "candidate_success_min": _round(float(args.candidate_success_min)),
            "near_miss_clean_delta_max": _round(float(args.near_miss_clean_delta_max)),
            "collision_clean_delta_max": _round(float(args.collision_clean_delta_max)),
            "fast_time_norm_max": _round(float(args.fast_time_norm_max)),
        },
        "interpretation_boundary": {
            "evidence_level": "aggregate_comparison",
            "waiting_until_dynamic_obstacles_passed": "trace_required",
            "near_miss_exposure": "longer horizons can create more exposure time",
        },
        "mechanism_counts": dict(Counter(str(case["mechanism"]) for case in cases)),
        "representative_cases": reps,
        "scenario_family_rollup": family_rows,
        "cases": cases,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "h500_solvability_mechanisms.json"
    csv_path = args.output_dir / "h500_solvability_cases.csv"
    family_csv_path = args.output_dir / "h500_solvability_family_rollup.csv"
    md_path = args.output_dir / "h500_solvability_mechanisms.md"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_csv(csv_path, cases)
    _write_csv(family_csv_path, family_rows)
    _write_markdown(md_path, payload=payload, cases=cases, reps=reps, family_rows=family_rows)

    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "family_csv": str(family_csv_path),
                "markdown": str(md_path),
                "case_count": len(cases),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
