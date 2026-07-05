"""Paired effect-size report builder for issue #3501 safety-wrapper ablations.

The report consumes completed ``wrapper_off``/``wrapper_on`` episode rows that
already satisfy the issue #3501 ablation row checker. It does not execute
episodes and does not promote benchmark or paper-facing claims by itself.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robot_sf.benchmark.safety_wrapper_ablation_manifest import (
    PAIRING_KEY_FIELDS,
    SAFETY_WRAPPER_MODE_FIELD,
    WRAPPER_OFF_ARM,
    WRAPPER_ON_ARM,
    check_factorial_ablation_rows,
)

REPORT_SCHEMA_VERSION = "robot_sf.issue_3501_safety_wrapper_factorial_report.v1"

SAFETY_WRAPPER_FACTORIAL_METRICS: tuple[str, ...] = (
    "exact_collision_probability",
    "near_miss_probability",
    "min_predicted_separation_m",
    "completion_probability",
    "progress_at_timeout",
    "false_positive_stop_rate",
    "stop_yield_latency_s",
    "wrapper_intervention_rate",
)

LOWER_IS_BETTER_METRICS = {
    "exact_collision_probability",
    "near_miss_probability",
    "false_positive_stop_rate",
    "stop_yield_latency_s",
}

HIGHER_IS_BETTER_METRICS = {
    "min_predicted_separation_m",
    "completion_probability",
    "progress_at_timeout",
}


def build_safety_wrapper_factorial_report(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a paired issue #3501 effect-size report from completed rows.

    Returns:
        Structured report with paired and per-planner effect rows, or fail-closed blockers.
    """

    row_check = check_factorial_ablation_rows(rows)
    if not row_check["complete"]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "issue": 3501,
            "status": "blocked",
            "reason": "ablation_rows_incomplete",
            "benchmark_evidence": False,
            "claim_boundary": _claim_boundary(),
            "row_check": row_check,
            "blocked_reasons": _row_check_blockers(row_check),
            "paired_effects": [],
            "per_planner_effects": [],
        }

    groups = _group_complete_pairs(rows)
    pair_rows: list[dict[str, Any]] = []
    missing_metrics: list[dict[str, Any]] = []
    for key, by_arm in sorted(groups.items(), key=lambda item: item[0]):
        off_row = by_arm[WRAPPER_OFF_ARM]
        on_row = by_arm[WRAPPER_ON_ARM]
        off_metrics = _extract_metrics(off_row)
        on_metrics = _extract_metrics(on_row)
        missing = [
            metric
            for metric in SAFETY_WRAPPER_FACTORIAL_METRICS
            if metric not in off_metrics or metric not in on_metrics
        ]
        if missing:
            missing_metrics.append(
                {
                    "pairing_key": dict(zip(PAIRING_KEY_FIELDS, key, strict=True)),
                    "metrics": missing,
                }
            )
            continue
        pair_rows.append(
            _paired_effect_row(
                key=key,
                off_metrics=off_metrics,
                on_metrics=on_metrics,
                off_row=off_row,
                on_row=on_row,
            )
        )

    if missing_metrics:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "issue": 3501,
            "status": "blocked",
            "reason": "metric_values_incomplete",
            "benchmark_evidence": False,
            "claim_boundary": _claim_boundary(),
            "row_check": row_check,
            "blocked_reasons": [
                "At least one complete off/on pair is missing primary outcome metrics."
            ],
            "missing_metrics": missing_metrics,
            "paired_effects": pair_rows,
            "per_planner_effects": [],
        }

    per_planner_effects = _per_planner_effects(pair_rows)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "issue": 3501,
        "status": "complete",
        "benchmark_evidence": True,
        "claim_boundary": _claim_boundary(),
        "row_check": row_check,
        "metric_directions": {
            "lower_is_better": sorted(LOWER_IS_BETTER_METRICS),
            "higher_is_better": sorted(HIGHER_IS_BETTER_METRICS),
            "diagnostic_only": ["wrapper_intervention_rate"],
        },
        "pair_count": len(pair_rows),
        "planner_count": len(per_planner_effects),
        "paired_effects": pair_rows,
        "per_planner_effects": per_planner_effects,
    }


def write_safety_wrapper_factorial_report(
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write summary JSON, per-planner CSV, and Markdown evidence packet.

    Returns:
        Paths to written summary, CSV, and README artifacts.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = out / "per_planner_effects.csv"
    _write_per_planner_csv(report.get("per_planner_effects", []), csv_path)

    readme_path = out / "README.md"
    readme_path.write_text(_format_readme(report), encoding="utf-8")

    return {
        "summary": summary_path,
        "per_planner_effects": csv_path,
        "readme": readme_path,
    }


def _claim_boundary() -> str:
    return (
        "Diagnostic paired safety-wrapper factorial report. Rows must already be complete "
        "native benchmark rows under the issue #3501 off/on contract. This report summarizes "
        "observed paired effects only; it does not execute campaigns, tune thresholds, or make "
        "paper/dissertation deployment-safety claims."
    )


def _row_check_blockers(row_check: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for field in (
        "missing_required_fields",
        "invalid_provenance_fields",
        "unexpected_wrapper_arms",
        "duplicate_pair_rows",
        "incomplete_pairs",
        "pair_provenance_mismatches",
    ):
        if row_check.get(field):
            blockers.append(f"{field}: {row_check[field]}")
    return blockers or ["Ablation row checker did not mark the paired rows complete."]


def _group_complete_pairs(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[Any, ...], dict[str, Mapping[str, Any]]]:
    groups: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = tuple(row[field] for field in PAIRING_KEY_FIELDS)
        groups[key][str(row["wrapper_arm"])] = row
    return dict(groups)


def _extract_metrics(row: Mapping[str, Any]) -> dict[str, float]:
    metric_values = row.get("metric_values")
    if not isinstance(metric_values, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for metric in SAFETY_WRAPPER_FACTORIAL_METRICS:
        value = (
            row.get(metric)
            if metric == "wrapper_intervention_rate" and metric not in metric_values
            else metric_values.get(metric)
        )
        if isinstance(value, bool):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            metrics[metric] = numeric
    return metrics


def _paired_effect_row(
    *,
    key: tuple[Any, ...],
    off_metrics: Mapping[str, float],
    on_metrics: Mapping[str, float],
    off_row: Mapping[str, Any],
    on_row: Mapping[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "planner": str(key[0]),
        "scenario_id": str(key[1]),
        "seed": int(key[2]),
        "wrapper_off_mode": off_row.get(SAFETY_WRAPPER_MODE_FIELD),
        "wrapper_on_mode": on_row.get(SAFETY_WRAPPER_MODE_FIELD),
        "software_commit": str(off_row["software_commit"]),
    }
    for metric in SAFETY_WRAPPER_FACTORIAL_METRICS:
        off_value = float(off_metrics[metric])
        on_value = float(on_metrics[metric])
        row[f"{metric}_wrapper_off"] = off_value
        row[f"{metric}_wrapper_on"] = on_value
        row[f"{metric}_delta_on_minus_off"] = on_value - off_value
    return row


def _per_planner_effects(pair_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_planner: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_planner[str(row["planner"])].append(row)

    results: list[dict[str, Any]] = []
    for planner, rows in sorted(by_planner.items()):
        result: dict[str, Any] = {"planner": planner, "pair_count": len(rows)}
        for metric in SAFETY_WRAPPER_FACTORIAL_METRICS:
            deltas = [float(row[f"{metric}_delta_on_minus_off"]) for row in rows]
            result[f"{metric}_mean_delta_on_minus_off"] = sum(deltas) / len(deltas)
        results.append(result)
    return results


def _write_per_planner_csv(rows: Any, path: Path) -> None:
    fieldnames = [
        "planner",
        "pair_count",
        *(f"{metric}_mean_delta_on_minus_off" for metric in SAFETY_WRAPPER_FACTORIAL_METRICS),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            for row in rows:
                if isinstance(row, Mapping):
                    writer.writerow({field: row.get(field, "") for field in fieldnames})


def _format_readme(report: Mapping[str, Any]) -> str:
    status = report.get("status", "unknown")
    lines = [
        "# Issue #3501 Safety Wrapper Factorial Report",
        "",
        str(report.get("claim_boundary", _claim_boundary())),
        "",
        f"- Status: `{status}`",
        f"- Benchmark evidence: `{bool(report.get('benchmark_evidence', False))}`",
        f"- Pair count: `{report.get('pair_count', 0)}`",
        f"- Planner count: `{report.get('planner_count', 0)}`",
        "",
    ]
    if status != "complete":
        lines.extend(["## Blockers", ""])
        for blocker in report.get("blocked_reasons", []):
            lines.append(f"- {blocker}")
        lines.append("")
    else:
        lines.extend(
            [
                "## Outputs",
                "",
                "- `summary.json`: complete paired effect report.",
                "- `per_planner_effects.csv`: planner-level mean on-minus-off deltas.",
                "",
            ]
        )
    return "\n".join(lines)


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "SAFETY_WRAPPER_FACTORIAL_METRICS",
    "build_safety_wrapper_factorial_report",
    "write_safety_wrapper_factorial_report",
]
