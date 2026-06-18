#!/usr/bin/env python3
"""Build an analysis-only comparison report from a campaign result store."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.tools.campaign_result_store import (
    ROW_STATUS_VALUES,
    read_parquet_frame,
    validate_result_store,
)

SCHEMA_VERSION = "campaign-comparison-report.v1"
BENCHMARK_VALID_ROW_STATUSES = frozenset({"native", "adapter"})
LIMITATION_ROW_STATUSES = frozenset(
    {"diagnostic_only", "fallback", "degraded", "unavailable", "failed"}
)
CORE_METRICS = (
    "success",
    "collision",
    "near_misses",
    "time_to_goal_norm",
    "path_efficiency",
)
SOCIAL_COMPLIANCE_METRICS = (
    "snqi",
    "comfort_exposure",
    "jerk",
)
METRIC_COLUMNS = CORE_METRICS + SOCIAL_COMPLIANCE_METRICS


def _finite_float(value: Any) -> float | None:
    """Return a finite float for loose result-store values."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _load_summary(result_store: Path) -> dict[str, Any]:
    """Read the canonical result-store summary payload."""
    return json.loads((result_store / "summary.json").read_text(encoding="utf-8"))


def _load_episode_frame(result_store: Path) -> pd.DataFrame:
    """Validate and load result-store episodes."""
    validation = validate_result_store(result_store)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    return read_parquet_frame(result_store / "episodes.parquet")


def _available_metrics(frame: pd.DataFrame) -> list[str]:
    """Return known metric columns that contain at least one finite value."""
    metrics: list[str] = []
    for metric in METRIC_COLUMNS:
        if metric not in frame.columns:
            continue
        if any(_finite_float(value) is not None for value in frame[metric].tolist()):
            metrics.append(metric)
    return metrics


def _metric_summary(values: list[Any]) -> dict[str, Any]:
    """Summarize one numeric metric with denominator and uncertainty fields."""
    parsed = [_finite_float(value) for value in values]
    finite = [value for value in parsed if value is not None]
    denominator = len(finite)
    missing = len(parsed) - denominator
    if not finite:
        return {
            "denominator": 0,
            "missing": missing,
            "mean": None,
            "std": None,
            "sem": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    mean = sum(finite) / denominator
    if denominator > 1:
        variance = sum((value - mean) ** 2 for value in finite) / (denominator - 1)
        std = math.sqrt(variance)
        sem = std / math.sqrt(denominator)
        half_width = 1.96 * sem
    else:
        std = None
        sem = None
        half_width = None
    return {
        "denominator": denominator,
        "missing": missing,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": None if half_width is None else mean - half_width,
        "ci95_high": None if half_width is None else mean + half_width,
    }


def _status_counts(rows: pd.DataFrame) -> dict[str, int]:
    """Count row statuses in a stable order."""
    counts = Counter(str(value) for value in rows["row_status"].tolist())
    return {status: counts[status] for status in sorted(counts)}


def _summarize_groups(
    frame: pd.DataFrame,
    *,
    group_fields: tuple[str, ...],
    metrics: list[str],
) -> list[dict[str, Any]]:
    """Build grouped metric summaries."""
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(group_fields), dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        payload = {field: str(key_tuple[index]) for index, field in enumerate(group_fields)}
        payload.update(
            {
                "episode_count": len(group),
                "benchmark_valid_episode_count": int(
                    group["row_status"].isin(BENCHMARK_VALID_ROW_STATUSES).sum()
                ),
                "excluded_or_limited_episode_count": int(
                    (~group["row_status"].isin(BENCHMARK_VALID_ROW_STATUSES)).sum()
                ),
                "row_status_counts": _status_counts(group),
                "metrics": {
                    metric: _metric_summary(group[metric].tolist())
                    for metric in metrics
                    if metric in group.columns
                },
            }
        )
        rows.append(payload)
    return rows


def _row_status_section(frame: pd.DataFrame) -> dict[str, Any]:
    """Build denominator and caveat payload for row-status interpretation."""
    counts = _status_counts(frame)
    caveats = []
    for status, count in counts.items():
        if status in BENCHMARK_VALID_ROW_STATUSES:
            interpretation = "benchmark_valid_denominator"
        elif status in LIMITATION_ROW_STATUSES:
            interpretation = "excluded_or_limited"
        else:
            interpretation = "unknown_status_fail_closed"
        caveats.append(
            {
                "row_status": status,
                "count": count,
                "interpretation": interpretation,
            }
        )
    return {
        "counts": counts,
        "benchmark_valid_episode_count": sum(
            count for status, count in counts.items() if status in BENCHMARK_VALID_ROW_STATUSES
        ),
        "excluded_or_limited_episode_count": sum(
            count for status, count in counts.items() if status not in BENCHMARK_VALID_ROW_STATUSES
        ),
        "known_row_status_values": list(ROW_STATUS_VALUES),
        "caveats": caveats,
    }


def _visual_summary(planner_rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    """Build compact bar-summary payloads for Markdown rendering."""
    summaries: list[dict[str, Any]] = []
    for metric in metrics:
        values: list[tuple[str, float, int]] = []
        for row in planner_rows:
            summary = row.get("metrics", {}).get(metric)
            if not isinstance(summary, dict) or summary.get("mean") is None:
                continue
            values.append(
                (str(row["planner"]), float(summary["mean"]), int(summary["denominator"]))
            )
        if not values:
            continue
        max_abs = max(abs(value) for _, value, _ in values) or 1.0
        rows = []
        for planner, value, denominator in sorted(values):
            width = round((abs(value) / max_abs) * 20)
            rows.append(
                {
                    "planner": planner,
                    "mean": value,
                    "denominator": denominator,
                    "bar": "#" * width if width else ".",
                }
            )
        summaries.append(
            {
                "metric": metric,
                "metric_family": ("core" if metric in CORE_METRICS else "social_compliance"),
                "rows": rows,
            }
        )
    return summaries


def _statistical_hooks(
    planner_rows: list[dict[str, Any]],
    metrics: list[str],
    *,
    min_sample: int,
) -> list[dict[str, Any]]:
    """Build descriptive pairwise hooks without claiming formal significance."""
    hooks: list[dict[str, Any]] = []
    rows_by_planner = {str(row["planner"]): row for row in planner_rows}
    for left, right in combinations(sorted(rows_by_planner), 2):
        left_row = rows_by_planner[left]
        right_row = rows_by_planner[right]
        for metric in metrics:
            left_metric = left_row.get("metrics", {}).get(metric)
            right_metric = right_row.get("metrics", {}).get(metric)
            if not isinstance(left_metric, dict) or not isinstance(right_metric, dict):
                continue
            if left_metric.get("mean") is None or right_metric.get("mean") is None:
                continue
            left_n = int(left_metric.get("denominator", 0) or 0)
            right_n = int(right_metric.get("denominator", 0) or 0)
            min_n = min(left_n, right_n)
            hooks.append(
                {
                    "comparison": f"{right} - {left}",
                    "metric": metric,
                    "left_planner": left,
                    "right_planner": right,
                    "left_denominator": left_n,
                    "right_denominator": right_n,
                    "mean_delta": float(right_metric["mean"]) - float(left_metric["mean"]),
                    "method": "descriptive_mean_delta",
                    "sample_gate": "met" if min_n >= min_sample else "underpowered",
                    "interpretation": (
                        "descriptive_only_formal_test_not_run"
                        if min_n >= min_sample
                        else "descriptive_only_underpowered"
                    ),
                }
            )
    return hooks


def build_report(
    result_store: Path,
    *,
    input_label: str | None = None,
    min_sample: int = 10,
) -> dict[str, Any]:
    """Build a comparison report payload from a campaign result store."""
    frame = _load_episode_frame(result_store)
    summary = _load_summary(result_store)
    metrics = _available_metrics(frame)
    planner_rows = _summarize_groups(frame, group_fields=("planner",), metrics=metrics)
    scenario_family_rows = _summarize_groups(
        frame,
        group_fields=("planner", "scenario_family"),
        metrics=metrics,
    )
    row_status = _row_status_section(frame)
    return {
        "schema_version": SCHEMA_VERSION,
        "report_status": "analysis_only",
        "claim_boundary": (
            "descriptive report only; row-status caveats and sample gates must be reviewed "
            "before benchmark or paper-facing claims"
        ),
        "input": {
            "result_store": (
                "transient_local_result_store" if input_label else result_store.as_posix()
            ),
            "durable_input_label": input_label,
            "study_id": summary.get("study_id"),
            "source_commit": summary.get("source_commit"),
            "episode_count": int(summary.get("episode_count", len(frame)) or 0),
        },
        "row_status": row_status,
        "available_metrics": metrics,
        "planner_summaries": planner_rows,
        "scenario_family_summaries": scenario_family_rows,
        "metric_visual_summaries": _visual_summary(planner_rows, metrics),
        "statistical_hooks": _statistical_hooks(planner_rows, metrics, min_sample=min_sample),
        "limitations": [
            "confidence intervals are normal-approximation descriptive intervals, not a "
            "significance claim",
            "fallback, degraded, unavailable, failed, and diagnostic-only rows are surfaced as "
            "limitations instead of successful benchmark evidence",
        ],
    }


def _fmt(value: Any) -> str:
    """Format optional numeric values for Markdown."""
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_markdown(payload: dict[str, Any]) -> str:
    """Render the report payload as Markdown."""
    lines = [
        "# Campaign Comparison Report",
        "",
        f"- Schema: `{payload['schema_version']}`",
        f"- Study: `{payload.get('input', {}).get('study_id', 'unknown')}`",
        f"- Report status: `{payload['report_status']}`",
        f"- Claim boundary: {payload['claim_boundary']}",
        "",
        "## Row Status Caveats",
        "",
        "| row_status | count | interpretation |",
        "|---|---:|---|",
    ]
    for caveat in payload.get("row_status", {}).get("caveats", []):
        lines.append(f"| {caveat['row_status']} | {caveat['count']} | {caveat['interpretation']} |")
    row_status = payload.get("row_status", {})
    lines.extend(
        [
            "",
            f"- Benchmark-valid denominator: `{row_status.get('benchmark_valid_episode_count', 0)}`",
            f"- Excluded or limited rows: `{row_status.get('excluded_or_limited_episode_count', 0)}`",
            "",
            "## Planner Metric Summaries",
            "",
            "| planner | row_status_counts | metric | denominator | mean | ci95_low | ci95_high |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload.get("planner_summaries", []):
        status_counts = ", ".join(
            f"{status}:{count}" for status, count in row.get("row_status_counts", {}).items()
        )
        for metric, summary in row.get("metrics", {}).items():
            lines.append(
                "| "
                f"{row['planner']} | {status_counts} | {metric} | "
                f"{summary['denominator']} | {_fmt(summary['mean'])} | "
                f"{_fmt(summary['ci95_low'])} | {_fmt(summary['ci95_high'])} |"
            )
    lines.extend(["", "## Metric Visual Summaries", ""])
    for summary in payload.get("metric_visual_summaries", []):
        lines.append(f"### {summary['metric']} ({summary['metric_family']})")
        lines.append("")
        lines.append("| planner | denominator | mean | visual |")
        lines.append("|---|---:|---:|---|")
        for row in summary.get("rows", []):
            lines.append(
                f"| {row['planner']} | {row['denominator']} | {_fmt(row['mean'])} | "
                f"`{row['bar']}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Statistical Hooks",
            "",
            "| comparison | metric | mean_delta | denominators | sample_gate | interpretation |",
            "|---|---|---:|---|---|---|",
        ]
    )
    hooks = payload.get("statistical_hooks", [])
    if hooks:
        for hook in hooks:
            lines.append(
                "| "
                f"{hook['comparison']} | {hook['metric']} | {_fmt(hook['mean_delta'])} | "
                f"{hook['left_denominator']}/{hook['right_denominator']} | "
                f"{hook['sample_gate']} | {hook['interpretation']} |"
            )
    else:
        lines.append("| NA | NA | NA | NA | underpowered | no comparable metric pairs |")
    lines.extend(["", "## Limitations", ""])
    for limitation in payload.get("limitations", []):
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def _write_outputs(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    """Write JSON and Markdown report artifacts."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(build_markdown(payload), encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-store", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--input-label",
        default=None,
        help=(
            "Optional durable source label recorded in the report when the result-store path is "
            "temporary local output."
        ),
    )
    parser.add_argument(
        "--min-sample",
        type=int,
        default=10,
        help="Minimum per-planner denominator before pairwise hooks pass the sample gate.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    try:
        payload = build_report(
            args.result_store,
            input_label=args.input_label,
            min_sample=args.min_sample,
        )
    except ValueError as exc:
        print(str(exc))
        return 1
    _write_outputs(payload, output_json=args.output_json, output_md=args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
