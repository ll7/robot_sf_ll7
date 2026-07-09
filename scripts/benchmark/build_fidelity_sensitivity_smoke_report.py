#!/usr/bin/env python3
"""Build a diagnostic report from issue #3207 fidelity-sensitivity smoke rows."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.fidelity_sensitivity import (
    DIAGNOSTIC_SMOKE_CLAIM_BOUNDARY,
    build_rank_stability_summary,
    metric_drift,
)
from robot_sf.benchmark.identity.hash_utils import read_jsonl as _load_jsonl

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "issue_3207_fidelity_sensitivity_diagnostic_smoke.v1"
DEFAULT_METRICS = (
    "success",
    "collisions",
    "min_distance",
    "mean_distance",
    "robot_ped_within_5m_frac",
)


@dataclass(frozen=True)
class SmokeReportOptions:
    """Configuration for a diagnostic fidelity-sensitivity smoke report."""

    baseline_variant: str
    ranking_metric: str = "min_distance"
    higher_is_better: bool = True
    metrics: Sequence[str] = DEFAULT_METRICS
    min_tau: float = 0.8
    git_head: str = ""
    date: str | None = None


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_result_spec(spec: str) -> tuple[str, str, pathlib.Path]:
    parts = spec.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError("--result must use VARIANT:PLANNER:PATH")
    return parts[0], parts[1], pathlib.Path(parts[2])


def _input_ref(path: pathlib.Path) -> str:
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return f"worktree-local ignored artifact summarized in this report ({path.name})"
    return rel.as_posix()


def _metric_means(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> dict[str, Any]:
    sums: dict[str, float] = dict.fromkeys(metrics, 0.0)
    counts: dict[str, int] = dict.fromkeys(metrics, 0)
    for row in rows:
        row_metrics = row.get("metrics")
        if not isinstance(row_metrics, dict):
            row_metrics = {}
        for metric in metrics:
            value = _number(row_metrics.get(metric))
            if value is None:
                continue
            sums[metric] += value
            counts[metric] += 1
    means = {
        metric: (sums[metric] / counts[metric] if counts[metric] else None) for metric in metrics
    }
    return {"means": means, "counts": counts}


def _non_null_means(summary: Mapping[str, Any]) -> dict[str, float]:
    means = summary["means"]
    return {str(key): float(value) for key, value in means.items() if value is not None}


def build_report(
    result_sets: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    inputs: Mapping[str, Mapping[str, str]],
    options: SmokeReportOptions,
) -> dict[str, Any]:
    """Build a compact diagnostic smoke report."""
    baseline_variant = options.baseline_variant
    ranking_metric = options.ranking_metric
    if baseline_variant not in result_sets:
        raise ValueError(f"baseline variant {baseline_variant!r} is missing")

    planner_summaries: dict[str, dict[str, Any]] = {}
    for variant, by_planner in sorted(result_sets.items()):
        if not by_planner:
            raise ValueError(f"variant {variant!r} has no planner rows")
        planner_summaries[variant] = {}
        for planner, rows in sorted(by_planner.items()):
            summary = _metric_means(rows, options.metrics)
            if summary["means"].get(ranking_metric) is None:
                raise ValueError(
                    f"{variant}:{planner} is missing ranking metric {ranking_metric!r}"
                )
            planner_summaries[variant][planner] = {
                "episode_count": len(rows),
                "scenario_ids": sorted({str(row.get("scenario_id", "")) for row in rows}),
                "seeds": sorted({row.get("seed") for row in rows}),
                "metrics": summary,
            }

    baseline_planners = set(planner_summaries[baseline_variant])
    baseline_scores = {
        planner: planner_summaries[baseline_variant][planner]["metrics"]["means"][ranking_metric]
        for planner in baseline_planners
    }

    comparisons: dict[str, Any] = {}
    for variant, by_planner in planner_summaries.items():
        if variant == baseline_variant:
            continue
        variant_planners = set(by_planner)
        if variant_planners != baseline_planners:
            raise ValueError(
                f"variant {variant!r} planners {sorted(variant_planners)} do not match "
                f"baseline planners {sorted(baseline_planners)}"
            )
        variant_scores = {
            planner: by_planner[planner]["metrics"]["means"][ranking_metric]
            for planner in variant_planners
        }
        comparisons[variant] = {
            "rank_stability": build_rank_stability_summary(
                baseline_scores,
                variant_scores,
                higher_is_better=options.higher_is_better,
                min_tau=options.min_tau,
            ),
            "metric_drift_by_planner": {
                planner: metric_drift(
                    _non_null_means(planner_summaries[baseline_variant][planner]["metrics"]),
                    _non_null_means(by_planner[planner]["metrics"]),
                )
                for planner in sorted(variant_planners)
            },
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 3207,
        "status": "diagnostic_smoke",
        "date": options.date,
        "git_head": options.git_head,
        "claim_boundary": DIAGNOSTIC_SMOKE_CLAIM_BOUNDARY,
        "baseline_variant": baseline_variant,
        "ranking": {
            "metric": ranking_metric,
            "higher_is_better": options.higher_is_better,
            "min_tau": options.min_tau,
        },
        "inputs": inputs,
        "variant_count": len(planner_summaries),
        "planner_count": len(baseline_planners),
        "planner_summaries": planner_summaries,
        "comparisons_vs_baseline": comparisons,
    }


def format_markdown(report: Mapping[str, Any]) -> str:
    """Render a smoke report as concise Markdown."""
    issue = report.get("issue", 3207)
    date_suffix = f" {report['date']}" if report.get("date") else ""
    lines = [
        f"# Issue #{issue} Fidelity Sensitivity Diagnostic Smoke{date_suffix}",
        "",
        f"- Status: `{report['status']}`",
        f"- Git head: `{report['git_head']}`",
        f"- Baseline variant: `{report['baseline_variant']}`",
        f"- Ranking metric: `{report['ranking']['metric']}`",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Variant Summary",
        "",
        "| Variant | Planner | Episodes | Seeds | Mean ranking metric |",
        "|---|---|---:|---|---:|",
    ]
    ranking_metric = str(report["ranking"]["metric"])
    for variant, by_planner in report["planner_summaries"].items():
        for planner, summary in by_planner.items():
            means = summary["metrics"]["means"]
            rank_value = means.get(ranking_metric)
            rank_text = f"{rank_value:.6g}" if isinstance(rank_value, float) else "`null`"
            seeds = ", ".join(str(seed) for seed in summary["seeds"])
            lines.append(
                f"| `{variant}` | `{planner}` | {summary['episode_count']} | "
                f"{seeds} | {rank_text} |"
            )

    lines.extend(["", "## Rank Stability", "", "| Variant | Kendall tau | Rank flips | Stable? |"])
    lines.append("|---|---:|---:|---|")
    for variant, comparison in report["comparisons_vs_baseline"].items():
        stability = comparison["rank_stability"]
        lines.append(
            f"| `{variant}` | {stability['kendall_tau_vs_baseline']:.6g} | "
            f"{stability['rank_flip_count']} | "
            f"`{stability['stable_by_tau_threshold']}` |"
        )

    lines.extend(
        [
            "",
            "This smoke is diagnostic only. It records sensitivity wiring, small-sample metric drift,",
            "and rank-stability calculations; it is not benchmark-strength evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], output_dir: pathlib.Path) -> None:
    """Write the JSON payload and Markdown summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(format_markdown(report), encoding="utf-8")


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Input JSONL in VARIANT:PLANNER:PATH form. Repeat once per pair.",
    )
    parser.add_argument("--baseline-variant", required=True)
    parser.add_argument("--ranking-metric", default="min_distance")
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--min-tau", type=float, default=0.8)
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Metric to summarize. Defaults to the compact smoke metric set.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/context/evidence/issue_3207_fidelity_sensitivity_smoke_2026-06-20",
    )
    parser.add_argument("--date", default=dt.datetime.now(tz=dt.UTC).date().isoformat())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke-report builder CLI."""
    args = _build_arg_parser().parse_args(argv)
    result_sets: dict[str, dict[str, list[dict[str, Any]]]] = {}
    inputs: dict[str, dict[str, str]] = {}
    for spec in args.result:
        variant, planner, path = _parse_result_spec(spec)
        rows = _load_jsonl(path)
        result_sets.setdefault(variant, {})[planner] = rows
        inputs.setdefault(variant, {})[planner] = _input_ref(path)

    metrics = tuple(args.metrics) if args.metrics else DEFAULT_METRICS
    if args.ranking_metric not in metrics:
        metrics = (*metrics, args.ranking_metric)
    report = build_report(
        result_sets,
        inputs=inputs,
        options=SmokeReportOptions(
            baseline_variant=args.baseline_variant,
            ranking_metric=args.ranking_metric,
            higher_is_better=not args.lower_is_better,
            metrics=metrics,
            min_tau=args.min_tau,
            git_head=_git_head(),
            date=args.date,
        ),
    )
    write_report(report, REPO_ROOT / args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
