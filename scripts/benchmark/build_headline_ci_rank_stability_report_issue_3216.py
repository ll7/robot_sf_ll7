#!/usr/bin/env python3
"""Build per-cell CI + rank-stability report for the 7x7 headline comparison (#3216).

This is the LOCAL report harness for issue #3216. It consumes already-measured
headline comparison rows (planner x scenario-family cells, multi-seed) and emits:

- per-cell point estimate + bootstrap/per-seed confidence interval on primary
  metrics, with fail-closed cell status (degraded/fallback/not_available cells
  never count as success), and
- a per-scenario planner-ranking rank-stability summary (Kendall-tau and
  rank-flip count across seed resamples) using the canonical owners.

It REUSES the canonical statistics owners and does NOT re-implement any
bootstrap / rank / Kendall logic:

- ``robot_sf.benchmark.seed_variance._stats_for_vals`` / ``_bootstrap_mean_ci``
  for per-cell bootstrap confidence intervals over per-seed means.
- ``robot_sf.benchmark.fidelity_rank_stability.rank_planners`` /
  ``kendall_tau`` / ``count_rank_flips`` for per-scenario rank stability across
  seed resamples.
- ``robot_sf.benchmark.canonical_table_export.load_rows_json`` for input rows.

It is analysis tooling. It makes NO benchmark or paper-grade claim by itself:
the paper-grade 7x7 headline run needs the increased seed budget (S20/S30 via
#1554) and is SLURM. On insufficient seed budget the harness classifies the
result ``blocked_until_run`` or ``diagnostic`` and never claims ``paper_grade``.

Coordination:

- Distinct from #3078 (Package A seed/planner-rank stability + held-out transfer
  campaign): this is the *grid-level per-cell* headline CI/rank-stability report.
- Distinct from #1554's per-planner-by-seed S20/S30 bundle: #3216 is the
  per-cell (planner x scenario) headline grid. The S20/S30 seed budget is
  supplied by #1554; this harness only consumes the resulting rows.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.canonical_table_export import load_rows_json
from robot_sf.benchmark.fidelity_rank_stability import (
    count_rank_flips,
    kendall_tau,
    rank_planners,
)
from robot_sf.benchmark.seed_variance import _stats_for_vals

SCHEMA_VERSION = "issue_3216_headline_ci_rank_stability.v1"
DEFAULT_ISSUE = 3216

#: Default primary metrics for per-cell CIs and ranking. ``snqi`` higher-is-better
#: is the rank metric; the rest are reported with CIs only.
DEFAULT_METRICS = ("success", "collisions", "near_misses", "snqi")
DEFAULT_RANK_METRIC = "snqi"

#: Planner execution modes that fail closed (never counted as successful cells).
_NON_SUCCESS_MODES = frozenset({"fallback", "degraded", "not_available"})
#: Row-status values that fail closed.
_NON_SUCCESS_STATUSES = frozenset(
    {
        "accepted_unavailable",
        "unexpected_failure",
        "fallback",
        "degraded",
        "blocked",
        "excluded",
        "revise",
        "completed_smoke_not_benchmark_evidence",
        "not_yet_populated",
        "not_available",
    }
)

#: Minimum per-cell seed count below which CIs/rank-stability are not paper-grade.
PAPER_GRADE_MIN_SEEDS = 20
#: Seed count at/above which a result is at least nominal (not just diagnostic).
NOMINAL_MIN_SEEDS = 10

#: Default bootstrap settings (reuses the canonical seed_variance bootstrap path).
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_SEED = 123
DEFAULT_RANK_RESAMPLES = 500

CLAIM_BOUNDARY = (
    "Per-cell confidence intervals and rank-stability are reported with explicit "
    "fail-closed cell status. This harness makes NO paper-grade or planner-ranking "
    "claim on its own: the paper-grade 7x7 headline run requires the increased "
    "seed budget (S20/S30 via #1554) and is SLURM. On insufficient seed budget the "
    "result is classified blocked_until_run or diagnostic."
)


@dataclass(frozen=True)
class ReportConfig:
    """Bootstrap + rank-stability settings for one report build."""

    metrics: tuple[str, ...] = DEFAULT_METRICS
    rank_metric: str = DEFAULT_RANK_METRIC
    higher_is_better: bool = True
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    confidence: float = DEFAULT_CONFIDENCE
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    resamples: int = DEFAULT_RANK_RESAMPLES
    rank_seed: int = DEFAULT_BOOTSTRAP_SEED


@dataclass(frozen=True)
class CellResult:
    """Per-cell (planner x scenario) CI + status payload."""

    scenario_id: str
    planner_key: str
    row_status: str
    counted: bool
    exclusion_reason: str | None
    seed_count: int
    metrics: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation.

        Returns:
            Mapping with cell identity, fail-closed status, and per-metric CIs.
        """
        return {
            "scenario_id": self.scenario_id,
            "planner_key": self.planner_key,
            "row_status": self.row_status,
            "counted": self.counted,
            "exclusion_reason": self.exclusion_reason,
            "seed_count": self.seed_count,
            "metrics": self.metrics,
        }


@dataclass
class ScenarioRankStability:
    """Per-scenario planner-ranking stability across seed resamples."""

    scenario_id: str
    rank_metric: str
    higher_is_better: bool
    point_ranking: list[str]
    rank_identifiable: bool
    reason: str | None
    resamples: int
    kendall_tau_mean: float | None
    kendall_tau_min: float | None
    rank_flip_rate: float | None
    top1_stable: bool | None
    sampled_orderings: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation.

        Returns:
            Mapping with point ranking and resample rank-stability statistics.
        """
        return {
            "scenario_id": self.scenario_id,
            "rank_metric": self.rank_metric,
            "higher_is_better": self.higher_is_better,
            "point_ranking": list(self.point_ranking),
            "rank_identifiable": self.rank_identifiable,
            "reason": self.reason,
            "resamples": self.resamples,
            "kendall_tau_mean": self.kendall_tau_mean,
            "kendall_tau_min": self.kendall_tau_min,
            "rank_flip_rate": self.rank_flip_rate,
            "top1_stable": self.top1_stable,
        }


def _git_head() -> str | None:
    """Return the current git HEAD commit, or None when unavailable.

    Returns:
        The 40-char commit SHA, or ``None`` outside a git checkout.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - env defensive
        return None


def _scenario_of(row: Mapping[str, Any]) -> str:
    """Return the scenario-family / scenario id for a row.

    Returns:
        The scenario grouping key, defaulting to ``"unknown"``.
    """
    for key in ("scenario_family", "scenario_id", "scenario"):
        value = row.get(key)
        if value:
            return str(value)
    return "unknown"


def _planner_of(row: Mapping[str, Any]) -> str:
    """Return the planner key for a row.

    Returns:
        The planner key, defaulting to ``"unknown"``.
    """
    for key in ("planner_key", "planner", "algo"):
        value = row.get(key)
        if value:
            return str(value)
    return "unknown"


def _cell_status(row: Mapping[str, Any]) -> tuple[str, bool, str | None]:
    """Classify a cell's fail-closed status.

    A cell counts only when neither its planner execution mode nor its row
    status is a non-success / unavailable value.

    Returns:
        Tuple of (row_status_label, counted, exclusion_reason).
    """
    row_status = str(row.get("row_status") or row.get("readiness_status") or "unspecified").strip()
    mode = str(row.get("execution_mode") or row.get("planner_mode") or "").strip().lower()
    if mode in _NON_SUCCESS_MODES:
        return row_status, False, f"execution_mode={mode}"
    if row_status.lower() in _NON_SUCCESS_STATUSES:
        return row_status, False, f"row_status={row_status}"
    return row_status, True, None


def _per_seed_metric_means(row: Mapping[str, Any], metric: str) -> list[float]:
    """Extract per-seed mean values for one metric from a headline row.

    Supports two shapes:
    - ``per_seed``: a list of ``{"seed", "metrics": {metric: value}}`` entries
      (the ``seed_variance.build_seed_variability_rows`` shape).
    - ``per_seed_metrics``: a mapping of ``{metric: [values...]}``.

    Returns:
        Finite per-seed mean values for the requested metric.
    """
    values: list[float] = []
    per_seed = row.get("per_seed")
    if isinstance(per_seed, Sequence) and not isinstance(per_seed, (str, bytes)):
        for entry in per_seed:
            if not isinstance(entry, Mapping):
                continue
            metrics = entry.get("metrics")
            if isinstance(metrics, Mapping) and metric in metrics:
                coerced = _coerce_float(metrics[metric])
                if coerced is not None:
                    values.append(coerced)
        return values
    per_seed_metrics = row.get("per_seed_metrics")
    if isinstance(per_seed_metrics, Mapping):
        raw = per_seed_metrics.get(metric)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            for item in raw:
                coerced = _coerce_float(item)
                if coerced is not None:
                    values.append(coerced)
    return values


def _coerce_float(value: Any) -> float | None:
    """Convert a value to a finite float when possible.

    Returns:
        Finite float, or ``None`` when conversion fails or value is non-finite.
    """
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def build_cell_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
    bootstrap_samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> list[CellResult]:
    """Build per-cell CI + status results for all rows.

    Reuses ``seed_variance._stats_for_vals`` for the bootstrap CI over per-seed
    means. Cells failing the fail-closed check are recorded with their status
    and never contribute success.

    Returns:
        One :class:`CellResult` per input row.
    """
    cells: list[CellResult] = []
    for row in rows:
        scenario = _scenario_of(row)
        planner = _planner_of(row)
        row_status, counted, reason = _cell_status(row)
        metric_payload: dict[str, dict[str, float]] = {}
        seed_count = 0
        for metric in metrics:
            per_seed = _per_seed_metric_means(row, metric)
            seed_count = max(seed_count, len(per_seed))
            if not per_seed:
                continue
            stats = _stats_for_vals(
                per_seed,
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
                bootstrap_seed=bootstrap_seed,
            )
            metric_payload[metric] = stats
        cells.append(
            CellResult(
                scenario_id=scenario,
                planner_key=planner,
                row_status=row_status,
                counted=counted,
                exclusion_reason=reason,
                seed_count=seed_count,
                metrics=metric_payload,
            )
        )
    return cells


def _counted_cells_by_scenario(
    cells: Sequence[CellResult],
) -> dict[str, list[CellResult]]:
    """Group counted (fail-closed-clean) cells by scenario.

    Returns:
        Mapping of scenario id to its counted cells.
    """
    grouped: dict[str, list[CellResult]] = defaultdict(list)
    for cell in cells:
        if cell.counted:
            grouped[cell.scenario_id].append(cell)
    return dict(grouped)


def _resample_per_seed_means(rows: Sequence[Mapping[str, Any]], cell: CellResult, metric: str):
    """Return the per-seed mean list for a cell from the original rows.

    Returns:
        Per-seed mean values for the cell's planner/scenario on ``metric``.
    """
    for row in rows:
        if _scenario_of(row) == cell.scenario_id and _planner_of(row) == cell.planner_key:
            return _per_seed_metric_means(row, metric)
    return []


def build_rank_stability(
    rows: Sequence[Mapping[str, Any]],
    cells: Sequence[CellResult],
    *,
    rank_metric: str,
    higher_is_better: bool,
    resamples: int,
    rng_seed: int,
) -> list[ScenarioRankStability]:
    """Build per-scenario rank-stability across seed resamples.

    For each scenario, the point ranking of counted planners is computed via the
    canonical ``rank_planners`` on per-cell metric means. Then per-seed values
    are bootstrap-resampled ``resamples`` times; each resample produces a new
    ranking whose ``kendall_tau`` / ``count_rank_flips`` against the point
    ranking are aggregated (all via the canonical fidelity_rank_stability
    owners).

    Returns:
        One :class:`ScenarioRankStability` per scenario with >= 2 counted cells.
    """
    rng = np.random.default_rng(rng_seed)
    grouped = _counted_cells_by_scenario(cells)
    out: list[ScenarioRankStability] = []
    for scenario in sorted(grouped):
        scenario_cells = grouped[scenario]
        if len(scenario_cells) < 2:
            out.append(
                ScenarioRankStability(
                    scenario_id=scenario,
                    rank_metric=rank_metric,
                    higher_is_better=higher_is_better,
                    point_ranking=[c.planner_key for c in scenario_cells],
                    rank_identifiable=False,
                    reason="fewer_than_two_counted_planners",
                    resamples=0,
                    kendall_tau_mean=None,
                    kendall_tau_min=None,
                    rank_flip_rate=None,
                    top1_stable=None,
                )
            )
            continue

        per_seed_by_planner: dict[str, list[float]] = {}
        for cell in scenario_cells:
            per_seed = _resample_per_seed_means(rows, cell, rank_metric)
            if per_seed:
                per_seed_by_planner[cell.planner_key] = per_seed
        if len(per_seed_by_planner) < 2:
            out.append(
                ScenarioRankStability(
                    scenario_id=scenario,
                    rank_metric=rank_metric,
                    higher_is_better=higher_is_better,
                    point_ranking=[c.planner_key for c in scenario_cells],
                    rank_identifiable=False,
                    reason="rank_metric_missing_per_seed_values",
                    resamples=0,
                    kendall_tau_mean=None,
                    kendall_tau_min=None,
                    rank_flip_rate=None,
                    top1_stable=None,
                )
            )
            continue

        point_table = {
            planner: {rank_metric: float(np.mean(values))}
            for planner, values in per_seed_by_planner.items()
        }
        point_ranking = rank_planners(point_table, rank_metric, higher_is_better=higher_is_better)

        taus: list[float] = []
        flip_counts: list[int] = []
        top1_changes = 0
        n_pairs = len(point_ranking) * (len(point_ranking) - 1) // 2
        for _ in range(resamples):
            sample_table = {}
            for planner, values in per_seed_by_planner.items():
                idx = rng.integers(0, len(values), size=len(values))
                sample_table[planner] = {rank_metric: float(np.mean([values[int(i)] for i in idx]))}
            sample_ranking = rank_planners(
                sample_table, rank_metric, higher_is_better=higher_is_better
            )
            taus.append(kendall_tau(point_ranking, sample_ranking))
            flip_counts.append(count_rank_flips(point_ranking, sample_ranking))
            if point_ranking[0] != sample_ranking[0]:
                top1_changes += 1

        flip_rate = (
            sum(flip_counts) / (len(flip_counts) * n_pairs) if flip_counts and n_pairs > 0 else 0.0
        )
        out.append(
            ScenarioRankStability(
                scenario_id=scenario,
                rank_metric=rank_metric,
                higher_is_better=higher_is_better,
                point_ranking=point_ranking,
                rank_identifiable=True,
                reason=None,
                resamples=resamples,
                kendall_tau_mean=float(np.mean(taus)) if taus else None,
                kendall_tau_min=float(np.min(taus)) if taus else None,
                rank_flip_rate=float(flip_rate),
                top1_stable=bool(top1_changes == 0),
            )
        )
    return out


def classify(
    cells: Sequence[CellResult],
    rank_stability: Sequence[ScenarioRankStability],
) -> tuple[str, str]:
    """Classify the overall result with fail-closed seed-budget gating.

    Returns:
        Tuple of (classification, rationale). Classification is one of
        ``paper_grade``, ``nominal``, ``diagnostic``, ``blocked_until_run``.
    """
    counted = [cell for cell in cells if cell.counted]
    if not counted:
        return (
            "blocked_until_run",
            "No fail-closed-clean cells; all cells were degraded/fallback/"
            "not_available or otherwise non-success. Run the S20/S30 headline "
            "campaign (#1554) to populate countable cells.",
        )
    min_seeds = min(cell.seed_count for cell in counted)
    max_seeds = max(cell.seed_count for cell in counted)
    identifiable_scenarios = [r for r in rank_stability if r.rank_identifiable]
    if min_seeds >= PAPER_GRADE_MIN_SEEDS and identifiable_scenarios:
        # Even at S20+, paper-grade promotion is gated on the run being the
        # actual SLURM headline campaign; this harness never self-certifies.
        return (
            "blocked_until_run",
            f"Per-cell seed budget reaches paper-grade threshold "
            f"(min_seeds={min_seeds} >= {PAPER_GRADE_MIN_SEEDS}), but paper-grade "
            "promotion requires the predeclared S20/S30 SLURM headline run (#1554) "
            "and claim-card review. This local harness emits the statistics only.",
        )
    if min_seeds < NOMINAL_MIN_SEEDS:
        return (
            "diagnostic",
            f"Per-cell seed budget is below the nominal threshold "
            f"(min_seeds={min_seeds} < {NOMINAL_MIN_SEEDS}; max_seeds={max_seeds}). "
            "CIs and rank-stability are diagnostic only; the S20/S30 run (#1554) "
            "is required before any planner-ranking or safety-delta claim.",
        )
    return (
        "diagnostic",
        f"Per-cell seed budget is nominal-range (min_seeds={min_seeds}) but below "
        f"the paper-grade threshold ({PAPER_GRADE_MIN_SEEDS}). Treat CIs/rank-"
        "stability as diagnostic; the S20/S30 SLURM run (#1554) is required for "
        "paper-grade planner-ranking claims.",
    )


def build_report(
    rows: Sequence[Mapping[str, Any]],
    config: ReportConfig | None = None,
    *,
    campaign: str | None = None,
    rows_path: str | None = None,
) -> dict[str, Any]:
    """Build the full report payload.

    Returns:
        JSON-safe report mapping with per-cell CIs, rank-stability, and the
        fail-closed classification.
    """
    config = config or ReportConfig()
    metrics = config.metrics
    rank_metric = config.rank_metric
    higher_is_better = config.higher_is_better
    cells = build_cell_results(
        rows,
        metrics=metrics,
        bootstrap_samples=config.bootstrap_samples,
        confidence=config.confidence,
        bootstrap_seed=config.bootstrap_seed,
    )
    rank_stability = build_rank_stability(
        rows,
        cells,
        rank_metric=rank_metric,
        higher_is_better=higher_is_better,
        resamples=config.resamples,
        rng_seed=config.rank_seed,
    )
    classification, rationale = classify(cells, rank_stability)
    counted = [cell for cell in cells if cell.counted]
    excluded = [cell for cell in cells if not cell.counted]
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": DEFAULT_ISSUE,
        "claim_boundary": CLAIM_BOUNDARY,
        "classification": classification,
        "classification_rationale": rationale,
        "git_head": _git_head(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "rows_path": rows_path,
            "campaign": campaign,
            "row_count": len(rows),
            "counted_cells": len(counted),
            "excluded_cells": len(excluded),
            "metrics": list(metrics),
            "rank_metric": rank_metric,
            "higher_is_better": higher_is_better,
        },
        "config": {
            "bootstrap_samples": config.bootstrap_samples,
            "confidence": config.confidence,
            "bootstrap_seed": config.bootstrap_seed,
            "rank_resamples": config.resamples,
            "rank_seed": config.rank_seed,
            "paper_grade_min_seeds": PAPER_GRADE_MIN_SEEDS,
            "nominal_min_seeds": NOMINAL_MIN_SEEDS,
        },
        "canonical_owners_reused": [
            "robot_sf.benchmark.seed_variance._stats_for_vals",
            "robot_sf.benchmark.seed_variance._bootstrap_mean_ci",
            "robot_sf.benchmark.fidelity_rank_stability.rank_planners",
            "robot_sf.benchmark.fidelity_rank_stability.kendall_tau",
            "robot_sf.benchmark.fidelity_rank_stability.count_rank_flips",
            "robot_sf.benchmark.canonical_table_export.load_rows_json",
        ],
        "cells": [cell.to_dict() for cell in cells],
        "rank_stability": [r.to_dict() for r in rank_stability],
        "excluded_cell_reasons": [
            {
                "scenario_id": cell.scenario_id,
                "planner_key": cell.planner_key,
                "row_status": cell.row_status,
                "exclusion_reason": cell.exclusion_reason,
            }
            for cell in excluded
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a markdown summary of the report.

    Returns:
        Markdown string with classification, per-cell CIs, and rank-stability.
    """
    inputs = report["inputs"]
    lines: list[str] = []
    lines.append("# Headline 7x7 CI + Rank-Stability Report (#3216)")
    lines.append("")
    lines.append(f"- **Classification**: `{report['classification']}`")
    lines.append(f"- **Rationale**: {report['classification_rationale']}")
    lines.append(f"- **Claim boundary**: {report['claim_boundary']}")
    lines.append(f"- **git HEAD**: `{report['git_head']}`")
    lines.append(
        f"- **Cells**: {inputs['counted_cells']} counted / "
        f"{inputs['excluded_cells']} excluded (of {inputs['row_count']} rows)"
    )
    lines.append(f"- **Rank metric**: `{inputs['rank_metric']}`")
    lines.append("")
    lines.append("## Canonical owners reused (not reinvented)")
    lines.append("")
    for owner in report["canonical_owners_reused"]:
        lines.append(f"- `{owner}`")
    lines.append("")
    lines.append("## Per-cell confidence intervals (counted cells)")
    lines.append("")
    lines.append("| scenario | planner | status | seeds | metric | mean | ci_low | ci_high |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for cell in report["cells"]:
        if not cell["counted"]:
            continue
        for metric, stats in sorted(cell["metrics"].items()):
            lines.append(
                f"| {cell['scenario_id']} | {cell['planner_key']} | {cell['row_status']} "
                f"| {cell['seed_count']} | {metric} | {stats['mean']:.4f} "
                f"| {_fmt(stats.get('ci_low'))} | {_fmt(stats.get('ci_high'))} |"
            )
    lines.append("")
    lines.append("## Per-scenario rank stability")
    lines.append("")
    lines.append(
        "| scenario | identifiable | resamples | tau_mean | tau_min | flip_rate | top1_stable |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for entry in report["rank_stability"]:
        lines.append(
            f"| {entry['scenario_id']} | {entry['rank_identifiable']} "
            f"| {entry['resamples']} | {_fmt(entry['kendall_tau_mean'])} "
            f"| {_fmt(entry['kendall_tau_min'])} | {_fmt(entry['rank_flip_rate'])} "
            f"| {entry['top1_stable']} |"
        )
    if report["excluded_cell_reasons"]:
        lines.append("")
        lines.append("## Excluded cells (fail-closed)")
        lines.append("")
        lines.append("| scenario | planner | row_status | reason |")
        lines.append("| --- | --- | --- | --- |")
        for entry in report["excluded_cell_reasons"]:
            lines.append(
                f"| {entry['scenario_id']} | {entry['planner_key']} "
                f"| {entry['row_status']} | {entry['exclusion_reason']} |"
            )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    """Format a float-or-None for markdown.

    Returns:
        Four-decimal string, or ``""`` for None/non-finite.
    """
    if value is None:
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{f:.4f}" if math.isfinite(f) else ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build per-cell CI + rank-stability report for the 7x7 headline "
            "planner comparison (#3216). Reuses canonical seed_variance, "
            "fidelity_rank_stability, and canonical_table_export owners. Emits "
            "blocked_until_run/diagnostic on insufficient seed budget; never "
            "self-certifies paper-grade."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--rows", type=str, help="Path to headline comparison rows JSON list.")
    src.add_argument(
        "--campaign",
        type=str,
        help="Campaign id/root; resolves to <root>/reports/headline_rows.json.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Primary metrics for per-cell CIs.",
    )
    parser.add_argument(
        "--rank-metric", type=str, default=DEFAULT_RANK_METRIC, help="Metric used for ranking."
    )
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Treat the rank metric as lower-is-better (default higher-is-better).",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--rank-resamples", type=int, default=DEFAULT_RANK_RESAMPLES)
    parser.add_argument("--rank-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/issue_3216_headline_ci/report",
        help="Directory for result.json and report.md.",
    )
    return parser.parse_args(argv)


def _resolve_rows_path(args: argparse.Namespace) -> str:
    """Resolve the rows path from --rows or --campaign.

    Returns:
        Filesystem path to the rows JSON list.
    """
    if args.rows:
        return args.rows
    return str(Path(args.campaign) / "reports" / "headline_rows.json")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    rows_path = _resolve_rows_path(args)
    rows = load_rows_json(rows_path)
    config = ReportConfig(
        metrics=tuple(args.metrics),
        rank_metric=args.rank_metric,
        higher_is_better=not args.lower_is_better,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        bootstrap_seed=args.bootstrap_seed,
        resamples=args.rank_resamples,
        rank_seed=args.rank_seed,
    )
    report = build_report(rows, config, campaign=args.campaign, rows_path=rows_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(
        f"classification={report['classification']} "
        f"counted={report['inputs']['counted_cells']} "
        f"excluded={report['inputs']['excluded_cells']} -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
