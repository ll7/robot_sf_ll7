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
from itertools import pairwise
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


@dataclass(frozen=True)
class AdjacentRankClaim:
    """Adjacent-rank CI overlap decision for one scenario ranking."""

    scenario_id: str
    rank_metric: str
    higher_is_better: bool
    higher_rank_planner: str
    lower_rank_planner: str
    higher_rank_mean: float
    lower_rank_mean: float
    higher_rank_ci_low: float
    higher_rank_ci_high: float
    lower_rank_ci_low: float
    lower_rank_ci_high: float
    decision: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe adjacent-rank claim decision."""

        return {
            "scenario_id": self.scenario_id,
            "rank_metric": self.rank_metric,
            "higher_is_better": self.higher_is_better,
            "higher_rank_planner": self.higher_rank_planner,
            "lower_rank_planner": self.lower_rank_planner,
            "higher_rank_mean": self.higher_rank_mean,
            "lower_rank_mean": self.lower_rank_mean,
            "higher_rank_ci_low": self.higher_rank_ci_low,
            "higher_rank_ci_high": self.higher_rank_ci_high,
            "lower_rank_ci_low": self.lower_rank_ci_low,
            "lower_rank_ci_high": self.lower_rank_ci_high,
            "decision": self.decision,
            "rationale": self.rationale,
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


def _metric_stats(cell: CellResult, metric: str) -> dict[str, float] | None:
    """Return complete metric statistics for CI decisions."""

    stats = cell.metrics.get(metric)
    if not stats:
        return None
    required = ("mean", "ci_low", "ci_high")
    if any(_coerce_float(stats.get(key)) is None for key in required):
        return None
    return stats


def _ci_overlap(a: dict[str, float], b: dict[str, float]) -> bool:
    """Return whether two confidence intervals overlap."""

    return max(float(a["ci_low"]), float(b["ci_low"])) <= min(
        float(a["ci_high"]),
        float(b["ci_high"]),
    )


def build_adjacent_rank_claims(
    cells: Sequence[CellResult],
    rank_stability: Sequence[ScenarioRankStability],
    *,
    rank_metric: str,
    higher_is_better: bool,
) -> list[AdjacentRankClaim]:
    """Build adjacent-rank downgrade decisions from per-cell CIs.

    Adjacent planners with overlapping confidence intervals are downgraded to
    ``not_statistically_distinguishable_budget`` so the packet cannot be read
    as support for a strict paper-facing ordering.
    """

    cell_by_key = {
        (cell.scenario_id, cell.planner_key): cell
        for cell in cells
        if cell.counted and _metric_stats(cell, rank_metric) is not None
    }
    claims: list[AdjacentRankClaim] = []
    for stability in rank_stability:
        ranking = stability.point_ranking
        if len(ranking) < 2:
            continue
        for higher_planner, lower_planner in pairwise(ranking):
            higher_cell = cell_by_key.get((stability.scenario_id, higher_planner))
            lower_cell = cell_by_key.get((stability.scenario_id, lower_planner))
            if higher_cell is None or lower_cell is None:
                continue
            higher_stats = _metric_stats(higher_cell, rank_metric)
            lower_stats = _metric_stats(lower_cell, rank_metric)
            if higher_stats is None or lower_stats is None:
                continue
            if _ci_overlap(higher_stats, lower_stats):
                decision = "not_statistically_distinguishable_budget"
                rationale = (
                    "Adjacent-rank confidence intervals overlap; downgrade any strict "
                    "planner-beats-planner claim at this seed budget."
                )
            else:
                decision = "ci_separable"
                rationale = "Adjacent-rank confidence intervals do not overlap."
            claims.append(
                AdjacentRankClaim(
                    scenario_id=stability.scenario_id,
                    rank_metric=rank_metric,
                    higher_is_better=higher_is_better,
                    higher_rank_planner=higher_planner,
                    lower_rank_planner=lower_planner,
                    higher_rank_mean=float(higher_stats["mean"]),
                    lower_rank_mean=float(lower_stats["mean"]),
                    higher_rank_ci_low=float(higher_stats["ci_low"]),
                    higher_rank_ci_high=float(higher_stats["ci_high"]),
                    lower_rank_ci_low=float(lower_stats["ci_low"]),
                    lower_rank_ci_high=float(lower_stats["ci_high"]),
                    decision=decision,
                    rationale=rationale,
                )
            )
    return claims


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
    adjacent_rank_claims = build_adjacent_rank_claims(
        cells,
        rank_stability,
        rank_metric=rank_metric,
        higher_is_better=higher_is_better,
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
        "adjacent_rank_claims": [claim.to_dict() for claim in adjacent_rank_claims],
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
    lines.append("")
    lines.append("## Adjacent-rank claim downgrades")
    lines.append("")
    lines.append(
        "| scenario | higher-rank planner | lower-rank planner | metric | decision | rationale |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for claim in report["adjacent_rank_claims"]:
        lines.append(
            f"| {claim['scenario_id']} | {claim['higher_rank_planner']} "
            f"| {claim['lower_rank_planner']} | {claim['rank_metric']} "
            f"| {claim['decision']} | {claim['rationale']} |"
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
    """Parse command-line arguments. Returns: Parsed argument namespace."""
    parser = argparse.ArgumentParser(
        description=(
            "Build per-cell CI + rank-stability report 7x7 headline "
            "planner comparison (#3216). Reuses canonical seed_variance, "
            "fidelity_rank_stability, canonical_table_export owners. Emits "
            "blocked_until_run/diagnostic on insufficient seed budget; never "
            "self-certifies paper-grade."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--rows", type=str, help="Path headline comparison rows JSON list.")
    src.add_argument(
        "--campaign",
        type=str,
        help="Campaign id/root; resolves <root>/reports/headline_rows.json.",
    )
    src.add_argument(
        "--dry-run",
        action="store_true",
        help="Use builtin deterministic fixture rows without external files.",
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+", default=list(DEFAULT_METRICS), help="Primary metrics per-cell CIs."
    )
    parser.add_argument(
        "--rank-metric", type=str, default=DEFAULT_RANK_METRIC, help="Metric used ranking."
    )
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Treat rank metric lower-is-better (default higher-is-better).",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--rank-resamples", type=int, default=DEFAULT_RANK_RESAMPLES)
    parser.add_argument("--rank-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--fail-on-decision-blocker",
        action="store_true",
        help="Exit non-zero when report contains a local preflight blocker.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/issue_3216_headline_ci/report",
        help="Directory result.json report.md.",
    )
    return parser.parse_args(argv)


def _dry_run_rows() -> list[dict[str, Any]]:
    """Deterministic fallback rows for local preflight smoke tests."""
    return [
        {
            "scenario_family": "merging",
            "planner_key": "best",
            "row_status": "successful_evidence",
            "execution_mode": "nominal",
            "per_seed": [
                {"seed": 111, "metrics": {"snqi": 0.90, "success": 0.84, "collisions": 0.20, "near_misses": 0.02}},
                {"seed": 112, "metrics": {"snqi": 0.91, "success": 0.85, "collisions": 0.20, "near_misses": 0.02}},
                {"seed": 113, "metrics": {"snqi": 0.92, "success": 0.86, "collisions": 0.21, "near_misses": 0.02}},
                {"seed": 114, "metrics": {"snqi": 0.93, "success": 0.83, "collisions": 0.19, "near_misses": 0.02}},
                {"seed": 115, "metrics": {"snqi": 0.94, "success": 0.82, "collisions": 0.19, "near_misses": 0.02}},
            ],
        },
        {
            "scenario_family": "merging",
            "planner_key": "mid",
            "row_status": "successful_evidence",
            "execution_mode": "nominal",
            "per_seed": [
                {"seed": 111, "metrics": {"snqi": 0.50, "success": 0.80, "collisions": 0.40, "near_misses": 0.10}},
                {"seed": 112, "metrics": {"snqi": 0.49, "success": 0.79, "collisions": 0.41, "near_misses": 0.11}},
                {"seed": 113, "metrics": {"snqi": 0.51, "success": 0.78, "collisions": 0.42, "near_misses": 0.09}},
                {"seed": 114, "metrics": {"snqi": 0.50, "success": 0.80, "collisions": 0.41, "near_misses": 0.10}},
                {"seed": 115, "metrics": {"snqi": 0.50, "success": 0.79, "collisions": 0.40, "near_misses": 0.10}},
            ],
        },
        {
            "scenario_family": "merging",
            "planner_key": "worst",
            "row_status": "successful_evidence",
            "execution_mode": "nominal",
            "per_seed": [
                {"seed": 111, "metrics": {"snqi": 0.10, "success": 0.30, "collisions": 0.70, "near_misses": 0.50}},
                {"seed": 112, "metrics": {"snqi": 0.11, "success": 0.31, "collisions": 0.71, "near_misses": 0.49}},
                {"seed": 113, "metrics": {"snqi": 0.09, "success": 0.29, "collisions": 0.72, "near_misses": 0.52}},
                {"seed": 114, "metrics": {"snqi": 0.12, "success": 0.28, "collisions": 0.73, "near_misses": 0.51}},
                {"seed": 115, "metrics": {"snqi": 0.10, "success": 0.30, "collisions": 0.74, "near_misses": 0.50}},
            ],
        },
    ]


def _resolve_rows_path(args: argparse.Namespace) -> str:
    """Resolve rows path from --rows --campaign."""
    if args.dry_run:
        return "builtin://issue3216-dry-run"
    if args.rows is not None:
        return args.rows
    return str(Path(args.campaign) / "reports" / "headline_rows.json")


def _resolve_rows(args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    """Resolve rows source and payload."""
    if args.dry_run:
        return _resolve_rows_path(args), _dry_run_rows()
    return _resolve_rows_path(args), load_rows_json(_resolve_rows_path(args))


def decision_packet_has_blocker(report: Mapping[str, Any]) -> bool:
    """Return whether a local preflight blocker should fail CI gate."""
    if report["classification"] == "blocked_until_run":
        return True
    if report["inputs"].get("invalid_rank_metric_reason"):
        return True
    for claim in report.get("adjacent_rank_claims", ()):  # pragma: no branch
        if claim.get("decision") == "not_statistically_distinguishable_budget":
            return True
    if report["inputs"].get("counted_cells", 0) == 0:
        return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns: Process exit code (0 on success)."""
    args = parse_args(argv)
    rows_path, rows = _resolve_rows(args)
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

    if args.fail_on_decision_blocker and decision_packet_has_blocker(report):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
