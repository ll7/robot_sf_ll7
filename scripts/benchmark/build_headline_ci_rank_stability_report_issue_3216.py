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
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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
CONSTRAINTS_FIRST_RANK_METRIC = "success"
CONSTRAINTS_FIRST_REQUIRED_METRICS = ("success", "collisions", "near_misses")
SNQI_WARNING_MARKERS = ("snqi contract", "snqi_contract")

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
    rank_profile: str = "snqi_diagnostic"
    higher_is_better: bool = True
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    confidence: float = DEFAULT_CONFIDENCE
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    resamples: int = DEFAULT_RANK_RESAMPLES
    rank_seed: int = DEFAULT_BOOTSTRAP_SEED
    invalid_rank_metric_reason: str | None = None
    expected_scenarios: tuple[str, ...] = ()
    expected_planners: tuple[str, ...] = ()


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


@dataclass(frozen=True)
class GridCompleteness:
    """Expected headline-grid coverage check for already-measured rows."""

    expected_scenarios: tuple[str, ...]
    expected_planners: tuple[str, ...]
    expected_cell_count: int
    observed_cell_count: int
    observed_expected_cell_count: int
    missing_cells: tuple[tuple[str, str], ...]
    unexpected_cells: tuple[tuple[str, str], ...]

    @property
    def complete(self) -> bool:
        """Return whether all expected scenario/planner cells were present."""

        return not self.missing_cells

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe completeness payload."""

        return {
            "expected_scenarios": list(self.expected_scenarios),
            "expected_planners": list(self.expected_planners),
            "expected_cell_count": self.expected_cell_count,
            "observed_cell_count": self.observed_cell_count,
            "observed_expected_cell_count": self.observed_expected_cell_count,
            "missing_cell_count": len(self.missing_cells),
            "missing_cells": [
                {"scenario_id": scenario, "planner_key": planner}
                for scenario, planner in self.missing_cells
            ],
            "unexpected_cell_count": len(self.unexpected_cells),
            "unexpected_cells": [
                {"scenario_id": scenario, "planner_key": planner}
                for scenario, planner in self.unexpected_cells
            ],
            "complete": self.complete,
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


def _merge_unique(*groups: Sequence[str]) -> tuple[str, ...]:
    """Return non-empty strings in first-seen order across groups."""
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            normalized = str(value).strip()
            if normalized and normalized not in seen:
                merged.append(normalized)
                seen.add(normalized)
    return tuple(merged)


def _planner_keys_from_benchmark_config(path: Path) -> tuple[str, ...]:
    """Read expected planner keys from a benchmark YAML config.

    The wrapper uses this as a local preflight guard: rows can be absent or
    diagnostic, but they must disclose which configured headline planners are
    missing before any manuscript-table or S30 decision packet is treated as
    reviewable.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Benchmark config {path} must be a YAML mapping")
    planners = payload.get("planners")
    if not isinstance(planners, Sequence) or isinstance(planners, (str, bytes)):
        raise ValueError(f"Benchmark config {path} must contain a planners list")

    keys: list[str] = []
    for index, planner in enumerate(planners):
        if isinstance(planner, Mapping):
            key = planner.get("key") or planner.get("planner_key") or planner.get("algo")
        else:
            key = planner
        normalized = str(key or "").strip()
        if not normalized:
            raise ValueError(f"Planner entry {index} in {path} does not define a key")
        keys.append(normalized)
    return _merge_unique(keys)


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


def _optional_int(value: Any) -> int | None:
    """Coerce a value to int, returning ``None`` for null/non-integer inputs."""

    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_csv_rows(path: Path, *, required_columns: set[str] | None = None) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries, failing closed on missing or empty inputs."""

    if not path.is_file():
        raise FileNotFoundError(f"required CSV not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"required CSV is empty: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        if required_columns:
            missing = sorted(required_columns - fieldnames)
            if missing:
                raise ValueError(f"{path} missing required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"required CSV has header but no rows: {path}")
    return rows


def _campaign_family_names(campaign_dir: Path) -> tuple[str, ...]:
    """Return scenario-family names already emitted by the campaign report."""

    family_csv = campaign_dir / "reports" / "scenario_family_breakdown.csv"
    if not family_csv.exists():
        return ()
    names = {
        str(row.get("scenario_family", "")).strip()
        for row in _read_csv_rows(family_csv)
        if str(row.get("scenario_family", "")).strip()
    }
    return tuple(sorted(names, key=lambda name: (-len(name), name)))


def _scenario_family_from_id(scenario_id: str, known_families: Sequence[str]) -> str:
    """Map a concrete scenario id to the campaign's scenario-family key."""

    normalized = scenario_id.strip()
    for prefix in ("classic_", "francis2023_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    for suffix in ("_low", "_medium", "_high"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    for family in known_families:
        if normalized == family or normalized.endswith(f"_{family}") or family in normalized:
            return family
    return normalized


def _campaign_planner_statuses(campaign_dir: Path) -> dict[str, dict[str, str]]:
    """Return planner row status metadata from the campaign table."""

    table_csv = campaign_dir / "reports" / "campaign_table.csv"
    if not table_csv.exists():
        return {}
    statuses: dict[str, dict[str, str]] = {}
    for row in _read_csv_rows(table_csv):
        planner = str(row.get("planner_key", "")).strip()
        if not planner:
            continue
        statuses[planner] = {
            "execution_mode": str(row.get("execution_mode", "")).strip() or "unknown",
            "row_status": str(row.get("status", "")).strip()
            or str(row.get("readiness_status", "")).strip()
            or "unspecified",
        }
    return statuses


def _campaign_metric_sources(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return headline metric source values from one seed episode CSV row."""

    return {
        "success": row.get("success"),
        "collisions": row.get("collision"),
        "near_misses": row.get("near_miss"),
        "snqi": row.get("snqi"),
    }


def _group_campaign_seed_metrics(
    seed_csv: Path, known_families: Sequence[str]
) -> dict[tuple[str, str], dict[int, dict[str, list[float]]]]:
    """Group completed campaign episode metrics by family, planner, and seed."""

    grouped: dict[tuple[str, str], dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in _read_csv_rows(
        seed_csv,
        required_columns={"planner_key", "scenario_id", "seed"},
    ):
        planner = str(row.get("planner_key", "")).strip()
        scenario_id = str(row.get("scenario_id", "")).strip()
        seed = _optional_int(row.get("seed"))
        if not planner or not scenario_id or seed is None:
            continue
        family = _scenario_family_from_id(scenario_id, known_families)
        for metric, raw_value in _campaign_metric_sources(row).items():
            value = _coerce_float(raw_value)
            if value is not None:
                grouped[(family, planner)][seed][metric].append(value)
    return grouped


def _headline_rows_from_campaign(campaign_dir: Path) -> list[dict[str, Any]]:
    """Build headline rows from completed camera-ready campaign artifacts."""

    seed_csv = campaign_dir / "reports" / "seed_episode_rows.csv"
    if not seed_csv.exists():
        raise FileNotFoundError(
            f"campaign seed episode rows not found: {seed_csv}; cannot recover headline rows"
        )

    known_families = _campaign_family_names(campaign_dir)
    planner_statuses = _campaign_planner_statuses(campaign_dir)
    grouped = _group_campaign_seed_metrics(seed_csv, known_families)

    headline_rows: list[dict[str, Any]] = []
    for (family, planner), seed_payload in sorted(grouped.items()):
        status = planner_statuses.get(planner, {})
        per_seed: list[dict[str, Any]] = []
        for seed, metrics in sorted(seed_payload.items()):
            metric_means = {
                metric: float(np.mean(values))
                for metric, values in sorted(metrics.items())
                if values
            }
            if metric_means:
                per_seed.append({"seed": seed, "metrics": metric_means})
        if per_seed:
            headline_rows.append(
                {
                    "scenario_family": family,
                    "planner_key": planner,
                    "row_status": status.get("row_status", "unspecified"),
                    "execution_mode": status.get("execution_mode", "unknown"),
                    "per_seed": per_seed,
                }
            )

    if not headline_rows:
        raise ValueError(f"no recoverable headline rows found in {seed_csv}")
    return headline_rows


def _ensure_campaign_headline_rows(campaign_dir: Path) -> Path:
    """Return campaign headline rows, generating them from completed artifacts if needed."""

    rows_path = campaign_dir / "reports" / "headline_rows.json"
    if rows_path.exists():
        return rows_path
    rows = _headline_rows_from_campaign(campaign_dir)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows_path


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


def build_grid_completeness(
    cells: Sequence[CellResult],
    *,
    expected_scenarios: Sequence[str],
    expected_planners: Sequence[str],
) -> GridCompleteness | None:
    """Check expected headline scenario/planner coverage."""

    scenarios = tuple(dict.fromkeys(str(value) for value in expected_scenarios if str(value)))
    planners = tuple(dict.fromkeys(str(value) for value in expected_planners if str(value)))
    if not scenarios and not planners:
        return None

    observed = {(cell.scenario_id, cell.planner_key) for cell in cells}
    if not scenarios:
        expected_planner_set = set(planners)
        observed_planners = {planner for _, planner in observed}
        missing = tuple(("*", planner) for planner in planners if planner not in observed_planners)
        unexpected = tuple(
            sorted(
                (scenario, planner)
                for scenario, planner in observed
                if planner not in expected_planner_set
            )
        )
        return GridCompleteness(
            expected_scenarios=scenarios,
            expected_planners=planners,
            expected_cell_count=len(planners),
            observed_cell_count=len(observed),
            observed_expected_cell_count=len(observed_planners & expected_planner_set),
            missing_cells=missing,
            unexpected_cells=unexpected,
        )

    if not planners:
        expected_scenario_set = set(scenarios)
        observed_scenarios = {scenario for scenario, _ in observed}
        missing = tuple(
            (scenario, "*") for scenario in scenarios if scenario not in observed_scenarios
        )
        unexpected = tuple(
            sorted(
                (scenario, planner)
                for scenario, planner in observed
                if scenario not in expected_scenario_set
            )
        )
        return GridCompleteness(
            expected_scenarios=scenarios,
            expected_planners=planners,
            expected_cell_count=len(scenarios),
            observed_cell_count=len(observed),
            observed_expected_cell_count=len(observed_scenarios & expected_scenario_set),
            missing_cells=missing,
            unexpected_cells=unexpected,
        )

    expected = tuple((scenario, planner) for scenario in scenarios for planner in planners)
    expected_set = set(expected)
    missing = tuple(
        (scenario, planner) for scenario, planner in expected if (scenario, planner) not in observed
    )
    unexpected = tuple(sorted(observed - expected_set))
    return GridCompleteness(
        expected_scenarios=scenarios,
        expected_planners=planners,
        expected_cell_count=len(expected),
        observed_cell_count=len(observed),
        observed_expected_cell_count=len(observed & expected_set),
        missing_cells=missing,
        unexpected_cells=unexpected,
    )


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
    invalid_rank_metric_reason: str | None = None,
) -> list[AdjacentRankClaim]:
    """Build adjacent-rank downgrade decisions from per-cell CIs.

    Adjacent planners with overlapping confidence intervals are downgraded to
    ``not_statistically_distinguishable_budget`` so the packet cannot be read
    as support for a strict paper-facing ordering. Adjacent planner statements
    below the paper-grade seed threshold are classified ``diagnostic_only``.
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
            min_pair_seeds = min(higher_cell.seed_count, lower_cell.seed_count)
            if invalid_rank_metric_reason:
                decision = "blocked_invalid_metric"
                rationale = (
                    f"rank metric {rank_metric!r} is contract-invalid: {invalid_rank_metric_reason}"
                )
            elif min_pair_seeds < PAPER_GRADE_MIN_SEEDS:
                decision = "diagnostic_only"
                rationale = (
                    "Adjacent-rank statement is diagnostic only because the paired cells "
                    f"have min_seeds={min_pair_seeds}, below the paper-grade threshold "
                    f"{PAPER_GRADE_MIN_SEEDS}."
                )
            elif _ci_overlap(higher_stats, lower_stats):
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


def load_job_evidence_packet(path: str | Path, *, job_id: int) -> dict[str, Any]:
    """Load the compact #1554/#3216 Slurm-evidence packet entry for one job.

    The packet is provenance only. It does not provide headline rows by itself,
    but it can carry warnings that must downgrade local rank claims.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    target_job = _optional_int(job_id)
    matches = [
        item
        for item in payload.get("evidence", [])
        if isinstance(item, Mapping)
        and target_job is not None
        and _optional_int(item.get("job")) == target_job
    ]
    if not matches:
        return {
            "status": "missing_job",
            "job": int(job_id),
            "packet_path": str(path),
            "snqi_contract_warning": False,
            "warnings": [],
            "limitations": [f"job {job_id} not found in evidence packet"],
        }

    item = dict(matches[0])
    artifact_summary = item.get("artifact_summary") or {}
    warnings = [str(value) for value in artifact_summary.get("warnings", [])]
    limitations = [str(value) for value in item.get("limitations", [])]
    warning_text = " ".join(warnings + limitations).lower()
    snqi_warning = any(marker in warning_text for marker in SNQI_WARNING_MARKERS)
    return {
        "status": "bound",
        "job": int(job_id),
        "packet_path": str(path),
        "campaign": item.get("campaign"),
        "config": item.get("config"),
        "role": item.get("role"),
        "slurm_state": item.get("slurm_state"),
        "exit_code": item.get("exit_code"),
        "public_commit": item.get("public_commit"),
        "planner_row_status_counts": artifact_summary.get("planner_row_status_counts", {}),
        "matrix_rows": artifact_summary.get("matrix_rows"),
        "planner_rows": artifact_summary.get("planner_rows"),
        "warnings": warnings,
        "limitations": limitations,
        "snqi_contract_warning": snqi_warning,
    }


def _constraints_first_metric_gaps(cells: Sequence[CellResult]) -> list[str]:
    """Return required constraints-first metrics missing from counted cells.

    Constraints-first ranking compares the safety constraint metrics across the
    full headline matrix, so a metric is only "present" when *every* counted
    cell reports it. If a single counted cell is missing a required metric (for
    example due to a logging failure) the cross-cell constraint comparison is
    incomplete and the metric is flagged as a gap. With no counted cells, all
    required metrics are gaps. This is the conservative, fail-closed reading.
    """

    counted = [cell for cell in cells if cell.counted]
    if not counted:
        return list(CONSTRAINTS_FIRST_REQUIRED_METRICS)
    gaps: list[str] = []
    for metric in CONSTRAINTS_FIRST_REQUIRED_METRICS:
        if all(_metric_stats(cell, metric) is not None for cell in counted):
            continue
        gaps.append(metric)
    return gaps


def _append_packet_reasons(
    manuscript_blockers: list[str],
    s30_reasons: list[str],
    *,
    overlap_claims: Sequence[AdjacentRankClaim],
    diagnostic_only_claims: Sequence[AdjacentRankClaim],
    invalid_rank_metric: bool,
    metric_gaps: Sequence[str],
    snqi_contract_warning: bool,
    unstable_rank_scenarios: Sequence[str],
) -> None:
    """Append downgrade/blocker reasons derived from rank-profile checks."""

    if overlap_claims:
        s30_reasons.append("adjacent_rank_ci_overlap_requires_claim_downgrade_or_more_data")
    if diagnostic_only_claims:
        s30_reasons.append("seed_budget_below_paper_grade_diagnostic_only")
    if invalid_rank_metric:
        manuscript_blockers.append("invalid_rank_metric_contract")
        s30_reasons.append("rank_metric_contract_invalid")
    if metric_gaps:
        manuscript_blockers.append("constraints_first_metrics_missing")
        s30_reasons.append("constraints_first_metric_gap")
    if snqi_contract_warning:
        s30_reasons.append("snqi_contract_warning_diagnostic_only")
    if unstable_rank_scenarios:
        s30_reasons.append("rank_resampling_instability_present")


def build_decision_packet(
    cells: Sequence[CellResult],
    rank_stability: Sequence[ScenarioRankStability],
    adjacent_rank_claims: Sequence[AdjacentRankClaim],
    *,
    rank_profile: str = "snqi_diagnostic",
    invalid_rank_metric_reason: str | None = None,
    job_evidence: Mapping[str, Any] | None = None,
    grid_completeness: GridCompleteness | None = None,
) -> dict[str, Any]:
    """Build a conservative manuscript/S30 decision packet."""
    counted = [cell for cell in cells if cell.counted]
    excluded = [cell for cell in cells if not cell.counted]
    identifiable = [entry for entry in rank_stability if entry.rank_identifiable]
    unstable_rank_scenarios = [
        entry.scenario_id
        for entry in identifiable
        if entry.top1_stable is False
        or (entry.rank_flip_rate is not None and entry.rank_flip_rate > 0.0)
    ]
    overlap_claims = [
        claim
        for claim in adjacent_rank_claims
        if claim.decision == "not_statistically_distinguishable_budget"
    ]
    diagnostic_only_claims = [
        claim for claim in adjacent_rank_claims if claim.decision == "diagnostic_only"
    ]
    invalid_metric_claims = [
        claim for claim in adjacent_rank_claims if claim.decision == "blocked_invalid_metric"
    ]
    invalid_rank_metric = bool(invalid_rank_metric_reason or invalid_metric_claims)
    metric_gaps = (
        _constraints_first_metric_gaps(cells) if rank_profile == "constraints_first" else []
    )
    snqi_contract_warning = bool(job_evidence and job_evidence.get("snqi_contract_warning"))

    min_seed_count = min((cell.seed_count for cell in counted), default=0)
    manuscript_blockers: list[str] = []
    s30_reasons: list[str] = []

    if not counted:
        manuscript_blockers.append("no_counted_cells")
        s30_reasons.append("missing_counted_headline_rows")
    if grid_completeness is not None and not grid_completeness.complete:
        manuscript_blockers.append("headline_grid_incomplete")
        s30_reasons.append("missing_expected_headline_cells")
    if excluded:
        manuscript_blockers.append("non_promotable_cells_present")
        s30_reasons.append("resolve_or_disclose_excluded_cells")
    if min_seed_count < PAPER_GRADE_MIN_SEEDS:
        manuscript_blockers.append("missing_increased_seed_budget")
        s30_reasons.append("minimum_seed_count_below_s20")
    if not identifiable:
        manuscript_blockers.append("no_identifiable_rankings")
        s30_reasons.append("rank_stability_not_identifiable")
    _append_packet_reasons(
        manuscript_blockers,
        s30_reasons,
        overlap_claims=overlap_claims,
        diagnostic_only_claims=diagnostic_only_claims,
        invalid_rank_metric=invalid_rank_metric,
        metric_gaps=metric_gaps,
        snqi_contract_warning=snqi_contract_warning,
        unstable_rank_scenarios=unstable_rank_scenarios,
    )

    if manuscript_blockers:
        manuscript_table_status = "blocked"
    else:
        manuscript_table_status = "ready_for_table_review_no_claim_promotion"

    if (
        not counted
        or excluded
        or not identifiable
        or (grid_completeness is not None and not grid_completeness.complete)
    ):
        s30_decision_status = "blocked"
    elif (
        min_seed_count < PAPER_GRADE_MIN_SEEDS
        or unstable_rank_scenarios
        or overlap_claims
        or invalid_rank_metric
    ):
        s30_decision_status = "needs_review"
    else:
        s30_decision_status = "not_required_by_local_preflight"

    return {
        "manuscript_table_status": manuscript_table_status,
        "manuscript_blockers": manuscript_blockers,
        "s30_decision_status": s30_decision_status,
        "s30_reasons": s30_reasons,
        "min_seed_count": min_seed_count,
        "paper_grade_min_seeds": PAPER_GRADE_MIN_SEEDS,
        "excluded_cell_count": len(excluded),
        "identifiable_scenario_count": len(identifiable),
        "unstable_rank_scenarios": unstable_rank_scenarios,
        "adjacent_overlap_count": len(overlap_claims),
        "diagnostic_only_claim_count": len(diagnostic_only_claims),
        "invalid_metric_claim_count": len(invalid_metric_claims),
        "invalid_rank_metric_reason": invalid_rank_metric_reason,
        "rank_profile": rank_profile,
        "constraints_first_required_metrics": list(CONSTRAINTS_FIRST_REQUIRED_METRICS),
        "constraints_first_metric_gaps": metric_gaps,
        "grid_completeness": (
            grid_completeness.to_dict() if grid_completeness is not None else None
        ),
        "job_evidence_status": (job_evidence or {}).get("status"),
        "job_id": (job_evidence or {}).get("job"),
        "snqi_contract_warning": snqi_contract_warning,
        "claim_boundary": (
            "Decision packet is local preflight only; no manuscript or paper claim is promoted."
        ),
    }


def build_report(
    rows: Sequence[Mapping[str, Any]],
    config: ReportConfig | None = None,
    *,
    campaign: str | None = None,
    rows_path: str | None = None,
    job_evidence: Mapping[str, Any] | None = None,
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
    grid_completeness = build_grid_completeness(
        cells,
        expected_scenarios=config.expected_scenarios,
        expected_planners=config.expected_planners,
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
        invalid_rank_metric_reason=config.invalid_rank_metric_reason,
    )
    decision_packet = build_decision_packet(
        cells,
        rank_stability,
        adjacent_rank_claims,
        rank_profile=config.rank_profile,
        invalid_rank_metric_reason=config.invalid_rank_metric_reason,
        job_evidence=job_evidence,
        grid_completeness=grid_completeness,
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
            "rank_profile": config.rank_profile,
            "higher_is_better": higher_is_better,
            "invalid_rank_metric_reason": config.invalid_rank_metric_reason,
        },
        "config": {
            "bootstrap_samples": config.bootstrap_samples,
            "confidence": config.confidence,
            "bootstrap_seed": config.bootstrap_seed,
            "rank_resamples": config.resamples,
            "rank_seed": config.rank_seed,
            "rank_profile": config.rank_profile,
            "invalid_rank_metric_reason": config.invalid_rank_metric_reason,
            "paper_grade_min_seeds": PAPER_GRADE_MIN_SEEDS,
            "nominal_min_seeds": NOMINAL_MIN_SEEDS,
            "expected_scenarios": list(config.expected_scenarios),
            "expected_planners": list(config.expected_planners),
        },
        "job_evidence": dict(job_evidence or {}),
        "grid_completeness": grid_completeness.to_dict() if grid_completeness else None,
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
        "decision_packet": decision_packet,
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


def _append_job_evidence_markdown(lines: list[str], job_evidence: Mapping[str, Any]) -> None:
    """Append compact job-evidence provenance lines to markdown."""

    if not job_evidence:
        return
    lines.append(
        f"- **Job evidence**: `{job_evidence.get('status')}` job "
        f"`{job_evidence.get('job')}` campaign `{job_evidence.get('campaign')}`"
    )
    if job_evidence.get("snqi_contract_warning"):
        lines.append("- **SNQI evidence status**: `diagnostic_only_contract_warning`")


def _append_grid_completeness_markdown(lines: list[str], report: Mapping[str, Any]) -> None:
    """Append expected-grid coverage line when the report checked one."""

    grid = report.get("grid_completeness")
    if not grid:
        return
    lines.append(
        f"- **Expected grid**: {grid['observed_expected_cell_count']} observed / "
        f"{grid['expected_cell_count']} expected; "
        f"{grid['missing_cell_count']} missing; "
        f"{grid['unexpected_cell_count']} unexpected"
    )


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
    _append_grid_completeness_markdown(lines, report)
    lines.append(f"- **Rank metric**: `{inputs['rank_metric']}`")
    lines.append(f"- **Rank profile**: `{inputs['rank_profile']}`")
    if inputs.get("invalid_rank_metric_reason"):
        lines.append(
            f"- **Rank metric contract**: `invalid` - {inputs['invalid_rank_metric_reason']}"
        )
    _append_job_evidence_markdown(lines, report.get("job_evidence") or {})
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
    _append_decision_packet_markdown(lines, report["decision_packet"])
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


def decision_packet_has_blocker(decision_packet: Mapping[str, Any]) -> bool:
    """Return whether a decision packet should fail a local preflight.

    The report itself is still useful when this returns ``True``; the non-zero
    CLI mode is for automation that wants a hard gate before table or S30 work.
    """
    return decision_packet.get("manuscript_table_status") == "blocked" or decision_packet.get(
        "s30_decision_status"
    ) in {"blocked", "needs_review"}


def _append_decision_packet_markdown(lines: list[str], decision_packet: Mapping[str, Any]) -> None:
    """Append the manuscript/S30 decision packet markdown section."""
    lines.append("## Manuscript/S30 decision packet")
    lines.append("")
    lines.append(f"- **Manuscript table status**: `{decision_packet['manuscript_table_status']}`")
    lines.append(f"- **S30 decision status**: `{decision_packet['s30_decision_status']}`")
    lines.append(f"- **Minimum seed count**: {decision_packet['min_seed_count']}")
    lines.append(
        f"- **Adjacent CI-overlap downgrades**: {decision_packet['adjacent_overlap_count']}"
    )
    lines.append(
        f"- **Diagnostic-only adjacent claims**: {decision_packet['diagnostic_only_claim_count']}"
    )
    if decision_packet["manuscript_blockers"]:
        blockers = ", ".join(decision_packet["manuscript_blockers"])
        lines.append(f"- **Manuscript blockers**: {blockers}")
    if decision_packet["s30_reasons"]:
        reasons = ", ".join(decision_packet["s30_reasons"])
        lines.append(f"- **S30 reasons**: {reasons}")
    lines.append(f"- **Packet boundary**: {decision_packet['claim_boundary']}")


def _dry_run_rows() -> list[dict[str, Any]]:
    """Return deterministic fixture rows for local CLI preflight.

    The fixture is intentionally below the nominal seed threshold. It proves the
    parser, CI, downgrade, and rank-stability paths execute without requiring a
    campaign artifact or cluster submission, but it cannot promote a benchmark
    claim.
    """

    def row(
        scenario: str,
        planner: str,
        snqi_values: Sequence[float],
        *,
        row_status: str = "successful_evidence",
        execution_mode: str = "nominal",
    ) -> dict[str, Any]:
        per_seed = [
            {
                "seed": 900 + index,
                "metrics": {
                    "success": 1.0 if value >= 0.5 else 0.0,
                    "collisions": 0.0 if value >= 0.5 else 1.0,
                    "near_misses": 0.0,
                    "snqi": value,
                },
            }
            for index, value in enumerate(snqi_values)
        ]
        return {
            "scenario_family": scenario,
            "planner_key": planner,
            "row_status": row_status,
            "execution_mode": execution_mode,
            "per_seed": per_seed,
        }

    return [
        row("dry_run_merging", "stable_top", [0.91, 0.92, 0.90, 0.93, 0.91]),
        row("dry_run_merging", "stable_mid", [0.56, 0.54, 0.55, 0.57, 0.56]),
        row("dry_run_merging", "stable_low", [0.20, 0.18, 0.21, 0.19, 0.20]),
        row("dry_run_bottleneck", "overlap_a", [0.50, 0.55, 0.45, 0.52, 0.48]),
        row("dry_run_bottleneck", "overlap_b", [0.51, 0.46, 0.54, 0.49, 0.53]),
        row(
            "dry_run_bottleneck",
            "excluded_degraded",
            [0.80, 0.81, 0.82, 0.83, 0.84],
            execution_mode="degraded",
        ),
    ]


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
        help=(
            "Campaign id/root; uses <root>/reports/headline_rows.json or "
            "recovers it from completed campaign report CSV artifacts."
        ),
    )
    src.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run deterministic built-in fixture rows through the report path. "
            "This writes output files but does not require campaign data."
        ),
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
        "--rank-profile",
        choices=("snqi_diagnostic", "constraints_first"),
        default="snqi_diagnostic",
        help=(
            "Ranking interpretation profile. constraints_first ranks by success "
            "unless --rank-metric overrides it and treats SNQI as diagnostic."
        ),
    )
    parser.add_argument(
        "--expected-scenario",
        action="append",
        default=[],
        help=(
            "Expected scenario-family key in the headline grid. Repeat for every "
            "required row family."
        ),
    )
    parser.add_argument(
        "--expected-planner",
        action="append",
        default=[],
        help="Expected planner key in the headline grid. Repeat for every required planner.",
    )
    parser.add_argument(
        "--expected-planners-from-config",
        type=str,
        default=None,
        help=(
            "Benchmark YAML config whose planners list defines expected "
            "headline planner rows. Explicit --expected-planner values are "
            "merged after config-derived values."
        ),
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
        "--invalid-rank-metric-reason",
        type=str,
        default=None,
        help=(
            "Fail closed adjacent rank statements when the rank metric contract is invalid "
            "(for example an SNQI contract warning/failure)."
        ),
    )
    parser.add_argument(
        "--evidence-packet",
        type=str,
        default=None,
        help="Optional compact Slurm evidence packet JSON to bind job provenance/warnings.",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        default=None,
        help="Job id to bind from --evidence-packet, for example 13198.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/issue_3216_headline_ci/report",
        help="Directory for result.json and report.md.",
    )
    parser.add_argument(
        "--fail-on-decision-blocker",
        action="store_true",
        help=(
            "Return exit code 4 after writing outputs when the local decision "
            "packet still blocks manuscript-table or S30 decisions."
        ),
    )
    return parser.parse_args(argv)


def _resolve_rows_path(args: argparse.Namespace) -> str:
    """Resolve the rows path from --rows or --campaign.

    Returns:
        Filesystem path to the rows JSON list.
    """
    if args.dry_run:
        return "builtin://issue3216-dry-run"
    if args.rows:
        return args.rows
    return str(_ensure_campaign_headline_rows(Path(args.campaign)))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    rows_path = _resolve_rows_path(args)
    rows = _dry_run_rows() if args.dry_run else load_rows_json(rows_path)
    if bool(args.evidence_packet) != bool(args.job_id):
        sys.stderr.write("--evidence-packet and --job-id must be supplied together\n")
        return 2
    job_evidence = (
        load_job_evidence_packet(args.evidence_packet, job_id=args.job_id)
        if args.evidence_packet
        else None
    )
    rank_metric = args.rank_metric
    if args.rank_profile == "constraints_first" and args.rank_metric == DEFAULT_RANK_METRIC:
        rank_metric = CONSTRAINTS_FIRST_RANK_METRIC
    invalid_rank_metric_reason = args.invalid_rank_metric_reason
    if (
        invalid_rank_metric_reason is None
        and rank_metric == "snqi"
        and job_evidence
        and job_evidence.get("snqi_contract_warning")
    ):
        invalid_rank_metric_reason = "SNQI contract warning in bound job evidence packet"
    expected_planners = tuple(args.expected_planner)
    if args.expected_planners_from_config:
        config_planners = _planner_keys_from_benchmark_config(
            Path(args.expected_planners_from_config)
        )
        expected_planners = _merge_unique(config_planners, expected_planners)
    config = ReportConfig(
        metrics=tuple(args.metrics),
        rank_metric=rank_metric,
        rank_profile=args.rank_profile,
        higher_is_better=not args.lower_is_better,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        bootstrap_seed=args.bootstrap_seed,
        resamples=args.rank_resamples,
        rank_seed=args.rank_seed,
        invalid_rank_metric_reason=invalid_rank_metric_reason,
        expected_scenarios=tuple(args.expected_scenario),
        expected_planners=expected_planners,
    )
    campaign = "dry-run" if args.dry_run else args.campaign
    report = build_report(
        rows,
        config,
        campaign=campaign,
        rows_path=rows_path,
        job_evidence=job_evidence,
    )
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
    if args.fail_on_decision_blocker and decision_packet_has_blocker(report["decision_packet"]):
        packet = report["decision_packet"]
        print(
            "decision_blocker=true "
            f"manuscript_table_status={packet['manuscript_table_status']} "
            f"s30_decision_status={packet['s30_decision_status']}",
            file=sys.stderr,
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
