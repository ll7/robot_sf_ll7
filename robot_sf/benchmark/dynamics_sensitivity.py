"""Dynamics-sensitivity ranking analysis for issue #3976.

The harness consumes already-measured planner metric tables keyed by dynamics
model. It does not run a benchmark campaign and does not change metric semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from robot_sf.benchmark.rank_metrics import kendall_tau, rank_order

DYNAMICS_SENSITIVITY_SCHEMA = "dynamics_sensitivity.v1"


MetricTable = Mapping[str, Mapping[str, object]]
DynamicsMetricTables = Mapping[str, MetricTable]


def _finite_metric_value(metrics: Mapping[str, object], metric: str) -> float | None:
    try:
        value = float(metrics[metric])
    except (KeyError, TypeError, ValueError):
        return None
    if not isfinite(value):
        return None
    return value


def _metric_values(table: MetricTable, metric: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for planner, metrics in table.items():
        value = _finite_metric_value(metrics, metric)
        if value is not None:
            values[str(planner)] = value
    return values


def _rank_planners(
    table: MetricTable,
    metric: str,
    *,
    higher_is_better: bool,
) -> list[str]:
    values = _metric_values(table, metric)
    return [str(planner) for planner in rank_order(values, higher_is_better=higher_is_better)]


def _project_table(
    table: MetricTable, metric_names: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for planner, metrics in table.items():
        row: dict[str, float] = {}
        for metric in metric_names:
            value = _finite_metric_value(metrics, metric)
            if value is not None:
                row[metric] = value
        rows[str(planner)] = row
    return rows


@dataclass(frozen=True)
class DynamicsRankStability:
    """Ranking comparison for one dynamics model against the reference model."""

    dynamics: str
    ranking: list[str]
    kendall_tau: float | None
    rank_flips: int | None
    top1_changed: bool | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""

        return {
            "dynamics": self.dynamics,
            "ranking": list(self.ranking),
            "kendall_tau": self.kendall_tau,
            "rank_flips": self.rank_flips,
            "top1_changed": self.top1_changed,
        }


@dataclass(frozen=True)
class DynamicsSensitivityReport:
    """Per-dynamics metric table plus planner-ranking stability summary."""

    reference_dynamics: str
    primary_metric: str
    higher_is_better: bool
    reference_ranking: list[str]
    metric_table: dict[str, dict[str, dict[str, float]]]
    dynamics: list[DynamicsRankStability]
    flipping_dynamics: list[str]
    rank_stable: bool

    def to_dict(self) -> dict[str, object]:
        """Return a ``dynamics_sensitivity.v1`` JSON-safe payload."""

        return {
            "schema_version": DYNAMICS_SENSITIVITY_SCHEMA,
            "reference_dynamics": self.reference_dynamics,
            "primary_metric": self.primary_metric,
            "higher_is_better": self.higher_is_better,
            "reference_ranking": list(self.reference_ranking),
            "rank_stable": self.rank_stable,
            "flipping_dynamics": list(self.flipping_dynamics),
            "metric_table": self.metric_table,
            "dynamics": [entry.to_dict() for entry in self.dynamics],
        }


def analyze_dynamics_sensitivity(
    tables: DynamicsMetricTables,
    *,
    reference_dynamics: str,
    primary_metric: str = "collision_rate",
    higher_is_better: bool = False,
    metric_names: tuple[str, ...] = (
        "collision_rate",
        "near_miss_rate",
        "tracking_error",
        "curvature_limit_violation",
        "braking_distance",
    ),
) -> DynamicsSensitivityReport:
    """Compare planner rankings across robot dynamics metric tables.

    Args:
        tables: Mapping ``dynamics -> planner -> metric -> value``.
        reference_dynamics: Dynamics name used as the ranking reference.
        primary_metric: Metric used to rank planners.
        higher_is_better: Direction for ``primary_metric``.
        metric_names: Metrics copied into the output table when present.

    Returns:
        A deterministic ranking-stability report for short local sensitivity runs.
    """

    if reference_dynamics not in tables:
        raise ValueError(f"reference_dynamics {reference_dynamics!r} is missing from tables.")

    reference_table = tables[reference_dynamics]
    reference_ranking = _rank_planners(
        reference_table,
        primary_metric,
        higher_is_better=higher_is_better,
    )
    if len(reference_ranking) < 2:
        raise ValueError("reference table must contain at least two finite planner metric values.")

    metric_table = {
        str(dynamics): _project_table(table, metric_names)
        for dynamics, table in sorted(tables.items(), key=lambda item: str(item[0]))
    }

    dynamics_entries: list[DynamicsRankStability] = []
    flipping_dynamics: list[str] = []
    for dynamics, table in sorted(tables.items(), key=lambda item: str(item[0])):
        dynamics_name = str(dynamics)
        ranking = _rank_planners(table, primary_metric, higher_is_better=higher_is_better)
        if set(ranking) != set(reference_ranking):
            tau = None
            rank_flips = None
            top1_changed = None
            flipping_dynamics.append(dynamics_name)
        else:
            tau = kendall_tau(reference_ranking, ranking, degenerate=1.0)
            rank_flips = _count_rank_flips(reference_ranking, ranking)
            top1_changed = bool(ranking and ranking[0] != reference_ranking[0])
            if rank_flips > 0:
                flipping_dynamics.append(dynamics_name)
        dynamics_entries.append(
            DynamicsRankStability(
                dynamics=dynamics_name,
                ranking=ranking,
                kendall_tau=tau,
                rank_flips=rank_flips,
                top1_changed=top1_changed,
            )
        )

    return DynamicsSensitivityReport(
        reference_dynamics=reference_dynamics,
        primary_metric=primary_metric,
        higher_is_better=higher_is_better,
        reference_ranking=reference_ranking,
        metric_table=metric_table,
        dynamics=dynamics_entries,
        flipping_dynamics=flipping_dynamics,
        rank_stable=not flipping_dynamics,
    )


def _count_rank_flips(reference: list[str], candidate: list[str]) -> int:
    position = {planner: index for index, planner in enumerate(candidate)}
    flips = 0
    for left_index, left in enumerate(reference):
        for right in reference[left_index + 1 :]:
            if position[left] > position[right]:
                flips += 1
    return flips


__all__ = [
    "DYNAMICS_SENSITIVITY_SCHEMA",
    "DynamicsRankStability",
    "DynamicsSensitivityReport",
    "analyze_dynamics_sensitivity",
]
