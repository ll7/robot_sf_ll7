"""Fidelity rank-stability and metric-drift analysis (bounded #3207 child).

Given planner metric tables measured under a *nominal* simulation fidelity plus
one table per *fidelity-perturbation axis* (the sweep output), this module reports
whether the planner **ranking** is stable across fidelity assumptions and which
axes flip it. That is the evidence behind a defensible benchmark validity
boundary: a ranking that survives a bounded fidelity sweep is benchmark-
strengthening; an axis that flips the ranking is a required reporting caveat and a
calibration candidate.

Pure and deterministic: it operates on already-measured metric tables and runs no
simulation. It is analysis tooling and makes no benchmark or sim-to-real claim;
fidelity perturbations are deliberate sensitivity probes, not fallback/degraded
runs. The fidelity *sweep execution* lives with parent issue #3207.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from robot_sf.benchmark.rank_metrics import kendall_tau as _shared_kendall_tau

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

FIDELITY_RANK_STABILITY_SCHEMA = "fidelity_rank_stability.v1"


def rank_planners(
    table: Mapping[str, Mapping[str, object]],
    metric: str,
    *,
    higher_is_better: bool = True,
) -> list[str]:
    """Return planner names ordered best to worst by ``metric``.

    Ties are broken by planner name for stable, deterministic output. Rows
    missing ``metric`` or carrying non-numeric/non-finite values sort last.

    Returns:
        Planner names ordered from best to worst on ``metric``.
    """

    def sort_key(planner: str) -> tuple[int, float, str]:
        value = _finite_metric_value(table[planner], metric)
        if value is None:
            return (1, 0.0, planner)
        return (0, -value if higher_is_better else value, planner)

    return sorted(table, key=sort_key)


def _finite_metric_value(metrics: Mapping[str, object], metric: str) -> float | None:
    """Return a finite numeric metric value, or ``None`` when unavailable."""
    raw_value = metrics.get(metric)
    if raw_value is None or isinstance(raw_value, bool):
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    """Return Kendall tau between two orderings of the same planners.

    ``1.0`` means identical order, ``-1.0`` fully reversed. Degenerate inputs
    (fewer than two planners) return ``1.0``.

    Returns:
        Kendall tau-b rank correlation in ``[-1.0, 1.0]``.
    """
    tau = _shared_kendall_tau(order_a, order_b, degenerate=1.0)
    if tau is None or math.isnan(tau):
        return 1.0
    return float(tau)


def count_rank_flips(order_a: list[str], order_b: list[str]) -> int:
    """Return the number of discordant planner pairs between two orderings.

    ``0`` means the two orderings agree on every pairwise comparison.

    Returns:
        Count of pairs whose relative order differs between ``order_a`` and ``order_b``.
    """
    position_b = {planner: index for index, planner in enumerate(order_b)}
    flips = 0
    for i in range(len(order_a)):
        for j in range(i + 1, len(order_a)):
            if position_b[order_a[i]] > position_b[order_a[j]]:
                flips += 1
    return flips


def metric_drift(
    nominal: Mapping[str, Mapping[str, object]],
    axis: Mapping[str, Mapping[str, object]],
    metrics: Iterable[str],
) -> dict[str, float]:
    """Return mean absolute relative drift per metric (axis vs nominal).

    Drift for one planner/metric is ``|axis - nominal| / max(|nominal|, 1)``,
    averaged across planners present in both tables with finite numeric values.

    Returns:
        Mapping of metric name to mean absolute relative drift.
    """
    drift: dict[str, float] = {}
    for metric in metrics:
        relatives: list[float] = []
        for planner, nominal_metrics in nominal.items():
            if planner not in axis:
                continue
            base = _finite_metric_value(nominal_metrics, metric)
            current = _finite_metric_value(axis[planner], metric)
            if base is None or current is None:
                continue
            denom = max(abs(base), 1.0)
            relatives.append(abs(current - base) / denom)
        drift[metric] = sum(relatives) / len(relatives) if relatives else 0.0
    return drift


@dataclass(frozen=True)
class AxisRankStability:
    """Rank-stability + drift result for one fidelity-perturbation axis."""

    axis: str
    ranking: list[str]
    kendall_tau: float
    rank_flips: int
    top1_changed: bool
    metric_drift: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe representation.

        Returns:
            Mapping with the axis ranking, tau, rank flips, top-1 change, and drift.
        """
        return {
            "axis": self.axis,
            "ranking": list(self.ranking),
            "kendall_tau": self.kendall_tau,
            "rank_flips": self.rank_flips,
            "top1_changed": self.top1_changed,
            "metric_drift": dict(self.metric_drift),
        }


@dataclass(frozen=True)
class FidelitySensitivityReport:
    """Validity-boundary report across all fidelity axes."""

    primary_metric: str
    higher_is_better: bool
    nominal_ranking: list[str]
    axes: list[AxisRankStability]
    flipping_axes: list[str]
    rank_stable: bool

    def to_dict(self) -> dict[str, object]:
        """Return the ``fidelity_rank_stability.v1`` JSON-safe payload.

        Returns:
            Mapping with the schema version, nominal ranking, per-axis results,
            flipping axes, and the validity-boundary verdict.
        """
        return {
            "schema_version": FIDELITY_RANK_STABILITY_SCHEMA,
            "primary_metric": self.primary_metric,
            "higher_is_better": self.higher_is_better,
            "nominal_ranking": list(self.nominal_ranking),
            "rank_stable": self.rank_stable,
            "flipping_axes": list(self.flipping_axes),
            "axes": [axis.to_dict() for axis in self.axes],
        }


def analyze_fidelity_sensitivity(
    nominal_table: Mapping[str, Mapping[str, object]],
    axis_tables: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    primary_metric: str,
    higher_is_better: bool = True,
    drift_metrics: Iterable[str] | None = None,
) -> FidelitySensitivityReport:
    """Analyze planner-ranking stability across fidelity-perturbation axes.

    Each axis table must contain the same planner set as ``nominal_table``. The
    nominal ranking (by ``primary_metric``) is the reference; each axis reports
    Kendall tau, rank-flip count, top-1 change, and per-metric drift relative to
    nominal. An axis with any rank flip is flagged as ranking-sensitive.

    Returns:
        A :class:`FidelitySensitivityReport` with per-axis results and the
        overall ``rank_stable`` validity-boundary verdict.
    """
    if not nominal_table:
        raise ValueError("nominal_table must contain at least one planner")
    nominal_planners = set(nominal_table)
    if drift_metrics is not None:
        metrics = list(drift_metrics)
    else:
        metrics = sorted({name for row in nominal_table.values() for name in row})
    nominal_ranking = rank_planners(
        nominal_table, primary_metric, higher_is_better=higher_is_better
    )

    axes: list[AxisRankStability] = []
    flipping_axes: list[str] = []
    for axis_name, table in axis_tables.items():
        if set(table) != nominal_planners:
            raise ValueError(
                f"fidelity axis '{axis_name}' planner set does not match nominal "
                f"({sorted(table)} vs {sorted(nominal_planners)})"
            )
        ranking = rank_planners(table, primary_metric, higher_is_better=higher_is_better)
        flips = count_rank_flips(nominal_ranking, ranking)
        axes.append(
            AxisRankStability(
                axis=axis_name,
                ranking=ranking,
                kendall_tau=kendall_tau(nominal_ranking, ranking),
                rank_flips=flips,
                top1_changed=bool(nominal_ranking and ranking and nominal_ranking[0] != ranking[0]),
                metric_drift=metric_drift(nominal_table, table, metrics),
            )
        )
        if flips > 0:
            flipping_axes.append(axis_name)

    return FidelitySensitivityReport(
        primary_metric=primary_metric,
        higher_is_better=higher_is_better,
        nominal_ranking=nominal_ranking,
        axes=axes,
        flipping_axes=flipping_axes,
        rank_stable=not flipping_axes,
    )
