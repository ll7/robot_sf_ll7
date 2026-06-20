"""Forecast-baseline comparison/leaderboard core (bounded #2915 child).

Given per-baseline forecast metrics (ADE / FDE / miss-rate, ...), rank the
baselines per metric and identify the best per metric, with fail-closed handling
of missing metrics. This gives the parent's CV / semantic-CV / interaction-aware
predictor outputs a consistent comparison surface.

Forecast error metrics are lower-is-better by default; any metric outside the
configured lower-is-better set is treated as higher-is-better. A baseline that
lacks a metric is reported as *not-comparable* for that metric (it is never
silently ranked as zero).

Pure and deterministic: it consumes already-measured metrics and runs no
forecasting. This is analysis tooling and makes no benchmark claim. Implementing
the predictor classes and running the forecast eval lives with parent #2915.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

FORECAST_BASELINE_COMPARISON_SCHEMA = "forecast_baseline_comparison.v1"

# Standard forecast error metrics where a smaller value is better.
DEFAULT_LOWER_IS_BETTER = ("ade", "fde", "min_ade", "min_fde", "miss_rate", "nll")


@dataclass(frozen=True)
class MetricRanking:
    """Per-metric ranking of forecast baselines."""

    metric: str
    lower_is_better: bool
    ranking: list[str]
    best: str | None
    not_comparable: list[str]
    deltas_vs_best: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe representation.

        Returns:
            Mapping with the metric, direction, ranking, best baseline,
            not-comparable baselines, and deltas vs the best.
        """
        return {
            "metric": self.metric,
            "lower_is_better": self.lower_is_better,
            "ranking": list(self.ranking),
            "best": self.best,
            "not_comparable": list(self.not_comparable),
            "deltas_vs_best": dict(self.deltas_vs_best),
        }


@dataclass(frozen=True)
class ForecastBaselineComparison:
    """Comparison report across all metrics."""

    baselines: list[str]
    metrics: list[str]
    rankings: list[MetricRanking]

    def best_by_metric(self) -> dict[str, str | None]:
        """Return the winning baseline per metric.

        Returns:
            Mapping of metric name to the best baseline (or ``None`` if none ranked).
        """
        return {ranking.metric: ranking.best for ranking in self.rankings}

    def to_dict(self) -> dict[str, Any]:
        """Return the ``forecast_baseline_comparison.v1`` JSON-safe payload.

        Returns:
            Mapping with the schema version, baselines, metrics, per-metric
            rankings, and best-by-metric.
        """
        return {
            "schema_version": FORECAST_BASELINE_COMPARISON_SCHEMA,
            "baselines": list(self.baselines),
            "metrics": list(self.metrics),
            "best_by_metric": self.best_by_metric(),
            "rankings": [ranking.to_dict() for ranking in self.rankings],
        }


def compare_forecast_baselines(
    metrics_by_baseline: Mapping[str, Mapping[str, float]],
    *,
    metrics: Iterable[str] | None = None,
    lower_is_better_metrics: Iterable[str] = DEFAULT_LOWER_IS_BETTER,
) -> ForecastBaselineComparison:
    """Rank forecast baselines per metric and pick the best per metric.

    ``metrics_by_baseline`` maps baseline name to ``{metric -> value}``. Baselines
    missing a metric are listed under that metric's ``not_comparable`` rather than
    ranked. Ties are broken by baseline name for deterministic output.

    Returns:
        A :class:`ForecastBaselineComparison` with per-metric rankings.
    """
    if not metrics_by_baseline:
        raise ValueError("metrics_by_baseline must contain at least one baseline")
    baselines = sorted(metrics_by_baseline)
    lower_set = set(lower_is_better_metrics)
    if metrics is not None:
        metric_names = list(metrics)
    else:
        metric_names = sorted({name for row in metrics_by_baseline.values() for name in row})

    rankings: list[MetricRanking] = []
    for metric in metric_names:
        lower_is_better = metric in lower_set
        present: dict[str, float] = {}
        not_comparable: list[str] = []
        for baseline in baselines:
            value = metrics_by_baseline[baseline].get(metric)
            if value is None:
                not_comparable.append(baseline)
            else:
                present[baseline] = float(value)

        ordered = sorted(
            present,
            key=lambda baseline: (
                present[baseline] if lower_is_better else -present[baseline],
                baseline,
            ),
        )
        best = ordered[0] if ordered else None
        deltas = (
            {baseline: present[baseline] - present[best] for baseline in ordered}
            if best is not None
            else {}
        )
        rankings.append(
            MetricRanking(
                metric=metric,
                lower_is_better=lower_is_better,
                ranking=ordered,
                best=best,
                not_comparable=not_comparable,
                deltas_vs_best=deltas,
            )
        )

    return ForecastBaselineComparison(baselines=baselines, metrics=metric_names, rankings=rankings)
