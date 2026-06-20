"""Tests for the forecast-baseline comparison core (issue #3244, child of #2915)."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.forecast_baseline_comparison import (
    FORECAST_BASELINE_COMPARISON_SCHEMA,
    compare_forecast_baselines,
)

_METRICS = {
    "cv": {"ade": 0.50, "fde": 1.20, "miss_rate": 0.30},
    "semantic_cv": {"ade": 0.40, "fde": 1.10, "miss_rate": 0.30},
    "interaction_aware": {"ade": 0.35, "fde": 1.30, "miss_rate": 0.25},
}


def _ranking(comparison, metric):
    return next(r for r in comparison.rankings if r.metric == metric)


def test_lower_is_better_ranking_and_best() -> None:
    """ADE is lower-is-better, so the smallest ADE ranks first."""
    comparison = compare_forecast_baselines(_METRICS)
    ade = _ranking(comparison, "ade")
    assert ade.lower_is_better is True
    assert ade.ranking == ["interaction_aware", "semantic_cv", "cv"]
    assert ade.best == "interaction_aware"
    assert comparison.best_by_metric()["fde"] == "semantic_cv"
    assert comparison.best_by_metric()["miss_rate"] == "interaction_aware"


def test_deltas_vs_best_are_relative_to_winner() -> None:
    """Deltas are each baseline's value minus the best value for that metric."""
    comparison = compare_forecast_baselines(_METRICS)
    fde = _ranking(comparison, "fde")
    assert fde.best == "semantic_cv"
    assert fde.deltas_vs_best["semantic_cv"] == pytest.approx(0.0)
    assert fde.deltas_vs_best["cv"] == pytest.approx(0.10)
    assert fde.deltas_vs_best["interaction_aware"] == pytest.approx(0.20)


def test_missing_metric_is_not_comparable() -> None:
    """A baseline lacking a metric is not-comparable for it, not ranked as zero."""
    metrics = {
        "cv": {"ade": 0.5, "fde": 1.2},
        "interaction_aware": {"ade": 0.4},  # no fde
    }
    comparison = compare_forecast_baselines(metrics)
    fde = _ranking(comparison, "fde")
    assert fde.not_comparable == ["interaction_aware"]
    assert fde.ranking == ["cv"]
    assert fde.best == "cv"


def test_ties_broken_by_name() -> None:
    """Equal metric values rank by baseline name for determinism."""
    metrics = {"b_planner": {"ade": 0.5}, "a_planner": {"ade": 0.5}}
    comparison = compare_forecast_baselines(metrics)
    assert _ranking(comparison, "ade").ranking == ["a_planner", "b_planner"]


def test_higher_is_better_metric_outside_default_set() -> None:
    """A metric not in the lower-is-better set ranks descending."""
    metrics = {"cv": {"coverage": 0.7}, "semantic_cv": {"coverage": 0.9}}
    comparison = compare_forecast_baselines(metrics)
    coverage = _ranking(comparison, "coverage")
    assert coverage.lower_is_better is False
    assert coverage.ranking == ["semantic_cv", "cv"]
    assert coverage.best == "semantic_cv"


def test_explicit_metrics_are_deduplicated_in_order() -> None:
    """Duplicate explicit metrics still emit one ranking per unique metric."""
    comparison = compare_forecast_baselines(_METRICS, metrics=["fde", "ade", "fde"])
    assert comparison.metrics == ["fde", "ade"]
    assert [ranking.metric for ranking in comparison.rankings] == ["fde", "ade"]


def test_empty_input_raises() -> None:
    """An empty comparison is rejected."""
    with pytest.raises(ValueError, match="at least one baseline"):
        compare_forecast_baselines({})


def test_to_dict_schema() -> None:
    """The payload carries the forecast_baseline_comparison.v1 schema."""
    payload = compare_forecast_baselines(_METRICS).to_dict()
    assert payload["schema_version"] == FORECAST_BASELINE_COMPARISON_SCHEMA
    assert payload["best_by_metric"]["ade"] == "interaction_aware"
    assert set(payload["baselines"]) == {"cv", "semantic_cv", "interaction_aware"}
