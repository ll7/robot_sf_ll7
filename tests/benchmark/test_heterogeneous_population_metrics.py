"""Tests for per-archetype metrics + mean-matched heterogeneity effect (issue #3574)."""

from __future__ import annotations

import math

import pytest

from robot_sf.benchmark.heterogeneous_population_metrics import (
    HETEROGENEOUS_POPULATION_METRICS_SCHEMA,
    PedestrianMetric,
    cvar,
    mean_matched_heterogeneity_effect,
    per_archetype_metrics,
)


def test_cvar_takes_lowest_tail_when_higher_is_safer() -> None:
    """For a higher-is-safer metric, CVaR must average the lowest (most dangerous) tail."""
    values = [1.0, 2.0, 3.0, 4.0, 10.0]
    # alpha=0.4 -> ceil(0.4*5)=2 worst (lowest) values -> mean(1,2)=1.5
    assert cvar(values, 0.4, higher_is_safer=True) == pytest.approx(1.5)


def test_cvar_takes_highest_tail_when_lower_is_safer() -> None:
    """For a lower-is-safer metric (e.g. exposure), CVaR must average the highest tail."""
    values = [1.0, 2.0, 3.0, 4.0, 10.0]
    assert cvar(values, 0.4, higher_is_safer=False) == pytest.approx(7.0)  # mean(4,10)


@pytest.mark.parametrize("alpha", [0.0, 1.5])
def test_cvar_rejects_invalid_alpha(alpha: float) -> None:
    """Alpha outside (0, 1] must fail closed."""
    with pytest.raises(ValueError):
        cvar([1.0, 2.0], alpha, higher_is_safer=True)


def test_per_archetype_aggregates_mean_worst_and_cvar() -> None:
    """Per-archetype aggregation must expose mean, worst-stratum, and CVaR."""
    observations = [
        PedestrianMetric("static", 0.2),
        PedestrianMetric("static", 0.4),
        PedestrianMetric("cooperative", 0.9),
        PedestrianMetric("cooperative", 1.1),
    ]
    report = per_archetype_metrics(observations, higher_is_safer=True, cvar_alpha=0.5)

    assert report["schema_version"] == HETEROGENEOUS_POPULATION_METRICS_SCHEMA
    static = report["per_archetype"]["static"]
    assert static["mean"] == pytest.approx(0.3)
    assert static["worst_stratum"] == pytest.approx(0.2)
    assert report["worst_archetype_by_mean"] == "static"


def test_per_archetype_worst_is_max_when_lower_is_safer() -> None:
    """When lower is safer, the worst stratum and worst archetype flip to the high end."""
    observations = [
        PedestrianMetric("a", 0.1),
        PedestrianMetric("b", 0.9),
        PedestrianMetric("b", 0.8),
    ]
    report = per_archetype_metrics(observations, higher_is_safer=False)

    assert report["per_archetype"]["b"]["worst_stratum"] == pytest.approx(0.9)
    assert report["worst_archetype_by_mean"] == "b"


def test_per_archetype_rejects_empty() -> None:
    """An empty observation set cannot be aggregated."""
    with pytest.raises(ValueError):
        per_archetype_metrics([])


def test_mean_matched_effect_is_isolated_when_baseline_is_mean_matched() -> None:
    """A mean-matched homogeneous baseline yields an isolated heterogeneity effect."""
    report = mean_matched_heterogeneity_effect(0.50, 0.62, homogeneous_is_mean_matched=True)

    assert report["heterogeneity_effect"] == pytest.approx(0.12)
    assert report["isolated_from_mean_shift"] is True
    assert report["validity"] == "isolated"


def test_effect_is_flagged_confounded_without_mean_matching() -> None:
    """Without a mean-matched baseline the effect must be flagged confounded."""
    report = mean_matched_heterogeneity_effect(0.50, 0.62, homogeneous_is_mean_matched=False)

    assert report["validity"] == "confounded_by_mean_shift"
    assert report["isolated_from_mean_shift"] is False


# --- non-finite (NaN/Inf) inputs must fail closed -----------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_cvar_rejects_non_finite_values(bad: float) -> None:
    """A NaN/Inf observation must raise, not let the tail-mean evaluate to NaN."""
    with pytest.raises(ValueError, match="finite"):
        cvar([1.0, 2.0, bad], 0.5, higher_is_safer=True)


def test_per_archetype_rejects_non_finite_value() -> None:
    """A non-finite per-pedestrian value must raise and name the archetype."""
    observations = [PedestrianMetric("static", 0.2), PedestrianMetric("static", math.nan)]
    with pytest.raises(ValueError, match="static"):
        per_archetype_metrics(observations)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_mean_matched_effect_rejects_non_finite_mean(bad: float) -> None:
    """A non-finite population mean must fail closed rather than propagate."""
    with pytest.raises(ValueError, match="finite"):
        mean_matched_heterogeneity_effect(bad, 0.62, homogeneous_is_mean_matched=True)
