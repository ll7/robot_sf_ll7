"""Tests for per-archetype metrics + mean-matched heterogeneity effect (issue #3574)."""

from __future__ import annotations

import math

import pytest

from robot_sf.benchmark.heterogeneous_population_metrics import (
    HETEROGENEOUS_POPULATION_METRICS_SCHEMA,
    PedestrianMetric,
    cvar,
    mean_matched_heterogeneity_effect,
    pedestrian_metric_observations_from_control_trace,
    per_archetype_metrics,
    per_archetype_metrics_from_control_trace,
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


def _control_trace() -> dict[str, object]:
    return {
        "schema_version": "pedestrian-control-trace.v1",
        "pedestrians": [
            {
                "id": "ped_cautious_a",
                "archetype": "cautious",
                "steps": [
                    {"step": 0, "speed_m_s": 0.0, "force_norm": 0.0},
                    {"step": 1, "speed_m_s": 0.4, "force_norm": 0.2},
                    {"step": 2, "speed_m_s": 0.6, "force_norm": 0.1},
                ],
            },
            {
                "id": "ped_cautious_b",
                "archetype": "cautious",
                "steps": [
                    {"step": 0, "speed_m_s": 0.0, "force_norm": 0.0},
                    {"step": 1, "speed_m_s": 0.2, "force_norm": 0.3},
                    {"step": 2, "speed_m_s": 0.4, "force_norm": 0.4},
                ],
            },
            {
                "id": "ped_hurried",
                "archetype": "hurried",
                "steps": [
                    {"step": 0, "speed_m_s": 0.0, "force_norm": 0.0},
                    {"step": 1, "speed_m_s": 1.0, "force_norm": 0.5},
                    {"step": 2, "speed_m_s": 1.4, "force_norm": 0.7},
                ],
            },
        ],
    }


def test_control_trace_extraction_builds_per_pedestrian_observations() -> None:
    """Control-trace rows can feed the per-archetype metric harness directly."""

    observations = pedestrian_metric_observations_from_control_trace(
        _control_trace(),
        "speed_m_s",
        reducer="mean",
    )

    assert observations == [
        PedestrianMetric("cautious", pytest.approx(1.0 / 3.0)),
        PedestrianMetric("cautious", pytest.approx(0.2)),
        PedestrianMetric("hurried", pytest.approx(0.8)),
    ]


def test_per_archetype_metrics_from_control_trace_keeps_trace_provenance() -> None:
    """Trace-derived report records source, metric, reducer, and per-archetype stats."""

    report = per_archetype_metrics_from_control_trace(
        _control_trace(),
        "force_norm",
        higher_is_safer=False,
        cvar_alpha=0.5,
        reducer="max",
    )

    assert report["schema_version"] == HETEROGENEOUS_POPULATION_METRICS_SCHEMA
    assert report["source"] == "pedestrian_control_trace"
    assert report["metric_key"] == "force_norm"
    assert report["pedestrian_metric_reducer"] == "max"
    assert report["worst_archetype_by_mean"] == "hurried"
    assert report["per_archetype"]["cautious"]["n"] == 2
    assert report["per_archetype"]["cautious"]["mean"] == pytest.approx(0.3)
    assert report["per_archetype"]["hurried"]["mean"] == pytest.approx(0.7)


def test_per_archetype_metrics_from_control_trace_normalizes_metric_key_provenance() -> None:
    """A padded metric key is normalized for lookup and recorded normalized in provenance."""

    report = per_archetype_metrics_from_control_trace(
        _control_trace(),
        "  speed_m_s  ",
    )

    assert report["metric_key"] == "speed_m_s"


def test_control_trace_extraction_supports_final_step_reducer() -> None:
    """Final-step extraction supports endpoint-style per-pedestrian metrics."""

    observations = pedestrian_metric_observations_from_control_trace(
        _control_trace(),
        "speed_m_s",
        reducer="final",
    )

    assert [observation.value for observation in observations] == pytest.approx([0.6, 0.4, 1.4])


def test_control_trace_extraction_rejects_missing_metric_key() -> None:
    """Missing trace metric keys fail closed rather than silently changing support."""

    trace = _control_trace()
    with pytest.raises(ValueError, match="missing 'clearance_m'"):
        pedestrian_metric_observations_from_control_trace(trace, "clearance_m")


def test_control_trace_extraction_rejects_non_finite_metric_value() -> None:
    """Non-finite trace metrics fail before per-archetype aggregation."""

    trace = _control_trace()
    trace["pedestrians"][0]["steps"][1]["speed_m_s"] = math.nan

    with pytest.raises(ValueError, match="finite"):
        pedestrian_metric_observations_from_control_trace(trace, "speed_m_s")


def test_control_trace_extraction_rejects_null_archetype() -> None:
    """An explicit null archetype fails closed instead of grouping under 'None'."""

    trace = _control_trace()
    trace["pedestrians"][0]["archetype"] = None

    with pytest.raises(ValueError, match="archetype must be non-empty"):
        pedestrian_metric_observations_from_control_trace(trace, "speed_m_s")


def test_control_trace_extraction_rejects_null_metric_value() -> None:
    """A null trace metric value fails closed with a descriptive error, not float(None)."""

    trace = _control_trace()
    trace["pedestrians"][0]["steps"][1]["speed_m_s"] = None

    with pytest.raises(ValueError, match="must not be null"):
        pedestrian_metric_observations_from_control_trace(trace, "speed_m_s")


def test_control_trace_extraction_rejects_non_mapping_trace() -> None:
    """A non-mapping control trace fails closed instead of raising a bare AttributeError."""

    with pytest.raises(ValueError, match="control_trace must be a mapping"):
        pedestrian_metric_observations_from_control_trace([], "speed_m_s")  # type: ignore[arg-type]


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
