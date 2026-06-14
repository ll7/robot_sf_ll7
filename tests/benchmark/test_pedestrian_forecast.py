"""Tests for deterministic pedestrian forecast baselines."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark.pedestrian_forecast import (
    PedestrianState,
    chi_square_2d_threshold,
    compute_batch_forecast_metrics,
    constant_velocity_gaussian_baseline,
    evaluate_forecast,
)


def test_constant_velocity_forecast_is_deterministic() -> None:
    """Forecast means and covariance are stable for fixed trace inputs."""

    state = PedestrianState(
        id=1,
        position=np.array([0.0, 1.0]),
        velocity=np.array([1.0, -0.5]),
        intent="crossing",
        signal="green",
        signal_available=True,
    )

    first = constant_velocity_gaussian_baseline(state, horizons_s=(0.5, 1.0))
    second = constant_velocity_gaussian_baseline(state, horizons_s=(0.5, 1.0))

    assert first.id == 1
    assert [prediction.horizon_s for prediction in first.predictions] == [0.5, 1.0]
    for left, right in zip(first.predictions, second.predictions, strict=True):
        np.testing.assert_allclose(left.mean, right.mean)
        np.testing.assert_allclose(left.covariance, right.covariance)
    np.testing.assert_allclose(first.predictions[0].mean, [0.5, 0.75])
    assert first.predictions[0].metadata["context_status"] == "available"


def test_unavailable_signal_widens_uncertainty_without_assuming_phase() -> None:
    """Unavailable signal metadata is exposed as uncertainty, not a phase guess."""

    aware = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        intent="crossing",
        signal="green",
        signal_available=True,
    )
    unknown_signal = PedestrianState.from_trace(
        {
            "id": 1,
            "position": [0.0, 0.0],
            "velocity": [1.0, 0.0],
            "intent_label": "crossing",
            "signal_state": {"available": False},
        }
    )

    aware_forecast = constant_velocity_gaussian_baseline(aware, horizons_s=(1.0,))
    unknown_forecast = constant_velocity_gaussian_baseline(unknown_signal, horizons_s=(1.0,))

    assert unknown_signal.signal is None
    assert unknown_forecast.predictions[0].metadata["signal_state"] == "unknown"
    assert unknown_forecast.predictions[0].metadata["context_status"] == "uncertain"
    assert unknown_forecast.predictions[0].metadata["std_m"] == pytest.approx(
        aware_forecast.predictions[0].metadata["std_m"] * 1.5
    )


def test_flat_signal_label_trace_path_is_supported() -> None:
    """Legacy flat signal labels are still represented as available signal context."""

    state = PedestrianState.from_trace(
        {
            "id": 4,
            "position": [1.0, 2.0],
            "velocity": [0.0, 0.0],
            "signal_label": "red",
        }
    )

    assert state.signal_available is True
    assert state.signal == "red"


def test_evaluate_forecast_reports_likelihood_calibration_and_miss_rate() -> None:
    """Single-sample forecast metrics include NLL, calibration, and miss rate."""

    forecast = constant_velocity_gaussian_baseline(
        PedestrianState(
            id=1,
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
            intent="crossing",
            signal="green",
            signal_available=True,
        ),
        horizons_s=(1.0,),
    )

    perfect = evaluate_forecast(forecast, {1.0: np.array([1.0, 0.0])})
    assert perfect["negative_log_likelihood_1s"] == pytest.approx(-perfect["log_likelihood_1s"])
    assert perfect["mahalanobis_dist_1s"] == 0.0
    assert perfect["miss_rate_1s"] == 0.0
    assert perfect["within_95ci_1s"] == 1.0
    assert perfect["calibration_error_1s"] == pytest.approx(0.05)

    missed = evaluate_forecast(forecast, {1.0: np.array([10.0, 0.0])})
    assert missed["miss_rate_1s"] == 1.0
    assert missed["within_95ci_1s"] == 0.0
    assert missed["calibration_error_1s"] == pytest.approx(0.95)


def test_evaluate_forecast_reports_collision_relevance_error() -> None:
    """Collision relevance compares predicted overlap against future robot proximity."""

    forecast = constant_velocity_gaussian_baseline(
        PedestrianState(
            id=7,
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
            intent="crossing",
            signal="green",
            signal_available=True,
        ),
        horizons_s=(1.0,),
    )

    matching = evaluate_forecast(
        forecast,
        {1.0: np.array([1.0, 0.0])},
        robot_positions={1.0: np.array([1.1, 0.0])},
        collision_distance_m=0.3,
    )
    assert matching["collision_relevance_error_1s"] == 0.0

    false_positive = evaluate_forecast(
        forecast,
        {1.0: np.array([5.0, 0.0])},
        robot_positions={1.0: np.array([1.1, 0.0])},
        collision_distance_m=0.3,
    )
    assert false_positive["collision_relevance_error_1s"] == 1.0


def test_compute_batch_forecast_metrics_uses_trace_steps_and_denominators() -> None:
    """Batch metrics consume trace steps and report denominator counts."""

    dt_s = 0.5
    trace_steps = []
    for index in range(5):
        x = index * dt_s
        trace_steps.append(
            {
                "step": index,
                "time_s": x,
                "robot": {"position": [x + 0.1, 0.0]},
                "pedestrians": [
                    {
                        "id": 1,
                        "position": [x, 0.0],
                        "velocity": [1.0, 0.0],
                        "intent_label": "crossing",
                        "signal_state": {"available": True, "label": "green"},
                    }
                ],
            }
        )

    summary = compute_batch_forecast_metrics(
        trace_steps,
        horizons_s=(0.5, 1.0),
        dt_s=dt_s,
        collision_distance_m=0.3,
    )

    assert summary["forecast_evaluable_samples"] == 4.0
    assert summary["mean_ade_0.5s"] == pytest.approx(0.0)
    assert summary["mean_ade_1s"] == pytest.approx(0.0)
    assert summary["mean_miss_rate_1s"] == 0.0
    assert summary["count_collision_relevance_error_0.5s"] == 4.0
    assert summary["count_collision_relevance_error_1s"] == 3.0


def test_compute_batch_forecast_metrics_excludes_cyclists() -> None:
    """Cyclist actors are excluded from scoring but tracked in metadata."""

    dt_s = 0.1
    trace_steps = [
        {
            "step": 0,
            "time_s": 0.0,
            "pedestrians": [
                {"id": 1, "position": [0, 0], "velocity": [1, 0], "actor_type": "pedestrian"},
                {"id": 2, "position": [0, 1], "velocity": [1, 0], "actor_type": "cyclist_like_vru"},
            ],
        },
        {
            "step": 1,
            "time_s": 0.1,
            "pedestrians": [
                {"id": 1, "position": [0.1, 0], "velocity": [1, 0], "actor_type": "pedestrian"},
                {
                    "id": 2,
                    "position": [0.1, 1],
                    "velocity": [1, 0],
                    "actor_type": "cyclist_like_vru",
                },
            ],
        },
        {
            "step": 2,
            "time_s": 0.2,
            "pedestrians": [
                {"id": 1, "position": [0.2, 0], "velocity": [1, 0], "actor_type": "pedestrian"},
                {
                    "id": 2,
                    "position": [0.2, 1],
                    "velocity": [1, 0],
                    "actor_type": "cyclist_like_vru",
                },
            ],
        },
    ]

    summary = compute_batch_forecast_metrics(
        trace_steps,
        horizons_s=(0.1,),
        dt_s=dt_s,
    )

    # Step 0: ped 1 (evaluable), cyclist 2 (excluded)
    # Step 1: ped 1 (evaluable), cyclist 2 (excluded)
    # Step 2: ped 1 (not evaluable - no future), cyclist 2 (excluded)
    assert summary["pedestrian_forecast_candidate_count"] == 6.0
    assert summary["pedestrian_forecast_included_actor_count"] == 3.0
    assert summary["pedestrian_forecast_excluded_actor_count"] == 3.0
    assert summary["pedestrian_forecast_excluded_cyclist_like_vru_count"] == 3.0
    assert summary["forecast_evaluable_samples"] == 2.0


def test_compute_batch_forecast_metrics_normalizes_excluded_actor_type_keys() -> None:
    """Excluded actor-type denominator keys stay flat even for display labels."""
    summary = compute_batch_forecast_metrics(
        [
            {
                "step": 0,
                "time_s": 0.0,
                "pedestrians": [
                    {"id": 1, "position": [0, 0], "velocity": [1, 0], "actor_type": "Bicycle/VRU"}
                ],
            },
            {
                "step": 1,
                "time_s": 0.1,
                "pedestrians": [
                    {
                        "id": 1,
                        "position": [0.1, 0],
                        "velocity": [1, 0],
                        "actor_type": "Bicycle/VRU",
                    }
                ],
            },
        ],
        horizons_s=(0.1,),
        dt_s=0.1,
    )

    assert summary["pedestrian_forecast_candidate_count"] == 2.0
    assert summary["pedestrian_forecast_included_actor_count"] == 0.0
    assert summary["pedestrian_forecast_excluded_actor_count"] == 2.0
    assert summary["pedestrian_forecast_excluded_bicycle_vru_count"] == 2.0
    assert summary["forecast_evaluable_samples"] == 0.0


def test_batch_metrics_report_empty_and_invalid_dt_cases() -> None:
    """Empty traces have an explicit denominator and invalid dt fails closed."""

    assert compute_batch_forecast_metrics([]) == {
        "forecast_evaluable_samples": 0.0,
        "pedestrian_forecast_candidate_count": 0.0,
        "pedestrian_forecast_included_actor_count": 0.0,
        "pedestrian_forecast_excluded_actor_count": 0.0,
    }
    with pytest.raises(ValueError, match="dt_s must be positive"):
        compute_batch_forecast_metrics([], dt_s=0.0)


def test_confidence_threshold_rejects_invalid_probability() -> None:
    """Confidence ellipse threshold rejects impossible probabilities."""

    with pytest.raises(ValueError, match="between 0 and 1"):
        chi_square_2d_threshold(0.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        chi_square_2d_threshold(1.0)
