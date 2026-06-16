"""Tests for deterministic pedestrian forecast baselines."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark.pedestrian_forecast import (
    NeighborContext,
    PedestrianState,
    chi_square_2d_threshold,
    compute_batch_forecast_metrics,
    constant_velocity_gaussian_baseline,
    evaluate_forecast,
    goal_aware_cv_baseline,
    interaction_aware_cv_baseline,
    risk_filtered_cv_baseline,
    semantic_cv_baseline,
    signal_aware_cv_baseline,
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


def test_signal_aware_cv_red_slows_mean() -> None:
    """Signal-aware CV with red signal reduces mean displacement."""
    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        intent="crossing",
        signal="red",
        signal_available=True,
    )
    forecast = signal_aware_cv_baseline(state, horizons_s=(1.0,))
    cv_forecast = constant_velocity_gaussian_baseline(state, horizons_s=(1.0,))

    assert forecast.predictions[0].mean[0] < cv_forecast.predictions[0].mean[0]
    assert forecast.predictions[0].metadata["signal_status"] == "red_slowed"
    assert forecast.predictions[0].metadata["model"] == "signal_aware_cv"


def test_signal_aware_cv_green_preserves_mean() -> None:
    """Signal-aware CV with green signal preserves CV mean."""
    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        intent="crossing",
        signal="green",
        signal_available=True,
    )
    forecast = signal_aware_cv_baseline(state, horizons_s=(1.0,))
    cv_forecast = constant_velocity_gaussian_baseline(state, horizons_s=(1.0,))

    np.testing.assert_allclose(forecast.predictions[0].mean, cv_forecast.predictions[0].mean)
    assert forecast.predictions[0].metadata["signal_status"] == "green_preserved"


def test_signal_aware_cv_unavailable_widens_uncertainty() -> None:
    """Signal-aware CV with unavailable signal widens uncertainty, not assuming phase."""
    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        intent="crossing",
        signal=None,
        signal_available=False,
    )
    cv_state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        intent="crossing",
        signal=None,
        signal_available=False,
    )
    forecast = signal_aware_cv_baseline(state, horizons_s=(1.0,))
    cv_forecast = constant_velocity_gaussian_baseline(cv_state, horizons_s=(1.0,))

    np.testing.assert_allclose(forecast.predictions[0].mean, cv_forecast.predictions[0].mean)
    assert forecast.predictions[0].metadata["signal_status"] == "unknown_widened"
    assert forecast.predictions[0].metadata["std_m"] == pytest.approx(
        cv_forecast.predictions[0].metadata["std_m"]
    )


def test_signal_aware_cv_unrecognized_signal_preserves_uncertainty() -> None:
    """Available but unrecognized signal values are labeled without widening uncertainty."""
    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        intent="crossing",
        signal="yellow",
        signal_available=True,
    )
    forecast = signal_aware_cv_baseline(state, horizons_s=(1.0,))
    cv_forecast = constant_velocity_gaussian_baseline(state, horizons_s=(1.0,))

    np.testing.assert_allclose(forecast.predictions[0].mean, cv_forecast.predictions[0].mean)
    assert forecast.predictions[0].metadata["signal_status"] == "unrecognized_preserved"
    assert forecast.predictions[0].metadata["std_m"] == pytest.approx(
        cv_forecast.predictions[0].metadata["std_m"]
    )


def test_goal_aware_cv_crossing_adjusts_mean() -> None:
    """Goal-aware CV with crossing intent increases mean displacement."""
    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        intent="crossing",
        signal="green",
        signal_available=True,
    )
    forecast = goal_aware_cv_baseline(state, horizons_s=(1.0,))
    cv_forecast = constant_velocity_gaussian_baseline(state, horizons_s=(1.0,))

    assert forecast.predictions[0].mean[0] > cv_forecast.predictions[0].mean[0]
    assert forecast.predictions[0].metadata["intent_status"] == "crossing_adjusted"
    assert forecast.predictions[0].metadata["model"] == "goal_aware_cv"


def test_goal_aware_cv_walking_along_adjusts_mean() -> None:
    """Goal-aware CV with walking_along intent decreases mean displacement."""
    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        intent="walking_along",
        signal="green",
        signal_available=True,
    )
    forecast = goal_aware_cv_baseline(state, horizons_s=(1.0,))
    cv_forecast = constant_velocity_gaussian_baseline(state, horizons_s=(1.0,))

    assert forecast.predictions[0].mean[0] < cv_forecast.predictions[0].mean[0]
    assert forecast.predictions[0].metadata["intent_status"] == "walking_along_adjusted"


def test_goal_aware_cv_unavailable_widens_uncertainty() -> None:
    """Goal-aware CV with absent intent widens uncertainty, not assuming a goal."""
    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        intent=None,
        signal="green",
        signal_available=True,
    )
    forecast = goal_aware_cv_baseline(state, horizons_s=(1.0,))

    assert forecast.predictions[0].metadata["intent_status"] == "unknown_widened"
    assert forecast.predictions[0].metadata["std_m"] == pytest.approx((0.3 + 0.4 * 1.0) * 1.3)


def test_goal_aware_cv_unrecognized_intent_preserves_uncertainty() -> None:
    """Present but unrecognized intent values are labeled without widening uncertainty."""
    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        intent="standing",
        signal="green",
        signal_available=True,
    )
    forecast = goal_aware_cv_baseline(state, horizons_s=(1.0,))
    cv_forecast = constant_velocity_gaussian_baseline(state, horizons_s=(1.0,))

    np.testing.assert_allclose(forecast.predictions[0].mean, cv_forecast.predictions[0].mean)
    assert forecast.predictions[0].metadata["intent_status"] == "unrecognized_preserved"
    assert forecast.predictions[0].metadata["std_m"] == pytest.approx(
        cv_forecast.predictions[0].metadata["std_m"]
    )


def test_semantic_cv_composes_signal_and_goal() -> None:
    """Semantic CV composes signal and goal factors correctly."""
    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        intent="crossing",
        signal="red",
        signal_available=True,
    )
    forecast = semantic_cv_baseline(state, horizons_s=(1.0,))
    pred = forecast.predictions[0]

    expected_mean_x = 0.0 + 1.0 * 1.0 * 0.4 * 1.2
    np.testing.assert_allclose(pred.mean[0], expected_mean_x)
    assert pred.metadata["signal_status"] == "red_slowed"
    assert pred.metadata["intent_status"] == "crossing_adjusted"
    assert pred.metadata["model"] == "semantic_cv"


def test_semantic_cv_unavailable_widens_both() -> None:
    """Semantic CV widens uncertainty for both unavailable signal and intent."""
    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        intent=None,
        signal=None,
        signal_available=False,
    )
    forecast = semantic_cv_baseline(state, horizons_s=(1.0,))
    pred = forecast.predictions[0]

    base_std = 0.3 + 0.4 * 1.0
    expected_std = base_std * 1.5 * 1.3
    assert pred.metadata["std_m"] == pytest.approx(expected_std)
    assert pred.metadata["signal_status"] == "unknown_widened"
    assert pred.metadata["intent_status"] == "unknown_widened"


def test_semantic_cv_unrecognized_context_preserves_uncertainty() -> None:
    """Semantic CV labels present unrecognized context without treating it as unavailable."""
    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        intent="standing",
        signal="yellow",
        signal_available=True,
    )
    forecast = semantic_cv_baseline(state, horizons_s=(1.0,))
    cv_forecast = constant_velocity_gaussian_baseline(state, horizons_s=(1.0,))
    pred = forecast.predictions[0]

    np.testing.assert_allclose(pred.mean, cv_forecast.predictions[0].mean)
    assert pred.metadata["std_m"] == pytest.approx(cv_forecast.predictions[0].metadata["std_m"])
    assert pred.metadata["signal_status"] == "unrecognized_preserved"
    assert pred.metadata["intent_status"] == "unrecognized_preserved"


def test_compute_batch_forecast_metrics_with_baseline_function() -> None:
    """Batch metrics accept an explicit baseline function."""
    dt_s = 0.5
    trace_steps = []
    for index in range(5):
        x = index * dt_s
        trace_steps.append(
            {
                "step": index,
                "time_s": x,
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

    cv_summary = compute_batch_forecast_metrics(trace_steps, horizons_s=(0.5, 1.0), dt_s=dt_s)
    signal_summary = compute_batch_forecast_metrics(
        trace_steps,
        horizons_s=(0.5, 1.0),
        dt_s=dt_s,
        baseline_function=signal_aware_cv_baseline,
    )

    assert cv_summary["forecast_evaluable_samples"] > 0
    assert signal_summary["forecast_evaluable_samples"] > 0
    assert cv_summary["forecast_evaluable_samples"] == signal_summary["forecast_evaluable_samples"]


def test_interaction_aware_cv_no_neighbors_matches_cv() -> None:
    """Without neighbors, interaction-aware CV degrades to plain CV."""
    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        intent="crossing",
        signal="green",
        signal_available=True,
    )
    forecast = interaction_aware_cv_baseline(state, horizons_s=(1.0,))
    cv_forecast = constant_velocity_gaussian_baseline(state, horizons_s=(1.0,))

    np.testing.assert_allclose(forecast.predictions[0].mean, cv_forecast.predictions[0].mean)
    np.testing.assert_allclose(
        forecast.predictions[0].covariance, cv_forecast.predictions[0].covariance
    )
    assert forecast.predictions[0].metadata["interaction_status"] == "no_neighbors"
    assert forecast.predictions[0].metadata["neighbor_count"] == 0.0
    assert forecast.predictions[0].metadata["model"] == "interaction_aware_cv"


def test_interaction_aware_cv_nearby_neighbor_deflects_mean() -> None:
    """A nearby neighbor pushes the forecast mean away (repulsion)."""
    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        intent="crossing",
        signal="green",
        signal_available=True,
    )
    neighbor = NeighborContext(
        position=np.array([0.5, 0.5]),
        velocity=np.array([-0.5, 0.0]),
    )
    forecast = interaction_aware_cv_baseline(state, horizons_s=(1.0,), neighbors=[neighbor])
    cv_forecast = constant_velocity_gaussian_baseline(state, horizons_s=(1.0,))

    # Mean should be deflected away from neighbor (repulsion is toward -y and -x relative to neighbor)
    pred = forecast.predictions[0]
    assert pred.metadata["interaction_status"] == "repulsion_active"
    assert pred.metadata["active_neighbor_count"] == 1.0
    # Repulsion pushes ego away from neighbor, so mean x < cv mean x
    assert pred.mean[0] < cv_forecast.predictions[0].mean[0]
    # Uncertainty increases due to crowding
    assert pred.metadata["std_m"] > cv_forecast.predictions[0].metadata["std_m"]


def test_interaction_aware_cv_far_neighbor_no_effect() -> None:
    """A neighbor outside the interaction radius has no effect."""
    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
    )
    far_neighbor = NeighborContext(
        position=np.array([10.0, 10.0]),
        velocity=np.array([0.0, 0.0]),
    )
    forecast = interaction_aware_cv_baseline(state, horizons_s=(1.0,), neighbors=[far_neighbor])
    cv_forecast = constant_velocity_gaussian_baseline(state, horizons_s=(1.0,))

    np.testing.assert_allclose(forecast.predictions[0].mean, cv_forecast.predictions[0].mean)
    assert forecast.predictions[0].metadata["interaction_status"] == "no_neighbors_in_radius"


def test_interaction_aware_cv_multiple_neighbors_increase_crowding() -> None:
    """More nearby neighbors increase uncertainty more than fewer."""
    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
    )
    one_neighbor = [NeighborContext(position=np.array([0.5, 0.0]), velocity=np.zeros(2))]
    three_neighbors = [
        NeighborContext(position=np.array([0.5, 0.0]), velocity=np.zeros(2)),
        NeighborContext(position=np.array([-0.3, 0.4]), velocity=np.zeros(2)),
        NeighborContext(position=np.array([0.2, -0.5]), velocity=np.zeros(2)),
    ]

    forecast_one = interaction_aware_cv_baseline(state, horizons_s=(1.0,), neighbors=one_neighbor)
    forecast_many = interaction_aware_cv_baseline(
        state, horizons_s=(1.0,), neighbors=three_neighbors
    )

    assert (
        forecast_many.predictions[0].metadata["std_m"]
        > forecast_one.predictions[0].metadata["std_m"]
    )
    assert forecast_many.predictions[0].metadata["active_neighbor_count"] == 3.0


def test_interaction_aware_cv_is_deterministic() -> None:
    """Interaction-aware forecast is deterministic for fixed inputs."""
    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
    )
    neighbors = [NeighborContext(position=np.array([0.5, 0.0]), velocity=np.zeros(2))]

    first = interaction_aware_cv_baseline(state, horizons_s=(0.5, 1.0), neighbors=neighbors)
    second = interaction_aware_cv_baseline(state, horizons_s=(0.5, 1.0), neighbors=neighbors)

    for left, right in zip(first.predictions, second.predictions, strict=True):
        np.testing.assert_allclose(left.mean, right.mean)
        np.testing.assert_allclose(left.covariance, right.covariance)


def test_interaction_aware_cv_composes_with_semantic_context() -> None:
    """Interaction-aware forecast respects signal/intent context when provided."""
    state_with_context = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        intent="crossing",
        signal="green",
        signal_available=True,
    )
    state_no_context = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        intent=None,
        signal=None,
        signal_available=False,
    )
    neighbors = [NeighborContext(position=np.array([0.5, 0.0]), velocity=np.zeros(2))]

    forecast_ctx = interaction_aware_cv_baseline(
        state_with_context, horizons_s=(1.0,), neighbors=neighbors
    )
    forecast_no_ctx = interaction_aware_cv_baseline(
        state_no_context, horizons_s=(1.0,), neighbors=neighbors
    )

    assert forecast_ctx.predictions[0].metadata["is_intent_aware"] is True
    assert forecast_ctx.predictions[0].metadata["is_signal_aware"] is True
    assert forecast_no_ctx.predictions[0].metadata["is_intent_aware"] is False
    assert forecast_no_ctx.predictions[0].metadata["is_signal_aware"] is False


def test_compute_batch_forecast_metrics_with_interaction_aware_baseline() -> None:
    """Batch metrics accept interaction_aware baseline via the registry."""
    dt_s = 0.5
    trace_steps = []
    for index in range(5):
        x = index * dt_s
        trace_steps.append(
            {
                "step": index,
                "time_s": x,
                "pedestrians": [
                    {
                        "id": 1,
                        "position": [x, 0.0],
                        "velocity": [1.0, 0.0],
                    },
                    {
                        "id": 2,
                        "position": [x, 1.0],
                        "velocity": [0.5, 0.0],
                    },
                ],
            }
        )

    summary = compute_batch_forecast_metrics(
        trace_steps,
        horizons_s=(0.5, 1.0),
        dt_s=dt_s,
        baseline_function=interaction_aware_cv_baseline,
    )

    assert summary["forecast_evaluable_samples"] > 0
    assert summary["mean_ade_0.5s"] >= 0.0


def test_risk_filtered_without_robot_matches_cv() -> None:
    """Risk-filtered baseline degrades to CV when no robot position is supplied."""

    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
    )
    risk_forecast = risk_filtered_cv_baseline(state, horizons_s=(1.0,))
    cv_forecast = constant_velocity_gaussian_baseline(state, horizons_s=(1.0,))

    assert risk_forecast.predictions[0].metadata["model"] == "risk_filtered_cv"
    assert risk_forecast.predictions[0].metadata["relevance_status"] == "robot_unavailable"
    np.testing.assert_allclose(risk_forecast.predictions[0].mean, cv_forecast.predictions[0].mean)
    np.testing.assert_allclose(
        risk_forecast.predictions[0].covariance,
        cv_forecast.predictions[0].covariance,
    )
    assert risk_forecast.predictions[0].metadata["std_m"] == pytest.approx(
        cv_forecast.predictions[0].metadata["std_m"]
    )


def test_risk_filtered_near_robot_stays_relevant() -> None:
    """A prediction close to the robot is marked collision_relevant."""

    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([0.5, 0.0]),
    )
    robot_position = np.array([0.6, 0.0])
    forecast = risk_filtered_cv_baseline(
        state, horizons_s=(1.0,), robot_position=robot_position, risk_distance_m=1.0
    )

    assert forecast.predictions[0].metadata["relevance_status"] == "collision_relevant"
    cv_forecast = constant_velocity_gaussian_baseline(state, horizons_s=(1.0,))
    np.testing.assert_allclose(
        forecast.predictions[0].covariance,
        cv_forecast.predictions[0].covariance,
    )
    assert forecast.predictions[0].metadata["std_m"] == pytest.approx(
        cv_forecast.predictions[0].metadata["std_m"]
    )


def test_risk_filtered_far_robot_gets_widened() -> None:
    """A prediction far from the robot is filtered and covariance is widened."""

    state = PedestrianState(
        id=1,
        position=np.array([0.0, 0.0]),
        velocity=np.array([5.0, 0.0]),
    )
    robot_position = np.array([0.0, 0.0])
    forecast = risk_filtered_cv_baseline(
        state, horizons_s=(1.0,), robot_position=robot_position, risk_distance_m=1.0
    )

    assert forecast.predictions[0].metadata["relevance_status"] == "filtered_low_relevance"
    cv_forecast = constant_velocity_gaussian_baseline(state, horizons_s=(1.0,))
    assert forecast.predictions[0].metadata["std_m"] > cv_forecast.predictions[0].metadata["std_m"]
