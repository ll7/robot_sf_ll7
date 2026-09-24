"""Contract tests for planner-visible force-residual prediction."""

from __future__ import annotations

import ast
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from robot_sf.nav.force_residual_intent_predictor import (
    ForceResidualIntentConfig,
    ForceResidualIntentPredictor,
    ObstacleSegment,
    RobotKinematics,
    StaticGeometry,
)
from robot_sf.planner.scenario_belief_adapter import (
    TRACK_EXISTENCE_PROBABILITY_SEMANTICS,
    PlannerTrackBelief,
)


def _track(  # noqa: PLR0913
    step: int,
    *,
    track_id: int = 1,
    position: tuple[float, float] = (0.0, 0.0),
    velocity: tuple[float, float] = (1.0, 0.0),
    namespace: str = "tracker",
    reset_epoch: int = 0,
    visibility: bool = True,
    missed_steps: int = 0,
    confidence: float = 1.0,
    covariance_scale: float = 0.1,
    age_steps: int | None = None,
) -> PlannerTrackBelief:
    """Construct an identity-safe, tracker-owned planner input."""
    return PlannerTrackBelief(
        track_id=track_id,
        mean_state=np.asarray((*position, *velocity, 0.3), dtype=float),
        covariance=np.eye(4, dtype=float) * covariance_scale,
        confidence=confidence,
        existence_probability=1.0,
        existence_probability_calibrated=False,
        existence_probability_semantics=TRACK_EXISTENCE_PROBABILITY_SEMANTICS,
        visibility=visibility,
        age_steps=max(step + 1, missed_steps) if age_steps is None else age_steps,
        source="test_tracker",
        tracker_namespace=namespace,
        reset_epoch=reset_epoch,
        lifecycle_token=json.dumps([namespace, reset_epoch, track_id], separators=(",", ":")),
        belief_step=step,
        last_observed_step=max(0, step - missed_steps),
        missed_steps=missed_steps,
        status="confirmed" if visibility else "lost",
    )


def _predictor(**config_overrides: object) -> ForceResidualIntentPredictor:
    return ForceResidualIntentPredictor(ForceResidualIntentConfig(**config_overrides))


def _robot(position: tuple[float, float] = (100.0, 100.0)) -> RobotKinematics:
    return RobotKinematics(position_xy=position, velocity_xy=(0.0, 0.0), radius_m=0.25)


def _update(
    predictor: ForceResidualIntentPredictor,
    step: int,
    tracks: dict[int, PlannerTrackBelief],
    *,
    geometry: StaticGeometry | None = None,
    robot: RobotKinematics | None = None,
) -> object:
    return predictor.update_and_predict(
        step=step,
        dt=0.1,
        tracks=tracks,
        robot_state=robot or _robot(),
        static_geometry=geometry or StaticGeometry(),
    )


def test_empty_input_emits_valid_empty_horizon() -> None:
    prediction = _update(_predictor(prediction_horizon_steps=3), 0, {})

    assert prediction.forecasts == {}
    assert prediction.horizon == 3
    assert prediction.prediction_horizon == pytest.approx(0.3)
    assert prediction.metadata["force_model"]["source"] == "analytic_estimator"
    assert prediction.metadata["force_model"]["simulator_parity_claim"] is False


def test_first_observation_is_cv_only_and_uses_identity_provenance() -> None:
    predictor = _predictor(prediction_horizon_steps=2)
    prediction = _update(predictor, 0, {1: _track(0)})
    forecast = prediction.forecasts[1]
    diagnostic = predictor.diagnostics["tracks"]["1"]

    assert [mode.mode_id for mode in forecast.modes] == ["constant_velocity"]
    np.testing.assert_allclose(forecast.modes[0].mean, [[0.1, 0.0], [0.2, 0.0]])
    assert forecast.metadata["identity_token"] == '["tracker",0,1]'
    assert forecast.existence_probability == 1.0
    assert diagnostic["tracker_age_steps"] == 1
    assert diagnostic["observation_ages_steps"] == [0]
    assert diagnostic["observation_ages_s"] == pytest.approx([0.0])
    assert diagnostic["measured_acceleration_mps2"] is None
    assert diagnostic["measured_acceleration_status"] == "insufficient_history"
    assert diagnostic["interaction_acceleration_mps2"] == pytest.approx([0.0, 0.0])
    assert diagnostic["interaction_acceleration_status"] == "available"
    assert diagnostic["residual_acceleration_mps2"] is None
    assert diagnostic["inferred_desired_velocity_mps"] is None
    assert diagnostic["acceleration_clipped"] is None
    assert diagnostic["desired_speed_clipped"] is None


def test_two_observations_enable_residual_and_predict_then_update() -> None:
    predictor = _predictor(prediction_horizon_steps=2)
    _update(predictor, 0, {1: _track(0, velocity=(0.5, 0.0))})
    prediction = _update(
        predictor,
        1,
        {1: _track(1, position=(0.05, 0.0), velocity=(0.5, 0.0))},
    )
    modes = prediction.forecasts[1].modes
    track_diagnostics = predictor.diagnostics["tracks"]["1"]

    assert {mode.mode_id for mode in modes} == {
        "constant_velocity",
        "force_residual_desired_motion",
    }
    assert sum(mode.probability for mode in modes) == pytest.approx(1.0)
    assert track_diagnostics["mode_update"]["constant_velocity"]["update_status"] == (
        "predict_then_update"
    )
    assert track_diagnostics["force_model"]["source"] == "analytic_estimator"
    assert np.isfinite(track_diagnostics["inferred_desired_velocity_mps"]).all()
    assert track_diagnostics["observation_ages_steps"] == [1, 0]
    assert track_diagnostics["observation_ages_s"] == pytest.approx([0.1, 0.0])
    assert track_diagnostics["measured_acceleration_status"] == "valid"
    assert track_diagnostics["interaction_acceleration_status"] == "available"
    assert track_diagnostics["residual_acceleration_status"] == "available"
    assert track_diagnostics["inferred_desired_velocity_status"] == "valid"
    assert isinstance(track_diagnostics["acceleration_clipped"], bool)
    assert isinstance(track_diagnostics["desired_speed_clipped"], bool)


def test_known_synthetic_acceleration_recovers_desired_velocity() -> None:
    predictor = _predictor(
        pedestrian_interaction_strength_mps2=0.0,
        robot_interaction_strength_mps2=0.0,
        obstacle_interaction_strength_mps2=0.0,
    )
    initial_velocity = np.asarray((0.5, -0.2))
    acceleration = np.asarray((0.4, -0.2))
    next_velocity = initial_velocity + 0.1 * acceleration
    _update(predictor, 0, {1: _track(0, velocity=tuple(initial_velocity))})
    _update(
        predictor,
        1,
        {
            1: _track(
                1,
                position=tuple(0.1 * initial_velocity),
                velocity=tuple(next_velocity),
            )
        },
    )

    inferred = np.asarray(predictor.diagnostics["tracks"]["1"]["inferred_desired_velocity_mps"])
    np.testing.assert_allclose(inferred, next_velocity + 0.6 * acceleration, atol=1e-6)


@pytest.mark.parametrize("interaction", ["pedestrian", "obstacle", "robot"])
def test_interaction_acceleration_is_subtracted_with_correct_sign(interaction: str) -> None:
    config = {
        "pedestrian_interaction_strength_mps2": 0.0,
        "obstacle_interaction_strength_mps2": 0.0,
        "robot_interaction_strength_mps2": 0.0,
    }
    config[f"{interaction}_interaction_strength_mps2"] = 1.0
    predictor = _predictor(**config)
    robot = RobotKinematics((1.0, 0.0), (0.0, 0.0), 0.25)
    geometry = StaticGeometry()
    tracks = {1: _track(0, velocity=(0.0, 0.0))}
    if interaction == "pedestrian":
        tracks[2] = _track(0, track_id=2, position=(1.0, 0.0), velocity=(0.0, 0.0))
    elif interaction == "obstacle":
        geometry = StaticGeometry(obstacles=(ObstacleSegment("right", (1.0, -1.0), (1.0, 1.0)),))
    _update(predictor, 0, tracks, geometry=geometry, robot=robot)

    tracks = {
        track_id: _track(
            1,
            track_id=track_id,
            position=tuple(track.mean_state[:2]),
            velocity=(0.0, 0.0),
        )
        for track_id, track in tracks.items()
    }
    _update(predictor, 1, tracks, geometry=geometry, robot=robot)
    diagnostic = predictor.diagnostics["tracks"]["1"]

    assert diagnostic["interaction_terms"][interaction]["acceleration_mps2"][0] < 0.0
    assert diagnostic["residual_acceleration_mps2"][0] > 0.0


def test_combined_interactions_recover_controlled_residual() -> None:
    predictor = _predictor(
        pedestrian_interaction_strength_mps2=0.2,
        obstacle_interaction_strength_mps2=0.2,
        robot_interaction_strength_mps2=0.2,
        interaction_decay_length_m=1.0,
    )
    geometry = StaticGeometry(obstacles=(ObstacleSegment("right", (0.2, -1.0), (0.2, 1.0)),))
    robot = RobotKinematics((0.5, 0.0), (0.0, 0.0), 0.25)
    _update(
        predictor,
        0,
        {
            1: _track(0, position=(0.0, 0.0), velocity=(0.0, 0.0)),
            2: _track(0, track_id=2, position=(0.5, 0.0), velocity=(0.0, 0.0)),
        },
        geometry=geometry,
        robot=robot,
    )
    _update(
        predictor,
        1,
        {
            1: _track(1, position=(0.0, 0.0), velocity=(-0.03, 0.0)),
            2: _track(1, track_id=2, position=(0.5, 0.0), velocity=(0.0, 0.0)),
        },
        geometry=geometry,
        robot=robot,
    )
    diagnostic = predictor.diagnostics["tracks"]["1"]

    np.testing.assert_allclose(diagnostic["interaction_acceleration_mps2"], [-0.6, 0.0])
    np.testing.assert_allclose(diagnostic["measured_acceleration_mps2"], [-0.3, 0.0])
    np.testing.assert_allclose(diagnostic["residual_acceleration_mps2"], [0.3, 0.0])


def test_predict_then_update_favors_observed_residual_mode() -> None:
    predictor = _predictor(
        pedestrian_interaction_strength_mps2=0.0,
        robot_interaction_strength_mps2=0.0,
        obstacle_interaction_strength_mps2=0.0,
        process_noise_position_m2_per_s=0.0,
        process_noise_velocity_m2_per_s=0.0,
        residual_model_noise_m2_per_s=0.0,
        likelihood_covariance_floor_m2=1e-6,
        restart_speed_threshold_mps=0.5,
    )
    _update(
        predictor,
        0,
        {1: _track(0, position=(0.0, 0.0), velocity=(0.0, 0.0), covariance_scale=1e-5)},
    )
    _update(
        predictor,
        1,
        {1: _track(1, position=(0.01, 0.0), velocity=(0.2, 0.0), covariance_scale=1e-5)},
    )
    prediction = _update(
        predictor,
        2,
        {1: _track(2, position=(0.05, 0.0), velocity=(0.4, 0.0), covariance_scale=1e-5)},
    )
    probabilities = {mode.mode_id: mode.probability for mode in prediction.forecasts[1].modes}

    assert probabilities["force_residual_desired_motion"] > probabilities["constant_velocity"]


def test_transition_prior_matrix_changes_mode_posterior() -> None:
    transition_priors = (
        (
            "constant_velocity",
            (
                ("constant_velocity", 0.05),
                ("force_residual_desired_motion", 0.95),
                ("yield_or_decelerate", 0.0),
            ),
        ),
        (
            "force_residual_desired_motion",
            (
                ("constant_velocity", 0.1),
                ("force_residual_desired_motion", 0.85),
                ("yield_or_decelerate", 0.05),
            ),
        ),
        (
            "yield_or_decelerate",
            (
                ("constant_velocity", 0.15),
                ("force_residual_desired_motion", 0.1),
                ("yield_or_decelerate", 0.75),
            ),
        ),
    )
    custom = _predictor(mode_transition_priors=transition_priors)
    default = _predictor()
    for predictor in (custom, default):
        _update(predictor, 0, {1: _track(0, velocity=(0.5, 0.0))})
        _update(predictor, 1, {1: _track(1, position=(0.05, 0.0), velocity=(0.5, 0.0))})
    custom_probabilities = custom.diagnostics["tracks"]["1"]["mode_probabilities"]
    default_probabilities = default.diagnostics["tracks"]["1"]["mode_probabilities"]

    assert (
        custom_probabilities["force_residual_desired_motion"]
        > custom_probabilities["constant_velocity"]
    )
    assert (
        default_probabilities["constant_velocity"]
        > default_probabilities["force_residual_desired_motion"]
    )
    with pytest.raises(ValueError, match="transition prior row must sum to 1"):
        ForceResidualIntentConfig(
            mode_transition_priors=(
                (
                    "constant_velocity",
                    (
                        ("constant_velocity", 0.1),
                        ("force_residual_desired_motion", 0.1),
                        ("yield_or_decelerate", 0.1),
                    ),
                ),
            )
        )


def test_mode_update_diagnostics_include_prediction_innovation_and_floor_status() -> None:
    predictor = _predictor()
    _update(predictor, 0, {1: _track(0, velocity=(0.5, 0.0))})
    _update(predictor, 1, {1: _track(1, position=(0.05, 0.0), velocity=(0.5, 0.0))})
    mode_diagnostics = predictor.diagnostics["tracks"]["1"]["mode_update"]
    cv = mode_diagnostics["constant_velocity"]
    new_residual = mode_diagnostics["force_residual_desired_motion"]

    assert cv["one_step_predicted_mean"] == pytest.approx([0.05, 0.0, 0.5, 0.0])
    assert cv["one_step_predicted_position"] == pytest.approx([0.05, 0.0])
    assert cv["one_step_predicted_velocity"] == pytest.approx([0.5, 0.0])
    assert cv["innovation"] == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert cv["position_innovation"] == pytest.approx([0.0, 0.0])
    assert cv["velocity_innovation"] == pytest.approx([0.0, 0.0])
    assert cv["observation_dimension"] == 4
    assert np.asarray(cv["innovation_covariance"]).shape == (4, 4)
    assert isinstance(cv["log_likelihood"], float)
    assert isinstance(cv["probability_floor_applied"], bool)
    assert cv["state_smoothing_applied"] is False
    assert new_residual["one_step_predicted_mean"] is None
    assert new_residual["one_step_predicted_velocity"] is None
    assert new_residual["prediction_status"] == "no_previous_mode_prediction"


def test_velocity_observation_contributes_to_mode_likelihood() -> None:
    likelihoods: list[float] = []
    velocity_innovations: list[list[float]] = []
    for observed_velocity in (0.5, 0.9):
        predictor = _predictor()
        _update(predictor, 0, {1: _track(0, velocity=(0.5, 0.0))})
        _update(
            predictor,
            1,
            {
                1: _track(
                    1,
                    position=(0.05, 0.0),
                    velocity=(observed_velocity, 0.0),
                )
            },
        )
        cv = predictor.diagnostics["tracks"]["1"]["mode_update"]["constant_velocity"]
        likelihoods.append(cv["log_likelihood"])
        velocity_innovations.append(cv["velocity_innovation"])

    assert likelihoods[0] != pytest.approx(likelihoods[1])
    assert velocity_innovations[0] == pytest.approx([0.0, 0.0])
    assert velocity_innovations[1] == pytest.approx([0.4, 0.0])


def test_unavailable_interaction_keeps_acceleration_diagnostics_explicit() -> None:
    predictor = _predictor()
    unavailable_geometry = StaticGeometry(available=False)
    _update(predictor, 0, {1: _track(0)}, geometry=unavailable_geometry)
    first = predictor.diagnostics["tracks"]["1"]
    assert first["measured_acceleration_mps2"] is None
    assert first["measured_acceleration_status"] == "insufficient_history"
    assert first["interaction_acceleration_mps2"] is None
    assert first["interaction_acceleration_status"] == "unavailable"
    assert first["residual_acceleration_mps2"] is None
    assert first["residual_acceleration_status"] == "insufficient_history"
    assert first["inferred_desired_velocity_mps"] is None
    assert first["inferred_desired_velocity_status"] == "unavailable"
    assert first["observation_ages_steps"] == [0]
    assert first["observation_ages_s"] == pytest.approx([0.0])

    _update(
        predictor,
        1,
        {1: _track(1, position=(0.1, 0.0))},
        geometry=unavailable_geometry,
    )
    later = predictor.diagnostics["tracks"]["1"]
    assert later["measured_acceleration_mps2"] == pytest.approx([0.0, 0.0])
    assert later["measured_acceleration_status"] == "valid"
    assert later["interaction_acceleration_mps2"] is None
    assert later["interaction_acceleration_status"] == "unavailable"
    assert later["residual_acceleration_mps2"] is None
    assert later["residual_acceleration_status"] == "interaction_unavailable"
    assert later["inferred_desired_velocity_mps"] is None
    assert later["inferred_desired_velocity_status"] == "unavailable"
    assert later["observation_ages_steps"] == [1, 0]
    assert later["observation_ages_s"] == pytest.approx([0.1, 0.0])


def test_velocity_covariance_floor_is_dimensioned_and_config_hashed() -> None:
    position_floor = ForceResidualIntentConfig(likelihood_covariance_floor_m2=0.2)
    velocity_floor = ForceResidualIntentConfig(
        likelihood_covariance_floor_m2=0.2,
        likelihood_velocity_covariance_floor_m2_per_s2=0.3,
    )
    assert position_floor.config_hash != velocity_floor.config_hash
    with pytest.raises(ValueError, match="likelihood_velocity_covariance_floor_m2_per_s2"):
        ForceResidualIntentConfig(likelihood_velocity_covariance_floor_m2_per_s2=0.0)

    covariances: list[np.ndarray] = []
    for velocity_floor_value in (0.3, 0.5):
        predictor = _predictor(
            process_noise_position_m2_per_s=0.0,
            process_noise_velocity_m2_per_s=0.0,
            residual_model_noise_m2_per_s=0.0,
            likelihood_covariance_floor_m2=0.2,
            likelihood_velocity_covariance_floor_m2_per_s2=velocity_floor_value,
        )
        _update(predictor, 0, {1: _track(0, covariance_scale=0.1)})
        _update(predictor, 1, {1: _track(1, position=(0.1, 0.0), covariance_scale=0.1)})
        covariances.append(
            np.asarray(
                predictor.diagnostics["tracks"]["1"]["mode_update"]["constant_velocity"][
                    "innovation_covariance"
                ]
            )
        )
    covariance_difference = covariances[1] - covariances[0]
    np.testing.assert_allclose(covariance_difference[:2], 0.0, atol=1e-12)
    np.testing.assert_allclose(covariance_difference[:, :2], 0.0, atol=1e-12)
    np.testing.assert_allclose(np.diag(covariance_difference), [0.0, 0.0, 0.2, 0.2])


def test_tiny_likelihoods_still_normalize_without_nan() -> None:
    predictor = _predictor(likelihood_collapse_threshold=-1e300)
    _update(predictor, 0, {1: _track(0, covariance_scale=1e-5)})
    prediction = _update(
        predictor,
        1,
        {1: _track(1, position=(100.0, 0.0), covariance_scale=1e-5)},
    )
    probabilities = [mode.probability for mode in prediction.forecasts[1].modes]

    assert all(np.isfinite(probabilities))
    assert sum(probabilities) == pytest.approx(1.0)


def test_history_length_and_track_order_are_bounded_and_deterministic() -> None:
    config = ForceResidualIntentConfig(history_steps=2, prediction_horizon_steps=2)
    forward = ForceResidualIntentPredictor(config)
    reverse = ForceResidualIntentPredictor(config)
    for step in range(4):
        tracks = {
            track_id: _track(
                step,
                track_id=track_id,
                position=(step * 0.05, float(track_id)),
                velocity=(0.5 + step * 0.1, 0.0),
            )
            for track_id in (1, 2)
        }
        first = _update(forward, step, tracks)
        second = _update(reverse, step, dict(reversed(tuple(tracks.items()))))
        assert first.to_dict() == second.to_dict()

    assert forward.diagnostics["tracks"]["1"]["history_length"] == 2


def test_reset_seed_reproduces_the_same_first_forecast() -> None:
    predictor = _predictor()
    _update(predictor, 0, {1: _track(0)})
    predictor.reset(seed=42)
    first = _update(predictor, 0, {1: _track(0)}).to_dict()
    predictor.reset(seed=42)
    second = _update(predictor, 0, {1: _track(0)}).to_dict()

    assert first == second


@pytest.mark.parametrize(
    ("config", "old_velocity", "new_velocity", "expected_reason"),
    [
        (
            {"heading_change_reset_rad": 1.0},
            (1.0, 0.0),
            (0.0, 1.0),
            "heading_change",
        ),
        ({}, (0.0, 0.0), (0.3, 0.0), "stop_restart"),
    ],
)
def test_observable_motion_changes_reset_track_state(
    config: dict[str, object],
    old_velocity: tuple[float, float],
    new_velocity: tuple[float, float],
    expected_reason: str,
) -> None:
    predictor = _predictor(**config)
    _update(predictor, 0, {1: _track(0, velocity=old_velocity)})
    prediction = _update(
        predictor,
        1,
        {1: _track(1, position=(0.1, 0.0), velocity=new_velocity)},
    )

    assert prediction.forecasts[1].metadata["reset_reason"] == expected_reason


def test_likelihood_collapse_and_long_occlusion_have_stable_reset_reasons() -> None:
    collapse = _predictor(likelihood_collapse_threshold=0.0)
    _update(collapse, 0, {1: _track(0)})
    collapsed = _update(collapse, 1, {1: _track(1, position=(0.1, 0.0))})
    assert collapsed.forecasts[1].metadata["reset_reason"] == "likelihood_collapse"

    occlusion = _predictor(max_coast_steps_for_intent=1)
    _update(occlusion, 0, {1: _track(0)})
    _update(occlusion, 1, {1: _track(1, visibility=False, missed_steps=1)})
    reappeared = _update(occlusion, 2, {1: _track(2, missed_steps=2)})
    assert reappeared.forecasts[1].metadata["reset_reason"] == "reappearance_after_gap"


def test_occluded_mode_update_reports_coasting_without_measurement_scoring() -> None:
    predictor = _predictor(max_coast_steps_for_intent=2)
    _update(predictor, 0, {1: _track(0)})
    _update(predictor, 1, {1: _track(1, position=(0.1, 0.0))})
    _update(predictor, 2, {1: _track(2, visibility=False, missed_steps=1)})

    for mode in predictor.diagnostics["tracks"]["1"]["mode_update"].values():
        assert mode["update_status"] == "coasted_no_measurement_update"
        assert mode["prediction_status"] == "no_measurement_to_score"


def test_invalid_timestep_is_rejected_with_diagnostic_and_no_state_change() -> None:
    predictor = _predictor()
    _update(predictor, 0, {1: _track(0)})
    before = predictor.snapshot_state()

    with pytest.raises(ValueError, match="invalid_dt"):
        predictor.update_and_predict(
            step=1,
            dt=float("nan"),
            tracks={1: _track(1)},
            robot_state=_robot(),
            static_geometry=StaticGeometry(),
        )

    assert predictor.snapshot_state() == before
    assert predictor.diagnostics["status"] == "rejected"


def test_duplicate_and_out_of_order_updates_fail_without_state_mutation() -> None:
    predictor = _predictor()
    _update(predictor, 0, {1: _track(0)})
    before = predictor.snapshot_state()

    with pytest.raises(ValueError, match="duplicate_step"):
        _update(predictor, 0, {1: _track(0, position=(50.0, 0.0))})
    assert predictor.snapshot_state() == before

    _update(predictor, 2, {1: _track(2, position=(0.2, 0.0))})
    with pytest.raises(ValueError, match="out_of_order_step"):
        _update(predictor, 1, {1: _track(1)})


def test_track_epoch_change_resets_history_and_posterior() -> None:
    predictor = _predictor()
    _update(predictor, 0, {1: _track(0)})
    _update(predictor, 1, {1: _track(1, position=(0.1, 0.0))})

    prediction = _update(
        predictor,
        2,
        {1: _track(2, position=(0.2, 0.0), reset_epoch=1)},
    )

    assert prediction.forecasts[1].metadata["reset_reason"] == "identity_generation_changed"
    assert predictor.diagnostics["reset_events"] == [
        {
            "track_id": 1,
            "identity_token": '["tracker",1,1]',
            "reason": "identity_generation_changed",
        }
    ]
    assert predictor.diagnostics["tracks"]["1"]["history_length"] == 1


def test_snapshot_restore_replays_identical_prediction() -> None:
    config = ForceResidualIntentConfig(prediction_horizon_steps=3)
    original = ForceResidualIntentPredictor(config)
    _update(original, 0, {1: _track(0)})
    snapshot = original.snapshot_state()
    assert json.loads(json.dumps(snapshot)) == snapshot

    next_tracks = {1: _track(1, position=(0.1, 0.0))}
    expected = _update(original, 1, next_tracks).to_dict()
    restored = ForceResidualIntentPredictor(config)
    restored.restore_state(snapshot)
    actual = _update(restored, 1, next_tracks).to_dict()

    assert actual == expected
    with pytest.raises(ValueError, match="snapshot_config_hash_mismatch"):
        ForceResidualIntentPredictor(ForceResidualIntentConfig(prediction_dt_s=0.2)).restore_state(
            snapshot
        )


def test_prediction_covariance_remains_psd_and_grows_with_uncertainty() -> None:
    config = ForceResidualIntentConfig(prediction_horizon_steps=2)
    nominal = (
        _update(ForceResidualIntentPredictor(config), 0, {1: _track(0, covariance_scale=0.1)})
        .forecasts[1]
        .modes[0]
        .covariance
    )
    assert np.all(np.linalg.eigvalsh(nominal) >= -1e-7)
    for uncertainty in (
        {"covariance_scale": 0.2},
        {"confidence": 0.5},
        {"visibility": False, "missed_steps": 1},
    ):
        forecast_covariance = (
            _update(
                ForceResidualIntentPredictor(config),
                0,
                {1: _track(0, **uncertainty)},
            )
            .forecasts[1]
            .modes[0]
            .covariance
        )
        assert np.all(np.linalg.eigvalsh(forecast_covariance) >= -1e-7)
        assert np.all(np.linalg.eigvalsh(forecast_covariance - nominal) >= -1e-6)

    older_tenure = (
        _update(
            ForceResidualIntentPredictor(config),
            0,
            {1: _track(0, age_steps=20)},
        )
        .forecasts[1]
        .modes[0]
        .covariance
    )
    # Tracker age is tenure; missed_steps/visibility describe staleness and drive inflation.
    np.testing.assert_allclose(older_tenure, nominal)


def test_pruned_track_id_reuse_with_new_lifecycle_does_not_inherit_posterior() -> None:
    predictor = _predictor(max_coast_steps_for_intent=1)
    _update(predictor, 0, {1: _track(0, velocity=(0.5, 0.0))})
    _update(predictor, 1, {1: _track(1, position=(0.05, 0.0), velocity=(0.5, 0.0))})
    _update(predictor, 2, {})
    _update(predictor, 3, {})

    reused = _update(
        predictor,
        4,
        {1: _track(4, position=(10.0, 0.0), reset_epoch=1)},
    )
    forecast = reused.forecasts[1]

    assert forecast.metadata["identity_token"] == '["tracker",1,1]'
    assert forecast.metadata["reset_reason"] == "initial_track"
    assert [mode.mode_id for mode in forecast.modes] == ["constant_velocity"]
    assert predictor.diagnostics["tracks"]["1"]["history_length"] == 1


def test_unavailable_geometry_and_degenerate_overlap_are_explicit() -> None:
    unavailable = _predictor()
    _update(
        unavailable,
        0,
        {1: _track(0)},
        geometry=StaticGeometry(available=False),
    )
    assert (
        unavailable.diagnostics["tracks"]["1"]["interaction_terms"]["obstacle"]["status"]
        == "unavailable_geometry"
    )

    overlapping = _predictor()
    _update(
        overlapping,
        0,
        {1: _track(0), 2: _track(0, track_id=2)},
        geometry=StaticGeometry(obstacles=(ObstacleSegment("at_track", (0.0, 0.0), (0.0, 0.0)),)),
    )
    terms = overlapping.diagnostics["tracks"]["1"]["interaction_terms"]
    assert terms["pedestrian"]["status"] == "unavailable_degenerate_geometry"
    assert terms["obstacle"]["status"] == "unavailable_degenerate_geometry"


def test_yield_mode_is_opt_in_and_decelerates_without_reversing() -> None:
    default_prediction = _update(_predictor(), 0, {1: _track(0)}).forecasts[1]
    assert all(mode.mode_id != "yield_or_decelerate" for mode in default_prediction.modes)

    predictor = _predictor(yield_mode_enabled=True)
    prediction = _update(predictor, 0, {1: _track(0, velocity=(0.5, 0.0))})
    yield_mode = next(
        mode for mode in prediction.forecasts[1].modes if mode.mode_id == "yield_or_decelerate"
    )

    assert np.all(np.diff(yield_mode.mean[:, 0]) >= 0.0)
    assert yield_mode.mean[-1, 0] < 0.5


def test_inputs_are_frozen_and_module_has_no_simulator_oracle_imports() -> None:
    config = ForceResidualIntentConfig()
    with pytest.raises(FrozenInstanceError):
        config.history_steps = 4  # type: ignore[misc]

    module_path = Path(__file__).parents[1] / "robot_sf/nav/force_residual_intent_predictor.py"
    module = ast.parse(module_path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert not any(
        name.startswith(("robot_sf.sim", "robot_sf.gym_env", "robot_sf.benchmark"))
        for name in imported
    )


def test_track_and_obstacle_work_caps_are_enforced() -> None:
    predictor = _predictor(max_tracks=1, max_static_obstacles=0)
    with pytest.raises(ValueError, match="track_limit_exceeded"):
        _update(predictor, 0, {1: _track(0), 2: _track(0, track_id=2)})
    with pytest.raises(ValueError, match="static_obstacle_limit_exceeded"):
        _update(
            predictor,
            0,
            {1: _track(0)},
            geometry=StaticGeometry(obstacles=(ObstacleSegment("wall", (1.0, 0.0), (1.0, 1.0)),)),
        )
