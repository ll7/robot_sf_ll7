"""Focused contract tests for the observation-only inverse-force slice."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from robot_sf.prediction import (
    GoalCandidateProvider,
    GoalCandidateProviderConfig,
    GoalCandidateSet,
    GoalCandidateSource,
    GoalForceInformationMode,
    GoalForceInverseConfig,
    GoalForceInverseEstimator,
    GoalForceObservation,
    GoalForceTrackingAdapter,
    ObservableForceComponent,
    PublicGoalCandidateRecord,
    reconstruct_observable_force,
)
from robot_sf.sensor.pedestrian_tracking import (
    PedestrianObservationSnapshot,
    PedestrianTracker,
    PedestrianTrackingConfig,
)


def _observation(
    timestamp_s: float,
    step_index: int,
    velocity: tuple[float, float],
    *,
    confidence: float = 1.0,
    blockers: tuple[str, ...] = (),
    status: str = "confirmed",
) -> GoalForceObservation:
    """Build one deterministic global-frame observation row."""

    return GoalForceObservation(
        track_id="track-1",
        tracking_epoch_id="epoch-1",
        timestamp_s=timestamp_s,
        step_index=step_index,
        position_xy=(timestamp_s * velocity[0], timestamp_s * velocity[1]),
        velocity_xy=velocity,
        confidence=confidence,
        blockers=blockers,
        status=status,
    )


def _complete_components(
    social: tuple[float, float] = (0.0, 0.0),
) -> tuple[ObservableForceComponent, ...]:
    """Declare every configured non-goal family, including known zeroes."""

    values = {
        "social": social,
        "obstacle": (0.0, 0.0),
        "pedestrian_robot": (0.0, 0.0),
        "group": (0.0, 0.0),
        "adversarial": (0.0, 0.0),
    }
    return tuple(
        ObservableForceComponent(
            component_id=component_type, component_type=component_type, force_xy=force
        )
        for component_type, force in values.items()
    )


def _config(**overrides: object) -> GoalForceInverseConfig:
    """Return an enabled fixture configuration."""

    values: dict[str, object] = {
        "enabled": True,
        "expected_force_component_types": (
            "social",
            "obstacle",
            "pedestrian_robot",
            "group",
            "adversarial",
        ),
        "preferred_speed_mps": 1.3,
    }
    values.update(overrides)
    return GoalForceInverseConfig(**values)


def test_config_is_default_off_immutable_and_strict() -> None:
    """The opt-in estimator cannot activate through an accidental mutation."""

    config = GoalForceInverseConfig()
    assert config.enabled is False
    assert len(config.config_hash) == 64
    with pytest.raises(FrozenInstanceError):
        config.enabled = True  # type: ignore[misc]
    with pytest.raises(ValueError):
        GoalForceInverseConfig.from_mapping({"enabled": True, "unknown": 1})
    parsed = GoalForceInverseConfig.from_mapping(
        {"schema_version": "goal_force_inverse.v1", "enabled": True}
    )
    assert parsed.enabled is True


def test_force_reconstruction_requires_explicit_zero_or_reports_missing() -> None:
    """Omitted force families become partial uncertainty rather than zero force."""

    complete = reconstruct_observable_force(
        _complete_components((0.2, -0.1)),
    )
    assert complete.mode is GoalForceInformationMode.OBSERVATION_RECONSTRUCTED
    assert complete.total_force_xy == pytest.approx((0.2, -0.1))
    assert complete.blockers == ()

    partial = reconstruct_observable_force(
        (ObservableForceComponent("social", "social", (0.2, 0.0)),),
    )
    assert partial.mode is GoalForceInformationMode.PARTIAL_OBSERVATION
    assert partial.total_force_xy == pytest.approx((0.2, 0.0))
    assert "force_component_missing:obstacle" in partial.blockers
    assert "non_goal_force_components_incomplete" in partial.blockers


def test_h1_heading_baseline_is_observation_only_and_has_no_force_claim() -> None:
    """One frame exposes heading evidence while explicitly withholding acceleration."""

    candidate_set = GoalCandidateSet.from_points(
        {"east": (10.0, 0.0), "north": (0.0, 10.0)},
        source="public_fixture",
    )
    estimate = GoalForceInverseEstimator(_config(history_length=1)).estimate(
        (_observation(0.0, 0, (1.0, 0.0)),),
        candidate_set=candidate_set,
    )

    assert estimate.mode is GoalForceInformationMode.HEADING_BASELINE
    assert estimate.force_estimate is None
    assert estimate.desired_direction_rad == pytest.approx(0.0)
    assert estimate.belief is not None
    assert estimate.belief.source.value == "observation_only"
    assert estimate.belief.force_estimate is None
    assert estimate.belief.candidate_probabilities[0].probability > 0.0


def test_candidate_provider_result_is_narrowed_to_actor_candidate_set() -> None:
    """The #8073 provenance envelope can feed H=2 without exposing oracle fields."""

    provider = GoalCandidateProvider(
        GoalCandidateProviderConfig(
            enabled_sources=(GoalCandidateSource.MAP_DESTINATION_ZONE,),
            unknown_enabled=True,
        )
    )
    generation = provider.generate(
        (
            PublicGoalCandidateRecord(
                source=GoalCandidateSource.MAP_DESTINATION_ZONE,
                source_id="east-exit",
                position=(10.0, 0.0),
            ),
        ),
        observed_position_global=(0.0, 0.0),
    )
    estimate = GoalForceInverseEstimator(_config(history_length=2)).estimate(
        (
            _observation(0.0, 0, (1.0, 0.0)),
            _observation(0.1, 1, (1.1, 0.0)),
        ),
        known_force_components=_complete_components(),
        candidate_set=generation,
        max_speed_mps=3.0,
    )

    assert estimate.belief is not None
    assert estimate.belief.source.value == "observation_only"
    provider_candidate_ids = {
        candidate.id
        for candidate in generation.candidate_set.candidates
        if candidate.id != "unknown"
    }
    belief_candidate_ids = {item.candidate_id for item in estimate.belief.candidate_probabilities}
    assert provider_candidate_ids & belief_candidate_ids
    assert estimate.belief.unknown_candidate_probability > 0.0


def test_h2_recovers_goal_force_after_explicit_non_goal_reconstruction() -> None:
    """The finite-difference inverse follows the desired-force algebra."""

    estimator = GoalForceInverseEstimator(
        _config(history_length=2, relaxation_time_s=0.5, desired_force_factor=1.0)
    )
    estimate = estimator.estimate(
        (
            _observation(0.0, 0, (1.0, 0.0)),
            _observation(0.1, 1, (1.2, 0.0)),
        ),
        known_force_components=_complete_components((0.5, 0.0)),
        max_speed_mps=3.0,
    )

    assert estimate.mode is GoalForceInformationMode.OBSERVATION_RECONSTRUCTED
    assert estimate.acceleration_xy == pytest.approx((2.0, 0.0))
    assert estimate.force_estimate is not None
    assert estimate.force_estimate.mean_xy == pytest.approx((1.5, 0.0))
    assert estimate.desired_velocity_xy == pytest.approx((1.75, 0.0))
    assert estimate.inferred_preferred_speed_mps == pytest.approx(1.75)
    assert estimate.to_dict()["inferred_preferred_speed_mps"] == pytest.approx(1.75)
    assert estimate.belief is not None
    assert estimate.belief.mode.value == "nominal"
    assert {term.name for term in estimate.covariance_terms} == {
        "acceleration",
        "known_force",
        "model_mismatch",
        "parameter",
        "tracking",
        "unmodeled_force",
    }


def test_h3_uses_causal_three_frame_fit_and_keeps_covariance_finite() -> None:
    """Three ordered frames recover constant acceleration without future data."""

    estimate = GoalForceInverseEstimator(_config(history_length=3)).estimate(
        (
            _observation(0.0, 0, (1.0, 0.0)),
            _observation(0.1, 1, (1.1, 0.0)),
            _observation(0.2, 2, (1.2, 0.0)),
        ),
        known_force_components=_complete_components(),
        max_speed_mps=3.0,
    )

    assert estimate.mode is GoalForceInformationMode.OBSERVATION_RECONSTRUCTED
    assert estimate.estimator_variant == "h3_causal_linear_fit"
    assert estimate.acceleration_xy == pytest.approx((1.0, 0.0))
    covariance = np.asarray(estimate.force_estimate.covariance_xy)
    assert np.all(np.isfinite(covariance))
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() > 0.0


def test_partial_force_and_low_confidence_increase_uncertainty() -> None:
    """Missing force and weak tracking confidence are visible in covariance."""

    history = (
        _observation(0.0, 0, (1.0, 0.0)),
        _observation(0.1, 1, (1.1, 0.0)),
    )
    complete = GoalForceInverseEstimator(_config(history_length=2)).estimate(
        history,
        known_force_components=_complete_components(),
        max_speed_mps=3.0,
    )
    partial = GoalForceInverseEstimator(_config(history_length=2)).estimate(
        history,
        known_force_components=(ObservableForceComponent("social", "social", (0.0, 0.0)),),
        max_speed_mps=3.0,
    )
    low_confidence = GoalForceInverseEstimator(_config(history_length=2)).estimate(
        (
            _observation(0.0, 0, (1.0, 0.0), confidence=0.2),
            _observation(0.1, 1, (1.1, 0.0), confidence=0.2),
        ),
        known_force_components=_complete_components(),
        max_speed_mps=3.0,
    )

    assert partial.mode is GoalForceInformationMode.PARTIAL_OBSERVATION
    assert partial.belief is not None and partial.belief.mode.value == "censored"
    assert partial.force_estimate.mean_xy == pytest.approx((1.0, 0.0))
    assert partial.force_estimate.covariance_xy[0][0] > complete.force_estimate.covariance_xy[0][0]
    assert (
        low_confidence.force_estimate.covariance_xy[0][0]
        > complete.force_estimate.covariance_xy[0][0]
    )


def test_speed_cap_is_actor_side_censoring_not_exact_cap_truth() -> None:
    """A public max-speed proximity produces a possible, not exact, cap label."""

    estimate = GoalForceInverseEstimator(_config(history_length=2)).estimate(
        (
            _observation(0.0, 0, (1.0, 0.0)),
            _observation(0.1, 1, (1.2, 0.0)),
        ),
        known_force_components=_complete_components(),
        max_speed_mps=1.2,
    )

    assert estimate.speed_cap_status.value == "possible"
    assert estimate.censoring_state.value == "saturated"
    assert "speed_cap_may_censor_transition" in estimate.blockers
    assert estimate.belief is not None and estimate.belief.mode.value == "censored"


def test_braking_residual_does_not_create_a_reversed_goal_direction() -> None:
    """A strong opposite residual is retained as local braking evidence."""

    estimate = GoalForceInverseEstimator(_config(history_length=2)).estimate(
        (
            _observation(0.0, 0, (1.0, 0.0)),
            _observation(0.1, 1, (0.7, 0.0)),
        ),
        known_force_components=_complete_components(),
        max_speed_mps=3.0,
    )

    assert estimate.braking_probability > 0.5
    assert estimate.arrival_probability > 0.0
    assert estimate.desired_velocity_xy == pytest.approx((0.0, 0.0))
    assert estimate.desired_direction_rad == pytest.approx(0.0)
    assert "braking_direction_preserved" in estimate.blockers


def test_tracking_adapter_scopes_history_by_epoch_and_reset() -> None:
    """The stateful bridge cannot carry a prior episode into a new identity epoch."""

    tracker = PedestrianTracker(
        PedestrianTrackingConfig(
            enabled=True,
            process_noise=0.0,
            initial_position_covariance=0.1,
            initial_velocity_covariance=0.1,
            measurement_position_covariance=0.01,
            measurement_velocity_covariance=0.01,
            position_gate_threshold=100.0,
            velocity_gate_threshold=100.0,
            confirmation_steps=1,
            max_missed_seconds=10.0,
            history_capacity=4,
        )
    )
    adapter = GoalForceTrackingAdapter(_config(history_length=2))
    first = tracker.update(
        PedestrianObservationSnapshot(
            timestamp_s=0.0,
            step_index=0,
            coordinate_frame="global_xy",
            positions=np.asarray([[0.0, 0.0]]),
            velocities=np.asarray([[1.0, 0.0]]),
            robot_pose_global=(0.0, 0.0, 0.0),
            valid_mask=np.asarray([True]),
            visible_mask=np.asarray([True]),
        )
    )
    second = tracker.update(
        PedestrianObservationSnapshot(
            timestamp_s=0.1,
            step_index=1,
            coordinate_frame="global_xy",
            positions=np.asarray([[0.1, 0.0]]),
            velocities=np.asarray([[1.1, 0.0]]),
            robot_pose_global=(0.0, 0.0, 0.0),
            valid_mask=np.asarray([True]),
            visible_mask=np.asarray([True]),
        )
    )

    first_estimate = adapter.update(first)[0]
    second_estimate = adapter.update(second)[0]
    assert first_estimate.mode is GoalForceInformationMode.UNAVAILABLE
    assert second_estimate.history_steps[-1].step_index == 1
    assert second_estimate.track_id == "track-1"

    adapter.reset("episode-2")
    reset_estimate = adapter.update(first)[0]
    assert reset_estimate.tracking_epoch_id == "1"
    assert reset_estimate.mode is GoalForceInformationMode.UNAVAILABLE
    assert reset_estimate.belief.tracking_epoch_id == "1"


def test_actor_estimator_rejects_oracle_shaped_force_input() -> None:
    """The actor API accepts only its public contribution type, never oracle records."""

    estimator = GoalForceInverseEstimator(_config(history_length=2))
    history = (_observation(0.0, 0, (1.0, 0.0)), _observation(0.1, 1, (1.1, 0.0)))
    with pytest.raises(TypeError, match="known_force_components"):
        estimator.estimate(history, known_force_components=object())  # type: ignore[arg-type]
