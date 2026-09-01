"""Contract tests for the actor-side goal-belief value (issue #8063)."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from robot_sf.prediction.goal_belief_contract import (
    ACTOR_FORBIDDEN_KEYS,
    ActorObservationStep,
    ActorSpeedCapStatus,
    CensoringState,
    CoordinateFrame,
    ForceEstimate2D,
    GoalBeliefMode,
    GoalBeliefObservation,
    GoalBeliefV1,
    GoalCandidateKind,
    GoalCandidateProbability,
    ObservationMask,
    stable_config_hash,
)

CONFIG_HASH = stable_config_hash({"model": "synthetic", "version": 1})


def _observation(*, track_id: str = "track-1") -> GoalBeliefObservation:
    """Build a finite nominal actor observation for the focused contract tests."""
    return GoalBeliefObservation(
        track_id=track_id,
        tracking_epoch_id="epoch-1",
        timestamp_s=0.1,
        step_index=1,
        config_hash=CONFIG_HASH,
        history_steps=(
            ActorObservationStep(
                timestamp_s=0.0,
                step_index=0,
                position_xy=(0.0, 0.0),
                velocity_xy=(1.0, 0.0),
            ),
            ActorObservationStep(
                timestamp_s=0.1,
                step_index=1,
                position_xy=(0.1, 0.0),
                velocity_xy=(1.0, 0.0),
            ),
        ),
        force_estimate=ForceEstimate2D(
            mean_xy=(0.2, 0.0),
            covariance_xy=((0.25, 0.01), (0.01, 0.5)),
        ),
        desired_velocity_xy=(1.0, 0.0),
        desired_direction_rad=0.0,
        candidate_probabilities=(
            GoalCandidateProbability("final", GoalCandidateKind.FINAL_DESTINATION, 0.3),
            GoalCandidateProbability("active", GoalCandidateKind.ACTIVE_WAYPOINT, 0.6),
        ),
        unknown_candidate_probability=0.1,
        arrival_probability=0.05,
        change_probability=0.2,
        track_confidence=0.9,
    )


def test_nominal_belief_is_finite_immutable_and_round_trips() -> None:
    """The nominal actor record is finite, deterministic, and strict enough to round-trip."""
    belief = GoalBeliefV1.from_observation(_observation())
    payload = belief.to_dict()

    assert payload["schema_version"] == "goal_belief.v1"
    assert payload["tracking_epoch_id"] == "epoch-1"
    assert payload["history_order"] == "oldest_to_newest"
    assert payload["candidate_probabilities"][0]["candidate_id"] == "active"
    assert math.isclose(
        sum(item["probability"] for item in payload["candidate_probabilities"])
        + payload["unknown_candidate_probability"],
        1.0,
    )
    assert GoalBeliefV1.from_dict(payload).to_json() == belief.to_json()
    assert belief.content_digest == GoalBeliefV1.from_dict(payload).content_digest
    decoded = json.loads(belief.to_json())
    assert all(
        math.isfinite(value)
        for value in (
            decoded["timestamp_s"],
            decoded["unknown_candidate_probability"],
            decoded["arrival_probability"],
            decoded["change_probability"],
        )
    )


def test_unavailable_belief_is_explicit_without_fake_vectors() -> None:
    """Unavailable sensing uses null estimates and a named blocker rather than NaN or zero data."""
    observation = GoalBeliefObservation(
        track_id="track-missing",
        tracking_epoch_id="epoch-1",
        timestamp_s=0.1,
        step_index=1,
        config_hash=CONFIG_HASH,
        mode=GoalBeliefMode.UNAVAILABLE,
        censoring_state=CensoringState.UNKNOWN,
        blockers=("not_visible",),
    )

    belief = GoalBeliefV1.from_observation(observation)

    assert belief.force_estimate is None
    assert belief.desired_velocity_xy is None
    assert belief.track_confidence is None
    assert belief.unknown_candidate_probability == 1.0
    assert belief.to_dict()["blockers"] == ["not_visible"]


def test_candidate_mass_must_include_unknown_probability() -> None:
    """Candidate probabilities that do not close to one fail before serialization."""
    with pytest.raises(ValueError, match="plus unknown mass must sum to 1"):
        GoalBeliefObservation(
            track_id="track-1",
            tracking_epoch_id="epoch-1",
            timestamp_s=0.1,
            step_index=1,
            config_hash=CONFIG_HASH,
            mode=GoalBeliefMode.CENSORED,
            censoring_state=CensoringState.CENSORED,
            blockers=("candidate_censored",),
            candidate_probabilities=(
                GoalCandidateProbability("active", GoalCandidateKind.ACTIVE_WAYPOINT, 0.2),
            ),
            unknown_candidate_probability=0.2,
        )


@pytest.mark.parametrize(
    "covariance, message",
    [
        (((0.0, 0.0), (0.0, 1.0)), "positive-definite"),
        (((1.0, 0.2), (0.1, 1.0)), "symmetric"),
        (((math.nan, 0.0), (0.0, 1.0)), "finite"),
    ],
)
def test_force_covariance_rejects_invalid_values(covariance, message: str) -> None:
    """Force covariance must be finite, symmetric, and positive definite."""
    with pytest.raises((TypeError, ValueError), match=message):
        ForceEstimate2D(mean_xy=(0.0, 0.0), covariance_xy=covariance)


def test_duplicate_candidate_ids_fail_closed() -> None:
    """Candidate identity is semantic and cannot be duplicated."""
    with pytest.raises(ValueError, match="duplicate candidate ID"):
        GoalBeliefObservation(
            track_id="track-1",
            tracking_epoch_id="epoch-1",
            timestamp_s=0.1,
            step_index=1,
            config_hash=CONFIG_HASH,
            mode=GoalBeliefMode.CENSORED,
            censoring_state=CensoringState.CENSORED,
            blockers=("ambiguous",),
            candidate_probabilities=(
                GoalCandidateProbability("same", GoalCandidateKind.ACTIVE_WAYPOINT, 0.4),
                GoalCandidateProbability("same", GoalCandidateKind.FINAL_DESTINATION, 0.4),
            ),
            unknown_candidate_probability=0.2,
        )


def test_actor_constructor_rejects_oracle_like_objects_and_future_data() -> None:
    """Actor construction accepts the exact typed observation only and rejects future rows."""
    with pytest.raises(TypeError, match="GoalBeliefObservation"):
        GoalBeliefV1.from_observation(object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="newer than the decision point"):
        GoalBeliefV1.from_observation(_observation(), current_step_index=0)


def test_invalid_mode_source_and_frame_fail_closed() -> None:
    """Enum fields cannot silently accept values outside the versioned contract."""
    with pytest.raises(TypeError, match="mode must be GoalBeliefMode"):
        replace(_observation(), mode="nominal")  # type: ignore[arg-type]

    payload = GoalBeliefV1.from_observation(_observation()).to_dict()
    payload["source"] = "not-a-source"
    with pytest.raises(ValueError, match="source must be one of"):
        GoalBeliefV1.from_dict(payload)
    payload["source"] = "oracle_upper_bound"
    privileged = GoalBeliefV1.from_dict(payload)
    assert privileged.source.value == "oracle_upper_bound"
    with pytest.raises(ValueError, match="source=observation_only"):
        privileged.to_actor_model_features()
    with pytest.raises(ValueError, match="source=observation_only"):
        privileged.to_model_features()
    payload["coordinate_frame"] = "robot_relative"
    with pytest.raises(ValueError, match="one of"):
        GoalBeliefV1.from_dict(payload)


def test_actor_payload_and_model_features_have_no_oracle_fields() -> None:
    """Actor JSON and model-ready features must not carry simulator goal or identity truth."""
    belief = GoalBeliefV1.from_observation(_observation())

    assert ACTOR_FORBIDDEN_KEYS.isdisjoint(belief.to_dict())
    assert belief.to_actor_model_features() == belief.to_model_features()
    assert ACTOR_FORBIDDEN_KEYS.isdisjoint(belief.to_model_features())


def test_actor_cap_status_is_uncertainty_not_oracle_truth() -> None:
    """Actor payloads expose only cap uncertainty and never an applied cap value."""
    observation = _observation()
    observation = replace(observation, speed_cap_status=ActorSpeedCapStatus.POSSIBLE)
    belief = GoalBeliefV1.from_observation(observation)

    assert belief.speed_cap_status is ActorSpeedCapStatus.POSSIBLE
    assert belief.to_dict()["speed_cap_status"] == "possible"
    assert "oracle_speed_cap_active" not in belief.to_dict()
    assert "applied_speed_mps" not in belief.to_model_features()


def test_invisible_and_padded_history_rows_cannot_smuggle_values() -> None:
    """Mask semantics make missing history explicit and prevent hidden zero-vector placeholders."""
    with pytest.raises(ValueError, match="must omit position and velocity"):
        ActorObservationStep(
            timestamp_s=0.1,
            step_index=1,
            position_xy=(0.0, 0.0),
            velocity_xy=(0.0, 0.0),
            mask=ObservationMask.PADDED,
        )

    padded = ActorObservationStep(
        timestamp_s=0.0,
        step_index=0,
        position_xy=None,
        velocity_xy=None,
        mask=ObservationMask.PADDED,
    )
    assert padded.to_dict()["position_xy"] is None
    assert CoordinateFrame.GLOBAL_XY.value == "global_xy"


def test_track_id_mapping_is_independent_of_observation_slot_order() -> None:
    """Reordering a batch cannot exchange state because slot index is absent from the payload."""
    observations = (_observation(track_id="track-a"), _observation(track_id="track-b"))
    forward = {
        observation.track_id: GoalBeliefV1.from_observation(observation).to_json()
        for observation in observations
    }
    reversed_slots = {
        observation.track_id: GoalBeliefV1.from_observation(observation).to_json()
        for observation in reversed(observations)
    }

    assert forward == reversed_slots
    assert all("slot_index" not in payload for payload in forward.values())


def test_unknown_external_actor_key_fails_closed() -> None:
    """Versioned external payloads reject silent schema extensions."""
    payload = GoalBeliefV1.from_observation(_observation()).to_dict()
    payload["goal_after_behavior"] = [1.0, 0.0]

    with pytest.raises(ValueError, match="unexpected key"):
        GoalBeliefV1.from_dict(payload)
