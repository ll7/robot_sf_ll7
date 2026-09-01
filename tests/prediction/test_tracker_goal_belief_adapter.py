"""Contract tests for the opt-in tracker-to-goal-belief adapter."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_sf.prediction import TrackerGoalBeliefAdapter as PackageTrackerGoalBeliefAdapter
from robot_sf.prediction.goal_belief_contract import ACTOR_FORBIDDEN_KEYS
from robot_sf.prediction.tracker_goal_belief_adapter import (
    TrackerGoalBeliefAdapter,
    TrackerGoalBeliefAdapterConfig,
)
from robot_sf.sensor.pedestrian_tracking import (
    PedestrianObservationSnapshot,
    PedestrianTracker,
    PedestrianTrackingConfig,
    TrackStatus,
)


def _tracker() -> PedestrianTracker:
    """Build an enabled tracker with deterministic fixture-friendly settings."""
    return PedestrianTracker(
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
            max_missed_steps=2,
            history_capacity=4,
        )
    )


def _snapshot(
    timestamp_s: float,
    step_index: int,
    positions: list[list[float]],
    velocities: list[list[float]],
    *,
    visible: list[bool] | None = None,
    velocity_coordinate_frame: str | None = None,
    robot_pose_global: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PedestrianObservationSnapshot:
    """Build one deterministic observation snapshot, including frame metadata."""
    row_count = len(positions)
    snapshot = PedestrianObservationSnapshot(
        timestamp_s=timestamp_s,
        step_index=step_index,
        coordinate_frame="global_xy",
        velocity_coordinate_frame=velocity_coordinate_frame,
        robot_pose_global=robot_pose_global,
        positions=np.asarray(positions, dtype=float),
        velocities=np.asarray(velocities, dtype=float),
        valid_mask=np.ones(row_count, dtype=bool),
        visible_mask=np.ones(row_count, dtype=bool)
        if visible is None
        else np.asarray(visible, dtype=bool),
    )
    return snapshot


def _tracking_result():
    """Build one observation-derived result for adapter contract tests."""
    tracker = _tracker()
    return tracker.update(
        _snapshot(
            0.0,
            0,
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        )
    )


def test_adapter_is_default_off_and_does_not_emit_beliefs() -> None:
    """Default-off prediction metadata must contain no belief records or tracker data."""

    channel = TrackerGoalBeliefAdapter().adapt(_tracking_result())

    assert channel.enabled is False
    assert channel.to_dict()["beliefs"] == {}
    assert channel.to_dict()["diagnostics"]["status"] == "disabled"


def test_adapter_is_available_from_the_lazy_prediction_export() -> None:
    """The package export resolves without importing the tracker during package startup."""
    assert PackageTrackerGoalBeliefAdapter is TrackerGoalBeliefAdapter


def test_enabled_adapter_emits_observation_only_unavailable_belief() -> None:
    """Enabled adaptation reaches the actor contract without inventing a goal posterior."""

    adapter = TrackerGoalBeliefAdapter(TrackerGoalBeliefAdapterConfig(enabled=True))
    channel = adapter.adapt(_tracking_result())

    assert channel.enabled is True
    assert channel.timestamp_s == 0.0
    assert channel.step_index == 0
    assert [belief.track_id for belief in channel.beliefs] == ["track-1"]
    belief = channel.beliefs[0]
    payload = belief.to_dict()
    assert belief.tracking_epoch_id == "0"
    assert payload["source"] == "observation_only"
    assert payload["coordinate_frame"] == "global_xy"
    assert payload["mode"] == "unavailable"
    assert payload["unknown_candidate_probability"] == 1.0
    assert payload["force_estimate"] is None
    assert payload["desired_velocity_xy"] is None
    assert "candidate_provider_not_configured" in payload["blockers"]
    assert ACTOR_FORBIDDEN_KEYS.isdisjoint(payload)
    assert payload["history_steps"][-1]["mask"] == "observed"
    assert channel.diagnostics["track_diagnostics"]["track-1"]["status"] == "confirmed"
    assert channel.diagnostics["track_diagnostics"]["track-1"]["position_covariance"]


def test_fixture_preserves_reorder_frame_and_brief_occlusion_semantics() -> None:
    """The adapter keeps identity, normalized velocity, and current masks explicit."""
    tracker = _tracker()
    first = tracker.update(
        _snapshot(
            0.0,
            0,
            [[1.0, 0.0], [3.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            velocity_coordinate_frame="robot_ego_xy",
            robot_pose_global=(0.0, 0.0, np.pi / 2.0),
        )
    )
    reordered = tracker.update(
        _snapshot(
            0.1,
            1,
            [[3.0, 0.0], [1.0, 0.0]],
            [[-1.0, 0.0], [1.0, 0.0]],
        )
    )
    lost = tracker.update(
        _snapshot(
            0.2,
            2,
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[np.nan, np.nan], [np.nan, np.nan]],
            visible=[False, False],
        )
    )
    reacquired = tracker.update(
        _snapshot(
            0.3,
            3,
            [[3.0, 0.0], [1.0, 0.0]],
            [[-1.0, 0.0], [1.0, 0.0]],
        )
    )
    adapter = TrackerGoalBeliefAdapter(TrackerGoalBeliefAdapterConfig(enabled=True))

    first_channel = adapter.adapt(first)
    reordered_channel = adapter.adapt(reordered)
    lost_channel = adapter.adapt(lost)
    reacquired_channel = adapter.adapt(reacquired)

    np.testing.assert_allclose(first.track(1).velocity_global_xy, [0.0, 1.0], atol=1e-9)
    assert [belief.track_id for belief in reordered_channel.beliefs] == ["track-1", "track-2"]
    np.testing.assert_allclose(reordered.track(1).position_global_xy, [1.0, 0.0], atol=0.02)
    np.testing.assert_allclose(reordered.track(2).position_global_xy, [3.0, 0.0], atol=0.02)
    assert first_channel.beliefs[0].to_dict()["history_steps"][0]["mask"] == "observed"
    assert reordered_channel.beliefs[0].to_dict()["history_steps"][0]["mask"] == "observed"
    assert lost.track(1).status is TrackStatus.LOST
    lost_history = lost_channel.beliefs[0].to_dict()["history_steps"]
    assert [row["mask"] for row in lost_history] == ["invisible"]
    assert lost_history[0]["position_xy"] is None
    assert "occluded" in lost_channel.beliefs[0].blockers
    reacquired_history = reacquired_channel.beliefs[0].to_dict()["history_steps"]
    assert [row["mask"] for row in reacquired_history] == ["observed"]
    assert reacquired_history[0]["velocity_xy"] is not None
    assert all(len(belief.history_steps) == 1 for belief in reacquired_channel.beliefs)
    assert first_channel.content_digest != lost_channel.content_digest


def test_reset_isolation_changes_epoch_without_retaining_prior_beliefs() -> None:
    """Reset provenance partitions channels while the stateless adapter retains no old tracks."""
    adapter = TrackerGoalBeliefAdapter(TrackerGoalBeliefAdapterConfig(enabled=True))
    old_channel = adapter.adapt(_tracking_result())

    adapter.reset("episode-2")
    new_channel = adapter.adapt(_tracking_result())

    assert old_channel.tracking_epoch_id == 0
    assert old_channel.beliefs[0].reset_provenance is None
    assert old_channel.beliefs[0].tracking_epoch_id == "0"
    assert new_channel.tracking_epoch_id == 1
    assert new_channel.beliefs[0].reset_provenance == "episode-2"
    assert new_channel.beliefs[0].tracking_epoch_id == "1"
    assert new_channel.beliefs[0].track_id == "track-1"
    assert (old_channel.beliefs[0].tracking_epoch_id, old_channel.beliefs[0].track_id) != (
        new_channel.beliefs[0].tracking_epoch_id,
        new_channel.beliefs[0].track_id,
    )
    assert new_channel.diagnostics["reset_provenance"] == "episode-2"


def test_adapter_is_deterministic_and_fails_closed_at_input_boundary() -> None:
    """Equivalent tracker results serialize identically and malformed inputs are rejected."""
    result = _tracking_result()
    adapter = TrackerGoalBeliefAdapter(TrackerGoalBeliefAdapterConfig(enabled=True))
    channel = adapter.adapt(result)
    repeated = adapter.adapt(result)

    assert channel.to_dict() == repeated.to_dict()
    assert channel.content_digest == repeated.content_digest
    assert ACTOR_FORBIDDEN_KEYS.isdisjoint(channel.to_dict()["diagnostics"])
    with pytest.raises(TypeError, match="PedestrianTrackingResult"):
        adapter.adapt(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="enabled must be a bool"):
        TrackerGoalBeliefAdapterConfig(enabled=1)  # type: ignore[arg-type]


def test_enabled_adapter_reports_disabled_tracker_without_fabricating_beliefs() -> None:
    """An enabled bridge reports an unavailable tracker rather than inventing actor state."""
    disabled_tracker = PedestrianTracker()
    result = disabled_tracker.update(_snapshot(0.0, 0, [[1.0, 0.0]], [[0.0, 0.0]]))

    channel = TrackerGoalBeliefAdapter(TrackerGoalBeliefAdapterConfig(enabled=True)).adapt(result)

    assert channel.beliefs == ()
    assert channel.diagnostics["status"] == "tracking_unavailable"
    assert channel.diagnostics["tracking_blockers"] == ["tracking_disabled"]


def test_adapter_does_not_promote_missing_velocity_to_an_observed_vector() -> None:
    """A visible position with no velocity becomes a conservative unavailable history row."""
    result = _tracker().update(_snapshot(0.0, 0, [[1.0, 0.0]], [[np.nan, np.nan]]))

    channel = TrackerGoalBeliefAdapter(TrackerGoalBeliefAdapterConfig(enabled=True)).adapt(result)
    belief = channel.beliefs[0]
    latest = belief.to_dict()["history_steps"][-1]

    assert latest["mask"] == "invisible"
    assert latest["position_xy"] is None
    assert latest["velocity_xy"] is None
    assert "velocity_unavailable" in belief.blockers
    assert channel.diagnostics["history_projection_counts"] == {
        "current_row_only_velocity_unavailable": 1
    }


def test_recovered_velocity_does_not_promote_prior_missing_velocity_history() -> None:
    """A recovered current velocity cannot establish provenance for an older tracker row."""
    tracker = _tracker()
    tracker.update(_snapshot(0.0, 0, [[1.0, 0.0]], [[np.nan, np.nan]]))
    recovered = tracker.update(_snapshot(0.1, 1, [[1.1, 0.0]], [[1.0, 0.0]]))

    track = recovered.track(1)
    assert "velocity_unavailable" not in track.blockers
    assert track.age_steps == 2

    channel = TrackerGoalBeliefAdapter(TrackerGoalBeliefAdapterConfig(enabled=True)).adapt(
        recovered
    )
    payload = channel.beliefs[0].to_dict()

    assert [row["step_index"] for row in payload["history_steps"]] == [1]
    assert payload["history_steps"][0]["mask"] == "observed"
    assert payload["history_steps"][0]["velocity_xy"] is not None
    assert "current_row_only_tracker_v1_velocity_provenance_unavailable" in payload["blockers"]


def test_adapter_orders_serialized_track_ids_lexically() -> None:
    """Beliefs and serialized mappings use the channel's textual identity order."""
    result = _tracking_result()
    source = result.tracks[0]
    result = replace(
        result,
        tracks=tuple(replace(source, track_id=value) for value in (1, 2, 10)),
    )

    channel = TrackerGoalBeliefAdapter(TrackerGoalBeliefAdapterConfig(enabled=True)).adapt(result)

    expected = ["track-1", "track-10", "track-2"]
    assert [belief.track_id for belief in channel.beliefs] == expected
    assert list(channel.to_dict()["beliefs"]) == expected
    assert len(set(expected)) == 3


def test_result_rejects_track_decision_point_mismatch() -> None:
    """A track from another decision point cannot be silently joined to a result batch."""
    result = _tracking_result()
    mismatched_track = replace(result.tracks[0], step_index=1)

    with pytest.raises(ValueError, match="tracks must match"):
        replace(result, tracks=(mismatched_track,))
