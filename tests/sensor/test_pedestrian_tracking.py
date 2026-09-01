"""Focused deterministic tests for the observation-derived tracker."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from robot_sf.sensor.pedestrian_tracking import (
    NormalizedPedestrianObservation,
    OracleTrackingEvaluator,
    PedestrianObservationSnapshot,
    PedestrianTracker,
    PedestrianTrackingConfig,
    TrackStatus,
)


def _snapshot(  # noqa: PLR0913
    timestamp_s: float,
    step_index: int,
    positions: list[list[float]] | np.ndarray,
    velocities: list[list[float]] | np.ndarray | None = None,
    *,
    valid: list[bool] | np.ndarray | None = None,
    visible: list[bool] | np.ndarray | None = None,
    coordinate_frame: str = "global_xy",
    velocity_coordinate_frame: str | None = None,
    robot_pose: list[float] | np.ndarray = (0.0, 0.0, 0.0),
    position_covariances: np.ndarray | None = None,
    velocity_valid_mask: list[bool] | np.ndarray | None = None,
) -> PedestrianObservationSnapshot:
    """Build a small observation snapshot for a single fixture step."""
    positions_array = np.asarray(positions, dtype=float)
    row_count = positions_array.shape[0]
    return PedestrianObservationSnapshot(
        timestamp_s=timestamp_s,
        step_index=step_index,
        coordinate_frame=coordinate_frame,
        velocity_coordinate_frame=velocity_coordinate_frame,
        robot_pose_global=robot_pose,
        positions=positions_array,
        velocities=None if velocities is None else np.asarray(velocities, dtype=float),
        valid_mask=np.ones(row_count, dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool),
        visible_mask=np.ones(row_count, dtype=bool)
        if visible is None
        else np.asarray(visible, dtype=bool),
        position_covariances=position_covariances,
        velocity_valid_mask=velocity_valid_mask,
    )


def _tracker(**overrides: object) -> PedestrianTracker:
    """Return an enabled tracker with deterministic fixture-friendly defaults."""
    values: dict[str, object] = {
        "enabled": True,
        "process_noise": 0.0,
        "initial_position_covariance": 0.1,
        "initial_velocity_covariance": 0.1,
        "measurement_position_covariance": 0.01,
        "measurement_velocity_covariance": 0.01,
        "position_gate_threshold": 100.0,
        "velocity_gate_threshold": 100.0,
        "max_missed_seconds": 10.0,
        "history_capacity": 4,
    }
    values.update(overrides)
    return PedestrianTracker(PedestrianTrackingConfig(**values))


def test_snapshot_normalizes_world_position_and_robot_ego_velocity() -> None:
    """The snapshot keeps current positions global and rotates ego velocities once."""
    snapshot = _snapshot(
        0.5,
        3,
        [[2.0, 0.0]],
        [[1.0, 0.0]],
        coordinate_frame="global_xy",
        velocity_coordinate_frame="robot_ego_xy",
        robot_pose=[10.0, 0.0, np.pi / 2.0],
    )
    normalized = snapshot.to_global_xy()
    np.testing.assert_allclose(normalized.positions_global_xy, [[2.0, 0.0]])
    np.testing.assert_allclose(normalized.velocities_global_xy, [[0.0, 1.0]], atol=1e-12)
    assert normalized.coordinate_frame == "global_xy"


def test_config_is_strict_immutable_and_default_off() -> None:
    """Configuration hashing is stable and tracking does not activate by default."""
    config = PedestrianTrackingConfig.from_mapping(
        {"enabled": False, "initial_covariance": 2.0, "history_capacity": 2}
    )
    assert config.enabled is False
    assert config.initial_position_covariance == 2.0
    assert len(config.config_hash) == 64
    with pytest.raises(FrozenInstanceError):
        config.enabled = True  # type: ignore[misc]
    disabled = PedestrianTracker().update(_snapshot(0.0, 0, [[1.0, 0.0]], [[0.0, 0.0]]))
    assert disabled.tracks == ()
    assert disabled.diagnostics.blockers == ("tracking_disabled",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"process_noise": -1.0},
        {"initial_position_covariance": -1.0},
        {"position_gate_threshold": 0.0},
        {"max_missed_seconds": 0.0},
        {"history_capacity": 0},
        {"gating_mode": "implicit_fallback"},
    ],
)
def test_config_rejects_invalid_contract_values(kwargs: dict[str, object]) -> None:
    """Negative noise, invalid gates, and invalid horizons fail closed."""
    with pytest.raises((TypeError, ValueError)):
        PedestrianTrackingConfig(**kwargs)


def test_static_reorder_keeps_track_identity_and_source_slot_is_diagnostic() -> None:
    """Nearest-first row reorder cannot exchange histories."""
    tracker = _tracker()
    first = tracker.update(_snapshot(0.0, 0, [[1.0, 0.0], [3.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]))
    second = tracker.update(_snapshot(0.1, 1, [[3.0, 0.0], [1.1, 0.0]], [[0.0, 0.0], [1.0, 0.0]]))
    assert [track.track_id for track in first.tracks] == [1, 2]
    assert [track.track_id for track in second.tracks] == [1, 2]
    np.testing.assert_allclose(second.track(1).position_global_xy, [1.1, 0.0], atol=1e-6)
    np.testing.assert_allclose(second.track(2).position_global_xy, [3.0, 0.0], atol=1e-6)
    assert second.track(1).last_observation_slot == 1
    assert second.track(2).last_observation_slot == 0
    assert second.track(1).history_valid_mask.tolist() == [False, False, True, True]


def test_crossing_trajectories_use_velocity_innovation() -> None:
    """Two actors exchanging distance rank retain IDs through a crossing."""
    tracker = _tracker()
    tracker.update(_snapshot(0.0, 0, [[-1.0, 0.0], [1.0, 0.0]], [[2.0, 0.0], [-2.0, 0.0]]))
    result = tracker.update(_snapshot(1.0, 1, [[-1.0, 0.0], [1.0, 0.0]], [[-2.0, 0.0], [2.0, 0.0]]))
    assert result.associations[0].used_velocity is True
    assert result.track(1).position_global_xy[0] > 0.0
    assert result.track(2).position_global_xy[0] < 0.0


def test_symmetric_tie_is_deterministic_under_input_reorder() -> None:
    """Canonical observation ordering plus epsilon gives repeatable tie decisions."""
    first = _snapshot(0.0, 0, [[-1.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]])
    ordered = _snapshot(1.0, 1, [[0.0, -0.1], [0.0, 0.1]], [[0.0, 0.0], [0.0, 0.0]])
    reversed_rows = _snapshot(1.0, 1, [[0.0, 0.1], [0.0, -0.1]], [[0.0, 0.0], [0.0, 0.0]])
    tracker_a = _tracker(use_velocity=False, gating_mode="euclidean", position_gate_threshold=2.0)
    tracker_b = _tracker(use_velocity=False, gating_mode="euclidean", position_gate_threshold=2.0)
    tracker_a.update(first)
    tracker_b.update(first)
    result_a = tracker_a.update(ordered)
    result_b = tracker_b.update(reversed_rows)
    for track_a, track_b in zip(result_a.tracks, result_b.tracks, strict=True):
        np.testing.assert_allclose(track_a.position_global_xy, track_b.position_global_xy)
    assert [association.track_id for association in result_a.associations] == [1, 2]


def test_brief_occlusion_reacquires_and_marks_prediction_gap() -> None:
    """A short invisible interval becomes lost memory and then the same track."""
    tracker = _tracker(max_missed_steps=2)
    tracker.update(_snapshot(0.0, 0, [[1.0, 0.0]], [[1.0, 0.0]]))
    lost = tracker.update(
        _snapshot(
            1.0,
            1,
            [[np.nan, np.nan]],
            [[np.nan, np.nan]],
            visible=[False],
        )
    )
    assert lost.track(1).status is TrackStatus.LOST
    assert lost.track(1).missed_steps == 1
    assert "prediction_only" in lost.track(1).blockers
    reacquired = tracker.update(_snapshot(2.0, 2, [[3.0, 0.0]], [[1.0, 0.0]]))
    assert reacquired.track(1).status is TrackStatus.CONFIRMED
    assert reacquired.track(1).missed_steps == 0
    assert reacquired.associations[0].track_id == 1
    assert int(np.count_nonzero(reacquired.track(1).history_valid_mask)) == 2


def test_long_occlusion_retires_and_new_track_id_is_not_reused() -> None:
    """Retirement occurs at the configured miss count and IDs remain monotonic."""
    tracker = _tracker(max_missed_steps=1)
    tracker.update(_snapshot(0.0, 0, [[1.0, 0.0]], [[0.0, 0.0]]))
    tracker.update(_snapshot(1.0, 1, [[np.nan, np.nan]], [[np.nan, np.nan]], visible=[False]))
    retired = tracker.update(
        _snapshot(2.0, 2, [[np.nan, np.nan]], [[np.nan, np.nan]], visible=[False])
    )
    assert retired.track(1).status is TrackStatus.RETIRED
    created = tracker.update(_snapshot(3.0, 3, [[5.0, 0.0]], [[0.0, 0.0]]))
    assert created.track(2).status is TrackStatus.TENTATIVE
    assert created.track(1) is None
    assert tracker.tracks[0].track_id == 2


def test_velocity_unavailable_is_explicit_and_lower_confidence() -> None:
    """Position-only detections remain usable but visibly carry lower confidence."""
    tracker = _tracker()
    result = tracker.update(_snapshot(0.0, 0, [[1.0, 0.0]], [[np.nan, np.nan]]))
    track = result.track(1)
    assert result.diagnostics.velocity_unavailable_count == 1
    assert "velocity_unavailable" in track.blockers
    assert track.association_confidence == pytest.approx(0.5)
    result = tracker.update(_snapshot(1.0, 1, [[1.0, 0.0]], [[np.nan, np.nan]]))
    assert result.associations[0].used_velocity is False
    assert result.associations[0].confidence < 1.0


def test_velocity_validity_cannot_claim_absent_velocity_payload() -> None:
    """A validity mask cannot smuggle a velocity into an absent payload."""
    with pytest.raises(ValueError, match="velocities are absent"):
        _snapshot(
            0.0,
            0,
            [[1.0, 0.0]],
            velocities=None,
            velocity_valid_mask=[True],
        )


def test_invalid_padding_is_ignored_without_nonfinite_state() -> None:
    """Padded rows are sanitized and cannot create or contaminate tracks."""
    tracker = _tracker()
    result = tracker.update(
        _snapshot(
            0.0,
            0,
            [[1.0, 0.0], [np.nan, np.nan]],
            [[0.0, 0.0], [np.nan, np.nan]],
            valid=[True, False],
            visible=[True, False],
        )
    )
    assert len(result.tracks) == 1
    assert result.diagnostics.invalid_row_count == 1
    assert np.all(np.isfinite(result.track(1).position_global_xy))


def test_non_monotonic_input_does_not_mutate_tracker_state() -> None:
    """Time and step regressions fail closed before changing the prior track."""
    tracker = _tracker()
    tracker.update(_snapshot(1.0, 1, [[1.0, 0.0]], [[0.0, 0.0]]))
    with pytest.raises(ValueError, match="timestamp"):
        tracker.update(_snapshot(0.5, 2, [[1.0, 0.0]], [[0.0, 0.0]]))
    with pytest.raises(ValueError, match="step_index"):
        tracker.update(_snapshot(2.0, 1, [[1.0, 0.0]], [[0.0, 0.0]]))
    result = tracker.update(_snapshot(2.0, 2, [[1.0, 0.0]], [[0.0, 0.0]]))
    assert result.track(1).status is TrackStatus.CONFIRMED


def test_timestamp_gap_grows_lost_covariance() -> None:
    """Constant-velocity process covariance scales with elapsed seconds."""
    tracker = _tracker(process_noise=1.0, max_missed_seconds=10.0)
    initial = tracker.update(_snapshot(0.0, 0, [[1.0, 0.0]], [[0.0, 0.0]])).track(1)
    gap = tracker.update(
        _snapshot(2.0, 1, [[np.nan, np.nan]], [[np.nan, np.nan]], visible=[False])
    ).track(1)
    assert gap.status is TrackStatus.LOST
    assert gap.position_covariance[0, 0] > initial.position_covariance[0, 0]


def test_reset_clears_history_and_restarts_episode_ids() -> None:
    """Reset cannot leak tracks or IDs into the next episode."""
    tracker = _tracker()
    tracker.update(_snapshot(0.0, 0, [[1.0, 0.0]], [[0.0, 0.0]]))
    tracker.reset()
    result = tracker.update(_snapshot(0.0, 0, [[9.0, 0.0]], [[0.0, 0.0]]))
    assert [track.track_id for track in result.tracks] == [1]
    np.testing.assert_allclose(result.track(1).position_global_xy, [9.0, 0.0])
    assert result.track(1).history_valid_mask.tolist() == [False, False, False, True]


def test_maximum_track_count_is_bounded_and_reported() -> None:
    """The configured cap bounds state creation and exposes the omitted-row blocker."""
    tracker = _tracker(max_tracks=2)
    result = tracker.update(
        _snapshot(
            0.0,
            0,
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        )
    )
    assert len(result.tracks) == 2
    assert result.diagnostics.new_track_count == 2
    assert "track_capacity_exceeded" in result.diagnostics.blockers


def test_public_arrays_are_read_only_and_oracle_is_post_tracking_only() -> None:
    """Immutable output arrays cannot be mutated and oracle scoring consumes results only."""
    tracker = _tracker(confirmation_steps=1)
    result = tracker.update(_snapshot(0.0, 0, [[1.0, 0.0]], [[0.0, 0.0]]))
    with pytest.raises(ValueError):
        result.track(1).position_global_xy[0] = 4.0
    evaluator = OracleTrackingEvaluator()
    metrics = evaluator.evaluate(
        result,
        {0: "sim-ped-a"},
        simulator_position_global_xy_by_identity={"sim-ped-a": [1.0, 0.0]},
    )
    assert metrics.assignment_accuracy is None
    next_result = tracker.update(_snapshot(1.0, 1, [[1.0, 0.0]], [[0.0, 0.0]]))
    metrics = evaluator.evaluate(next_result, {0: "sim-ped-a"})
    assert metrics.assignment_accuracy == pytest.approx(1.0)
    assert metrics.identity_switches == 0
    assert metrics.error_by_visibility["occluded"] is None


def test_oracle_assignment_accuracy_stays_bounded_across_frames() -> None:
    """Cumulative assignment accuracy uses associations, not unique track count."""
    tracker = _tracker(confirmation_steps=1)
    evaluator = OracleTrackingEvaluator()
    for step_index in range(3):
        result = tracker.update(
            _snapshot(float(step_index), step_index, [[1.0, 0.0]], [[0.0, 0.0]])
        )
        metrics = evaluator.evaluate(result, {0: "sim-ped-a"})
        if step_index == 0:
            assert metrics.assignment_accuracy is None
        else:
            assert metrics.assignment_accuracy == pytest.approx(1.0)


def test_normalized_contract_validates_optional_geometry() -> None:
    """Direct normalized construction cannot bypass covariance or radius checks."""
    common = {
        "timestamp_s": 0.0,
        "step_index": 0,
        "robot_pose_global": [0.0, 0.0, 0.0],
        "positions_global_xy": np.array([[1.0, 0.0]]),
        "velocities_global_xy": np.array([[0.0, 0.0]]),
        "valid_mask": np.array([True]),
        "visible_mask": np.array([True]),
        "velocity_valid_mask": np.array([True]),
        "position_covariances_global_xy": np.zeros((1, 2, 2)),
        "velocity_covariances_global_xy": np.zeros((1, 2, 2)),
        "radius": np.array([1.0]),
    }
    negative_covariance = dict(common)
    negative_covariance["position_covariances_global_xy"] = np.array([[[1.0, 2.0], [2.0, -1.0]]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        NormalizedPedestrianObservation(**negative_covariance)
    negative_radius = dict(common)
    negative_radius["radius"] = np.array([-1.0])
    with pytest.raises(ValueError, match="radius"):
        NormalizedPedestrianObservation(**negative_radius)
