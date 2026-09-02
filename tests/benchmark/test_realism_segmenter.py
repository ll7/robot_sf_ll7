"""Synthetic fixture tests for interaction-conditioned realism segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_sf.benchmark.pedestrian_realism_validation import (
    INTERACTION_CLASSES,
    InteractionSegmentationConfig,
    RealismInteractionContext,
    RealismMetricConfig,
    RealismObstacle,
    RealismSceneGeometry,
    build_dataset_scorecard,
    render_scorecard_markdown,
    run_realism_validation_from_track_set,
    segment_interactions,
)
from robot_sf.benchmark.realism_validation_contract import load_realism_validation_contract
from robot_sf.data.external.eth_ucy_trajectories import EthUcyTrack, EthUcyTrackSet


def _track_set(tracks: tuple[EthUcyTrack, ...]) -> EthUcyTrackSet:
    """Build a minimal in-memory ETH/UCY-like track set without external data."""

    return EthUcyTrackSet(
        asset_id="fixture",
        group="fixture",
        split="known-mix",
        format="synthetic",
        docs_path="tests/benchmark/test_realism_segmenter.py",
        tracks=tracks,
        skipped_formats=(),
        frame_period_s=0.2,
    )


def _resolved_context(times: np.ndarray) -> RealismInteractionContext:
    """Build explicit empty-scene context for a no-robot-interaction fixture."""

    return RealismInteractionContext(
        robot_time_s=times,
        robot_positions=np.column_stack((np.full_like(times, -10.0), np.full_like(times, -10.0))),
        scene_geometry=RealismSceneGeometry(bounds_m=((-20.0, -20.0), (20.0, 20.0))),
    )


def test_segmenter_recovers_free_and_crossing_windows_from_synthetic_tracks() -> None:
    """A planted perpendicular encounter is counted separately from free walking."""

    times = np.arange(0.0, 4.01, 0.2)
    tracks = _track_set(
        (
            EthUcyTrack(
                pedestrian_id=1,
                time_s=times,
                positions=np.column_stack((times - 2.0, np.zeros_like(times))),
            ),
            EthUcyTrack(
                pedestrian_id=2,
                time_s=times,
                positions=np.column_stack((np.zeros_like(times), times - 2.0)),
            ),
            EthUcyTrack(
                pedestrian_id=3,
                time_s=times,
                positions=np.column_stack((times, np.full_like(times, 5.0))),
            ),
        )
    )

    result = segment_interactions(
        tracks,
        context=_resolved_context(times),
        config=InteractionSegmentationConfig(
            frame_window_s=0.8,
            frame_stride_s=0.8,
            crossing_distance_m=0.8,
            ped_interaction_distance_m=1.0,
        ),
    )

    assert result.status == "ok"
    assert set(result.counts) == set(INTERACTION_CLASSES)
    assert result.counts["crossing_conflict"] >= 1
    assert result.counts["free_walking"] >= 1
    crossing_windows = [window for window in result.windows if window.label == "crossing_conflict"]
    assert crossing_windows[0].track_ids == (1, 2)


def test_event_floors_deduplicate_overlapping_interaction_windows() -> None:
    """Overlapping windows from one encounter count as one independent event."""

    times = np.arange(0.0, 4.01, 0.2)
    track_set = _track_set(
        (
            EthUcyTrack(
                pedestrian_id=1,
                time_s=times,
                positions=np.column_stack((times - 2.0, np.zeros_like(times))),
            ),
            EthUcyTrack(
                pedestrian_id=2,
                time_s=times,
                positions=np.column_stack((np.zeros_like(times), times - 2.0)),
            ),
        )
    )
    segmentation = segment_interactions(
        track_set,
        config=InteractionSegmentationConfig(
            frame_window_s=0.8,
            frame_stride_s=0.4,
            crossing_distance_m=0.8,
            ped_interaction_distance_m=0.5,
        ),
    )
    minimums = dict.fromkeys(INTERACTION_CLASSES, 0)
    minimums["crossing_conflict"] = 2

    scorecard = build_dataset_scorecard(
        dataset_id="fixture/known-mix",
        config=RealismMetricConfig(),
        rmse_metrics=None,
        fundamental_diagram=None,
        lane_formation=None,
        reference_source="synthetic fixture only",
        interaction_segmentation=segmentation,
        interaction_minimum_event_counts=minimums,
    )

    interaction = scorecard.metrics["interaction_conditioned_segmentation"]
    assert segmentation.counts["crossing_conflict"] > 1
    assert segmentation.event_counts["crossing_conflict"] == 1
    assert interaction["event_count_status"]["rows"]["crossing_conflict"] == {
        "observed": 1,
        "minimum": 2,
        "status": "insufficient_events",
    }


def test_stationary_windows_are_excluded_instead_of_free_walking() -> None:
    """A stationary track lacks the motion evidence required for free walking."""

    times = np.arange(0.0, 3.21, 0.2)
    result = segment_interactions(
        _track_set(
            (
                EthUcyTrack(
                    pedestrian_id=1,
                    time_s=times,
                    positions=np.zeros((times.shape[0], 2)),
                ),
            )
        ),
        config=InteractionSegmentationConfig(frame_window_s=0.8, frame_stride_s=0.4),
    )

    assert result.status == "empty"
    assert result.counts["free_walking"] == 0
    assert result.excluded_window_counts["insufficient_motion_evidence"] > 0


def test_context_unresolvable_windows_are_excluded_instead_of_free_walking() -> None:
    """A moving track without robot/scene context cannot prove free walking."""

    times = np.arange(0.0, 3.21, 0.2)
    result = segment_interactions(
        _track_set(
            (
                EthUcyTrack(
                    pedestrian_id=1,
                    time_s=times,
                    positions=np.column_stack((times, np.zeros_like(times))),
                ),
            )
        ),
        config=InteractionSegmentationConfig(frame_window_s=0.8, frame_stride_s=0.4),
    )

    assert result.status == "empty"
    assert result.counts["free_walking"] == 0
    assert result.excluded_window_counts["insufficient_context"] > 0


def test_incomplete_track_windows_cannot_form_interaction_events() -> None:
    """A center sample is not enough to admit a track into a complete window."""

    full_times = np.arange(0.0, 4.01, 0.2)
    partial_times = np.arange(1.8, 2.21, 0.2)
    result = segment_interactions(
        _track_set(
            (
                EthUcyTrack(
                    pedestrian_id=1,
                    time_s=full_times,
                    positions=np.column_stack((full_times - 2.0, np.zeros_like(full_times))),
                ),
                EthUcyTrack(
                    pedestrian_id=2,
                    time_s=partial_times,
                    positions=np.column_stack((np.zeros_like(partial_times), partial_times - 2.0)),
                ),
            )
        ),
        config=InteractionSegmentationConfig(
            frame_window_s=0.8,
            frame_stride_s=0.8,
            crossing_distance_m=0.8,
            ped_interaction_distance_m=0.5,
        ),
    )

    assert result.counts["crossing_conflict"] == 0
    assert all(2 not in window.track_ids for window in result.windows)


def test_separating_opposing_tracks_are_not_crossing_conflicts() -> None:
    """Opposing headings with increasing separation are not crossing encounters."""

    times = np.arange(1.0, 3.01, 0.2)
    result = segment_interactions(
        _track_set(
            (
                EthUcyTrack(
                    pedestrian_id=1,
                    time_s=times,
                    positions=np.column_stack((0.5 - times, np.zeros_like(times))),
                ),
                EthUcyTrack(
                    pedestrian_id=2,
                    time_s=times,
                    positions=np.column_stack((-0.5 + times, np.zeros_like(times))),
                ),
            )
        ),
        config=InteractionSegmentationConfig(
            frame_window_s=0.8,
            frame_stride_s=0.8,
            crossing_distance_m=2.0,
            ped_interaction_distance_m=0.5,
        ),
    )

    assert result.counts["crossing_conflict"] == 0


def test_segmenter_labels_a_co_moving_cluster_as_group() -> None:
    """Three nearby pedestrians with matched headings form a group stratum."""

    times = np.arange(0.0, 3.21, 0.2)
    tracks = _track_set(
        tuple(
            EthUcyTrack(
                pedestrian_id=index,
                time_s=times,
                positions=np.column_stack((times, np.full_like(times, 0.3 * index))),
            )
            for index in (1, 2, 3)
        )
    )

    result = segment_interactions(
        tracks,
        config=InteractionSegmentationConfig(frame_window_s=0.8, frame_stride_s=0.8),
    )

    assert result.counts["group"] >= 1
    assert all(window.label == "group" for window in result.windows)
    assert result.windows[0].track_ids == (1, 2, 3)


def test_segmenter_labels_a_faster_same_direction_pedestrian_as_overtaking() -> None:
    """A faster pedestrian behind a slower one is separated from generic interaction."""

    times = np.arange(0.0, 3.21, 0.2)
    tracks = _track_set(
        (
            EthUcyTrack(
                pedestrian_id=1,
                time_s=times,
                positions=np.column_stack((1.5 * times - 2.25, np.zeros_like(times))),
            ),
            EthUcyTrack(
                pedestrian_id=2,
                time_s=times,
                positions=np.column_stack((times, np.zeros_like(times))),
            ),
        )
    )

    result = segment_interactions(
        tracks,
        config=InteractionSegmentationConfig(
            frame_window_s=0.8,
            frame_stride_s=0.4,
            group_distance_m=1.0,
            overtaking_distance_m=2.0,
        ),
    )

    overtaking_windows = [window for window in result.windows if window.label == "overtaking"]
    assert overtaking_windows
    assert overtaking_windows[0].track_ids == (1, 2)


def test_segmenter_requires_explicit_robot_context_for_robot_approach() -> None:
    """A moving robot trajectory enables robot labels without inferring missing obstacles."""

    times = np.arange(0.0, 3.21, 0.2)
    tracks = _track_set(
        (
            EthUcyTrack(
                pedestrian_id=7,
                time_s=times,
                positions=np.column_stack((np.full_like(times, 2.0), np.zeros_like(times))),
            ),
        )
    )
    context = RealismInteractionContext(
        robot_time_s=times,
        robot_positions=np.column_stack((times, np.zeros_like(times))),
    )

    result = segment_interactions(
        tracks,
        context=context,
        config=InteractionSegmentationConfig(
            frame_window_s=0.8,
            frame_stride_s=0.8,
            robot_distance_m=1.0,
            robot_approach_min_speed_mps=0.1,
        ),
    )

    assert result.counts["robot_approach"] >= 1
    assert all("obstacle_avoidance" in blocker for blocker in result.blockers)


def test_partial_robot_context_excludes_context_dependent_windows() -> None:
    """A robot trace must cover the window before it can support a positive label."""

    track_times = np.arange(0.0, 4.01, 0.2)
    result = segment_interactions(
        _track_set(
            (
                EthUcyTrack(
                    pedestrian_id=8,
                    time_s=track_times,
                    positions=np.column_stack((track_times, np.zeros_like(track_times))),
                ),
            )
        ),
        context=RealismInteractionContext(
            robot_time_s=np.asarray([0.0, 1.0]),
            robot_positions=np.asarray([[-10.0, -10.0], [-9.0, -10.0]]),
            scene_geometry=RealismSceneGeometry(
                bounds_m=((-20.0, -20.0), (20.0, 20.0)),
            ),
        ),
        config=InteractionSegmentationConfig(frame_window_s=0.8, frame_stride_s=0.8),
    )

    assert result.counts["free_walking"] == 0
    assert result.counts["robot_approach"] == 0
    assert result.excluded_window_counts["insufficient_context"] > 0


def test_partial_robot_context_does_not_admit_robot_approach() -> None:
    """A nearby pedestrian cannot be labeled from a robot trace ending mid-window."""

    track_times = np.arange(0.0, 4.01, 0.2)
    result = segment_interactions(
        _track_set(
            (
                EthUcyTrack(
                    pedestrian_id=9,
                    time_s=track_times,
                    positions=np.column_stack(
                        (np.ones_like(track_times), np.zeros_like(track_times))
                    ),
                ),
            )
        ),
        context=RealismInteractionContext(
            robot_time_s=np.asarray([0.0, 1.0]),
            robot_positions=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        ),
        config=InteractionSegmentationConfig(
            frame_window_s=0.8,
            frame_stride_s=0.8,
            robot_distance_m=1.2,
            robot_approach_min_speed_mps=0.1,
        ),
    )

    assert result.counts["robot_approach"] == 0


def test_free_walking_excludes_track_positions_outside_scene_bounds() -> None:
    """A supplied scene contract cannot certify an out-of-bounds free-walking track."""

    track_times = np.arange(0.0, 3.21, 0.2)
    result = segment_interactions(
        _track_set(
            (
                EthUcyTrack(
                    pedestrian_id=10,
                    time_s=track_times,
                    positions=np.column_stack((10.0 + track_times, np.zeros_like(track_times))),
                ),
            )
        ),
        context=RealismInteractionContext(
            robot_time_s=track_times,
            robot_positions=np.column_stack(
                (np.full_like(track_times, -10.0), np.full_like(track_times, -10.0))
            ),
            scene_geometry=RealismSceneGeometry(bounds_m=((-1.0, -1.0), (1.0, 1.0))),
        ),
        config=InteractionSegmentationConfig(frame_window_s=0.8, frame_stride_s=0.8),
    )

    assert result.counts["free_walking"] == 0
    assert result.excluded_window_counts["insufficient_context"] > 0


def test_segmenter_labels_a_turn_near_a_trusted_obstacle() -> None:
    """Obstacle labels require supplied geometry and a measurable trajectory turn."""

    times = np.arange(0.0, 4.01, 0.2)
    x = np.where(times <= 1.0, times, 1.0 + 0.5 * (times - 1.0))
    y = np.where(times <= 1.0, 0.0, 0.5 * (times - 1.0))
    tracks = _track_set(
        (
            EthUcyTrack(
                pedestrian_id=4,
                time_s=times,
                positions=np.column_stack((x, y)),
            ),
        )
    )
    geometry = RealismSceneGeometry(
        bounds_m=((-1.0, -1.0), (4.0, 4.0)),
        obstacles=(
            RealismObstacle(
                obstacle_id="corner",
                polygon_m=((0.7, -0.2), (1.1, -0.2), (1.1, 0.2), (0.7, 0.2)),
            ),
        ),
    )

    result = segment_interactions(
        tracks,
        context=RealismInteractionContext(scene_geometry=geometry),
        config=InteractionSegmentationConfig(
            frame_window_s=0.8,
            frame_stride_s=0.8,
            obstacle_distance_m=0.75,
            obstacle_turn_angle_deg=10.0,
        ),
    )

    assert result.counts["obstacle_avoidance"] >= 1


def test_segmenter_is_fail_closed_when_real_tracks_are_not_staged() -> None:
    """An absent external track set cannot produce a free-walking success row."""

    result = segment_interactions(None)

    assert result.status == "not_available"
    assert result.windows == ()
    assert result.counts == dict.fromkeys(INTERACTION_CLASSES, 0)
    assert any("not provided" in blocker for blocker in result.blockers)


def test_scorecard_exposes_per_class_rows_and_event_floor_status() -> None:
    """Scorecards retain sparse interaction classes instead of pooling them away."""

    times = np.arange(0.0, 2.01, 0.2)
    track_set = _track_set(
        (
            EthUcyTrack(
                pedestrian_id=1,
                time_s=times,
                positions=np.column_stack((times, np.zeros_like(times))),
            ),
        )
    )
    segmentation = segment_interactions(
        track_set,
        context=_resolved_context(times),
        config=InteractionSegmentationConfig(frame_window_s=0.8, frame_stride_s=0.8),
    )
    contract = load_realism_validation_contract(
        Path(__file__).resolve().parents[2]
        / "configs"
        / "benchmark"
        / "realism_validation_contract.v1.yaml"
    )

    scorecard = build_dataset_scorecard(
        dataset_id="fixture/known-mix",
        config=RealismMetricConfig(),
        rmse_metrics=None,
        fundamental_diagram=None,
        lane_formation=None,
        reference_source="synthetic fixture only",
        interaction_segmentation=segmentation,
        interaction_minimum_event_counts=contract.minimum_event_counts,
    )

    interaction = scorecard.to_dict()["metrics"]["interaction_conditioned_segmentation"]
    assert interaction["counts"]["free_walking"] == segmentation.counts["free_walking"]
    assert interaction["event_count_status"]["rows"]["crossing_conflict"]["status"] == (
        "insufficient_events"
    )
    markdown = render_scorecard_markdown(scorecard)
    assert "Interaction-Conditioned Segmentation" in markdown
    assert "insufficient_events" in markdown


def test_track_set_scorecard_runs_segmentation_without_external_side_effects() -> None:
    """The canonical track-set wrapper carries segmentation into its scorecard."""

    times = np.arange(0.0, 2.01, 0.2)
    track_set = _track_set(
        (
            EthUcyTrack(
                pedestrian_id=1,
                time_s=times,
                positions=np.column_stack((times, np.zeros_like(times))),
            ),
        )
    )

    scorecard = run_realism_validation_from_track_set(
        dataset_id="fixture/known-mix",
        track_set=track_set,
        interaction_context=_resolved_context(times),
        interaction_minimum_event_counts=dict.fromkeys(INTERACTION_CLASSES, 1),
    )

    interaction = scorecard.metrics["interaction_conditioned_segmentation"]
    assert interaction["status"] == "ok"
    assert interaction["event_count_status"]["status"] == "insufficient_events"
    assert interaction["event_count_status"]["rows"]["free_walking"]["status"] == "sufficient"
    assert interaction["event_count_status"]["rows"]["crossing_conflict"]["status"] == (
        "insufficient_events"
    )
