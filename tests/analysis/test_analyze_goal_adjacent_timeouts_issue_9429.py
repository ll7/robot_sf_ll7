"""Tests for the issue #9429 frozen-campaign analyzer."""

from __future__ import annotations

from scripts.analysis.analyze_goal_adjacent_timeouts_issue_9429 import (
    FinalWaypointMeasurement,
    classify_goal_adjacent_timeout,
    is_noncollision_timeout,
    summarize_wall_distances,
)


def _episode(positions: list[list[float]] | None) -> dict:
    metadata = {}
    if positions is not None:
        metadata["simulation_step_trace"] = {
            "steps": [{"robot": {"position": position}} for position in positions]
        }
    return {
        "outcome": {"timeout_event": True, "collision_event": False},
        "algorithm_metadata": metadata,
    }


def test_goal_adjacent_predicate_applies_strict_completion_boundary() -> None:
    positions = [[5.0, 0.0]] * 100
    positions[-1] = [3.0, 0.0]
    result = classify_goal_adjacent_timeout(
        _episode(positions),
        final_waypoint=(0.0, 0.0),
        completion_radius_m=2.0,
    )
    assert result.value is True
    assert result.min_tail_distance_m == 3.0
    assert result.min_episode_distance_m == 3.0

    positions[-1] = [2.0, 0.0]
    completed = classify_goal_adjacent_timeout(
        _episode(positions),
        final_waypoint=(0.0, 0.0),
        completion_radius_m=2.0,
    )
    assert completed.value is False


def test_goal_adjacent_predicate_fails_closed_without_trace() -> None:
    result = classify_goal_adjacent_timeout(
        _episode(None),
        final_waypoint=(0.0, 0.0),
        completion_radius_m=2.0,
    )
    assert result.value is None
    assert result.reason == "missing_simulation_step_trace"


def test_timeout_predicate_requires_no_collision() -> None:
    row = _episode([[5.0, 0.0]] * 100)
    assert is_noncollision_timeout(row) is True
    row["outcome"]["collision_event"] = True
    assert is_noncollision_timeout(row) is False


def test_wall_distance_summary_counts_completion_radius_plus_margin() -> None:
    rows = [
        FinalWaypointMeasurement("s1", "head_on", 1, 0.0, 0.0, 2.4, 2.0),
        FinalWaypointMeasurement("s1", "head_on", 2, 0.0, 0.0, 2.6, 2.0),
    ]
    summary = summarize_wall_distances(rows, wall_margin_m=0.5)
    family = next(row for row in summary if row["scenario_family"] == "head_on")
    overall = next(row for row in summary if row["scenario_family"] == "ALL")
    assert family["within_threshold"] == 1
    assert family["within_threshold_fraction"] == 0.5
    assert family["median_m"] == 2.5
    assert overall["final_waypoints"] == 2
