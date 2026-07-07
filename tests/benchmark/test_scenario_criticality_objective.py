"""Tests for scenario criticality objective (Issue #4362)."""

from __future__ import annotations

import math

import pytest

from robot_sf.benchmark.scenario_criticality_objective import (
    CriticalityObjectiveConfig,
    apply_criticality_parameters,
    compute_criticality_score,
)


class _MockEpisodeData:
    """Mock EpisodeData for testing."""

    def __init__(self, metrics: dict):
        self.metrics = metrics


def test_criticality_score_collision_increases() -> None:
    """Criticality score increases with collision count."""
    no_collision = _MockEpisodeData(
        metrics={
            "collision_count": 0,
            "near_misses": 0,
            "min_clearance": 0.6,
            "failure_to_progress": 0,
            "stalled_time": 0,
        }
    )
    with_collision = _MockEpisodeData(
        metrics={
            "collision_count": 2,
            "near_misses": 0,
            "min_clearance": 0.6,
            "failure_to_progress": 0,
            "stalled_time": 0,
        }
    )

    result_no = compute_criticality_score(no_collision)
    result_coll = compute_criticality_score(with_collision)

    assert result_no.status == "evaluated"
    assert result_coll.status == "evaluated"
    assert result_coll.criticality_score > result_no.criticality_score
    assert result_coll.collision_term == 20.0


def test_criticality_score_near_miss_increases() -> None:
    """Criticality score increases with near-miss count."""
    no_near_miss = _MockEpisodeData(
        metrics={
            "collision_count": 0,
            "near_misses": 0,
            "min_clearance": 0.6,
            "failure_to_progress": 0,
            "stalled_time": 0,
        }
    )
    with_near_miss = _MockEpisodeData(
        metrics={
            "collision_count": 0,
            "near_misses": 3,
            "min_clearance": 0.6,
            "failure_to_progress": 0,
            "stalled_time": 0,
        }
    )

    result_no = compute_criticality_score(no_near_miss)
    result_nm = compute_criticality_score(with_near_miss)

    assert result_nm.criticality_score > result_no.criticality_score
    assert result_nm.near_miss_term == 6.0


def test_criticality_score_low_clearance_increases() -> None:
    """Criticality score increases when clearance is below margin."""
    high_clearance = _MockEpisodeData(
        metrics={
            "collision_count": 0,
            "near_misses": 0,
            "min_clearance": 0.8,
            "failure_to_progress": 0,
            "stalled_time": 0,
        }
    )
    low_clearance = _MockEpisodeData(
        metrics={
            "collision_count": 0,
            "near_misses": 0,
            "min_clearance": 0.2,
            "failure_to_progress": 0,
            "stalled_time": 0,
        }
    )

    result_high = compute_criticality_score(high_clearance)
    result_low = compute_criticality_score(low_clearance)

    assert result_low.clearance_term > result_high.clearance_term
    assert result_low.criticality_score > result_high.criticality_score


def test_criticality_score_missing_metric_fail_closed() -> None:
    """Missing required metric produces not_evaluable with fail_closed=True."""
    incomplete = _MockEpisodeData(
        metrics={
            "collision_count": 0,
            "near_misses": 0,
        }
    )

    result = compute_criticality_score(incomplete)

    assert result.status == "not_evaluable"
    assert result.reason is not None
    assert math.isnan(result.criticality_score)


def test_criticality_score_custom_weights() -> None:
    """Custom objective weights are applied correctly."""
    config = CriticalityObjectiveConfig(
        collision_weight=20.0,
        near_miss_weight=5.0,
        clearance_margin=1.0,
        clearance_weight=2.0,
    )
    episode = _MockEpisodeData(
        metrics={
            "collision_count": 1,
            "near_misses": 2,
            "min_clearance": 0.3,
            "failure_to_progress": 0,
            "stalled_time": 0,
        }
    )

    result = compute_criticality_score(episode, config)

    assert result.collision_term == 20.0
    assert result.near_miss_term == 10.0
    assert result.clearance_term == pytest.approx(1.4)


def test_apply_criticality_parameters_copy_on_write() -> None:
    """Scenario patching does not mutate the source."""
    original = {
        "id": "test_scenario",
        "pedestrians": [
            {"speed_multiplier": 1.0, "start_delay_s": 0.0, "waypoints": [{"x": 0, "y": 0}]}
        ],
        "robot": {"start_x": 0.0},
    }

    params = {
        "pedestrian_speed_scale": 1.2,
        "pedestrian_start_delay_s": 1.5,
        "crossing_waypoint_y_offset_m": 0.3,
    }

    patched = apply_criticality_parameters(original, params)

    assert original["pedestrians"][0]["speed_multiplier"] == 1.0
    assert original["pedestrians"][0]["start_delay_s"] == 0.0
    assert original["pedestrians"][0]["waypoints"][0]["y"] == 0

    assert patched["pedestrians"][0]["speed_multiplier"] == 1.2
    assert patched["pedestrians"][0]["start_delay_s"] == 1.5
    assert patched["pedestrians"][0]["waypoints"][0]["y"] == 0.3


def test_apply_criticality_parameters_metadata_attached() -> None:
    """Parameter metadata is attached to patched scenario."""
    original = {"id": "test"}
    params = {"pedestrian_speed_scale": 1.2}

    patched = apply_criticality_parameters(original, params)

    assert "issue_4362_criticality_parameters" in patched["metadata"]
    assert patched["metadata"]["issue_4362_criticality_parameters"] == params
    assert "issue_4362_candidate_id" in patched["metadata"]
    assert patched["id"].startswith("test_crit_")


def test_apply_criticality_parameters_invalid_input() -> None:
    """Invalid inputs raise ValueError."""
    with pytest.raises(ValueError, match="scenario must be a dict"):
        apply_criticality_parameters("not a dict", {})

    with pytest.raises(ValueError, match="params must be a dict"):
        apply_criticality_parameters({}, "not a dict")


def test_criticality_score_progress_failure() -> None:
    """Criticality score increases with failure to progress."""
    no_failure = _MockEpisodeData(
        metrics={
            "collision_count": 0,
            "near_misses": 0,
            "min_clearance": 0.6,
            "failure_to_progress": 0,
            "stalled_time": 0,
        }
    )
    with_failure = _MockEpisodeData(
        metrics={
            "collision_count": 0,
            "near_misses": 0,
            "min_clearance": 0.6,
            "failure_to_progress": 1,
            "stalled_time": 0,
        }
    )

    result_no = compute_criticality_score(no_failure)
    result_fail = compute_criticality_score(with_failure)

    assert result_fail.progress_failure_term == 5.0
    assert result_fail.criticality_score > result_no.criticality_score


def test_criticality_score_stalled_time() -> None:
    """Criticality score increases with stalled time."""
    no_stall = _MockEpisodeData(
        metrics={
            "collision_count": 0,
            "near_misses": 0,
            "min_clearance": 0.6,
            "failure_to_progress": 0,
            "stalled_time": 0,
        }
    )
    with_stall = _MockEpisodeData(
        metrics={
            "collision_count": 0,
            "near_misses": 0,
            "min_clearance": 0.6,
            "failure_to_progress": 0,
            "stalled_time": 10.0,
        }
    )

    result_no = compute_criticality_score(no_stall)
    result_stall = compute_criticality_score(with_stall)

    assert result_stall.stalled_time_term == 5.0
    assert result_stall.criticality_score > result_no.criticality_score
