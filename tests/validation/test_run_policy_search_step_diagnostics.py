"""Tests for policy-search step diagnostics helpers."""

from __future__ import annotations

import pytest

from scripts.validation.run_policy_search_step_diagnostics import _trace_progress_summary


def test_trace_progress_summary_exposes_progress_stagnation_and_risk() -> None:
    """Step diagnostics should summarize progress and clearance without scanning every row."""
    rows = [
        {
            "step": 0,
            "goal_distance": 10.0,
            "post_step_goal_distance": 9.5,
            "min_robot_ped_distance": 1.2,
            "post_step_min_robot_ped_distance": 0.9,
            "is_pedestrian_collision": False,
            "is_obstacle_collision": False,
            "is_robot_collision": False,
        },
        {
            "step": 1,
            "goal_distance": 9.5,
            "post_step_goal_distance": 9.5,
            "min_robot_ped_distance": 0.8,
            "post_step_min_robot_ped_distance": 0.7,
            "is_pedestrian_collision": False,
            "is_obstacle_collision": False,
            "is_robot_collision": False,
        },
        {
            "step": 2,
            "goal_distance": 9.5,
            "post_step_goal_distance": 9.5,
            "min_robot_ped_distance": 0.72,
            "post_step_min_robot_ped_distance": 0.65,
            "is_pedestrian_collision": True,
            "is_obstacle_collision": False,
            "is_robot_collision": False,
        },
        {
            "step": 3,
            "goal_distance": 9.5,
            "post_step_goal_distance": 9.8,
            "min_robot_ped_distance": None,
            "post_step_min_robot_ped_distance": 0.6,
            "is_pedestrian_collision": False,
            "is_obstacle_collision": True,
            "is_robot_collision": False,
        },
    ]

    summary = _trace_progress_summary(rows)

    assert summary["steps_observed"] == 4
    assert summary["initial_goal_distance"] == pytest.approx(10.0)
    assert summary["final_goal_distance"] == pytest.approx(9.8)
    assert summary["best_goal_distance"] == pytest.approx(9.5)
    assert summary["net_goal_progress"] == pytest.approx(0.2)
    assert summary["best_goal_progress"] == pytest.approx(0.5)
    assert summary["progress_step_count"] == 1
    assert summary["regression_step_count"] == 1
    assert summary["stagnant_step_count"] == 2
    assert summary["longest_stagnant_run"] == 2
    assert summary["closest_robot_ped_distance"] == pytest.approx(0.6)
    assert summary["closest_robot_ped_step"] == 3
    assert summary["collision_flag_counts"] == {
        "pedestrian": 1,
        "obstacle": 1,
        "robot": 0,
    }


def test_trace_progress_summary_handles_empty_trace() -> None:
    """Empty traces should still produce a stable report payload."""
    summary = _trace_progress_summary([])

    assert summary == {
        "steps_observed": 0,
        "initial_goal_distance": None,
        "final_goal_distance": None,
        "best_goal_distance": None,
        "net_goal_progress": None,
        "best_goal_progress": None,
        "progress_step_count": 0,
        "regression_step_count": 0,
        "stagnant_step_count": 0,
        "longest_stagnant_run": 0,
        "closest_robot_ped_distance": None,
        "closest_robot_ped_step": None,
        "collision_flag_counts": {
            "pedestrian": 0,
            "obstacle": 0,
            "robot": 0,
        },
    }


def test_trace_progress_summary_ignores_invalid_numeric_values() -> None:
    """Invalid diagnostic numbers should not poison progress or clearance summaries."""
    rows = [
        {
            "step": 0,
            "goal_distance": 10.0,
            "post_step_goal_distance": 9.0,
            "min_robot_ped_distance": 2.0,
            "post_step_min_robot_ped_distance": 1.5,
        },
        {
            "step": True,
            "goal_distance": float("nan"),
            "post_step_goal_distance": float("inf"),
            "min_robot_ped_distance": True,
            "post_step_min_robot_ped_distance": "",
        },
        {
            "step": 2,
            "goal_distance": None,
            "post_step_goal_distance": None,
            "min_robot_ped_distance": 1.2,
            "post_step_min_robot_ped_distance": 1.1,
        },
    ]

    summary = _trace_progress_summary(rows)

    assert summary["initial_goal_distance"] == pytest.approx(10.0)
    assert summary["final_goal_distance"] == pytest.approx(9.0)
    assert summary["best_goal_distance"] == pytest.approx(9.0)
    assert summary["progress_step_count"] == 1
    assert summary["regression_step_count"] == 0
    assert summary["stagnant_step_count"] == 0
    assert summary["longest_stagnant_run"] == 0
    assert summary["closest_robot_ped_distance"] == pytest.approx(1.1)
    assert summary["closest_robot_ped_step"] == 2


def test_trace_progress_summary_missing_steps_break_stagnant_runs() -> None:
    """Unavailable progress data should split contiguous stagnant step runs."""
    rows = [
        {"step": 0, "goal_distance": 5.0, "post_step_goal_distance": 5.0},
        {"step": 1, "goal_distance": None, "post_step_goal_distance": None},
        {"step": 2, "goal_distance": 4.0, "post_step_goal_distance": 4.0},
    ]

    summary = _trace_progress_summary(rows)

    assert summary["stagnant_step_count"] == 2
    assert summary["longest_stagnant_run"] == 1
