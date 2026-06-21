"""Tests for policy-search step diagnostics helpers."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from scripts.validation.run_policy_search_step_diagnostics import (
    _apply_observed_pedestrians_to_policy_obs,
    _diagnostics_stdout_payload,
    _false_positive_actor_state_from_args,
    _format_planner_summary_lines,
    _observation_perturbation_spec,
    _occlusion_mask_by_distance,
    _pedestrian_state_from_sim,
    _trace_observation_payload,
    _trace_progress_summary,
)


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


def test_planner_summary_lines_render_nested_diagnostics() -> None:
    """Markdown reports should expose planner summaries without opening trace JSON."""
    lines = _format_planner_summary_lines(
        {
            "calls": 4,
            "selected_sources": {"route_guide": 2, "dynamic_window": 2},
        }
    )

    assert lines == [
        "## Planner Summary",
        "",
        "- `calls`: `4`",
        '- `selected_sources`: `{"dynamic_window": 2, "route_guide": 2}`',
    ]


def test_stdout_payload_includes_planner_summary(tmp_path) -> None:
    """Machine-readable stdout should surface the aggregate planner diagnostics."""
    payload = _diagnostics_stdout_payload(
        metadata={
            "trace": tmp_path / "trace.json",
            "report": tmp_path / "report.md",
            "scenario_id": "planner_sanity_simple",
            "family": "nominal",
            "seed": 111,
            "decision_counts": {"fallback": 1},
            "selected_head_counts": {"risk_dwa": 1},
        },
        progress_summary={"steps_observed": 1},
        planner_summary={"fallback_count": 1},
        done_info={"success": False},
    )

    assert payload["planner_summary"] == {"fallback_count": 1}
    assert payload["progress_summary"] == {"steps_observed": 1}


class _DummySimulator:
    def __init__(self) -> None:
        self.robot_pos = [[0.0, 0.0]]
        self.ped_pos = [[1.0, 0.0], [5.0, 0.0]]
        self.ped_vel = [[0.1, 0.0], [0.0, 0.2]]


class _DummyEnv:
    def __init__(self) -> None:
        self.simulator = _DummySimulator()


def test_pedestrian_state_from_sim_uses_stable_synthetic_ids() -> None:
    """Diagnostic traces should expose simulator pedestrian state with stable IDs."""
    positions, velocities, actor_ids = _pedestrian_state_from_sim(_DummyEnv())

    assert positions.tolist() == [[1.0, 0.0], [5.0, 0.0]]
    assert velocities.tolist() == [[0.1, 0.0], [0.0, 0.2]]
    assert actor_ids == ["ped_0", "ped_1"]


def test_occlusion_mask_by_distance_marks_out_of_range_pedestrians() -> None:
    """Range-limited occlusion should hide pedestrians beyond the diagnostic threshold."""
    env = _DummyEnv()
    positions, _velocities, _actor_ids = _pedestrian_state_from_sim(env)

    mask = _occlusion_mask_by_distance(env, positions, occlusion_distance_m=2.0)

    assert mask.tolist() == [False, True]


def test_false_positive_actor_state_from_args_offsets_from_robot() -> None:
    """False-positive diagnostic actors should be deterministic robot-relative observations."""
    env = _DummyEnv()
    args = argparse.Namespace(
        false_positive_actor_count=2,
        false_positive_offset_x_m=1.25,
        false_positive_offset_y_m=-0.5,
        false_positive_spacing_y_m=0.25,
    )

    positions, velocities, actor_ids = _false_positive_actor_state_from_args(args, env)

    np.testing.assert_array_equal(positions, [[1.25, -0.5], [1.25, -0.25]])
    np.testing.assert_array_equal(velocities, np.zeros((2, 2)))
    assert actor_ids == ["false_positive_0", "false_positive_1"]


def test_observation_perturbation_spec_includes_false_positive_actors() -> None:
    """CLI perturbation spec should surface false positives separately from misses."""
    env = _DummyEnv()
    args = argparse.Namespace(
        observation_noise_std_m=0.0,
        observation_noise_bound_m=0.0,
        missed_detection_probability=0.5,
        occlusion_distance_m=None,
        false_positive_actor_count=1,
        false_positive_offset_x_m=1.0,
        false_positive_offset_y_m=0.0,
        false_positive_spacing_y_m=0.5,
        observation_delay_steps=0,
        observation_perturbation_seed=42,
    )

    spec = _observation_perturbation_spec(args, env)

    assert spec.missed_detection_probability == pytest.approx(0.5)
    assert spec.false_positive_actor_count == 1
    np.testing.assert_array_equal(spec.false_positive_positions, [[1.0, 0.0]])
    assert spec.false_positive_ids == ["false_positive_0"]


def test_policy_obs_uses_observed_pedestrian_payload() -> None:
    """Perturbed observations should be passed to policy input without mutating the source obs."""
    original = {
        "robot": {"position": [0.0, 0.0]},
        "pedestrians": {
            "positions": [[1.0, 0.0], [5.0, 0.0]],
            "velocities": [[0.1, 0.0], [0.0, 0.2]],
            "count": [2.0],
        },
        "pedestrians_positions": [[1.0, 0.0], [5.0, 0.0]],
        "pedestrians_velocities": [[0.1, 0.0], [0.0, 0.2]],
        "pedestrians_count": [2.0],
    }
    perturbation = {
        "observed": {
            "positions": [[9.0, 9.0]],
            "velocities": [[0.0, 0.0]],
            "ids": ["ped_0"],
        }
    }

    policy_obs = _apply_observed_pedestrians_to_policy_obs(original, perturbation)

    assert policy_obs["pedestrians"]["positions"].tolist() == [[9.0, 9.0]]
    assert policy_obs["pedestrians"]["velocities"].tolist() == [[0.0, 0.0]]
    assert policy_obs["pedestrians"]["count"].tolist() == [1.0]
    assert policy_obs["pedestrians_positions"].tolist() == [[9.0, 9.0]]
    assert policy_obs["pedestrians_velocities"].tolist() == [[0.0, 0.0]]
    assert policy_obs["pedestrians_count"].tolist() == [1.0]
    assert original["pedestrians"]["positions"] == [[1.0, 0.0], [5.0, 0.0]]


def test_trace_observation_payload_separates_ground_truth_and_observed() -> None:
    """Trace rows should keep ideal and perception-limited evidence separate."""
    payload = _trace_observation_payload(
        {
            "ground_truth": {
                "positions": [[1.0, 0.0], [5.0, 0.0]],
                "velocities": [[0.1, 0.0], [0.0, 0.2]],
                "ids": ["ped_0", "ped_1"],
            },
            "observed": {
                "positions": [[1.0, 0.0]],
                "velocities": [[0.1, 0.0]],
                "ids": ["ped_0"],
            },
            "missing_ids": ["ped_1"],
            "metadata": {
                "evidence_class": "perception_limited",
                "noise_profile": "missed_detection",
                "actor_count": 2,
                "observed_actor_count": 1,
            },
        }
    )

    assert payload["ground_truth_observation"]["ids"] == ["ped_0", "ped_1"]
    assert payload["ground_truth_observation"]["evidence_class"] == "ideal_state"
    assert payload["observed_observation"]["ids"] == ["ped_0"]
    assert payload["observed_observation"]["missing_ids"] == ["ped_1"]
    assert payload["observed_observation"]["evidence_class"] == "perception_limited"
