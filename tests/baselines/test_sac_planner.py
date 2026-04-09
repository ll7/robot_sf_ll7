"""Tests for SAC benchmark adapter observation handling."""

from __future__ import annotations

import numpy as np

from robot_sf.baselines.sac import SACPlanner


def _planner_without_model(*, relative_obs: bool = True) -> SACPlanner:
    planner = SACPlanner(
        {
            "model_path": "output/models/sac/does_not_exist.zip",
            "fallback_to_goal": True,
            "relative_obs": relative_obs,
        }
    )
    planner._model = None
    return planner


def test_build_model_obs_dict_applies_relative_socnav_transform_without_model() -> None:
    """Inference adapter should match the relative observation contract used in training."""
    planner = _planner_without_model(relative_obs=True)

    converted = planner._build_model_obs_dict(
        {
            "robot_position": [10.0, 5.0],
            "goal_current": [13.0, 7.0],
            "goal_next": [15.0, 9.0],
            "pedestrians_positions": [[11.0, 6.0], [0.0, 0.0]],
            "robot_speed": [0.25, 0.0],
        }
    )

    assert np.allclose(converted["robot_position"], np.array([0.0, 0.0], dtype=np.float32))
    assert np.allclose(converted["goal_current"], np.array([3.0, 2.0], dtype=np.float32))
    assert np.allclose(converted["goal_next"], np.array([5.0, 4.0], dtype=np.float32))
    assert np.allclose(
        converted["pedestrians_positions"][0],
        np.array([1.0, 1.0], dtype=np.float32),
    )
    assert np.allclose(
        converted["pedestrians_positions"][1],
        np.array([0.0, 0.0], dtype=np.float32),
    )


def test_build_model_obs_dict_can_leave_absolute_socnav_transform_disabled() -> None:
    """The relative transform should stay opt-out for compatibility experiments."""
    planner = _planner_without_model(relative_obs=False)

    converted = planner._build_model_obs_dict(
        {
            "robot_position": [10.0, 5.0],
            "goal_current": [13.0, 7.0],
        }
    )

    assert np.allclose(converted["robot_position"], np.array([10.0, 5.0]))
    assert np.allclose(converted["goal_current"], np.array([13.0, 7.0]))
