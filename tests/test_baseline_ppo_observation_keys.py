"""Regression tests for PPO dict observation key normalization."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from gymnasium import spaces as gym_spaces

from robot_sf.baselines.ppo import PPOPlanner, PPOPlannerConfig


class _FakePPOModel:
    """Small Stable-Baselines3-like model double for observation adapter tests."""

    def __init__(self, observation_space: gym_spaces.Dict) -> None:
        self.observation_space = observation_space
        self.last_obs: dict[str, Any] | None = None

    def predict(
        self,
        obs: dict[str, Any],
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray, None]:
        """Capture model-ready observation and return a deterministic velocity action."""
        self.last_obs = obs
        return np.array([0.4, -0.2], dtype=np.float32), None


def _box(shape: tuple[int, ...]) -> gym_spaces.Box:
    """Return a float32 Box leaf for PPO Dict observation spaces."""
    return gym_spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)


def _planner_with_model(model: _FakePPOModel) -> PPOPlanner:
    """Build a PPOPlanner with a fake loaded model and no checkpoint IO."""
    planner = PPOPlanner.__new__(PPOPlanner)
    planner.config = PPOPlannerConfig(
        obs_mode="dict",
        action_space="velocity",
        fallback_to_goal=False,
    )
    planner._seed = None
    planner._model = model
    planner._status = "ok"
    planner._fallback_reason = None
    planner._predictive_foresight = None
    planner._runtime_observation_space = None
    return planner


def test_ppo_dict_obs_maps_flat_runner_keys_to_nested_model_space() -> None:
    """Flat map-runner observations should satisfy nested robot_env-style PPO spaces."""
    model = _FakePPOModel(
        gym_spaces.Dict(
            {
                "robot": gym_spaces.Dict(
                    {
                        "position": _box((2,)),
                        "velocity": _box((2,)),
                    }
                ),
                "goal": gym_spaces.Dict({"current": _box((2,))}),
                "pedestrians": gym_spaces.Dict(
                    {
                        "positions": _box((2, 2)),
                        "velocities": _box((2, 2)),
                        "count": _box((1,)),
                        "radius": _box((1,)),
                    }
                ),
            }
        )
    )
    planner = _planner_with_model(model)
    flat_obs = {
        "robot_position": np.array([1.0, 2.0], dtype=np.float32),
        "robot_velocity": np.array([0.1, 0.2], dtype=np.float32),
        "goal_current": np.array([4.0, 5.0], dtype=np.float32),
        "pedestrians_positions": np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        "pedestrians_velocities": np.zeros((2, 2), dtype=np.float32),
        "pedestrians_count": np.array([2.0], dtype=np.float32),
        "pedestrians_radius": np.array([0.35], dtype=np.float32),
    }

    action = planner.step(flat_obs)

    assert action == {"vx": pytest.approx(0.4), "vy": pytest.approx(-0.2)}
    assert model.last_obs is not None
    np.testing.assert_allclose(model.last_obs["robot"]["position"], [1.0, 2.0])
    np.testing.assert_allclose(model.last_obs["robot"]["velocity"], [0.1, 0.2])
    np.testing.assert_allclose(model.last_obs["goal"]["current"], [4.0, 5.0])
    np.testing.assert_allclose(
        model.last_obs["pedestrians"]["positions"],
        [[2.0, 2.0], [3.0, 3.0]],
    )


def test_ppo_dict_obs_keeps_nested_env_obs_compatible_with_flat_model_space() -> None:
    """Nested env observations should continue to satisfy flattened SB3-compatible spaces."""
    model = _FakePPOModel(
        gym_spaces.Dict(
            {
                "robot_position": _box((2,)),
                "goal_current": _box((2,)),
                "pedestrians_positions": _box((2, 2)),
            }
        )
    )
    planner = _planner_with_model(model)
    nested_obs = {
        "robot": {"position": np.array([1.0, 2.0], dtype=np.float32)},
        "goal": {"current": np.array([4.0, 5.0], dtype=np.float32)},
        "pedestrians": {"positions": np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32)},
    }

    action = planner.step(nested_obs)

    assert action == {"vx": pytest.approx(0.4), "vy": pytest.approx(-0.2)}
    assert model.last_obs is not None
    assert set(model.last_obs) == {
        "robot_position",
        "goal_current",
        "pedestrians_positions",
    }
    np.testing.assert_allclose(model.last_obs["robot_position"], [1.0, 2.0])
    np.testing.assert_allclose(model.last_obs["goal_current"], [4.0, 5.0])
