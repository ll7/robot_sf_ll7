"""Tests for observation helper utilities."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from gymnasium import Env, spaces

from robot_sf.training.observation_wrappers import (
    LegacyRun023ObsAdapter,
    adapt_dict_observation_to_policy_space,
    maybe_flatten_env_observations,
    resolve_policy_obs_adapter,
)


class _DummyEnv(Env):
    """Minimal gymnasium env for exercising wrappers in tests."""

    metadata = {"render_modes": []}

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space) -> None:
        """TODO docstring. Document this function.

        Args:
            observation_space: TODO docstring.
            action_space: TODO docstring.
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        """TODO docstring. Document this function.

        Args:
            seed: TODO docstring.
            options: TODO docstring.
        """
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):  # type: ignore[override]
        """TODO docstring. Document this function.

        Args:
            action: TODO docstring.
        """
        observation = self.observation_space.sample()
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, float] = {}
        return observation, reward, terminated, truncated, info

    def render(self):  # type: ignore[override]
        """TODO docstring. Document this function."""
        return None


def test_maybe_flatten_env_noop_for_box_space():
    """TODO docstring. Document this function."""
    observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    env = _DummyEnv(observation_space, action_space)

    wrapped = maybe_flatten_env_observations(env, context="unit-test")

    assert wrapped is env
    obs, _ = wrapped.reset()
    assert obs.shape == (4,)


def test_maybe_flatten_env_wraps_dict_space():
    """TODO docstring. Document this function."""
    observation_space = spaces.Dict(
        {
            "drive_state": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "rays": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
        }
    )
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    env = _DummyEnv(observation_space, action_space)

    wrapped = maybe_flatten_env_observations(env, context="unit-test")

    assert wrapped is not env
    assert wrapped.observation_space.shape == (5,)
    obs, _ = wrapped.reset()
    assert obs.shape == (5,)
    wrapped.close()


def test_adapt_dict_observation_to_policy_space_drops_extra_keys_and_preserves_expected():
    """Dict-policy adapter should filter live env payloads down to model-declared keys."""
    policy_model = SimpleNamespace(
        observation_space=spaces.Dict(
            {
                "robot_speed": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "goal_current": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            }
        )
    )

    adapted = adapt_dict_observation_to_policy_space(
        {
            "robot_speed": [0.1, 0.2],
            "goal_current": [1.0, 2.0],
            "robot_velocity_xy": [0.1, 0.2],
            "robot_heading": [0.0],
        },
        policy_model,
    )

    assert set(adapted) == {"robot_speed", "goal_current"}
    assert adapted["robot_speed"].dtype == np.float32
    assert adapted["goal_current"].shape == (2,)


def test_adapt_dict_observation_to_policy_space_backfills_robot_speed_from_velocity_xy():
    """Compatibility alias should keep older PPO checkpoints runnable on newer env payloads."""
    policy_model = SimpleNamespace(
        observation_space=spaces.Dict(
            {
                "robot_speed": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            }
        )
    )

    adapted = adapt_dict_observation_to_policy_space(
        {"robot_velocity_xy": [0.3, -0.2]},
        policy_model,
    )

    assert np.allclose(adapted["robot_speed"], np.array([0.3, -0.2], dtype=np.float32))


def test_resolve_policy_obs_adapter_aligns_dict_observations():
    """Dict PPO checkpoints should receive a compatibility adapter instead of raw env payloads."""
    policy_model = SimpleNamespace(
        observation_space=spaces.Dict(
            {
                "robot_speed": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            }
        )
    )

    adapter = resolve_policy_obs_adapter(policy_model)

    assert adapter is not None
    adapted = adapter({"robot_velocity_xy": [0.0, 0.5], "robot_heading": [0.0]})
    assert set(adapted) == {"robot_speed"}


def test_legacy_run023_obs_adapter_flattens_drive_and_ray_state() -> None:
    """The shared legacy adapter should preserve run_023 observation semantics."""
    captured: dict[str, object] = {}

    class _Model:
        action_space = "stub"

        def predict(self, obs, deterministic: bool = True):
            captured["obs"] = obs
            captured["deterministic"] = deterministic
            return "action", None

    adapter = LegacyRun023ObsAdapter(_Model())
    obs = {
        "drive_state": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        "rays": np.array([[9.0, 8.0]], dtype=np.float32),
    }

    action, _ = adapter.predict(obs, deterministic=False)

    assert action == "action"
    assert captured["deterministic"] is False
    assert np.allclose(
        captured["obs"],
        np.array([9.0, 8.0, 1.0, 2.0, 30.0], dtype=np.float32),
    )
