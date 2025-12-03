"""Tests for observation helper utilities."""

from __future__ import annotations

import numpy as np
from gymnasium import Env, spaces

from robot_sf.training.observation_wrappers import maybe_flatten_env_observations


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
