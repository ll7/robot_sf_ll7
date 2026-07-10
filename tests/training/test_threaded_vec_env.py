"""Contract tests for the in-process threaded Stable-Baselines3 vector environment."""

from __future__ import annotations

from threading import Barrier

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from robot_sf.training.threaded_vec_env import ThreadedVecEnv


class _BarrierEnv(gym.Env):
    """Small environment that proves sibling steps execute concurrently."""

    metadata = {"render_modes": []}

    def __init__(self, barrier: Barrier, *, terminate: bool = False) -> None:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        self._barrier = barrier
        self._terminate = terminate
        self.reset_count = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.reset_count += 1
        return np.array([self.reset_count], dtype=np.float32), {"seed": seed}

    def step(self, action: np.ndarray):
        self._barrier.wait(timeout=2)
        observation = np.asarray(action, dtype=np.float32)
        return observation, float(observation[0]), self._terminate, False, {"stepped": True}


def test_threaded_vec_env_steps_sibling_environments_concurrently() -> None:
    """Threaded mode must complete mutually waiting sibling steps and batch their results."""
    barrier = Barrier(2)
    vec_env = ThreadedVecEnv([lambda: _BarrierEnv(barrier), lambda: _BarrierEnv(barrier)])
    try:
        observations = vec_env.reset()
        assert observations.shape == (2, 1)

        next_obs, rewards, dones, infos = vec_env.step(np.array([[0.25], [0.75]], dtype=np.float32))

        np.testing.assert_allclose(next_obs, [[0.25], [0.75]])
        np.testing.assert_allclose(rewards, [0.25, 0.75])
        assert not dones.any()
        assert infos == [
            {"stepped": True, "TimeLimit.truncated": False},
            {"stepped": True, "TimeLimit.truncated": False},
        ]
    finally:
        vec_env.close()


def test_threaded_vec_env_preserves_sb3_terminal_observation_and_auto_reset() -> None:
    """Done environments must expose terminal observations before SB3-compatible auto-reset."""
    barrier = Barrier(1)
    vec_env = ThreadedVecEnv([lambda: _BarrierEnv(barrier, terminate=True)])
    try:
        vec_env.reset()
        observations, _rewards, dones, infos = vec_env.step(np.array([[0.5]], dtype=np.float32))

        assert dones.tolist() == [True]
        np.testing.assert_allclose(infos[0]["terminal_observation"], [0.5])
        np.testing.assert_allclose(observations, [[2.0]])
        assert infos[0]["TimeLimit.truncated"] is False
    finally:
        vec_env.close()


def test_threaded_vec_env_rejects_overlapping_steps_and_invalid_worker_count() -> None:
    """The asynchronous VecEnv lifecycle must fail closed on invalid concurrent use."""
    barrier = Barrier(1)
    with pytest.raises(ValueError, match="at least 1"):
        ThreadedVecEnv([lambda: _BarrierEnv(barrier)], max_workers=0)

    vec_env = ThreadedVecEnv([lambda: _BarrierEnv(barrier)])
    try:
        vec_env.reset()
        vec_env.step_async(np.array([[0.0]], dtype=np.float32))
        with pytest.raises(RuntimeError, match="while a step is pending"):
            vec_env.step_async(np.array([[0.0]], dtype=np.float32))
        vec_env.step_wait()
        with pytest.raises(RuntimeError, match="before step_async"):
            vec_env.step_wait()
    finally:
        vec_env.close()
