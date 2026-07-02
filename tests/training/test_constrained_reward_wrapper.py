"""Tests for the constrained reward wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import Env, spaces

from robot_sf.training.constrained_reward_wrapper import ConstrainedRewardWrapper
from robot_sf.training.safety_constraints import SafetyConstraintSpec


class _DummyEnv(Env):
    """Small deterministic environment for wrapper behavior tests."""

    metadata = {"render_modes": []}

    def __init__(self, *, terminal: bool = False) -> None:
        """Initialize a one-step test environment."""
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)
        self.terminal = terminal

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return the deterministic initial observation."""
        super().reset(seed=seed)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Return a fixed reward and near-miss cost payload."""
        del action
        return (
            np.array([0.5], dtype=np.float32),
            2.0,
            self.terminal,
            False,
            {"meta": {"near_misses": 2.0}},
        )


def _near_miss_spec() -> SafetyConstraintSpec:
    return SafetyConstraintSpec(
        name="near_miss",
        source_key="near_miss",
        budget_per_episode=0.5,
        multiplier_init=0.25,
        multiplier_lr=0.1,
        multiplier_max=1.0,
    )


def test_wrapper_preserves_observation_and_termination_outputs() -> None:
    """Observation and terminal flags pass through unchanged."""
    env = ConstrainedRewardWrapper(_DummyEnv(terminal=True), [_near_miss_spec()])

    observation, reward, terminated, truncated, info = env.step(0)

    assert observation.shape == (1,)
    assert reward == 1.5
    assert terminated is True
    assert truncated is False
    assert info["raw_task_reward"] == 2.0


def test_constrained_reward_decreases_when_cost_multiplier_is_positive() -> None:
    """Positive safety costs reduce the training reward."""
    env = ConstrainedRewardWrapper(_DummyEnv(), [_near_miss_spec()])

    _, reward, _, _, info = env.step(0)

    assert reward < info["raw_task_reward"]
    assert info["constrained_reward"] == reward
    assert info["constraint_costs"] == {"near_miss": 2.0}
    assert info["constraint_multipliers"] == {"near_miss": 0.25}


def test_episode_cost_summary_appears_on_terminal_info() -> None:
    """Terminal steps include episode cost, budget, and violation diagnostics."""
    env = ConstrainedRewardWrapper(_DummyEnv(terminal=True), [_near_miss_spec()])

    _, _, _, _, info = env.step(0)

    episode = info["constraint_episode"]
    assert episode["costs"] == {"near_miss": 2.0}
    assert episode["budgets"] == {"near_miss": 0.5}
    assert episode["violations"] == {"near_miss": 1.5}
    assert episode["multipliers_before_update"] == {"near_miss": 0.25}


def test_multiplier_update_is_explicit_not_step_side_effect() -> None:
    """Vectorized training callbacks update multipliers explicitly after episodes."""
    env = ConstrainedRewardWrapper(_DummyEnv(terminal=True), [_near_miss_spec()])

    _, _, _, _, info = env.step(0)
    assert env.multiplier_state.values == {"near_miss": 0.25}

    updated = env.update_multipliers_from_episode(
        info["constraint_episode"]["costs"],
        episode_steps=info["constraint_episode"]["episode_steps"],
    )

    assert updated == {"near_miss": 0.4}
