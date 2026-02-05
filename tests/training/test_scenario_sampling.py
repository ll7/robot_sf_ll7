"""Tests for scenario sampling utilities and switching env wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from gymnasium import Env, spaces

from robot_sf.training.scenario_sampling import (
    ScenarioSampler,
    ScenarioSwitchingEnv,
    _spaces_compatible,
    scenario_id_from_definition,
)


class _DummyEnv(Env):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.closed = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros(self.observation_space.shape, dtype=np.float32),
            0.0,
            True,
            False,
            {},
        )

    def close(self) -> None:
        self.closed = True


def _env_factory(*, config: dict[str, Any], **_kwargs) -> Env:
    low = np.asarray(config["low"], dtype=np.float32)
    high = np.asarray(config["high"], dtype=np.float32)
    obs_space = spaces.Box(low=low, high=high, dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return _DummyEnv(obs_space, action_space)


def test_scenario_id_requires_name() -> None:
    """Scenario ID derivation should fail without identifying fields."""
    with pytest.raises(ValueError):
        scenario_id_from_definition({"foo": "bar"}, index=0)


def test_spaces_compatible_allows_bounds_mismatch() -> None:
    """Box compatibility should be strict unless bounds mismatches are allowed."""
    base = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
    other = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    assert _spaces_compatible(base, other, allow_box_bounds_mismatch=False) is False
    assert _spaces_compatible(base, other, allow_box_bounds_mismatch=True) is True


def test_scenario_sampler_cycles_and_filters() -> None:
    """Sampler should honor include filters and cycle strategy."""
    scenarios = [{"name": "A"}, {"name": "B"}]
    sampler = ScenarioSampler(scenarios, include_scenarios=("A", "B"), strategy="cycle")
    first = sampler.sample()[1]
    second = sampler.sample()[1]
    assert first != second


def test_scenario_switching_env_handles_bounds_mismatch() -> None:
    """Switching env should allow bounds mismatches but reject shape changes."""
    scenarios = [
        {"name": "A", "low": [0.0, 0.0], "high": [1.0, 1.0]},
        {"name": "B", "low": [-1.0, -1.0], "high": [1.0, 1.0]},
    ]
    sampler = ScenarioSampler(scenarios, strategy="cycle")
    env = ScenarioSwitchingEnv(
        scenario_sampler=sampler,
        scenario_path="dummy",
        env_factory=_env_factory,
        config_builder=lambda scenario: scenario,
        switch_per_reset=True,
    )
    env.reset()
    env.reset()
    assert set(env.scenario_coverage) == {"A", "B"}

    bad_scenarios = [
        {"name": "C", "low": [0.0, 0.0], "high": [1.0, 1.0]},
        {"name": "D", "low": [0.0, 0.0, 0.0], "high": [1.0, 1.0, 1.0]},
    ]
    bad_sampler = ScenarioSampler(bad_scenarios, strategy="cycle")
    bad_env = ScenarioSwitchingEnv(
        scenario_sampler=bad_sampler,
        scenario_path="dummy",
        env_factory=_env_factory,
        config_builder=lambda scenario: scenario,
        switch_per_reset=True,
    )
    bad_env.reset()
    with pytest.raises(ValueError):
        bad_env.reset()
