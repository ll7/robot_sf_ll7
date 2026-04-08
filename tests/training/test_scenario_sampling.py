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
    """Minimal env with configurable observation bounds."""

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space) -> None:
        """Initialize deterministic spaces for wrapper assertions."""
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.closed = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Return one zero observation."""
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        """Return one terminal transition."""
        return (
            np.zeros(self.observation_space.shape, dtype=np.float32),
            0.0,
            True,
            False,
            {"action_seen": np.asarray(action)},
        )

    def close(self) -> None:
        """Mark the env closed."""
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
    sampler = ScenarioSampler(
        [{"name": "A"}, {"name": "B"}],
        include_scenarios=("A", "B"),
        strategy="cycle",
    )
    assert sampler.sample()[1] == "A"
    assert sampler.sample()[1] == "B"


def test_scenario_sampler_rejects_bad_weights() -> None:
    """Weights should be non-negative and include at least one positive entry."""
    scenarios = [{"name": "A"}, {"name": "B"}]
    with pytest.raises(ValueError):
        ScenarioSampler(scenarios, weights={"A": -1.0})
    with pytest.raises(ValueError):
        ScenarioSampler(scenarios, weights={"A": 0.0, "B": 0.0})


def test_scenario_sampler_invalid_filters_raise() -> None:
    """Unknown include/exclude filters should raise a validation error."""
    scenarios = [{"name": "A"}]
    with pytest.raises(ValueError):
        ScenarioSampler(scenarios, include_scenarios=("missing",))
    with pytest.raises(ValueError):
        ScenarioSampler(scenarios, exclude_scenarios=("missing",))


def test_scenario_switching_env_handles_bounds_mismatch() -> None:
    """Switching env should allow bounds mismatches but reject shape changes."""
    scenarios = [
        {"name": "A", "low": [0.0, 0.0], "high": [1.0, 1.0]},
        {"name": "B", "low": [-1.0, -1.0], "high": [1.0, 1.0]},
    ]
    env = ScenarioSwitchingEnv(
        scenario_sampler=ScenarioSampler(scenarios, strategy="cycle"),
        scenario_path="dummy",
        env_factory=_env_factory,
        config_builder=dict,
        switch_per_reset=True,
    )
    env.reset()
    env.reset()
    assert set(env.scenario_coverage) == {"A", "B"}

    bad_scenarios = [
        {"name": "C", "low": [0.0, 0.0], "high": [1.0, 1.0]},
        {"name": "D", "low": [0.0, 0.0, 0.0], "high": [1.0, 1.0, 1.0]},
    ]
    bad_env = ScenarioSwitchingEnv(
        scenario_sampler=ScenarioSampler(bad_scenarios, strategy="cycle"),
        scenario_path="dummy",
        env_factory=_env_factory,
        config_builder=dict,
        switch_per_reset=True,
    )
    bad_env.reset()
    with pytest.raises(ValueError):
        bad_env.reset()


def test_scenario_switching_env_step_requires_active_env() -> None:
    """Step should raise when the active environment is missing."""
    env = ScenarioSwitchingEnv(
        scenario_sampler=ScenarioSampler(
            [{"name": "A", "low": [0.0, 0.0], "high": [1.0, 1.0]}],
            strategy="cycle",
        ),
        scenario_path="dummy",
        env_factory=_env_factory,
        config_builder=dict,
        switch_per_reset=False,
    )
    env.close()
    with pytest.raises(RuntimeError):
        env.step(np.array([0.0, 0.0], dtype=np.float32))
