"""Tests for scenario sampling utilities used in training."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from gymnasium import Env, spaces

from robot_sf.training.scenario_sampling import ScenarioSampler, ScenarioSwitchingEnv


class _DummyEnv(Env):
    """Minimal env used to validate scenario switching behavior."""

    def __init__(
        self, obs_space: spaces.Space, act_space: spaces.Space, scenario_name: str
    ) -> None:
        super().__init__()
        self.observation_space = obs_space
        self.action_space = act_space
        self.scenario_name = scenario_name
        self.state = SimpleNamespace(max_sim_steps=1)

    def reset(self, *, seed=None, options=None):
        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action):
        obs = self.observation_space.sample()
        return obs, 0.0, True, False, {}

    def close(self) -> None:
        return None


def test_scenario_sampler_cycles_over_filtered_ids() -> None:
    """ScenarioSampler cycles in order after include filtering."""
    scenarios = [{"name": "alpha"}, {"name": "beta"}, {"name": "gamma"}]
    sampler = ScenarioSampler(
        scenarios,
        include_scenarios=("alpha", "gamma"),
        strategy="cycle",
    )
    assert sampler.scenario_ids == ("alpha", "gamma")
    first, first_id = sampler.sample()
    _second, second_id = sampler.sample()
    _third, third_id = sampler.sample()
    assert first_id == "alpha"
    assert second_id == "gamma"
    assert third_id == "alpha"
    assert first["name"] == "alpha"


def test_scenario_switching_env_tracks_coverage() -> None:
    """ScenarioSwitchingEnv increments scenario coverage on each switch."""
    scenarios = [{"name": "sc_a"}, {"name": "sc_b"}]
    sampler = ScenarioSampler(scenarios, strategy="cycle")
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def _factory(*, config, scenario_name, **kwargs):
        return _DummyEnv(obs_space, act_space, scenario_name)

    env = ScenarioSwitchingEnv(
        scenario_sampler=sampler,
        scenario_path="unused",
        env_factory=_factory,
        config_builder=lambda scenario: {"name": scenario["name"]},
        switch_per_reset=True,
        seed=123,
    )

    env.reset()
    env.reset()

    coverage = env.scenario_coverage
    assert set(coverage.keys()) == {"sc_a", "sc_b"}
    assert sum(coverage.values()) == 2

    env.close()
