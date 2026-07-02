"""Scenario-switching integration tests for issue #4018 density curriculum."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import Env, spaces

from robot_sf.training.density_curriculum import build_density_curriculum_schedule
from robot_sf.training.scenario_sampling import ScenarioSampler, ScenarioSwitchingEnv


class _ConfigCaptureEnv(Env):
    """Minimal env that keeps the built config for assertions."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}


def _env_factory(*, config: dict[str, Any], **_kwargs: object) -> Env:
    return _ConfigCaptureEnv(config)


def test_curriculum_timestep_changes_stage_on_next_reset() -> None:
    """Stage updates apply when the next scenario env is constructed."""
    schedule = build_density_curriculum_schedule(
        {
            "enabled": True,
            "stages": [
                {"id": "sparse", "until_timesteps": 10, "density_m2": 0.04},
                {"id": "dense", "until_timesteps": None, "density_m2": 0.12},
            ],
        }
    )
    env = ScenarioSwitchingEnv(
        scenario_sampler=ScenarioSampler(
            [{"name": "A", "simulation_config": {"dt": 0.1}}],
            strategy="cycle",
        ),
        scenario_path="dummy",
        env_factory=_env_factory,
        config_builder=dict,
        density_curriculum=schedule,
    )

    assert env.current_curriculum_stage_id == "sparse"
    assert env._current_env.config["simulation_config"]["ped_density_by_difficulty"] == [0.04]
    env.reset()

    previous_env = env._current_env
    env.set_curriculum_timestep(10)
    assert env._current_env is previous_env
    assert env.current_curriculum_stage_id == "dense"

    env.reset()
    assert env._current_env is not previous_env
    assert env._current_env.config["simulation_config"]["ped_density_by_difficulty"] == [0.12]
