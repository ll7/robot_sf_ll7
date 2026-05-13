"""Regression tests for observation-owned stack history configuration."""

from __future__ import annotations

import pytest

from robot_sf.gym_env.env_config import EnvSettings as LegacyEnvSettings
from robot_sf.gym_env.observation_config import (
    ObservationStackSettings,
    get_observation_stack_steps,
    set_observation_stack_steps,
)
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.sim.sim_config import SimulationSettings


def test_unified_config_observation_stack_is_source_of_truth() -> None:
    """Observation stack settings should own stack depth for unified configs."""
    config = RobotSimulationConfig(
        observation_stack=ObservationStackSettings(stack_steps=5),
    )

    assert get_observation_stack_steps(config) == 5
    assert config.sim_config.stack_steps == 5


def test_unified_config_hydrates_observation_stack_from_legacy_sim_setting() -> None:
    """Existing SimulationSettings.stack_steps values should migrate without shape drift."""
    config = RobotSimulationConfig(sim_config=SimulationSettings(stack_steps=2))

    assert get_observation_stack_steps(config) == 2
    assert config.observation_stack.stack_steps == 2


def test_set_observation_stack_steps_keeps_legacy_alias_in_sync() -> None:
    """Policy compatibility code should update new and legacy paths together."""
    config = RobotSimulationConfig()

    set_observation_stack_steps(config, 4)

    assert config.observation_stack.stack_steps == 4
    assert config.sim_config.stack_steps == 4


def test_legacy_env_settings_support_observation_stack_config() -> None:
    """Legacy EnvSettings should expose the same observation-owned stack contract."""
    config = LegacyEnvSettings(observation_stack=ObservationStackSettings(stack_steps=6))

    assert get_observation_stack_steps(config) == 6
    assert config.sim_config.stack_steps == 6


def test_observation_stack_rejects_non_positive_depth() -> None:
    """Stack depth must stay positive so observation spaces remain valid."""
    with pytest.raises(ValueError, match="stack_steps must be > 0"):
        ObservationStackSettings(stack_steps=0)
