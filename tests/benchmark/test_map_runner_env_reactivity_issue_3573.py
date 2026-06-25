"""Tests for the scenario-driven pedestrian-reactivity toggle in build_env_config (#3573)."""

from __future__ import annotations

from robot_sf.benchmark.map_runner_env import apply_pedestrian_reactivity_to_env_config
from robot_sf.gym_env.unified_config import RobotSimulationConfig


def test_reactivity_absent_key_preserves_reactive_default() -> None:
    """No scenario key leaves the robot->pedestrian force untouched (reactive default)."""
    config = RobotSimulationConfig()
    before = config.sim_config.prf_config.is_active
    apply_pedestrian_reactivity_to_env_config(config, scenario={})
    assert config.sim_config.prf_config.is_active is before


def test_reactivity_true_keeps_robot_repulsion_active() -> None:
    """``peds_have_robot_repulsion: True`` keeps the robot-response force on (reactive)."""
    config = RobotSimulationConfig()
    apply_pedestrian_reactivity_to_env_config(config, scenario={"peds_have_robot_repulsion": True})
    assert config.sim_config.prf_config.is_active is True
    assert config.peds_have_robot_repulsion is True


def test_reactivity_false_disables_robot_repulsion_for_replay() -> None:
    """``peds_have_robot_repulsion: False`` disables the force (open-loop non-reactive replay)."""
    config = RobotSimulationConfig()
    apply_pedestrian_reactivity_to_env_config(config, scenario={"peds_have_robot_repulsion": False})
    assert config.sim_config.prf_config.is_active is False
    assert config.peds_have_robot_repulsion is False
