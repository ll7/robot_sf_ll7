"""Tests for force flag deprecation and syncing."""

from __future__ import annotations

from robot_sf.gym_env.unified_config import RobotSimulationConfig


def test_force_flags_default_sync() -> None:
    """Keep new force flags in sync with legacy defaults to avoid regressions."""
    cfg = RobotSimulationConfig()
    assert cfg.peds_have_static_obstacle_forces is True
    assert cfg.peds_have_robot_repulsion is True
    assert cfg.sim_config.prf_config.is_active is True
    assert cfg.peds_have_obstacle_forces is True


def test_force_flags_deprecated_alias_sets_static() -> None:
    """Map deprecated field onto static obstacle forces for compatibility."""
    cfg = RobotSimulationConfig(peds_have_obstacle_forces=False)
    assert cfg.peds_have_static_obstacle_forces is False
    assert cfg.peds_have_obstacle_forces is False


def test_force_flags_robot_repulsion_updates_sim_config() -> None:
    """Ensure robot repulsion flag updates the simulator config as expected."""
    cfg = RobotSimulationConfig(peds_have_robot_repulsion=False)
    assert cfg.peds_have_robot_repulsion is False
    assert cfg.sim_config.prf_config.is_active is False
