"""TODO docstring. Document this module."""

import pytest

from robot_sf.gym_env.env_config import (
    BicycleDriveRobot,
    BicycleDriveSettings,
    DifferentialDriveRobot,
    DifferentialDriveSettings,
    EnvSettings,
    LidarScannerSettings,
    MapDefinitionPool,
    SimulationSettings,
)


def test_env_settings_initialization():
    """TODO docstring. Document this function."""
    env_settings = EnvSettings()
    assert isinstance(env_settings.sim_config, SimulationSettings)
    assert isinstance(env_settings.lidar_config, LidarScannerSettings)
    assert isinstance(env_settings.robot_config, DifferentialDriveSettings)
    assert isinstance(env_settings.map_pool, MapDefinitionPool)


def test_env_settings_post_init():
    """TODO docstring. Document this function."""
    with pytest.raises(ValueError):
        _env_settings = EnvSettings(sim_config=None)  # type: ignore


def test_robot_factory():
    """TODO docstring. Document this function."""
    env_settings = EnvSettings()
    robot = env_settings.robot_factory()
    assert isinstance(robot, DifferentialDriveRobot)

    env_settings.robot_config = BicycleDriveSettings()
    robot = env_settings.robot_factory()
    assert isinstance(robot, BicycleDriveRobot)

    with pytest.raises(NotImplementedError):
        env_settings.robot_config = "unsupported type"  # type: ignore
        env_settings.robot_factory()
