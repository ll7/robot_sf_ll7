"""Module sim_config_test auto-generated docstring."""

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
    """Test env settings initialization.

    Returns:
        Any: Auto-generated placeholder description.
    """
    env_settings = EnvSettings()
    assert isinstance(env_settings.sim_config, SimulationSettings)
    assert isinstance(env_settings.lidar_config, LidarScannerSettings)
    assert isinstance(env_settings.robot_config, DifferentialDriveSettings)
    assert isinstance(env_settings.map_pool, MapDefinitionPool)


def test_env_settings_post_init():
    """Test env settings post init.

    Returns:
        Any: Auto-generated placeholder description.
    """
    with pytest.raises(ValueError):
        _env_settings = EnvSettings(sim_config=None)  # type: ignore


def test_robot_factory():
    """Test robot factory.

    Returns:
        Any: Auto-generated placeholder description.
    """
    env_settings = EnvSettings()
    robot = env_settings.robot_factory()
    assert isinstance(robot, DifferentialDriveRobot)

    env_settings.robot_config = BicycleDriveSettings()
    robot = env_settings.robot_factory()
    assert isinstance(robot, BicycleDriveRobot)

    with pytest.raises(NotImplementedError):
        env_settings.robot_config = "unsupported type"  # type: ignore
        env_settings.robot_factory()
