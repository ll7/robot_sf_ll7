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


def test_forecast_variant_default_and_validation():
    """forecast_variant should default to none and reject unknown variants."""

    env_settings = EnvSettings()
    assert env_settings.forecast_variant == "none"

    valid_settings = EnvSettings(forecast_variant="cv")
    assert valid_settings.forecast_variant == "cv"

    with pytest.raises(ValueError, match="forecast_variant"):
        EnvSettings(forecast_variant="unknown_variant")


def test_non_reactive_response_multiplier_default_and_validation():
    """non_reactive_response_multiplier should default to 0.0 and reject invalid values.

    This is a regression test (issue #4850) ensuring the default behavior is preserved:
    non-reactive/non-yielding pedestrians do not respond to the robot.
    """

    env_settings = EnvSettings()
    assert env_settings.sim_config.non_reactive_response_multiplier == 0.0

    # Valid multiplier values
    env_settings.sim_config = SimulationSettings(non_reactive_response_multiplier=0.1)
    assert env_settings.sim_config.non_reactive_response_multiplier == 0.1

    env_settings.sim_config = SimulationSettings(non_reactive_response_multiplier=0.3)
    assert env_settings.sim_config.non_reactive_response_multiplier == 0.3

    # Invalid multiplier values
    with pytest.raises(ValueError, match="non_reactive_response_multiplier"):
        SimulationSettings(non_reactive_response_multiplier=-0.1)

    with pytest.raises(ValueError, match="non_reactive_response_multiplier"):
        SimulationSettings(non_reactive_response_multiplier=float("inf"))
