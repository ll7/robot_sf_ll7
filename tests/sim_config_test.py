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


def test_desired_speed_default_preserves_legacy_behavior():
    """Default SimulationSettings must not set a decoupled desired speed (issue #4972).

    With no tier / explicit mean, desired_speed_mean stays None so the simulator
    keeps the legacy ``peds_speed_mult * initial_speed`` derivation.
    """
    settings = SimulationSettings()
    assert settings.ped_speed_tier is None
    assert settings.desired_speed_mean is None
    assert settings.desired_speed_std is None


def test_ped_speed_tier_derives_desired_speed_params():
    """A speed tier should derive literature-calibrated desired-speed params (issue #4972)."""
    settings = SimulationSettings(ped_speed_tier="typical")
    assert settings.ped_speed_tier == "typical"
    assert settings.desired_speed_mean == pytest.approx(1.3)
    assert settings.desired_speed_std == pytest.approx(0.2)

    slow = SimulationSettings(ped_speed_tier="slow")
    assert slow.desired_speed_mean == pytest.approx(0.65)

    brisk = SimulationSettings(ped_speed_tier="Brisk")
    assert brisk.ped_speed_tier == "brisk"
    assert brisk.desired_speed_mean == pytest.approx(1.6)


def test_explicit_desired_speed_overrides_tier():
    """Explicit desired_speed_mean/std must take precedence over the tier mapping."""
    settings = SimulationSettings(
        ped_speed_tier="typical", desired_speed_mean=1.0, desired_speed_std=0.1
    )
    assert settings.desired_speed_mean == pytest.approx(1.0)
    assert settings.desired_speed_std == pytest.approx(0.1)


def test_desired_speed_validation_rejects_invalid_values():
    """Bad tier keys and negative/non-finite desired-speed values must raise."""
    with pytest.raises(ValueError, match="Unsupported ped_speed_tier"):
        SimulationSettings(ped_speed_tier="sprint")

    with pytest.raises(ValueError, match="desired_speed_mean"):
        SimulationSettings(desired_speed_mean=-0.1)

    with pytest.raises(ValueError, match="desired_speed_mean"):
        SimulationSettings(desired_speed_mean=float("inf"))

    with pytest.raises(ValueError, match="desired_speed_std"):
        SimulationSettings(desired_speed_mean=1.3, desired_speed_std=-0.2)
