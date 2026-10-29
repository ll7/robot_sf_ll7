"""The opt-in compiled RNG contract reaches native noisy reset/step observations."""

from copy import deepcopy

import numpy as np
import pytest

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.sensor import range_sensor


@pytest.mark.parametrize("drive_type", [DifferentialDriveSettings, BicycleDriveSettings])
def test_seeded_lidar_repeats_native_reset_and_steps_without_changing_physics(drive_type):
    """Default noisy observations repeat while legacy physical evolution is retained."""

    def collect(*, seed_compiled):
        config = RobotSimulationConfig(robot_config=drive_type())
        assert config.lidar_config.scan_noise == [0.005, 0.002]
        if seed_compiled:
            range_sensor.seed_compiled_lidar_rng(1001)
        env = make_robot_env(config=config, seed=1001)
        try:
            observation, _ = env.reset(seed=1001)
            frames = [(deepcopy(observation), deepcopy(env.unwrapped.simulator.robots[0].pose))]
            outcomes = []
            for _ in range(4):
                action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
                observation, reward, terminated, truncated, _ = env.step(action)
                frames.append(
                    (deepcopy(observation), deepcopy(env.unwrapped.simulator.robots[0].pose))
                )
                outcomes.append((reward, terminated, truncated))
                if terminated or truncated:
                    break
            return frames, outcomes
        finally:
            env.close()

    first, first_outcomes = collect(seed_compiled=True)
    second, second_outcomes = collect(seed_compiled=True)
    legacy, legacy_outcomes = collect(seed_compiled=False)
    assert first_outcomes == second_outcomes == legacy_outcomes
    assert len(first) == len(second) == len(legacy) == 5
    for (first_obs, first_pose), (second_obs, second_pose), (_, legacy_pose) in zip(
        first, second, legacy, strict=True
    ):
        np.testing.assert_equal(first_obs, second_obs)
        np.testing.assert_equal(first_pose, second_pose)
        np.testing.assert_equal(first_pose, legacy_pose)
