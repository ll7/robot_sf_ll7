"""
test the new environment implementation
"""

from gymnasium import spaces
import numpy as np

from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.robot_env_from_base import RobotEnvFromBase
from robot_sf.gym_env.env_config import RobotEnvSettings
from robot_sf.sensor.sensor_fusion import OBS_RAYS, OBS_DRIVE_STATE


def test_new_robot_env():
    """
    Test the new robot environment implementation.
    """
    new_env_config = RobotEnvSettings()
    new_env = RobotEnvFromBase(
        env_config=new_env_config,
    )
    assert new_env is not None


def test_can_return_valid_observation():
    env = RobotEnvFromBase()
    drive_state_spec: spaces.Box = env.observation_space[OBS_DRIVE_STATE]
    lidar_state_spec: spaces.Box = env.observation_space[OBS_RAYS]

    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert OBS_DRIVE_STATE in obs and OBS_RAYS in obs
    assert drive_state_spec.shape == obs[OBS_DRIVE_STATE].shape
    assert lidar_state_spec.shape == obs[OBS_RAYS].shape


def test_can_simulate_with_pedestrians():
    total_steps = 1000
    env = RobotEnvFromBase()
    env.reset()
    for _ in range(total_steps):
        rand_action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(rand_action)
        done = terminated or truncated
        if done:
            env.reset()


if __name__ == "__main__":
    test_new_robot_env()
    test_can_return_valid_observation()
    test_can_simulate_with_pedestrians()
