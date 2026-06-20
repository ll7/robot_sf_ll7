"""Gymnasium contract checks for the public multi-robot environment."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from robot_sf.gym_env.environment_factory import make_multi_robot_env, make_robot_env
from robot_sf.gym_env.unified_config import MultiRobotConfig, RobotSimulationConfig
from robot_sf.sensor.range_sensor import LidarScannerSettings


def test_multi_robot_observation_space_contains_reset_and_step_observations() -> None:
    """MultiRobotEnv should publish the vectorized observation contract it returns."""
    env = make_multi_robot_env(config=MultiRobotConfig(num_robots=2), seed=123)

    try:
        assert isinstance(env.observation_space, spaces.Dict)

        reset_obs, reset_info = env.reset(seed=123)
        assert reset_info["seed"] == 123
        assert env.observation_space.contains(reset_obs)

        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        step_obs, reward, terminated, truncated, info = env.step(action)

        assert env.observation_space.contains(step_obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert truncated is False
        assert "agents" in info
    finally:
        env.close()


def test_public_robot_envs_pass_gymnasium_checker_with_deterministic_lidar() -> None:
    """Public envs should pass check_env when stochastic lidar noise is disabled."""
    lidar = LidarScannerSettings(scan_noise=[0.0, 0.0])
    envs = [
        make_robot_env(config=RobotSimulationConfig(lidar_config=lidar), seed=321),
        make_multi_robot_env(config=MultiRobotConfig(num_robots=2, lidar_config=lidar), seed=321),
    ]

    for env in envs:
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()
