"""
pytest test for robot_env_with_pedestrian_obstacle_forces.py
"""

# from gymnasium import spaces
# from stable_baselines3 import PPO

# from robot_sf.gym_env.pedestrian_env import PedestrianEnv
# from robot_sf.sensor.sensor_fusion import OBS_RAYS, OBS_DRIVE_STATE
from functools import partial

import numpy as np
import pytest
from loguru import logger

from robot_sf.benchmark.seed_bands import DEV_SEEDS
from robot_sf.gym_env import robot_env_with_pedestrian_obstacle_forces as obstacle_env
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.env_util import global_reset_seed
from robot_sf.gym_env.robot_env_with_pedestrian_obstacle_forces import (
    RobotEnvWithPedestrianObstacleForces,
)
from robot_sf.sim.sim_config import SimulationSettings


@pytest.fixture(autouse=True)
def deterministic_population(monkeypatch):
    """Seed population construction before reset and restore legacy RNG state afterward."""
    monkeypatch.setattr(
        obstacle_env,
        "EnvSettings",
        partial(EnvSettings, sim_config=SimulationSettings(pedestrian_seed=DEV_SEEDS[0])),
    )
    with global_reset_seed(DEV_SEEDS[0]):
        yield


def test_initial_population_is_repeatable():
    """Smoke-test construction must not depend on ambient population entropy."""
    positions = []
    for _ in range(2):
        env = RobotEnvWithPedestrianObstacleForces()
        try:
            positions.append(env.simulator.pysf_sim.peds.pos().copy())
        finally:
            env.close()
    np.testing.assert_array_equal(positions[0], positions[1])


def test_can_create_env():
    """
    Test that we can create an environment with pedestrian obstacle forces
    """
    env = RobotEnvWithPedestrianObstacleForces()
    try:
        assert env is not None
    finally:
        env.close()


def test_can_simulate_with_pedestrians():
    """
    Test that we can simulate with pedestrians in the environment
    """
    total_steps = 1000
    env = RobotEnvWithPedestrianObstacleForces()
    try:
        env.action_space.seed(DEV_SEEDS[0])
        env.reset(seed=DEV_SEEDS[0])
        for _ in range(total_steps):
            rand_action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(rand_action)
            done = terminated or truncated
            if done:
                env.reset(seed=DEV_SEEDS[0])
    finally:
        env.close()


if __name__ == "__main__":
    test_can_create_env()
    test_can_simulate_with_pedestrians()
    logger.info("All tests passed")
