"""
pytest test for robot_env_with_pedestrian_obstacle_forces.py
"""

# from gymnasium import spaces
# from stable_baselines3 import PPO

from robot_sf.gym_env.robot_env_with_pedestrian_obstacle_forces import (
    RobotEnvWithPedestrianObstacleForces
)
# from robot_sf.gym_env.pedestrian_env import PedestrianEnv
# from robot_sf.sensor.sensor_fusion import OBS_RAYS, OBS_DRIVE_STATE
from loguru import logger


def test_can_create_env():
    """
    Test that we can create an environment with pedestrian obstacle forces
    """
    env = RobotEnvWithPedestrianObstacleForces()
    assert env is not None


def test_can_simulate_with_pedestrians():
    """
    Test that we can simulate with pedestrians in the environment
    """
    total_steps = 1000
    env = RobotEnvWithPedestrianObstacleForces()
    env.reset()
    for _ in range(total_steps):
        rand_action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(rand_action)
        done = terminated or truncated
        if done:
            env.reset()

if __name__ == "__main__":
    test_can_create_env()
    test_can_simulate_with_pedestrians()
    logger.info("All tests passed")
