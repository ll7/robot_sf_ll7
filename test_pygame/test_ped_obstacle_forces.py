"""
Visually test the Pedestrian and Obstacle forces
"""

from robot_sf.gym_env.robot_env_with_pedestrian_obstacle_forces import (
    RobotEnvWithPedestrianObstacleForces,
)
from loguru import logger

def test_pedestrian_obstacle_avoidance():
    env = RobotEnvWithPedestrianObstacleForces(
        map_def="maps/svg/map"
    )
    env.reset()
    for _ in range(1000):
        rand_action = env.action_space.sample()
        _, _, done, _, _ = env.step(rand_action)
        env.render()
        if done:
            env.reset()
    env.close()

if __name__ == "__main__":
    test_pedestrian_obstacle_avoidance()
    logger.info("All tests passed")
