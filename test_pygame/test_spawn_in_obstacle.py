"""
Test what happens if pedestrian and robot spawn in an obstacle.
"""

from loguru import logger

from robot_sf.gym_env.robot_env_with_pedestrian_obstacle_forces import (
    RobotEnvWithPedestrianObstacleForces,
)
from robot_sf.nav.svg_map_parser import convert_map


def test_spawn_in_obstacle():
    logger.info("Testing Spawn in Obstacle")
    map_def = convert_map("maps/svg_maps/test_spawn_in_obstacle.svg")
    logger.debug(f"type map_def: {type(map_def)}")
    env = RobotEnvWithPedestrianObstacleForces(map_def=map_def, debug=True)
    logger.info("created environment")
    env.reset()
    for _ in range(1000):
        rand_action = env.action_space.sample()
        _, _, done, _, _ = env.step(rand_action)
        env.render()
        if done:
            env.reset()
    env.close()


if __name__ == "__main__":
    logger.info("Testing Spawn in Obstacle")
    test_spawn_in_obstacle()
    logger.info("All tests passed")
