"""
Visually test the Pedestrian and Obstacle forces
"""

from loguru import logger

from robot_sf.gym_env.robot_env_with_pedestrian_obstacle_forces import (
    RobotEnvWithPedestrianObstacleForces,
)
from robot_sf.nav.svg_map_parser import convert_map


def test_pedestrian_obstacle_avoidance():
    logger.info("Testing Pedestrian and Obstacle forces")
    map_def = convert_map("maps/svg_maps/example_map_with_obstacles.svg")
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
    logger.info("Testing Pedestrian and Obstacle forces")
    test_pedestrian_obstacle_avoidance()
    logger.info("All tests passed")
