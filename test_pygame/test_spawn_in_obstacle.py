"""
Test what happens if pedestrian and robot spawn in an obstacle.
"""

import pytest
from loguru import logger

from robot_sf.gym_env.robot_env_with_pedestrian_obstacle_forces import (
    RobotEnvWithPedestrianObstacleForces,
)
from robot_sf.nav.svg_map_parser import convert_map


def test_spawn_in_obstacle():
    """Ensure obstacle-aware spawning fails fast on invalid zones."""
    logger.info("Testing Spawn in Obstacle")
    map_def = convert_map("maps/svg_maps/test_spawn_in_obstacle.svg")
    logger.debug(f"type map_def: {type(map_def)}")
    with pytest.raises(RuntimeError, match="Failed to sample"):
        RobotEnvWithPedestrianObstacleForces(map_def=map_def, debug=True)


if __name__ == "__main__":
    logger.info("Testing Spawn in Obstacle")
    test_spawn_in_obstacle()
    logger.info("All tests passed")
