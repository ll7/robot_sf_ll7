"""
Create a robot environment with pedestrian obstacle forces
"""

from loguru import logger
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.nav.map_config import MapDefinitionPool

# specify default map:
from robot_sf.nav.svg_map_parser import convert_map


class RobotEnvWithPedestrianObstacleForces(RobotEnv):
    """
    Robot environment with pedestrian obstacle forces
    This increases the simulation time by roughly 40%
    """

    def __init__(self, map_def=None, debug=False):
        """
        Initialize the Robot Environment with pedestrian obstacle forces
        """
        # Load the default map
        if map_def is None:
            logger.warning("No map_def provided. Using default map")
            default_map_path = "maps/svg_maps/example_map_with_obstacles.svg"
            try:
                map_def = convert_map(default_map_path)
            except FileNotFoundError:
                logger.error(f"Default map not found at {default_map_path}")
                raise
            except Exception as e:
                logger.error(f"Failed to load default map: {str(e)}")
                raise

        # create map pool with one map
        map_pool = MapDefinitionPool(map_defs={"my_map": map_def})

        # create environment settings
        env_config = EnvSettings(map_pool=map_pool)

        super().__init__(env_config=env_config, debug=debug, peds_have_obstacle_forces=True)
