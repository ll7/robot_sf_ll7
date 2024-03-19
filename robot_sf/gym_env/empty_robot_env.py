"""
An empty environment for the robot to drive to several goals.
"""

import gymnasium
from robot_sf.gym_env.env_config import EnvSettings


class EmptyRobotEnv(gymnasium.Env):
    """
    A simple robot environment based on gymnasium.Env.
    """

    def __init__(
            self,
            env_config: EnvSettings = EnvSettings(),
            debug: bool = False
            ):
        """
        Initialize the EmptyRobotEnv environment.
        :param env_config: Environment settings.
        :param debug: Debug flag.
        """

        self.env_config = env_config
        self.debug = debug

        map_def = env_config.map_pool.map_defs[0]

        
        
        
        self.info = {}

        

        # Action space