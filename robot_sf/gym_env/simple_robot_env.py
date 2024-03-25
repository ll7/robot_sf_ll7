"""
Module to write a simple robot environment that is compatible with gymnasium.
The focus of this module is to use more "of-the-shelf" components from sb3 and
to be natively compatible with gymnasium.
"""

import gymnasium

from robot_sf.gym_env.env_config import EnvSettings



class SimpleRobotEnv(gymnasium.Env):
    """
    A simple robot environment based on gymnasium.Env.
    """

    def __init__(
            self,
            env_config: EnvSettings = EnvSettings()
            ):
        
        self.info = {}

        # Action space
        # TODO: Action space depends on the robot type? Differential drive or bicycle model?
        self.action_space = self._define_action_space()
        
        # Observation Space
        self.observation_space = self._define_observation_space()
        
    def _define_action_space(self):
        """
        Action Space:
        - 2 actions: move forward, turn
        """
        return gymnasium.spaces.Discrete(2)


    def _define_observation_space(self):
        """
        Observation Space:
        - x, y, theta, v_x, v_y, omega
        - goal_x, goal_y
        - distance and angle to goal
        - lidar observations
        """
        return gymnasium.spaces.Discrete(5)
    
    def _get_observation(self):
        """
        Get the current observation
        """
        return self.observation
    
    def _get_info(self):
        """
        Get the current info
        """
        return self.info


    def step(self, action):
        pass

    def reset(self, seed=None):
        # seed for self.np_random
        super().reset(seed=seed)



        
        observation = self_get_observation()
        info = self._get_info()
        return observation, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

