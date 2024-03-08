"""
Module to write a simple robot environment that is compatible with gymnasium.
The focus of this module is to use more "of-the-shelf" components from sb3 and
to be natively compatible with gymnasium.
"""

import gymnasium

class SimpleRobotEnv(gymnasium.Env):
    """
    A simple robot environment based on gymnasium.Env.
    """

    def __init__(self):
        
        self.action_space = gymnasium.spaces.Discrete(2)

        self.observation_space = gymnasium.spaces.Discrete(5)

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass

