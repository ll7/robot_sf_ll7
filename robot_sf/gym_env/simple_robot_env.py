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

    def __init__(self, env_config: EnvSettings = EnvSettings()):
        """TODO docstring. Document this function.

        Args:
            env_config: TODO docstring.
        """
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

        Returns:
            Discrete action space with 2 actions.
        """
        return gymnasium.spaces.Discrete(2)

    def _define_observation_space(self):
        """
        Observation Space:
        - x, y, theta, v_x, v_y, omega
        - goal_x, goal_y
        - distance and angle to goal
        - lidar observations

        Returns:
            Discrete observation space with 5 states.
        """
        return gymnasium.spaces.Discrete(5)

    def _get_observation(self):
        """
        Get the current observation

        Returns:
            Current observation state from the environment.
        """
        return self.observation

    def _get_info(self):
        """
        Get the current info

        Returns:
            Dictionary containing metadata about the current state.
        """
        return self.info

    def step(self, action):
        """TODO docstring. Document this function.

        Args:
            action: TODO docstring.
        """
        pass

    def reset(self, seed=None):
        # seed for self.np_random
        """TODO docstring. Document this function.

        Args:
            seed: TODO docstring.

        Returns:
            Tuple of (observation, info) after resetting the environment.
        """
        super().reset(seed=seed)

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def render(self, mode="human"):
        """TODO docstring. Document this function.

        Args:
            mode: TODO docstring.
        """
        pass

    def close(self):
        """TODO docstring. Document this function."""
        pass
