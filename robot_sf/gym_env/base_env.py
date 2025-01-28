"""
Base environment for the simulation environment.
Provides common functionality for all robot-based environments.
"""

# from typing import List, Optional
# import datetime
# import pickle

from typing import Callable
from loguru import logger

from gymnasium import Env
from robot_sf.gym_env.reward import simple_reward
from robot_sf.gym_env.env_config import EnvSettings

# from robot_sf.gym_env.env_config import BaseEnvSettings
# from robot_sf.render.sim_view import (
#     SimulationView,
#     VisualizableSimState,
# )
# from robot_sf.sim.simulator import init_simulators


class BaseEnv(Env):
    """Base environment class that handles common functionality."""

    def __init__(
        self,
        env_config: EnvSettings = EnvSettings(),
        reward_func: Callable[[dict], float] = simple_reward,
        debug: bool = False,
        recording_enabled: bool = False,
        record_video: bool = False,
        video_path: str = None,
        video_fps: float = None,
    ):
        super().__init__()

        # Environment configuration details
        self.env_config = env_config

        # Set video FPS if not provided
        if video_fps is None:
            video_fps = 1 / self.env_config.sim_config.time_per_step_in_secs
            logger.info(f"Video FPS not provided, setting to {video_fps}")

        # Extract first map definition; currently only supports using the first map
        self.map_def = env_config.map_pool.choose_random_map()

    def render(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
