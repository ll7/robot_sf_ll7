"""
Base environment for the simulation environment.
Provides common functionality for all robot-based environments.
"""

# from typing import List, Optional
# import datetime
# import pickle

from typing import List
from loguru import logger

from gymnasium import Env
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.render.sim_view import SimulationView, VisualizableSimState
from robot_sf.sim.simulator import init_simulators


class BaseEnv(Env):
    """Base environment class that handles common functionality."""

    def __init__(
        self,
        env_config: EnvSettings = EnvSettings(),
        debug: bool = False,
        recording_enabled: bool = False,
        record_video: bool = False,
        video_path: str = None,
        video_fps: float = None,
        peds_have_obstacle_forces: bool = False,
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

        self.debug = debug

        # Initialize the list to store recorded states
        self.recorded_states: List[VisualizableSimState] = []
        self.recording_enabled = recording_enabled

        # Initialize simulator with a random start position
        self.simulator = init_simulators(
            env_config,
            self.map_def,
            random_start_pos=True,
            peds_have_obstacle_forces=peds_have_obstacle_forces,
        )[0]

        # Store last action executed by the robot
        self.last_action = None

        # If in debug mode or video recording is enabled, create simulation view
        if debug or record_video:
            self.sim_ui = SimulationView(
                scaling=10,
                map_def=self.map_def,
                obstacles=self.map_def.obstacles,
                robot_radius=env_config.robot_config.radius,
                ped_radius=env_config.sim_config.ped_radius,
                goal_radius=env_config.sim_config.goal_radius,
                record_video=record_video,
                video_path=video_path,
                video_fps=video_fps,
            )

    def render(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def exit(self):
        """
        Clean up and exit the simulation UI, if it exists.
        """
        if self.sim_ui:
            self.sim_ui.exit_simulation()
