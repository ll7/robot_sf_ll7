"""
Base environment for the simulation environment.
Provides common functionality for all environments.
"""

import datetime
import os
import pickle
from typing import List

from gymnasium import Env
from loguru import logger

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

    def save_recording(self, filename: str = None, reward_dict: dict = None):
        """
        Save the recorded states to a file in dictionary format.

        Args:
            filename (str, optional): Path where the recording will be saved.
                If None, a timestamped filename in the recordings directory is used.
            reward_dict (dict, optional): Dictionary containing reward information
                to save with the recording.

        Returns:
            str: Path to the saved file, or None if no states were recorded

        Notes:
            - The recording is saved in a dictionary format with keys:
              'states', 'map_def', 'rewards', 'metadata'
            - The recorded states list is reset after saving
        """
        if filename is None:
            now = datetime.datetime.now()
            # get current working directory
            cwd = os.getcwd()
            filename = f"{cwd}/recordings/{now.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"

        # only save if there are recorded states
        if len(self.recorded_states) == 0:
            logger.warning("No states recorded, skipping save")
            # TODO: First env.reset will always have no recorded states
            return None

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Create metadata with recording timestamp
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "num_states": len(self.recorded_states),
            "environment_type": self.__class__.__name__,
            "time_per_step": self.env_config.sim_config.time_per_step_in_secs,
        }

        # Prepare data dictionary
        recording_data = {
            "states": self.recorded_states,
            "map_def": self.map_def,
            "rewards": reward_dict,
            "metadata": metadata,
        }

        with open(filename, "wb") as f:  # write binary
            pickle.dump(recording_data, f)
            logger.info(f"Recording saved to {filename}")
            logger.info(f"Saved {len(self.recorded_states)} states with metadata")
            logger.info("Reset state list")
            self.recorded_states = []

        return filename
