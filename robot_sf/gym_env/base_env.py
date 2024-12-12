#!/usr/bin/env python3

"""
Base environment for the robot simulation environment.
"""

from loguru import logger
from gymnasium import Env
from typing import List

from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.render.sim_view import VisualizableSimState, SimulationView
from robot_sf.sim.simulator import init_simulators


class BaseEnv(Env):
    """
    Represents the base class of robot_sf environment.
    It should be possible to simulate this environment without any action.
    So this could be a standalone pedestrian simulation.
    """

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
        """
        Initializes the base environment.
        """
        super().__init__()
        self.env_config = env_config

        self.debug = debug

        self.recording_enabled = recording_enabled
        if self.recording_enabled:
            self.recorded_states: List[VisualizableSimState] = []
            self.record_video = record_video
            if self.record_video:
                self.video_path = video_path
                self.video_fps = video_fps
                self._set_video_fps()

        self.map_def = env_config.map_pool.choose_random_map()

        # Initialize simulator with a random start position
        self.simulator = init_simulators(
            env_config,
            self.map_def,
            random_start_pos=True,
            peds_have_obstacle_forces=peds_have_obstacle_forces,
        )[0]

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

    def _set_video_fps(self):
        """
        This methods sets a default value for the video_fps attribute if it was not provided.
        """

        if video_fps is None:
            video_fps = 1 / self.env_config.sim_config.time_per_step_in_secs
            logger.info(f"Video FPS not provided, setting to {video_fps}")

    def step(self, action=None):
        """
        Advances the simulation by one step.
        Does not take any action.
        Returns only dummy values.
        """
        # todo: the simulator in this configuration requires a robot action
        # Instead the robot should be redesigned to not require a pedestrian.
        self.simulator.step_once()

        obs = None
        reward = None
        terminal = None
        truncated = None
        info = None

        return (obs, reward, terminal, truncated, info)
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        """
        super().reset(seed=seed, options=options)
        
        # Reset internal simulator state
        self.simulator.reset_state()

        if self.recording_enabled:
            self.save_recording()
        return None