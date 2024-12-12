#!/usr/bin/env python3

"""
Base environment for the robot simulation environment.
"""

from loguru import logger
from gymnasium import Env
from typing import List

from robot_sf.render.sim_view import VisualizableSimState
from robot_sf.gym_env.env_config import EnvSettings


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

    def _set_video_fps(self):
        """
        This methods sets a default value for the video_fps attribute if it was not provided.
        """

        if video_fps is None:
            video_fps = 1 / self.env_config.sim_config.time_per_step_in_secs
            logger.info(f"Video FPS not provided, setting to {video_fps}")
