"""
Extended robot environment with image-based observation space support.
"""

from typing import Callable

from robot_sf.gym_env.env_config import RobotEnvSettings
from robot_sf.gym_env.env_util import (
    create_spaces_with_image,
    init_collision_and_sensors_with_image,
)
from robot_sf.gym_env.reward import simple_reward
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.robot.robot_state import RobotState


class RobotEnvWithImage(RobotEnv):
    """
    Extended Robot Environment that supports image-based observations.

    This environment captures visual information from the pygame rendering system
    and includes it in the observation space for reinforcement learning.
    """

    def __init__(
        self,
        env_config: RobotEnvSettings = RobotEnvSettings(),
        reward_func: Callable[[dict], float] = simple_reward,
        debug: bool = False,
        recording_enabled: bool = False,
        record_video: bool = False,
        video_path: str = None,
        video_fps: float = None,
        peds_have_obstacle_forces: bool = False,
    ):
        """
        Initialize the Robot Environment with Image Observations.

        Parameters:
        - env_config (RobotEnvSettings): Configuration for environment settings including image config.
        - reward_func (Callable[[dict], float]): Reward function.
        - debug (bool): If True, enables debugging information and visualization.
        - recording_enabled (bool): If True, enables recording of the simulation.
        - record_video: If True, saves simulation as video file.
        - video_path: Path where to save the video file.
        """
        # Force debug mode if image observations are enabled to ensure SimulationView is created
        if env_config.use_image_obs:
            debug = True

        # Initialize the base robot environment
        super().__init__(
            env_config=env_config,
            reward_func=reward_func,
            debug=debug,
            recording_enabled=recording_enabled,
            record_video=record_video,
            video_path=video_path,
            video_fps=video_fps,
            peds_have_obstacle_forces=peds_have_obstacle_forces,
        )

        # Store configuration for factory pattern compatibility
        self.config = env_config

        # Override spaces initialization to include image observations
        self.action_space, self.observation_space, orig_obs_space = create_spaces_with_image(
            env_config, self.map_def
        )

        # Override sensor initialization to include image sensors
        occupancies, sensors = init_collision_and_sensors_with_image(
            self.simulator, env_config, orig_obs_space, sim_view=getattr(self, "sim_ui", None)
        )

        # Setup initial state of the robot with the new sensor fusion
        self.state = RobotState(
            self.simulator.robot_navs[0],
            occupancies[0],
            sensors[0],
            env_config.sim_config.time_per_step_in_secs,
            env_config.sim_config.sim_time_in_secs,
        )
