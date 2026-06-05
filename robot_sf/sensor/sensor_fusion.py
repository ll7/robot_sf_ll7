"""
The `sensor_fusion.py` file defines a `SensorFusion` class that combines data from multiple sensors.
It also provides a function `fused_sensor_space` to create a combined observation space.
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from gymnasium import spaces

from robot_sf.common.types import PolarVec2D
from robot_sf.sensor.history_stack import (
    append_history_row,
    fill_history_stack,
    reset_history_stack,
)

OBS_DRIVE_STATE = "drive_state"
OBS_RAYS = "rays"
OBS_IMAGE = "image"


def fused_sensor_space(
    timesteps: int,
    robot_obs: spaces.Box,
    target_obs: spaces.Box,
    lidar_obs: spaces.Box,
) -> tuple[spaces.Dict, spaces.Dict]:
    """
    Create a combined observation space for the robot, target, and LiDAR sensors.

    Parameters
    ----------
    timesteps : int
        The number of **stacked** timesteps in the observation (history length).
        This should align with ``observation_stack.stack_steps`` on the environment config.
    robot_obs : spaces.Box
        The observation space for the robot.
    target_obs : spaces.Box
        The observation space for the target.
    lidar_obs : spaces.Box
        The observation space for the LiDAR sensor.

    Returns
    -------
    Tuple[spaces.Dict, spaces.Dict]
        The normalized and original combined observation spaces.
    """
    # Build one row of bounds and repeat it, avoiding per-timestep Python lists.
    drive_high = np.concatenate(
        (
            np.asarray(robot_obs.high, dtype=np.float32),
            np.asarray(target_obs.high, dtype=np.float32),
        )
    )
    drive_low = np.concatenate(
        (
            np.asarray(robot_obs.low, dtype=np.float32),
            np.asarray(target_obs.low, dtype=np.float32),
        )
    )
    max_drive_state = np.repeat(drive_high[np.newaxis, :], timesteps, axis=0)
    min_drive_state = np.repeat(drive_low[np.newaxis, :], timesteps, axis=0)

    lidar_high = np.asarray(lidar_obs.high, dtype=np.float32)
    lidar_low = np.asarray(lidar_obs.low, dtype=np.float32)
    max_lidar_state = np.repeat(lidar_high[np.newaxis, :], timesteps, axis=0)
    min_lidar_state = np.repeat(lidar_low[np.newaxis, :], timesteps, axis=0)

    # Create the original observation spaces for the drive and LiDAR states
    orig_box_drive_state = spaces.Box(low=min_drive_state, high=max_drive_state, dtype=np.float32)
    orig_box_lidar_state = spaces.Box(low=min_lidar_state, high=max_lidar_state, dtype=np.float32)
    orig_obs_space = spaces.Dict(
        {OBS_DRIVE_STATE: orig_box_drive_state, OBS_RAYS: orig_box_lidar_state},
    )

    # Create the normalized observation spaces for the drive and LiDAR states
    box_drive_state = spaces.Box(
        low=min_drive_state / max_drive_state,
        high=max_drive_state / max_drive_state,
        dtype=np.float32,
    )
    box_lidar_state = spaces.Box(
        low=min_lidar_state / max_lidar_state,
        high=max_lidar_state / max_lidar_state,
        dtype=np.float32,
    )
    norm_obs_space = spaces.Dict({OBS_DRIVE_STATE: box_drive_state, OBS_RAYS: box_lidar_state})

    return norm_obs_space, orig_obs_space


def fused_sensor_space_with_image(
    timesteps: int,
    robot_obs: spaces.Box,
    target_obs: spaces.Box,
    lidar_obs: spaces.Box,
    image_obs: spaces.Box | None = None,
) -> tuple[spaces.Dict, spaces.Dict]:
    """
    Create a combined observation space for the robot, target, LiDAR, and optionally image sensors.

    Parameters
    ----------
    timesteps : int
        The number of **stacked** timesteps in the observation.
    robot_obs : spaces.Box
        The observation space for the robot.
    target_obs : spaces.Box
        The observation space for the target.
    lidar_obs : spaces.Box
        The observation space for the LiDAR sensor.
    image_obs : spaces.Box, optional
        The observation space for the image sensor.

    Returns
    -------
    Tuple[spaces.Dict, spaces.Dict]
        The normalized and original combined observation spaces.
    """
    # Start with the basic sensor fusion
    norm_obs_space, orig_obs_space = fused_sensor_space(timesteps, robot_obs, target_obs, lidar_obs)

    # Add image observation space if provided
    if image_obs is not None:
        # Convert back to dict to add image space
        orig_dict = dict(orig_obs_space.spaces)
        norm_dict = dict(norm_obs_space.spaces)

        orig_dict[OBS_IMAGE] = image_obs
        norm_dict[OBS_IMAGE] = image_obs  # Images are already normalized in the sensor

        # Convert back to spaces.Dict
        orig_obs_space = spaces.Dict(orig_dict)
        norm_obs_space = spaces.Dict(norm_dict)

    return norm_obs_space, orig_obs_space


@dataclass
class SensorFusion:
    """
    A class that combines data from multiple sensors into a single observation.

    Attributes
    ----------
    lidar_sensor : Callable[[], np.ndarray]
        A function that returns the current LiDAR sensor data.
    robot_speed_sensor : Callable[[], PolarVec2D]
        A function that returns the current robot speed.
    target_sensor : Callable[[], Tuple[float, float, float]]
        A function that returns the current target sensor data.
    unnormed_obs_space : spaces.Dict
        The unnormalized observation space.
    use_next_goal : bool
        Whether to use the next goal in the observation.
    drive_state_cache : deque[None]
        Warm-history markers for drive state. Concrete temporal values are stored
        in ``stacked_drive_state``.
    lidar_state_cache : List[np.ndarray]
        A cache of previous LiDAR states.
    cache_steps : int
        The number of steps to cache.
    """

    lidar_sensor: Callable[[], np.ndarray]
    robot_speed_sensor: Callable[[], PolarVec2D]
    target_sensor: Callable[[], tuple[float, float, float]]
    unnormed_obs_space: spaces.Dict
    use_next_goal: bool
    drive_state_cache: deque[None] = field(init=False, default_factory=deque)
    lidar_state_cache: list[np.ndarray] = field(init=False, default_factory=list)
    cache_steps: int = field(init=False)

    def __post_init__(self):
        # Initialize the number of steps to cache based on the LiDAR observation space
        """Initialize the sensor fusion after dataclass field setup.

        Sets up the observation caches and determines the cache size based
        on the LiDAR observation space shape.
        """
        self.cache_steps = self.unnormed_obs_space[OBS_RAYS].shape[0]
        self.stacked_drive_state = np.zeros((self.cache_steps, 5), dtype=np.float32)
        self._drive_state_buffer = np.zeros(5, dtype=np.float32)
        self.stacked_lidar_state = np.zeros(
            (self.cache_steps, len(self.lidar_sensor())),
            dtype=np.float32,
        )
        self.drive_state_cache = deque(maxlen=self.cache_steps)
        self.lidar_state_cache = deque(maxlen=self.cache_steps)

    def next_obs(self) -> dict[str, np.ndarray]:
        """
        Get the next observation by combining data from all sensors.

        Returns
        -------
        Dict[str, np.ndarray]
            The next observation, consisting of the drive state and LiDAR state.
            Stacked rows are ordered oldest-to-newest, with the current sample
            stored at index ``-1``.
        """
        # Get the current LiDAR state
        lidar_state = self.lidar_sensor()

        # Get the current robot speed
        speed_x, speed_rot = self.robot_speed_sensor()

        # Get the current target sensor data
        target_distance, target_angle, next_target_angle = self.target_sensor()

        # If not using the next goal, set the next target angle to 0.0
        next_target_angle = next_target_angle if self.use_next_goal else 0.0

        # Combine the robot speed and target sensor data into the reusable current-state buffer.
        drive_state = self._drive_state_buffer
        drive_state[:] = (speed_x, speed_rot, target_distance, target_angle, next_target_angle)

        # Populate history with the current state on first call to avoid zeros.
        if len(self.drive_state_cache) == 0:
            for _ in range(self.cache_steps):
                self.drive_state_cache.append(None)
                self.lidar_state_cache.append(lidar_state)
            self.stacked_drive_state = fill_history_stack(self.stacked_drive_state, drive_state)
            self.stacked_lidar_state = fill_history_stack(self.stacked_lidar_state, lidar_state)
        else:
            # Add current states as the newest row so temporal stacks are oldest-to-newest.
            self.drive_state_cache.append(None)
            self.lidar_state_cache.append(lidar_state)
            self.stacked_drive_state = append_history_row(self.stacked_drive_state, drive_state)
            self.stacked_lidar_state = append_history_row(self.stacked_lidar_state, lidar_state)

        # Normalize the stacked states by the maximum values in the observation space
        max_drive = self.unnormed_obs_space[OBS_DRIVE_STATE].high
        max_lidar = self.unnormed_obs_space[OBS_RAYS].high
        return {
            OBS_DRIVE_STATE: (self.stacked_drive_state / max_drive).astype(np.float32, copy=False),
            OBS_RAYS: (self.stacked_lidar_state / max_lidar).astype(np.float32, copy=False),
        }

    def reset_cache(self):
        """
        Clear the caches of previous drive and LiDAR states.
        """
        self.drive_state_cache.clear()
        self.lidar_state_cache.clear()
        self.stacked_drive_state = reset_history_stack(self.stacked_drive_state)
        self.stacked_lidar_state = reset_history_stack(self.stacked_lidar_state)
