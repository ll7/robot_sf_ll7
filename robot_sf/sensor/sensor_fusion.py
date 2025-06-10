"""
The `sensor_fusion.py` file defines a `SensorFusion` class that combines data from multiple sensors.
It also provides a function `fused_sensor_space` to create a combined observation space.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from robot_sf.util.types import PolarVec2D

OBS_DRIVE_STATE = "drive_state"
OBS_RAYS = "rays"
OBS_IMAGE = "image"


def fused_sensor_space(
    timesteps: int, robot_obs: spaces.Box, target_obs: spaces.Box, lidar_obs: spaces.Box
) -> Tuple[spaces.Dict, spaces.Dict]:
    """
    Creates combined observation spaces for stacked robot, target, and LiDAR sensor data.
    
    The function constructs both normalized and original (unnormalized) observation spaces by stacking the bounds of the provided robot, target, and LiDAR observation spaces over a specified number of timesteps. The normalized space scales each dimension to the range [0, 1] based on the maximum values.
    
    Args:
        timesteps: Number of consecutive timesteps to stack in the observation.
        robot_obs: Observation space for the robot sensor.
        target_obs: Observation space for the target sensor.
        lidar_obs: Observation space for the LiDAR sensor.
    
    Returns:
        A tuple containing the normalized and original combined observation spaces as `spaces.Dict` objects.
    """
    # Create the maximum and minimum drive states for each timestep
    max_drive_state = np.array(
        [robot_obs.high.tolist() + target_obs.high.tolist() for t in range(timesteps)],
        dtype=np.float32,
    )
    min_drive_state = np.array(
        [robot_obs.low.tolist() + target_obs.low.tolist() for t in range(timesteps)],
        dtype=np.float32,
    )

    # Create the maximum and minimum LiDAR states for each timestep
    max_lidar_state = np.array(
        [lidar_obs.high.tolist() for t in range(timesteps)], dtype=np.float32
    )
    min_lidar_state = np.array([lidar_obs.low.tolist() for t in range(timesteps)], dtype=np.float32)

    # Create the original observation spaces for the drive and LiDAR states
    orig_box_drive_state = spaces.Box(low=min_drive_state, high=max_drive_state, dtype=np.float32)
    orig_box_lidar_state = spaces.Box(low=min_lidar_state, high=max_lidar_state, dtype=np.float32)
    orig_obs_space = spaces.Dict(
        {OBS_DRIVE_STATE: orig_box_drive_state, OBS_RAYS: orig_box_lidar_state}
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
    image_obs: Optional[spaces.Box] = None,
) -> Tuple[spaces.Dict, spaces.Dict]:
    """
    Creates combined observation spaces for robot, target, LiDAR, and optionally image sensors over a specified number of stacked timesteps.
    
    If an image observation space is provided, it is included in both the normalized and original observation spaces. The function returns a tuple containing the normalized and original observation spaces as `spaces.Dict` objects.
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
    drive_state_cache : List[np.ndarray]
        A cache of previous drive states.
    lidar_state_cache : List[np.ndarray]
        A cache of previous LiDAR states.
    cache_steps : int
        The number of steps to cache.
    """

    lidar_sensor: Callable[[], np.ndarray]
    robot_speed_sensor: Callable[[], PolarVec2D]
    target_sensor: Callable[[], Tuple[float, float, float]]
    unnormed_obs_space: spaces.Dict
    use_next_goal: bool
    drive_state_cache: List[np.ndarray] = field(init=False, default_factory=list)
    lidar_state_cache: List[np.ndarray] = field(init=False, default_factory=list)
    cache_steps: int = field(init=False)

    def __post_init__(self):
        # Initialize the number of steps to cache based on the LiDAR observation space
        self.cache_steps = self.unnormed_obs_space[OBS_RAYS].shape[0]
        self.stacked_drive_state = np.zeros((self.cache_steps, 5), dtype=np.float32)
        self.stacked_lidar_state = np.zeros(
            (self.cache_steps, len(self.lidar_sensor())), dtype=np.float32
        )
        self.drive_state_cache = deque(maxlen=self.cache_steps)
        self.lidar_state_cache = deque(maxlen=self.cache_steps)

    def next_obs(self) -> Dict[str, np.ndarray]:
        """
        Get the next observation by combining data from all sensors.

        Returns
        -------
        Dict[str, np.ndarray]
            The next observation, consisting of the drive state and LiDAR state.
        """
        # Get the current LiDAR state
        lidar_state = self.lidar_sensor()
        # TODO: append beginning at the end for conv feature extractor

        # Get the current robot speed
        speed_x, speed_rot = self.robot_speed_sensor()

        # Get the current target sensor data
        target_distance, target_angle, next_target_angle = self.target_sensor()

        # If not using the next goal, set the next target angle to 0.0
        next_target_angle = next_target_angle if self.use_next_goal else 0.0

        # Combine the robot speed and target sensor data into the drive state
        drive_state = np.array(
            [speed_x, speed_rot, target_distance, target_angle, next_target_angle]
        )

        # info: populate cache with same states -> no movement
        # If the caches are empty, fill them with the current states
        if len(self.drive_state_cache) == 0:
            for _ in range(self.cache_steps):
                self.drive_state_cache.append(drive_state)
                self.lidar_state_cache.append(lidar_state)

        # Add the current states to the caches and remove the oldest states
        self.drive_state_cache.append(drive_state)
        self.lidar_state_cache.append(lidar_state)
        self.stacked_drive_state = np.roll(self.stacked_drive_state, -1, axis=0)
        self.stacked_drive_state[-1] = drive_state
        self.stacked_lidar_state = np.roll(self.stacked_lidar_state, -1, axis=0)
        self.stacked_lidar_state[-1] = lidar_state

        # Normalize the stacked states by the maximum values in the observation space
        max_drive = self.unnormed_obs_space[OBS_DRIVE_STATE].high
        max_lidar = self.unnormed_obs_space[OBS_RAYS].high
        return {
            OBS_DRIVE_STATE: self.stacked_drive_state / max_drive,
            OBS_RAYS: self.stacked_lidar_state / max_lidar,
        }

    def reset_cache(self):
        """
        Clear the caches of previous drive and LiDAR states.
        """
        self.drive_state_cache.clear()
        self.lidar_state_cache.clear()
