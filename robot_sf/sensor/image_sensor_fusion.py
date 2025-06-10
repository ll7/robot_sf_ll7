"""
Extended sensor fusion that includes image observations.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces

from robot_sf.sensor.image_sensor import ImageSensor
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_IMAGE, OBS_RAYS
from robot_sf.util.types import PolarVec2D


@dataclass
class ImageSensorFusion:
    """
    Extended sensor fusion that includes image observations from the pygame rendering system.

    This class extends the basic sensor fusion to include visual observations captured
    from the simulation rendering.
    """

    lidar_sensor: Callable[[], np.ndarray]
    robot_speed_sensor: Callable[[], PolarVec2D]
    target_sensor: Callable[[], Tuple[float, float, float]]
    image_sensor: Optional[ImageSensor]
    unnormed_obs_space: spaces.Dict
    use_next_goal: bool
    use_image_obs: bool = False

    # Inherited from SensorFusion
    drive_state_cache: deque = field(init=False, default_factory=lambda: deque())
    lidar_state_cache: deque = field(init=False, default_factory=lambda: deque())
    image_state_cache: deque = field(init=False, default_factory=lambda: deque())
    cache_steps: int = field(init=False)

    def __post_init__(self):
        # Initialize the number of steps to cache based on the LiDAR observation space
        """
        Initializes internal caches and stacked state arrays for drive, LiDAR, and image data.
        
        Determines the number of cache steps and observation shapes from the unnormalized observation space. Sets up numpy arrays for stacking recent sensor data and initializes deques for caching drive, LiDAR, and optionally image states.
        """
        rays_space = self.unnormed_obs_space[OBS_RAYS]
        if hasattr(rays_space, "shape") and rays_space.shape is not None:
            self.cache_steps = rays_space.shape[0]
            lidar_shape = rays_space.shape[1]  # Number of rays
        else:
            # Fallback values
            self.cache_steps = 3
            lidar_shape = 5

        self.stacked_drive_state = np.zeros((self.cache_steps, 5), dtype=np.float32)
        self.stacked_lidar_state = np.zeros((self.cache_steps, lidar_shape), dtype=np.float32)

        # Initialize image cache if image observations are enabled
        if self.use_image_obs and self.image_sensor is not None:
            # Get image dimensions from the observation space
            image_space = self.unnormed_obs_space[OBS_IMAGE]
            if hasattr(image_space, "shape") and image_space.shape is not None:
                image_shape = image_space.shape
                if len(image_shape) == 2:  # Grayscale
                    self.stacked_image_state = np.zeros(
                        (self.cache_steps,) + image_shape, dtype=np.float32
                    )
                else:  # RGB
                    self.stacked_image_state = np.zeros(
                        (self.cache_steps,) + image_shape, dtype=np.float32
                    )

        self.drive_state_cache = deque(maxlen=self.cache_steps)
        self.lidar_state_cache = deque(maxlen=self.cache_steps)
        if self.use_image_obs:
            self.image_state_cache = deque(maxlen=self.cache_steps)

    def next_obs(self) -> Dict[str, np.ndarray]:
        """
        Combines the latest readings from LiDAR, robot speed, target, and optionally image sensors into a normalized observation dictionary.
        
        Returns:
            A dictionary containing the stacked and normalized drive state and LiDAR state, and, if enabled, the current image frame.
        """
        # Get the current LiDAR state
        lidar_state = self.lidar_sensor()

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

        # Get image state if enabled
        image_state = None
        if self.use_image_obs and self.image_sensor is not None:
            image_state = self.image_sensor.capture_frame()

        # Initialize caches if empty
        if len(self.drive_state_cache) == 0:
            for _ in range(self.cache_steps):
                self.drive_state_cache.append(drive_state)
                self.lidar_state_cache.append(lidar_state)
                if self.use_image_obs and image_state is not None:
                    self.image_state_cache.append(image_state)

        # Add the current states to the caches
        self.drive_state_cache.append(drive_state)
        self.lidar_state_cache.append(lidar_state)

        # Update stacked states
        self.stacked_drive_state = np.roll(self.stacked_drive_state, -1, axis=0)
        self.stacked_drive_state[-1] = drive_state
        self.stacked_lidar_state = np.roll(self.stacked_lidar_state, -1, axis=0)
        self.stacked_lidar_state[-1] = lidar_state

        # Prepare the observation dictionary
        obs = {}

        # Normalize the stacked states by the maximum values in the observation space
        drive_space = self.unnormed_obs_space[OBS_DRIVE_STATE]
        lidar_space = self.unnormed_obs_space[OBS_RAYS]

        if hasattr(drive_space, "high") and drive_space.high is not None:
            max_drive = drive_space.high
        else:
            max_drive = np.ones(5)  # Default normalization

        if hasattr(lidar_space, "high") and lidar_space.high is not None:
            max_lidar = lidar_space.high
        else:
            max_lidar = np.ones(self.stacked_lidar_state.shape[1])  # Default normalization

        obs[OBS_DRIVE_STATE] = self.stacked_drive_state / max_drive
        obs[OBS_RAYS] = self.stacked_lidar_state / max_lidar

        # Add image observation if enabled
        if self.use_image_obs and image_state is not None:
            self.image_state_cache.append(image_state)
            self.stacked_image_state = np.roll(self.stacked_image_state, -1, axis=0)
            self.stacked_image_state[-1] = image_state
            # Images are already normalized in the sensor
            obs[OBS_IMAGE] = image_state  # Use current frame, not stacked for now

        return obs

    def reset_cache(self):
        """
        Clears all cached drive, LiDAR, and image states.
        
        Resets the internal caches used for stacking recent sensor observations, including image state cache if image observations are enabled.
        """
        self.drive_state_cache.clear()
        self.lidar_state_cache.clear()
        if self.use_image_obs:
            self.image_state_cache.clear()
