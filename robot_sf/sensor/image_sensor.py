"""
Image sensor for capturing visual observations from the pygame rendering system.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pygame
from gymnasium import spaces

from robot_sf.render.sim_view import SimulationView


@dataclass
class ImageSensorSettings:
    """Configuration settings for the image sensor."""

    width: int = 84  # Standard size for RL vision tasks
    height: int = 84  # Standard size for RL vision tasks
    channels: int = 3  # RGB channels
    normalize: bool = True  # Whether to normalize pixel values to [0, 1]
    grayscale: bool = False  # Whether to convert to grayscale


class ImageSensor:
    """
    Sensor that captures images from the pygame rendering system.

    This sensor captures the current state of the simulation as a visual image
    that can be used as part of the observation space for reinforcement learning.
    """

    def __init__(self, settings: ImageSensorSettings, sim_view: Optional[SimulationView] = None):
        """
        Initializes the image sensor with specified settings and an optional simulation view.
        
        Args:
            settings: Configuration for image capture, resizing, normalization, and grayscale conversion.
            sim_view: Optional simulation view to capture images from.
        """
        self.settings = settings
        self.sim_view = sim_view

    def set_sim_view(self, sim_view: SimulationView):
        """
        Sets the simulation view from which images will be captured.
        
        Args:
        	sim_view: The simulation view providing the rendering surface for image capture.
        """
        self.sim_view = sim_view

    def capture_frame(self) -> np.ndarray:
        """
        Captures and processes the current frame from the simulation view as a numpy array.
        
        Returns:
            np.ndarray: The processed image with shape (height, width, channels) for color or (height, width) for grayscale, normalized to [0, 1] if specified.
            
        Raises:
            ValueError: If the simulation view is not set.
        """
        if self.sim_view is None:
            raise ValueError("SimulationView not set. Call set_sim_view() first.")

        # Capture frame using pygame's surfarray
        frame_data = pygame.surfarray.array3d(self.sim_view.screen)
        # pygame.surfarray returns (width, height, channels), we need (height, width, channels)
        frame_data = frame_data.swapaxes(0, 1)

        # Resize if needed
        if (
            frame_data.shape[0] != self.settings.height
            or frame_data.shape[1] != self.settings.width
        ):
            frame_data = self._resize_frame(frame_data)

        # Convert to grayscale if requested
        if self.settings.grayscale:
            frame_data = self._to_grayscale(frame_data)

        # Normalize if requested
        if self.settings.normalize:
            frame_data = frame_data.astype(np.float32) / 255.0
        else:
            frame_data = frame_data.astype(np.uint8)

        return frame_data

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resizes an image frame to the configured width and height.
        
        Args:
            frame: The input image as a NumPy array.
        
        Returns:
            The resized image frame as a NumPy array with shape (height, width, channels).
        """
        # Convert to pygame surface for resizing
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        resized_surface = pygame.transform.scale(
            surface, (self.settings.width, self.settings.height)
        )

        # Convert back to numpy array
        resized_frame = pygame.surfarray.array3d(resized_surface).swapaxes(0, 1)
        return resized_frame

    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """
        Converts an RGB image to a grayscale image using standard luminance weights.
        
        Args:
            frame: An RGB image as a NumPy array of shape (height, width, 3).
        
        Returns:
            A 2D NumPy array of shape (height, width) representing the grayscale image.
        """
        # Use standard RGB to grayscale conversion weights
        gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        return gray.astype(frame.dtype)


def image_sensor_space(settings: ImageSensorSettings) -> spaces.Box:
    """
    Constructs a Gymnasium Box observation space matching the output of the image sensor.
    
    The space shape and data type reflect the sensor's grayscale and normalization settings.
    Returns a Box with shape (height, width) for grayscale or (height, width, channels) for color images.
    """
    if settings.grayscale:
        shape = (settings.height, settings.width)
    else:
        shape = (settings.height, settings.width, settings.channels)

    if settings.normalize:
        low = 0.0
        high = 1.0
        dtype = np.float32
    else:
        low = 0
        high = 255
        dtype = np.uint8

    return spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
