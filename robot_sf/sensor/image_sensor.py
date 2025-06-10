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
        Initialize the image sensor.

        Parameters
        ----------
        settings : ImageSensorSettings
            Configuration settings for the image sensor.
        sim_view : SimulationView, optional
            The simulation view to capture images from.
        """
        self.settings = settings
        self.sim_view = sim_view

    def set_sim_view(self, sim_view: SimulationView):
        """Set the simulation view to capture images from."""
        self.sim_view = sim_view

    def capture_frame(self) -> np.ndarray:
        """
        Capture the current frame from the simulation view.

        Returns
        -------
        np.ndarray
            The captured image as a numpy array with shape (height, width, channels)
            or (height, width) if grayscale is True.
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
        Resize the frame to the target dimensions.

        Parameters
        ----------
        frame : np.ndarray
            The input frame to resize.

        Returns
        -------
        np.ndarray
            The resized frame.
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
        Convert RGB frame to grayscale.

        Parameters
        ----------
        frame : np.ndarray
            RGB frame with shape (height, width, 3).

        Returns
        -------
        np.ndarray
            Grayscale frame with shape (height, width).
        """
        # Use standard RGB to grayscale conversion weights
        gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        return gray.astype(frame.dtype)


def image_sensor_space(settings: ImageSensorSettings) -> spaces.Box:
    """
    Create the observation space for the image sensor.

    Parameters
    ----------
    settings : ImageSensorSettings
        Configuration settings for the image sensor.

    Returns
    -------
    spaces.Box
        The observation space for the image sensor.
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
