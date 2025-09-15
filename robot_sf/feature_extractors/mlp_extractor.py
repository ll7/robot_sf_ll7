"""
MLP-based feature extractor for robot environments.

This extractor uses simple Multi-Layer Perceptrons (MLPs) to process both
LiDAR rays and drive state, providing a lightweight alternative to the
convolutional approach of the original DynamicsExtractor.
"""

from typing import List

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS


class MLPFeatureExtractor(BaseFeaturesExtractor):
    """
    MLP-based feature extractor for robot sensor data.

    This extractor flattens both ray and drive state observations and processes
    them through separate MLPs before concatenating the results.

    Advantages:
    - Much fewer parameters than convolutional approach
    - Faster training and inference
    - Simple architecture that's easy to understand and debug

    Attributes:
        observation_space: The space of possible observations from the environment
        ray_hidden_dims: Hidden layer dimensions for ray processing MLP
        drive_hidden_dims: Hidden layer dimensions for drive state processing MLP
        dropout_rate: Dropout rate for regularization
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        ray_hidden_dims: List[int] = [128, 64],
        drive_hidden_dims: List[int] = [32, 16],
        dropout_rate: float = 0.1,
    ):
        # Extract observation spaces
        rays_space: spaces.Box = observation_space.spaces[OBS_RAYS]
        drive_state_space: spaces.Box = observation_space.spaces[OBS_DRIVE_STATE]

        # Calculate input and output dimensions
        ray_input_dim = np.prod(rays_space.shape)
        drive_input_dim = np.prod(drive_state_space.shape)
        ray_output_dim = ray_hidden_dims[-1] if ray_hidden_dims else ray_input_dim
        drive_output_dim = drive_hidden_dims[-1] if drive_hidden_dims else drive_input_dim

        total_features = ray_output_dim + drive_output_dim

        # Initialize the base feature extractor
        super().__init__(observation_space, features_dim=total_features)

        # Build ray processing MLP
        ray_layers = []
        ray_dims = [ray_input_dim] + ray_hidden_dims

        for i in range(len(ray_dims) - 1):
            ray_layers.extend(
                [nn.Linear(ray_dims[i], ray_dims[i + 1]), nn.ReLU(), nn.Dropout(dropout_rate)]
            )

        self.ray_extractor = nn.Sequential(nn.Flatten(), *ray_layers)

        # Build drive state processing MLP
        drive_layers = []
        drive_dims = [drive_input_dim] + drive_hidden_dims

        for i in range(len(drive_dims) - 1):
            drive_layers.extend(
                [nn.Linear(drive_dims[i], drive_dims[i + 1]), nn.ReLU(), nn.Dropout(dropout_rate)]
            )

        self.drive_state_extractor = nn.Sequential(nn.Flatten(), *drive_layers)

    def forward(self, obs: dict) -> th.Tensor:
        """
        Extract features from observations.

        Args:
            obs: Dictionary containing ray and drive state observations

        Returns:
            Concatenated features from ray and drive state processing
        """
        ray_features = self.ray_extractor(obs[OBS_RAYS])
        drive_features = self.drive_state_extractor(obs[OBS_DRIVE_STATE])
        return th.cat([ray_features, drive_features], dim=1)
