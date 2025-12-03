"""
Lightweight CNN-based feature extractor for robot environments.

This extractor uses a simplified convolutional approach with fewer parameters
than the original DynamicsExtractor while still leveraging spatial relationships
in the LiDAR data.
"""

from typing import cast

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS


class LightweightCNNExtractor(BaseFeaturesExtractor):
    """
    Lightweight CNN-based feature extractor for robot sensor data.

    This extractor uses a simplified convolutional approach with fewer layers
    and parameters compared to the original DynamicsExtractor.

    Advantages:
    - Preserves spatial relationships in LiDAR data
    - Fewer parameters than original DynamicsExtractor
    - Faster training and inference than full CNN approach
    - Still leverages convolution benefits

    Attributes:
        observation_space: The space of possible observations from the environment
        num_filters: List of filter counts for each conv layer
        kernel_sizes: List of kernel sizes for each conv layer
        dropout_rate: Dropout rate for regularization
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        num_filters: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        dropout_rate: float = 0.1,
        drive_hidden_dims: list[int] | None = None,
    ):
        """TODO docstring. Document this function.

        Args:
            observation_space: TODO docstring.
            num_filters: TODO docstring.
            kernel_sizes: TODO docstring.
            dropout_rate: TODO docstring.
            drive_hidden_dims: TODO docstring.
        """
        if num_filters is None:
            num_filters = [32, 16]
        if kernel_sizes is None:
            kernel_sizes = [5, 3]
        if drive_hidden_dims is None:
            drive_hidden_dims = [32, 16]
        # Extract observation spaces
        rays_space = cast("spaces.Box", observation_space.spaces[OBS_RAYS])
        drive_state_space = cast("spaces.Box", observation_space.spaces[OBS_DRIVE_STATE])

        # Calculate dimensions
        drive_input_dim = int(np.prod(drive_state_space.shape))
        drive_output_dim = drive_hidden_dims[-1] if drive_hidden_dims else drive_input_dim

        # Calculate ray features after convolutions
        # Simplified calculation assuming stride=1 and appropriate padding
        ray_features = num_filters[-1] * (rays_space.shape[1] // (2 ** len(num_filters)))

        total_features = ray_features + drive_output_dim

        # Initialize the base feature extractor
        super().__init__(observation_space, features_dim=total_features)

        def conv_block(in_channels: int, out_channels: int, kernel_size: int) -> list[nn.Module]:
            """Create a lightweight convolutional block."""
            padding = kernel_size // 2  # Maintain spatial dimension
            return [
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),  # Reduce spatial dimension by half
                nn.Dropout(dropout_rate),
            ]

        # Build ray processing CNN
        ray_layers = []
        in_channels = [rays_space.shape[0]] + num_filters[:-1]

        for i, (in_ch, out_ch, kernel_size) in enumerate(
            zip(in_channels, num_filters, kernel_sizes, strict=False)
        ):
            ray_layers.extend(conv_block(in_ch, out_ch, kernel_size))

        # Add adaptive pooling to handle variable input sizes
        ray_layers.extend(
            [nn.AdaptiveAvgPool1d(rays_space.shape[1] // (2 ** len(num_filters))), nn.Flatten()]
        )

        self.ray_extractor = nn.Sequential(*ray_layers)

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
        Extract features using lightweight CNN for rays and MLP for drive state.

        Args:
            obs: Dictionary containing ray and drive state observations

        Returns:
            Concatenated features from CNN and MLP processing
        """
        ray_features = self.ray_extractor(obs[OBS_RAYS])
        drive_features = self.drive_state_extractor(obs[OBS_DRIVE_STATE])
        return th.cat([ray_features, drive_features], dim=1)
