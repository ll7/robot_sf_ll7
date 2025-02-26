# WARNING: don't move this script or else loading trained SB3 policies might not work

from typing import List

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS


class DynamicsExtractor(BaseFeaturesExtractor):
    """
    A class used to extract features from the dynamics of the environment.

    Attributes
    ----------
    observation_space : spaces.Dict
        The space of possible observations from the environment.
    use_ray_conv : bool, optional
        Whether to use ray convolution for feature extraction (default is True).
    num_filters : list of int, optional
        The number of filters to use in each convolutional layer (default is [64, 16, 16, 16]).
    kernel_sizes : list of int, optional
        The sizes of the kernels to use in each convolutional layer (default is [3, 3, 3, 3]).
    dropout_rates : list of float, optional
        The dropout rates to use after each convolutional layer (default is [0.3, 0.3, 0.3, 0.3]).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        use_ray_conv: bool = True,
        num_filters: List[int] = [64, 16, 16, 16],
        kernel_sizes: List[int] = [3, 3, 3, 3],
        dropout_rates: List[float] = [0.3, 0.3, 0.3, 0.3],
    ):
        # Extract the ray and drive state spaces from the observation space
        rays_space: spaces.Box = observation_space.spaces[OBS_RAYS]
        drive_state_space: spaces.Box = observation_space.spaces[OBS_DRIVE_STATE]

        # Calculate the number of features for the drive state and rays
        drive_state_features = np.prod(drive_state_space.shape)
        num_rays = rays_space.shape[1]
        ray_features = (
            num_filters[3] * (num_rays // 16) if use_ray_conv else np.prod(rays_space.shape)
        )

        # Calculate the total number of features
        total_features = ray_features + drive_state_features

        # Initialize the base feature extractor
        super().__init__(observation_space, features_dim=total_features)

        def padding(kernel_size: int):
            """
            Calculate the padding needed for a given kernel size.

            Parameters
            ----------
            kernel_size : int
                The size of the kernel.

            Returns
            -------
            int
                The padding needed for the kernel.
            """
            if kernel_size % 2 == 0:
                raise ValueError("kernel size must be odd!")
            return int((kernel_size - 1) / 2)

        def conv_block(in_channels: int, out_channels: int, kernel_size: int, dropout_rate: float):
            """
            Create a convolutional block.

            Parameters
            ----------
            in_channels : int
                The number of input channels.
            out_channels : int
                The number of output channels.
            kernel_size : int
                The size of the kernel.
            dropout_rate : float
                The dropout rate after the convolution.

            Returns
            -------
            list
                A list containing the convolutional layer, ReLU activation, and dropout layer.
            """
            return [
                nn.Conv1d(in_channels, out_channels, kernel_size, 2, padding(kernel_size)),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]

        if use_ray_conv:
            in_channels = [rays_space.shape[0]] + num_filters[:-1]
            out_channels = num_filters
            args_of_blocks = zip(in_channels, out_channels, kernel_sizes, dropout_rates)
            layers = [layer for args in args_of_blocks for layer in conv_block(*args)] + [
                nn.Flatten()
            ]
            self.ray_extractor = nn.Sequential(*layers)
        else:
            self.ray_extractor = nn.Sequential(nn.Flatten())

        self.drive_state_extractor = nn.Sequential(nn.Flatten())

    def forward(self, obs: dict) -> th.Tensor:
        ray_x = self.ray_extractor(obs[OBS_RAYS])
        drive_x = self.drive_state_extractor(obs[OBS_DRIVE_STATE])
        return th.cat([ray_x, drive_x], dim=1)
