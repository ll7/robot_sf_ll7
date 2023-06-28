
# WARNING: don't move this script or else loading trained SB3 policies might not work

import numpy as np
from gym import spaces

import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from robot_sf.robot_env import OBS_DRIVE_STATE, OBS_RAYS


class DynamicsExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        rays_space: spaces.Box = observation_space.spaces[OBS_RAYS]
        drive_state_space: spaces.Box = observation_space.spaces[OBS_DRIVE_STATE]
        drive_state_features = np.prod(drive_state_space.shape)
        ray_features = 16 * (rays_space.shape[1] // 16)
        total_features = ray_features + drive_state_features
        super().__init__(observation_space, features_dim=total_features)

        self.ray_extractor = nn.Sequential(
            nn.Conv1d(rays_space.shape[0], 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Flatten())

        self.drive_state_extractor = nn.Sequential(nn.Flatten())

    def forward(self, obs: dict) -> th.Tensor:
        ray_x = self.ray_extractor(obs[OBS_RAYS])
        drive_x = self.drive_state_extractor(obs[OBS_DRIVE_STATE])
        return th.cat([ray_x, drive_x], dim=1)
