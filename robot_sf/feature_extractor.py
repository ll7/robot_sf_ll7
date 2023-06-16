import numpy as np
from gym import spaces

import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from robot_sf.robot_env import OBS_DRIVE_STATE, OBS_RAYS


class DynamicsExtractor(BaseFeaturesExtractor):
    def __init__(
            self, observation_space: spaces.Dict,
            ray_features: int = 256,
            drive_state_features: int = 64):
        super().__init__(observation_space, features_dim=ray_features+drive_state_features)

        rays_space: spaces.Box = observation_space.spaces[OBS_RAYS]
        drive_state_space: spaces.Box = observation_space.spaces[OBS_DRIVE_STATE]

        self.ray_extractor = nn.Sequential(
            nn.Conv1d(rays_space.shape[0], 64, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear((rays_space.shape[1] // 16) * 64, ray_features))

        self.drive_state_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(drive_state_space.shape), 64),
            nn.Linear(64, drive_state_features))

        self.extractors = nn.ModuleDict(
            { OBS_RAYS: self.ray_extractor, OBS_DRIVE_STATE: self.drive_state_extractor })

    def forward(self, obs: dict) -> th.Tensor:
        ray_x = self.ray_extractor(obs[OBS_RAYS])
        drive_x = self.drive_state_extractor(obs[OBS_DRIVE_STATE])
        return th.cat([ray_x, drive_x], dim=1)
