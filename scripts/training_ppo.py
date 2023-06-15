import numpy as np
from gym import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from robot_sf.robot_env import RobotEnv


class DynamicsExtractor(BaseFeaturesExtractor):
    def __init__(
            self, observation_space: spaces.Dict,
            ray_features: int = 256,
            drive_state_features: int = 64):
        super().__init__(observation_space, features_dim=ray_features+drive_state_features)

        rays_space: spaces.Box = observation_space.spaces["rays"]
        drive_state_space: spaces.Box = observation_space.spaces["drive_state"]

        ray_extractor = nn.Sequential(
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

        drive_state_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(drive_state_space.shape), 64),
            nn.Linear(64, drive_state_features))

        self.extractors = nn.ModuleDict(
            { "rays": ray_extractor, "drive_state": drive_state_extractor })

    def forward(self, obs: dict) -> th.Tensor:
        features_vecs = [extractor(obs[key]) for key, extractor in self.extractors.items()]
        return th.cat(features_vecs, dim=1)


def training():
    n_envs = 64
    env = make_vec_env(lambda: RobotEnv(), n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    policy_kwargs = dict(features_extractor_class=DynamicsExtractor)
    model = PPO("MultiInputPolicy", env, tensorboard_log="./logs/ppo_logs/", policy_kwargs=policy_kwargs, n_steps=512)
    save_model = CheckpointCallback(1_000_000 // n_envs, "./model/backup", "ppo_model")

    model.learn(total_timesteps=50_000_000, progress_bar=True, callback=save_model)
    model.save("./model/ppo_model")


if __name__ == '__main__':
    training()
