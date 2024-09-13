"train a robot in robot_sf on a slurm server"

"""
This script is used to train a PPO model on the CartPole-v1 environment.
"""
import sys
from datetime import datetime

import gymnasium as gym
from loguru import logger

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.feature_extractor import DynamicsExtractor
from robot_sf.tb_logging import DrivingMetricsCallback


def training():
    n_envs = 64
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 2


    def make_env():
        config = EnvSettings()
        config.sim_config.ped_density_by_difficulty = ped_densities
        config.sim_config.difficulty = difficulty
        return RobotEnv(config)

    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    policy_kwargs = dict(features_extractor_class=DynamicsExtractor)
    model = PPO(
        "MultiInputPolicy",
        env,
        tensorboard_log="./logs/ppo_logs/",
        policy_kwargs=policy_kwargs
        )
    save_model_callback = CheckpointCallback(
        500_000 // n_envs,
        "./model/backup",
        "ppo_model"
        )
    collect_metrics_callback = DrivingMetricsCallback(n_envs)
    combined_callback = CallbackList(
        [save_model_callback, collect_metrics_callback]
        )

    logger.info("start learning")
    model.learn(
        total_timesteps=10_000_000,
        progress_bar=True,
        callback=combined_callback
        )
    logger.info("save model")
    model.save("./model/ppo_model")


if __name__ == '__main__':
    logger.info(f"python path: {sys.executable}")
    logger.info(f"python version: {sys.version}")

    logger.info("start training")
    training()
    logger.info("end training")
