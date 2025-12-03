"""Train a robot in robot_sf on a SLURM server with resource tracking."""

import os
import sys

import GPUtil  # type: ignore[import]
import psutil
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from robot_sf.feature_extractor import DynamicsExtractor
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.tb_logging import DrivingMetricsCallback


class LogResourceUsageCallback(BaseCallback):
    """Custom callback to log CPU and GPU usage to TensorBoard."""

    def _on_step(self) -> bool:
        """Log CPU and GPU usage and memory utilization at each step."""
        cpu_usage = psutil.cpu_percent()
        gpus = GPUtil.getGPUs()
        gpu_usage = [gpu.load * 100 for gpu in gpus] if gpus else [0]
        gpu_memory_util = [gpu.memoryUtil * 100 for gpu in gpus] if gpus else [0]

        # Log to TensorBoard
        self.logger.record("cpu_usage", cpu_usage)
        for idx, (usage, mem_util) in enumerate(zip(gpu_usage, gpu_memory_util, strict=False)):
            self.logger.record(f"gpu_{idx}_usage", usage)
            self.logger.record(f"gpu_{idx}_memory_util", mem_util)

        return True


def training(
    n_envs: int | None = None,
    ped_densities: list[float] | None = None,
    difficulty: int = 2,
):
    """TODO docstring. Document this function.

    Args:
        n_envs: TODO docstring.
        ped_densities: TODO docstring.
        difficulty: TODO docstring.
    """
    if n_envs is None:
        n_envs = os.cpu_count() or 1
    """Train a robot in robot_sf.
    Args:
        n_envs: Number of environments to run in parallel.
        ped_densities: List of pedestrian densities to use.
        difficulty: Difficulty of the simulation.
    """
    logger.info(f"Number of CPUs: {n_envs}")
    if ped_densities is None:
        ped_densities = [0.01, 0.02, 0.04, 0.08]

    def make_env():
        """TODO docstring. Document this function."""
        config = EnvSettings()
        config.sim_config.ped_density_by_difficulty = ped_densities
        config.sim_config.difficulty = difficulty
        return RobotEnv(config)

    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    policy_kwargs = {"features_extractor_class": DynamicsExtractor}
    model = PPO(
        "MultiInputPolicy",
        env,
        tensorboard_log="./logs/ppo_logs/",
        policy_kwargs=policy_kwargs,
    )
    save_model_callback = CheckpointCallback(500_000 // n_envs, "./model/backup", "ppo_model")
    collect_metrics_callback = DrivingMetricsCallback(n_envs)
    combined_callback = CallbackList(
        [save_model_callback, collect_metrics_callback, LogResourceUsageCallback()],
    )

    logger.info("Start learning")

    model.learn(total_timesteps=1_000_000, progress_bar=True, callback=combined_callback)

    logger.info("Save model")
    model.save("./model/ppo_model")


if __name__ == "__main__":
    logger.info(f"Python path: {sys.executable}")
    logger.info(f"Python version: {sys.version}")

    logger.info("Start training")
    training()
    logger.info("End training")
