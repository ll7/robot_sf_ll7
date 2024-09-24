"""Train a robot in robot_sf on a SLURM server with resource tracking."""

import sys
import psutil
import GPUtil
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.feature_extractor import DynamicsExtractor
from robot_sf.tb_logging import DrivingMetricsCallback

class LogResourceUsageCallback(BaseCallback):
    """Custom callback to log CPU and GPU usage to TensorBoard."""

    def __init__(self, verbose=0):
        super(LogResourceUsageCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """Log CPU and GPU usage at each step."""
        cpu_usage = psutil.cpu_percent()
        gpus = GPUtil.getGPUs()
        gpu_usage = gpus[0].load * 100 if gpus else 0  # Assuming using the first GPU

        # Log to TensorBoard
        self.logger.record('cpu_usage', cpu_usage)
        self.logger.record('gpu_usage', gpu_usage)

        return True

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
        [save_model_callback, collect_metrics_callback, LogResourceUsageCallback()]
    )

    logger.info("Start learning")

    model.learn(
        total_timesteps=1_000_000,
        progress_bar=True,
        callback=combined_callback
    )


    logger.info("Save model")
    model.save("./model/ppo_model")

if __name__ == '__main__':
    logger.info(f"Python path: {sys.executable}")
    logger.info(f"Python version: {sys.version}")

    logger.info("Start training")
    training()
    logger.info("End training")
