"""
Train ppo robot and log to wandb
Documentation can be found in `docs/wandb.md`
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import wandb
from robot_sf.feature_extractor import DynamicsExtractor
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.tb_logging import DrivingMetricsCallback
from wandb.integration.sb3 import WandbCallback

wandb_config = {
    "env": "robot_sf",
    "algorithm": "ppo",
    "difficulty": 2,
    "ped_densities": [0.01, 0.02, 0.04, 0.08],
    "n_envs": 32,
    "total_timesteps": 10_000_000,
}

# Start a new run to track and log to W&B.
wandb_run = wandb.init(
    project="robot_sf",
    config=wandb_config,
    save_code=True,
    group="ppo robot_sf",
    job_type="initial training",
    tags=["ppo", "robot_sf"],
    name="init ppo robot_sf",
    notes="Initial training of ppo robot_sf",
    resume="allow",
    mode="online",
    sync_tensorboard=True,
    monitor_gym=True,
)


N_ENVS = wandb_config["n_envs"]
ped_densities = wandb_config["ped_densities"]
DIFFICULTY = wandb_config["difficulty"]


def make_env():
    config = EnvSettings()
    config.sim_config.ped_density_by_difficulty = ped_densities
    config.sim_config.difficulty = DIFFICULTY
    return RobotEnv(config)


env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

policy_kwargs = dict(features_extractor_class=DynamicsExtractor)
model = PPO(
    "MultiInputPolicy", env, tensorboard_log="./logs/ppo_logs/", policy_kwargs=policy_kwargs
)
save_model_callback = CheckpointCallback(500_000 // N_ENVS, "./model/backup", "ppo_model")
collect_metrics_callback = DrivingMetricsCallback(N_ENVS)

wandb_callback = WandbCallback(
    gradient_save_freq=20_000,
    model_save_path=f"models/{wandb_run.id}",
    verbose=2,
)

combined_callback = CallbackList([save_model_callback, collect_metrics_callback, wandb_callback])

model.learn(
    total_timesteps=wandb_config["total_timesteps"], progress_bar=True, callback=combined_callback
)
model.save("./model/ppo_model")

wandb_run.finish()
