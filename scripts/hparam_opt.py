import sys
import logging
from typing import List

import optuna
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, BaseCallback

from robot_sf.robot_env import RobotEnv
from robot_sf.sim_config import EnvSettings
from robot_sf.feature_extractor import DynamicsExtractor
from robot_sf.tb_logging import DrivingMetricsCallback, VecEnvMetrics


class DriveQualityCallback(BaseCallback):
    def __init__(self, metrics: VecEnvMetrics, thresholds: List[float], max_steps: int):
        super(DriveQualityCallback, self).__init__()
        self.metrics = metrics
        self.completion_thresholds = thresholds
        self.max_steps = max_steps
        self.steps_to_reach_threshold = np.full((len(thresholds)), self.max_steps)
        self.log_freq = 1000

    @property
    def score(self) -> float:
        return sum([t * (self.max_steps - s) / self.max_steps
                    for t, s in zip(self.completion_thresholds, self.steps_to_reach_threshold)])

    def _on_training_start(self):
        pass

    def _on_step(self) -> bool:
        curr_step = self.n_calls
        if curr_step % self.log_freq == 0:
            for i, completion_threshold in enumerate(self.completion_thresholds):
                if self.metrics.route_completion_rate >= completion_threshold:
                    self.steps_to_reach_threshold[i] = min(curr_step, self.steps_to_reach_threshold[i])
        return True # info: don't request early abort


def training_score(
        hparams: dict, max_steps: int=5_000_000, difficulty: int=1,
        route_completion_thresholds: List[float]=[
            0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
            0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 1.0]):

    def make_env():
        config = EnvSettings()
        config.sim_config.difficulty = difficulty
        return RobotEnv(config)

    env = make_vec_env(make_env, n_envs=hparams["n_envs"], vec_env_cls=SubprocVecEnv)

    policy_kwargs = dict(
        features_extractor_class=DynamicsExtractor,
        features_extractor_kwargs=dict(
            use_ray_conv = hparams["use_ray_conv"],
            num_filters = hparams["num_filters"],
            kernel_sizes = hparams["kernel_sizes"],
            dropout_rates = hparams["dropout_rates"]
        ))
    model = PPO("MultiInputPolicy", env, tensorboard_log="./logs/optuna_logs/",
                n_steps=hparams["n_steps"], n_epochs=hparams["n_epochs"],
                use_sde=hparams["use_sde"], policy_kwargs=policy_kwargs)
    collect_metrics_callback = DrivingMetricsCallback(hparams["n_envs"])
    threshold_callback = DriveQualityCallback(
        collect_metrics_callback.metrics, route_completion_thresholds, max_steps)
    combined_callback = CallbackList([collect_metrics_callback, threshold_callback])

    model.learn(total_timesteps=max_steps, progress_bar=True, callback=combined_callback)
    return threshold_callback.score


def objective(trial: optuna.Trial) -> float:
    n_envs = trial.suggest_categorical("n_envs", [32, 40, 48, 56, 64])
    n_epochs = trial.suggest_int("n_epochs", 2, 20)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096])
    use_sde = trial.suggest_categorical("use_sde", [True, False])
    use_ray_conv = trial.suggest_categorical("use_ray_conv", [True, False])
    if use_ray_conv:
        num_filters = [trial.suggest_categorical(f"num_filters_{i}", [8, 16, 32, 64, 128, 256]) for i in range(4)]
        kernel_sizes = [trial.suggest_categorical(f"kernel_sizes_{i}", [3, 5, 7, 9]) for i in range(4)]
        dropout_rates = [trial.suggest_float(f"dropout_rates_{i}", 0.0, 1.0) for i in range(4)]
    else:
        num_filters = []
        kernel_sizes = []
        dropout_rates = []

    sugg_params = {
        "n_envs": n_envs,
        "n_epochs": n_epochs,
        "n_steps": n_steps,
        "use_sde": use_sde,
        "use_ray_conv": use_ray_conv,
        "num_filters": num_filters,
        "kernel_sizes": kernel_sizes,
        "dropout_rates": dropout_rates
    }

    return training_score(sugg_params)


def generate_storage_url(study_name: str) -> str:
    return "sqlite:///logs/{}.db".format(study_name)


def tune_hparams(study_name: str):
    study = optuna.create_study(
        study_name=study_name, direction="maximize",
        storage=generate_storage_url(study_name))
    study.optimize(objective, n_trials=100, gc_after_trial=True)


if __name__ == '__main__':
    tune_hparams("diffdrive-opt")
