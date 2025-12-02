"""Module hparam_opt auto-generated docstring."""

import logging
import sys

import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from robot_sf.feature_extractor import DynamicsExtractor
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import RobotEnv, simple_reward
from robot_sf.tb_logging import DrivingMetricsCallback, VecEnvMetrics

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))


class DriveQualityCallback(BaseCallback):
    """DriveQualityCallback class."""

    def __init__(self, metrics: VecEnvMetrics, thresholds: list[float], max_steps: int):
        """Init.

        Args:
            metrics: Auto-generated placeholder description.
            thresholds: Auto-generated placeholder description.
            max_steps: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        super().__init__()
        self.metrics = metrics
        self.completion_thresholds = thresholds
        self.max_steps = max_steps
        self.steps_to_reach_threshold = np.full((len(thresholds)), self.max_steps)
        self.log_freq = 1000

    @property
    def score(self) -> float:
        """Score.

        Returns:
            float: Auto-generated placeholder description.
        """
        steps_per_threshold = zip(
            self.completion_thresholds,
            self.steps_to_reach_threshold,
            strict=False,
        )
        reached_thresholds = [(t, s) for t, s in steps_per_threshold if s < self.max_steps]
        threshold_scores = sum(
            [t * (2 + (self.max_steps - s) / self.max_steps) for t, s in reached_thresholds],
        )
        return threshold_scores / (sum(self.completion_thresholds) * 3)

    def _on_training_start(self):
        """On training start.

        Returns:
            Any: Auto-generated placeholder description.
        """
        pass

    def _on_step(self) -> bool:
        """On step.

        Returns:
            bool: Auto-generated placeholder description.
        """
        curr_step = self.n_calls
        if curr_step % self.log_freq == 0:
            for i, completion_threshold in enumerate(self.completion_thresholds):
                if self.metrics.route_completion_rate >= completion_threshold:
                    self.steps_to_reach_threshold[i] = min(
                        curr_step,
                        self.steps_to_reach_threshold[i],
                    )
        return True  # info: don't request early abort


def training_score(
    study_name: str,
    hparams: dict,
    max_steps: int = 5_000_000,
    difficulty: int = 1,
    route_completion_thresholds: list[float] | None = None,
):
    """Training score.

    Args:
        study_name: Auto-generated placeholder description.
        hparams: Auto-generated placeholder description.
        max_steps: Auto-generated placeholder description.
        difficulty: Auto-generated placeholder description.
        route_completion_thresholds: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    if route_completion_thresholds is None:
        route_completion_thresholds = [i / 100 for i in range(1, 101)]

    def make_env():
        """Make env.

        Returns:
            Any: Auto-generated placeholder description.
        """
        config = EnvSettings()
        config.sim_config.difficulty = difficulty
        config.sim_config.stack_steps = hparams["num_stacked_steps"]
        config.sim_config.time_per_step_in_secs = hparams["d_t"]
        config.sim_config.use_next_goal = hparams["use_next_goal"]

        def reward_func(meta):
            """Reward func.

            Args:
                meta: Auto-generated placeholder description.

            Returns:
                Any: Auto-generated placeholder description.
            """
            return simple_reward(
                meta,
                hparams["step_discount"],
                hparams["ped_coll_penalty"],
                hparams["obst_coll_penalty"],
                hparams["reach_wp_reward"],
            )

        return RobotEnv(config, reward_func=reward_func)

    env = make_vec_env(make_env, n_envs=hparams["n_envs"], vec_env_cls=SubprocVecEnv)

    policy_kwargs = {
        "features_extractor_class": DynamicsExtractor,
        "features_extractor_kwargs": {
            "use_ray_conv": hparams["use_ray_conv"],
            "num_filters": hparams["num_filters"],
            "kernel_sizes": hparams["kernel_sizes"],
            "dropout_rates": hparams["dropout_rates"],
        },
    }
    model = PPO(
        "MultiInputPolicy",
        env,
        tensorboard_log=f"./logs/{study_name}/",
        n_steps=hparams["n_steps"],
        n_epochs=hparams["n_epochs"],
        use_sde=hparams["use_sde"],
        policy_kwargs=policy_kwargs,
    )
    collect_metrics_callback = DrivingMetricsCallback(hparams["n_envs"])
    threshold_callback = DriveQualityCallback(
        collect_metrics_callback.metrics,
        route_completion_thresholds,
        max_steps,
    )
    combined_callback = CallbackList([collect_metrics_callback, threshold_callback])

    model.learn(total_timesteps=max_steps, progress_bar=True, callback=combined_callback)
    return threshold_callback.score


def suggest_ppo_params(trial: optuna.Trial, tune: bool = False) -> dict:
    """Suggest ppo params.

    Args:
        trial: Auto-generated placeholder description.
        tune: Auto-generated placeholder description.

    Returns:
        dict: Auto-generated placeholder description.
    """
    if tune:
        n_envs = trial.suggest_categorical("n_envs", [32, 40, 48, 56, 64])
        n_epochs = trial.suggest_int("n_epochs", 2, 20)
        n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 1536, 2048])
        use_sde = trial.suggest_categorical("use_sde", [True, False])
        use_ray_conv = trial.suggest_categorical("use_ray_conv", [True, False])
        if use_ray_conv:
            num_filters = [
                trial.suggest_categorical(f"num_filters_{i}", [8, 16, 32, 64, 128, 256])
                for i in range(4)
            ]
            kernel_sizes = [
                trial.suggest_categorical(f"kernel_sizes_{i}", [3, 5, 7, 9]) for i in range(4)
            ]
            dropout_rates = [trial.suggest_float(f"dropout_rates_{i}", 0.0, 1.0) for i in range(4)]
        else:
            num_filters = []
            kernel_sizes = []
            dropout_rates = []
    else:  # use defaults
        n_envs = 64
        n_epochs = 10
        n_steps = 1024
        use_sde = False
        use_ray_conv = True
        num_filters = [256, 256, 64, 32]
        kernel_sizes = [5, 5, 5, 3]
        dropout_rates = [0.1, 0.1, 0.3, 0.3]

    return {
        "n_envs": n_envs,
        "n_epochs": n_epochs,
        "n_steps": n_steps,
        "use_sde": use_sde,
        "use_ray_conv": use_ray_conv,
        "num_filters": num_filters,
        "kernel_sizes": kernel_sizes,
        "dropout_rates": dropout_rates,
    }


def suggest_simulation_params(trial: optuna.Trial, tune: bool = False) -> dict:
    """Suggest simulation params.

    Args:
        trial: Auto-generated placeholder description.
        tune: Auto-generated placeholder description.

    Returns:
        dict: Auto-generated placeholder description.
    """
    if tune:
        num_stacked_steps = trial.suggest_int("num_stacked_steps", 1, 5)
        num_lidar_rays = trial.suggest_categorical("num_lidar_rays", [144, 176, 208, 272])
        d_t = trial.suggest_categorical("d_t", [0.1, 0.2, 0.3, 0.4, 0.5])
        use_next_goal = trial.suggest_categorical("use_next_goal", [True, False])
    else:  # use defaults
        num_stacked_steps = 3
        num_lidar_rays = 272
        d_t = 0.1
        use_next_goal = True

    return {
        "num_stacked_steps": num_stacked_steps,
        "num_lidar_rays": num_lidar_rays,
        "d_t": d_t,
        "use_next_goal": use_next_goal,
    }


def suggest_reward_params(trial: optuna.Trial, tune: bool = False) -> dict:
    """Suggest reward params.

    Args:
        trial: Auto-generated placeholder description.
        tune: Auto-generated placeholder description.

    Returns:
        dict: Auto-generated placeholder description.
    """
    if tune:
        ped_coll_penalty = trial.suggest_int("ped_coll_penalty", -10, -1)
        obst_coll_penalty = trial.suggest_int("obst_coll_penalty", -10, -1)
        step_discount = trial.suggest_float("step_discount", -1.0, 0.0)
        reach_wp_reward = 1.0
        # reach_wp_reward = trial.suggest_int("reach_wp_reward", 1, 10)
    else:  # use defaults
        ped_coll_penalty = -2.0
        obst_coll_penalty = -2.0
        step_discount = -0.1
        reach_wp_reward = 1.0

    return {
        "ped_coll_penalty": ped_coll_penalty,
        "obst_coll_penalty": obst_coll_penalty,
        "step_discount": step_discount,
        "reach_wp_reward": reach_wp_reward,
    }


def objective(trial: optuna.Trial, study_name: str) -> float:
    """Objective.

    Args:
        trial: Auto-generated placeholder description.
        study_name: Auto-generated placeholder description.

    Returns:
        float: Auto-generated placeholder description.
    """
    ppo_params = suggest_ppo_params(trial, tune=False)
    sim_params = suggest_simulation_params(trial, tune=False)
    rew_params = suggest_reward_params(trial, tune=True)

    def merge_dicts(dicts: list[dict]) -> dict:
        """Merge dicts.

        Args:
            dicts: Auto-generated placeholder description.

        Returns:
            dict: Auto-generated placeholder description.
        """
        return {k: d[k] for d in dicts for k in d}

    sugg_params = merge_dicts([ppo_params, sim_params, rew_params])
    return training_score(study_name, sugg_params)


def generate_storage_url(study_name: str) -> str:
    """Generate storage url.

    Args:
        study_name: Auto-generated placeholder description.

    Returns:
        str: Auto-generated placeholder description.
    """
    return f"sqlite:///logs/{study_name}.db"


def tune_hparams(study_name: str):
    """Tune hparams.

    Args:
        study_name: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=generate_storage_url(study_name),
        load_if_exists=True,
    )
    study.optimize(lambda t: objective(t, study_name), n_trials=100, gc_after_trial=True)


if __name__ == "__main__":
    tune_hparams("reward-opt")
