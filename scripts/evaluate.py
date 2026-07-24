"""Evaluate a trained driving policy in the robot_sf Gymnasium environment.

Loads a stable-baselines3 model (PPO or A2C), wraps the :class:`RobotEnv` with
an observation/action adapter, runs rollouts across configured pedestrian-density
difficulties, and writes per-difficulty collision/route-completion metrics to
``results.json``.
"""

import json
from dataclasses import dataclass
from typing import Union

import gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3 import A2C, PPO

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

from robot_sf.eval import EnvMetrics
from robot_sf.gym_env.observation_config import set_observation_stack_steps
from robot_sf.gym_env.robot_env import EnvSettings, RobotEnv
from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS

DriveModel = Union[PPO, A2C]
VehicleConfig = Union[DifferentialDriveSettings, BicycleDriveSettings]


@dataclass
class GymAdapterSettings:
    """Configuration for adapting :class:`RobotEnv` observations to a model.

    Attributes:
        obs_space: Target observation space exposed to the model.
        action_space: Target action space exposed to the model.
        obs_timesteps: Number of stacked observation timesteps expected by the model.
        squeeze_obs: When true, drop singleton dimensions from observation arrays.
        cut_2nd_target_angle: When true, drop the second target angle component.
        return_dict: When true, pass the raw dict observation through unchanged.
    """

    obs_space: spaces.Space
    action_space: spaces.Space
    obs_timesteps: int
    squeeze_obs: bool
    cut_2nd_target_angle: bool
    return_dict: bool

    def obs_adapter(self, obs):
        """Adapt a raw :class:`RobotEnv` observation for the model.

        When ``return_dict`` is set the observation is returned unchanged.
        Otherwise the drive-state and ray components are optionally trimmed and
        squeezed, then concatenated into a single array along the time axis.

        Args:
            obs: Raw observation dict produced by :class:`RobotEnv`.

        Returns:
            Observation in the layout expected by the model.
        """
        if self.return_dict:
            return obs
        else:
            drive_state = obs[OBS_DRIVE_STATE]
            ray_state = obs[OBS_RAYS]

            if self.cut_2nd_target_angle:
                drive_state = drive_state[:, :-1]

            if self.squeeze_obs:
                drive_state = np.squeeze(drive_state)
                ray_state = np.squeeze(ray_state)

            axis = 0 if self.obs_timesteps == 1 else 1
            return np.concatenate((ray_state, drive_state), axis=axis)


@dataclass
class EvalSettings:
    """Top-level configuration for a policy evaluation run.

    Attributes:
        num_episodes: Number of episodes to roll out per difficulty.
        ped_densities: Pedestrian density applied at each difficulty index.
        vehicle_config: Drive-model vehicle settings (differential or bicycle).
        prf_config: Pedestrian-robot-force configuration.
        gym_config: Observation/action adapter configuration.
    """

    num_episodes: int
    ped_densities: list[float]
    vehicle_config: VehicleConfig
    prf_config: PedRobotForceConfig
    gym_config: GymAdapterSettings


@dataclass
class AdaptedEnv(gymnasium.Env):
    """Gymnasium wrapper that exposes an adapted :class:`RobotEnv`.

    Delegates stepping and reset to the wrapped environment while rewriting the
    observation through :meth:`GymAdapterSettings.obs_adapter` and projecting
    the configured observation/action spaces.

    Attributes:
        orig_env: Underlying :class:`RobotEnv` being adapted.
        config: Adapter settings controlling observation transforms and spaces.
    """

    orig_env: RobotEnv
    config: GymAdapterSettings

    @property
    def observation_space(self):
        """Return the configured observation space."""
        return self.config.obs_space

    @property
    def action_space(self):
        """Return the configured action space."""
        return self.config.action_space

    def step(self, action):
        """Apply an action to the wrapped environment and adapt the result.

        Args:
            action: Action to apply in the wrapped environment's action space.

        Returns:
            Adapted observation, reward, termination flag, and metadata tuple.
        """
        obs, reward, done, meta = self.orig_env.step(action)
        obs = self.config.obs_adapter(obs)
        return obs, reward, done, meta

    def reset(self):
        """Reset the wrapped environment and adapt the initial observation.

        Returns:
            Adapted initial observation.
        """
        obs = self.orig_env.reset()
        return self.config.obs_adapter(obs)


def evaluate(env: gymnasium.Env, model: DriveModel, num_episodes: int) -> EnvMetrics:
    """Roll out a policy and collect aggregate evaluation metrics.

    Runs ``num_episodes`` episodes using deterministic model predictions,
    resetting the environment at each route end, and accumulates per-episode
    metadata (collisions, route completion, timeouts) into an
    :class:`EnvMetrics` instance.

    Args:
        env: Gymnasium environment to evaluate in.
        model: Trained stable-baselines3 policy (PPO or A2C).
        num_episodes: Number of episodes to roll out.

    Returns:
        Aggregated evaluation metrics across all rolled-out episodes.
    """
    eval_metrics = EnvMetrics(cache_size=num_episodes)

    iterator = tqdm(range(num_episodes)) if tqdm is not None else range(num_episodes)
    for _ in iterator:
        is_end_of_route = False
        obs = env.reset()
        while not is_end_of_route:
            action, _ = model.predict(obs, deterministic=True)
            # Env step returns obs, reward, terminated, truncated, info
            obs, _reward, terminated, truncated, meta = env.step(action)
            done = terminated or truncated
            meta = meta["meta"]
            eval_metrics.update(meta)
            if done:
                obs = env.reset()
                is_end_of_route = (
                    meta["is_pedestrian_collision"]
                    or meta["is_obstacle_collision"]
                    or meta["is_route_complete"]
                    or meta["is_timesteps_exceeded"]
                )

    return eval_metrics


def prepare_env(settings: EvalSettings, difficulty: int) -> gymnasium.Env:
    """Build an evaluation environment for one configured difficulty.

    Args:
        settings: Evaluation settings containing gym, force, and vehicle configuration.
        difficulty: Pedestrian-density difficulty index to apply to the environment.

    Returns:
        Adapted Gymnasium environment ready for policy evaluation.
    """
    env_settings = EnvSettings()
    env_settings.sim_config.prf_config = settings.prf_config
    env_settings.sim_config.ped_density_by_difficulty = settings.ped_densities
    env_settings.sim_config.difficulty = difficulty
    set_observation_stack_steps(env_settings, settings.gym_config.obs_timesteps)
    env_settings.robot_config = settings.vehicle_config
    orig_env = RobotEnv(env_settings)
    return AdaptedEnv(orig_env, settings.gym_config)


def prepare_model(model_path: str, env: gymnasium.Env) -> DriveModel:
    """Load a trained A2C policy bound to the given environment.

    Args:
        model_path: Filesystem path to the saved A2C model.
        env: Gymnasium environment the model will interact with.

    Returns:
        Loaded stable-baselines3 model ready for prediction.
    """
    return A2C.load(model_path, env=env)


def evaluation_series(model_path: str, settings: EvalSettings):
    """Run evaluation across all configured difficulties.

    For each pedestrian-density difficulty, builds an environment, loads the
    model, evaluates it, prints the metrics, and writes the accumulated
    per-difficulty results to ``results.json``.

    Args:
        model_path: Filesystem path to the saved model to evaluate.
        settings: Evaluation settings describing episodes, densities, and adapters.
    """
    all_metrics = {}

    for difficulty in range(len(settings.ped_densities)):
        env = prepare_env(settings, difficulty)
        model = prepare_model(model_path, env)
        eval_metrics = evaluate(env, model, settings.num_episodes)

        metrics = {
            "route_completion_rate": eval_metrics.route_completion_rate,
            "obstacle_collision_rate": eval_metrics.obstacle_collision_rate,
            "pedestrian_collision_rate": eval_metrics.pedestrian_collision_rate,
            "timeout_rate": eval_metrics.timeout_rate,
        }
        print(f"run with difficulty {difficulty} completed with metrics:", metrics)

        all_metrics[difficulty] = metrics
        with open("results.json", "w") as f:
            json.dump(all_metrics, f)


def main():
    """Configure and run a default evaluation series for the bundled A2C model."""
    model_path = "./model/a2c_model"
    obs_space, action_space = prepare_gym_spaces()

    gym_settings = GymAdapterSettings(
        obs_space=obs_space,
        action_space=action_space,
        obs_timesteps=1,
        squeeze_obs=True,
        cut_2nd_target_angle=True,
        return_dict=False,
    )

    vehicle_config = DifferentialDriveSettings(
        radius=1.0,
        max_linear_speed=0.5,
        max_angular_speed=0.5,
        wheel_radius=0.05,
        interaxis_length=0.3,
    )

    prf_config = PedRobotForceConfig(
        is_active=True,
        robot_radius=1.0,
        activation_threshold=2.0,
        force_multiplier=10.0,
    )

    settings = EvalSettings(
        num_episodes=100,
        ped_densities=[0.00, 0.02, 0.08, 1.00],
        vehicle_config=vehicle_config,
        prf_config=prf_config,
        gym_config=gym_settings,
    )

    evaluation_series(model_path, settings)


def prepare_gym_spaces():
    """Build empty placeholder observation and action Gymnasium spaces.

    Returns:
        A ``(obs_space, action_space)`` pair of empty unbounded ``Box`` spaces.
    """
    obs_low = np.array([])
    obs_high = np.array([])
    action_low = np.array([])
    action_high = np.array([])

    obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)
    action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float64)
    return obs_space, action_space


if __name__ == "__main__":
    main()
