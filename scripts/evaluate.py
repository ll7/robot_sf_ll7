import json
from dataclasses import dataclass
from typing import List, Union

import gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3 import A2C, PPO
from tqdm import tqdm

from robot_sf.eval import EnvMetrics
from robot_sf.gym_env.robot_env import EnvSettings, RobotEnv
from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS

DriveModel = Union[PPO, A2C]
VehicleConfig = Union[DifferentialDriveSettings, BicycleDriveSettings]


@dataclass
class GymAdapterSettings:
    obs_space: spaces.Space
    action_space: spaces.Space
    obs_timesteps: int
    squeeze_obs: bool
    cut_2nd_target_angle: bool
    return_dict: bool

    def obs_adapter(self, obs):
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
    num_episodes: int
    ped_densities: List[float]
    vehicle_config: VehicleConfig
    prf_config: PedRobotForceConfig
    gym_config: GymAdapterSettings


@dataclass
class AdaptedEnv(gymnasium.Env):
    orig_env: RobotEnv
    config: GymAdapterSettings

    @property
    def observation_space(self):
        return self.config.obs_space

    @property
    def action_space(self):
        return self.config.action_space

    def step(self, action):
        obs, reward, done, meta = self.orig_env.step(action)
        obs = self.config.obs_adapter(obs)
        return obs, reward, done, meta

    def reset(self):
        obs = self.orig_env.reset()
        return self.config.obs_adapter(obs)


def evaluate(env: gymnasium.Env, model: DriveModel, num_episodes: int) -> EnvMetrics:
    eval_metrics = EnvMetrics(cache_size=num_episodes)

    for _ in tqdm(range(num_episodes)):
        is_end_of_route = False
        obs = env.reset()
        while not is_end_of_route:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, meta = env.step(action)
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
    env_settings = EnvSettings()
    env_settings.sim_config.prf_config = settings.prf_config
    env_settings.sim_config.ped_density_by_difficulty = settings.ped_densities
    env_settings.sim_config.difficulty = difficulty
    env_settings.sim_config.stack_steps = settings.gym_config.obs_timesteps
    env_settings.robot_config = settings.vehicle_config
    orig_env = RobotEnv(env_settings)
    return AdaptedEnv(orig_env, settings.gym_config)


def prepare_model(model_path: str, env: gymnasium.Env) -> DriveModel:
    return A2C.load(model_path, env=env)


def evaluation_series(model_path: str, settings: EvalSettings):
    all_metrics = dict()

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
        is_active=True, robot_radius=1.0, activation_threshold=2.0, force_multiplier=10.0
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
    obs_low = np.array([])
    obs_high = np.array([])
    action_low = np.array([])
    action_high = np.array([])

    obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)
    action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float64)
    return obs_space, action_space


if __name__ == "__main__":
    main()
