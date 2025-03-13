"""
Evaluation script for trained RL models in the Robot Social Force environment.

This script provides functionality to:
1. Load trained models (A2C/PPO)
2. Configure and prepare the environment with different difficulty levels
3. Evaluate models across multiple episodes
4. Record and report performance metrics

Usage:
    python evaluate.py

Key metrics collected:
    - Route completion rate
    - Obstacle collision rate
    - Pedestrian collision rate
    - Timeout rate
"""

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

# Type aliases for better readability
DriveModel = Union[PPO, A2C]
VehicleConfig = Union[DifferentialDriveSettings, BicycleDriveSettings]


@dataclass
class GymAdapterSettings:
    """
    Configuration for adapting the environment observation space to model requirements.

    Attributes:
        obs_space: Observation space for the model
        action_space: Action space for the model
        obs_timesteps: Number of observation timesteps to stack
        squeeze_obs: Whether to squeeze the observation dimensions
        cut_2nd_target_angle: Whether to remove the second target angle from drive state
        return_dict: Whether to return observations as a dictionary
    """

    obs_space: spaces.Space
    action_space: spaces.Space
    obs_timesteps: int
    squeeze_obs: bool
    cut_2nd_target_angle: bool
    return_dict: bool

    def obs_adapter(self, obs):
        """
        Adapt environment observations to the format expected by the model.

        Args:
            obs: Raw observation from the environment

        Returns:
            Processed observation in the format expected by the model
        """
        if self.return_dict:
            # Return observations as a dictionary
            return obs
        else:
            # Extract drive state and ray (sensor) state from observation dictionary
            drive_state = obs[OBS_DRIVE_STATE]
            ray_state = obs[OBS_RAYS]

            # Optionally remove the second target angle
            if self.cut_2nd_target_angle:
                drive_state = drive_state[:, :-1]

            # Optionally squeeze observation dimensions
            if self.squeeze_obs:
                drive_state = np.squeeze(drive_state)
                ray_state = np.squeeze(ray_state)

            # Determine concatenation axis based on number of timesteps
            axis = 0 if self.obs_timesteps == 1 else 1
            return np.concatenate((ray_state, drive_state), axis=axis)


@dataclass
class EvalSettings:
    """
    Settings for the evaluation process.

    Attributes:
        num_episodes: Number of episodes to evaluate
        ped_densities: List of pedestrian densities for different difficulty levels
        vehicle_config: Configuration for the vehicle (robot)
        prf_config: Configuration for pedestrian-robot force interactions
        gym_config: Configuration for adapting the environment
    """

    num_episodes: int
    ped_densities: List[float]
    vehicle_config: VehicleConfig
    prf_config: PedRobotForceConfig
    gym_config: GymAdapterSettings


@dataclass
class AdaptedEnv(gymnasium.Env):
    """
    Environment wrapper that adapts the original environment to the expected format.

    Attributes:
        orig_env: Original environment to wrap
        config: Configuration for adapting the environment
    """

    orig_env: RobotEnv
    config: GymAdapterSettings

    @property
    def observation_space(self):
        """The observation space of the environment."""
        return self.config.obs_space

    @property
    def action_space(self):
        """The action space of the environment."""
        return self.config.action_space

    def step(self, action):
        """
        Take a step in the environment with the given action.

        Args:
            action: Action to take in the environment

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Take a step in the original environment
        obs, reward, done, meta = self.orig_env.step(action)
        # Adapt the observation
        obs = self.config.obs_adapter(obs)
        return obs, reward, done, meta

    def reset(self):
        """
        Reset the environment.

        Returns:
            Initial observation after reset
        """
        obs = self.orig_env.reset()
        return self.config.obs_adapter(obs)


def evaluate(env: gymnasium.Env, model: DriveModel, num_episodes: int) -> EnvMetrics:
    """
    Evaluate a model on the given environment for a number of episodes.

    Args:
        env: Environment to evaluate on
        model: Model to evaluate
        num_episodes: Number of episodes to evaluate

    Returns:
        Metrics collected during evaluation
    """
    # Initialize metrics collector with appropriate cache size
    eval_metrics = EnvMetrics(cache_size=num_episodes)

    # Run evaluation for specified number of episodes
    for _ in tqdm(range(num_episodes)):
        is_end_of_route = False
        obs = env.reset()

        # Continue until route is complete or failure occurs
        while not is_end_of_route:
            # Get deterministic prediction from model
            action, _ = model.predict(obs, deterministic=True)
            # Take step in environment
            obs, _, done, meta = env.step(action)
            # Extract metadata from info dict
            meta = meta["meta"]
            # Update metrics with latest data
            eval_metrics.update(meta)

            # If episode is done, reset and check if route is complete
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
    """
    Prepare the environment with the given settings and difficulty level.

    Args:
        settings: Evaluation settings
        difficulty: Difficulty level index (corresponds to pedestrian density)

    Returns:
        Configured environment
    """
    # Create environment settings
    env_settings = EnvSettings()
    # Configure pedestrian-robot force interactions
    env_settings.sim_config.prf_config = settings.prf_config
    # Set pedestrian densities for different difficulty levels
    env_settings.sim_config.ped_density_by_difficulty = settings.ped_densities
    # Set specific difficulty level
    env_settings.sim_config.difficulty = difficulty
    # Configure observation stacking
    env_settings.sim_config.stack_steps = settings.gym_config.obs_timesteps
    # Set robot configuration
    env_settings.robot_config = settings.vehicle_config

    # Create original environment
    orig_env = RobotEnv(env_settings)
    # Return wrapped environment
    return AdaptedEnv(orig_env, settings.gym_config)


def prepare_model(model_path: str, env: gymnasium.Env) -> DriveModel:
    """
    Load a trained model from the given path.

    Args:
        model_path: Path to the saved model
        env: Environment for the model

    Returns:
        Loaded model
    """
    return A2C.load(model_path, env=env)


def evaluation_series(model_path: str, settings: EvalSettings):
    """
    Run a series of evaluations with different difficulty levels.

    Args:
        model_path: Path to the saved model
        settings: Evaluation settings
    """
    # Dictionary to store metrics for each difficulty level
    all_metrics = dict()

    # Evaluate for each difficulty level
    for difficulty in range(len(settings.ped_densities)):
        # Prepare environment and model
        env = prepare_env(settings, difficulty)
        model = prepare_model(model_path, env)
        # Run evaluation
        eval_metrics = evaluate(env, model, settings.num_episodes)

        # Extract key metrics
        metrics = {
            "route_completion_rate": eval_metrics.route_completion_rate,
            "obstacle_collision_rate": eval_metrics.obstacle_collision_rate,
            "pedestrian_collision_rate": eval_metrics.pedestrian_collision_rate,
            "timeout_rate": eval_metrics.timeout_rate,
        }
        print(f"run with difficulty {difficulty} completed with metrics:", metrics)

        # Store metrics and save to file
        all_metrics[difficulty] = metrics
        with open("results.json", "w") as f:
            json.dump(all_metrics, f)


def main():
    """Main function to run the evaluation."""
    # Path to the saved model
    model_path = "./model/a2c_model"
    # Prepare observation and action spaces
    obs_space, action_space = prepare_gym_spaces()

    # Configure gym adapter settings
    gym_settings = GymAdapterSettings(
        obs_space=obs_space,
        action_space=action_space,
        obs_timesteps=1,
        squeeze_obs=True,
        cut_2nd_target_angle=True,
        return_dict=False,
    )

    # Configure vehicle settings (differential drive)
    vehicle_config = DifferentialDriveSettings(
        radius=1.0,
        max_linear_speed=0.5,
        max_angular_speed=0.5,
        wheel_radius=0.05,
        interaxis_length=0.3,
    )

    # Configure pedestrian-robot force settings
    prf_config = PedRobotForceConfig(
        is_active=True, robot_radius=1.0, activation_threshold=2.0, force_multiplier=10.0
    )

    # Create evaluation settings
    settings = EvalSettings(
        num_episodes=100,
        # Increasing pedestrian density for different difficulty levels
        ped_densities=[0.00, 0.02, 0.08, 1.00],
        vehicle_config=vehicle_config,
        prf_config=prf_config,
        gym_config=gym_settings,
    )

    # Run evaluation series
    evaluation_series(model_path, settings)


def prepare_gym_spaces():
    """
    Prepare observation and action spaces for the gym environment.

    Returns:
        Tuple of (observation_space, action_space)
    """
    # Initialize arrays for space bounds
    # Note: These are empty and should be filled with appropriate values
    obs_low = np.array([])
    obs_high = np.array([])
    action_low = np.array([])
    action_high = np.array([])

    # Create spaces
    obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)
    action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float64)
    return obs_space, action_space


if __name__ == "__main__":
    main()
