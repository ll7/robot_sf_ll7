"""Training script for pedestrian avoidance using PPO with a differential-drive robot.

This module provides training functionality for a robot navigation policy
using Proximal Policy Optimization (PPO) in environments with pedestrians
of varying densities.
"""

import datetime
from collections.abc import Mapping

import loguru
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from robot_sf.feature_extractor import DynamicsExtractor
from robot_sf.gym_env.environment_factory import make_pedestrian_env
from robot_sf.gym_env.reward import stationary_collision_ped_reward
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.sensor.range_sensor import LidarScannerSettings
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.tb_logging import AdversarialPedestrianMetricsCallback

logger = loguru.logger


class LegacyRun023ObsAdapter:
    """Wrap a PPO model so run_023 gets the legacy flattened observation format."""

    def __init__(self, model: PPO):
        """Store the wrapped PPO model and expose its action space."""
        self._model = model
        self.action_space = getattr(model, "action_space", None)

    def set_action_space(self, action_space) -> None:
        """Allow pedestrian env to sync action-space expectations."""
        self.action_space = action_space
        if hasattr(self._model, "set_action_space"):
            self._model.set_action_space(action_space)

    def predict(self, obs, deterministic: bool = True):
        """Adapt dict observations to run_023 format before model inference."""
        adapted_obs = obs
        if isinstance(obs, Mapping):
            drive_state = np.asarray(obs[OBS_DRIVE_STATE])[:, :-1].copy()
            ray_state = np.asarray(obs[OBS_RAYS])
            drive_state[:, 2] *= 10
            drive_state = np.squeeze(drive_state).reshape(-1)
            ray_state = np.squeeze(ray_state).reshape(-1)
            adapted_obs = np.concatenate((ray_state, drive_state), axis=0)
        return self._model.predict(adapted_obs, deterministic=deterministic)


def training(svg_map_path: str) -> None:
    """Train a PPO policy for pedestrian avoidance on a given SVG map.

    This function builds a vectorized pedestrian environment using the provided SVG map,
    loads a pretrained robot model, and trains a PPO agent with adversarial metrics
    logging and periodic checkpointing. The trained policy is saved with a timestamp.

    Args:
        svg_map_path: Path to the SVG map file used to build the environment.

    Side Effects:
        - Creates multiple environment subprocesses.
        - Writes TensorBoard logs under ``./logs/ppo_logs/``.
        - Writes checkpoints under ``./model/backup``.
        - Saves the final model under ``./model_ped/`` with a timestamp.

    Raises:
        FileNotFoundError: If ``svg_map_path`` or the pretrained model path is invalid.
        OSError: If the environment cannot spawn subprocesses or write output files.
    """
    n_envs = 8
    # Match differential-drive defensive policy profile (run_023).
    ped_densities = [0.04]
    difficulty = 0

    def make_env():
        map_definition = convert_map(svg_map_path)
        robot_model = LegacyRun023ObsAdapter(PPO.load("./model/run_023", env=None))

        # Configure ego pedestrian lidar with longer range and 120 degree view
        ego_ped_lidar = LidarScannerSettings.ego_pedestrian_lidar()

        config = PedestrianSimulationConfig(
            map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
            sim_config=SimulationSettings(
                stack_steps=1,
                difficulty=difficulty,
                ped_density_by_difficulty=ped_densities,
            ),
            robot_config=DifferentialDriveSettings(radius=1.0, max_angular_speed=0.5),
            spawn_near_robot=False,
            ego_ped_lidar_config=ego_ped_lidar,
        )
        env = make_pedestrian_env(
            config=config,
            robot_model=robot_model,
            debug=False,
            recording_enabled=False,
            reward_func=stationary_collision_ped_reward,
        )
        return env

    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    try:
        policy_kwargs = {"features_extractor_class": DynamicsExtractor}
        model = PPO(
            "MultiInputPolicy",
            env,
            tensorboard_log="./logs/ppo_logs/",
            policy_kwargs=policy_kwargs,
        )
        save_model_callback = CheckpointCallback(500_000 // n_envs, "./model/backup", "ppo_model")
        collect_metrics_callback = AdversarialPedestrianMetricsCallback(n_envs)
        combined_callback = CallbackList([save_model_callback, collect_metrics_callback])

        model.learn(total_timesteps=1_500_000, progress_bar=True, callback=combined_callback)
        now = datetime.datetime.now()
        filename = now.strftime("%Y-%m-%d_%H-%M-%S")
        model.save(f"./model_ped/ppo_{filename}")
        logger.info(f"Model saved as ppo_{filename}")
    finally:
        env.close()


if __name__ == "__main__":
    SVG_MAP = "maps/svg_maps/masterthesis/headon.svg"

    training(SVG_MAP)
