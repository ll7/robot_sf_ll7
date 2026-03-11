"""Training script for pedestrian avoidance using PPO with adversarial metrics.

This module provides training functionality for a robot navigation policy
using Proximal Policy Optimization (PPO) in environments with pedestrians
of varying densities.
"""

import datetime

import loguru
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from robot_sf.feature_extractor import DynamicsExtractor
from robot_sf.gym_env.environment_factory import make_pedestrian_env
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.tb_logging import AdversialPedestrianMetricsCallback

logger = loguru.logger


def training(svg_map_path: str):
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
    n_envs = 10
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 2

    def make_env():
        map_definition = convert_map(svg_map_path)
        robot_model = PPO.load("./model/run_043", env=None)

        config = PedestrianSimulationConfig(
            map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
            sim_config=SimulationSettings(
                difficulty=difficulty,
                ped_density_by_difficulty=ped_densities,
            ),
            robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
            spawn_near_robot=True,
        )
        env = make_pedestrian_env(
            config=config,
            robot_model=robot_model,
            debug=False,
            recording_enabled=False,
        )
        return env

    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    policy_kwargs = {"features_extractor_class": DynamicsExtractor}
    model = PPO(
        "MultiInputPolicy", env, tensorboard_log="./logs/ppo_logs/", policy_kwargs=policy_kwargs
    )
    save_model_callback = CheckpointCallback(500_000 // n_envs, "./model/backup", "ppo_model")
    collect_metrics_callback = AdversialPedestrianMetricsCallback(n_envs)
    combined_callback = CallbackList([save_model_callback, collect_metrics_callback])

    model.learn(total_timesteps=1_000_000, progress_bar=True, callback=combined_callback)
    now = datetime.datetime.now()
    filename = now.strftime("%Y-%m-%d_%H-%M-%S")
    model.save(f"./model_ped/ppo_{filename}")
    logger.info(f"Model saved as ppo_{filename}")


if __name__ == "__main__":
    SVG_MAP = "maps/svg_maps/debug_06.svg"

    training(SVG_MAP)
