"""Debug runner for pedestrian policy models with a differential-drive robot policy.

This script mirrors ``scripts/debug_ped_policy.py`` while applying the
run_023 differential-drive robot profile and legacy observation adapter logic
used in ``scripts/debug_ped_apf.py``.
"""

import os
from pathlib import Path

import loguru
import numpy as np
from stable_baselines3 import PPO

from robot_sf.gym_env.environment_factory import make_pedestrian_env
from robot_sf.gym_env.reward import stationary_collision_ped_reward
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.sensor.range_sensor import LidarScannerSettings
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS
from robot_sf.sim.sim_config import SimulationSettings

logger = loguru.logger


class LegacyRun023ObsAdapter:
    """Wrap a PPO model so run_023 receives its legacy flattened observation format."""

    def __init__(self, model: PPO):
        """Store the wrapped model and expose action-space compatibility hooks."""
        self._model = model
        self.action_space = getattr(model, "action_space", None)

    def set_action_space(self, action_space) -> None:
        """Allow env-side action-space synchronization."""
        self.action_space = action_space
        if hasattr(self._model, "set_action_space"):
            self._model.set_action_space(action_space)

    def predict(self, obs, deterministic: bool = True):
        """Adapt dict observations to the run_023 flattened format before inference."""
        adapted_obs = obs
        if isinstance(obs, dict):
            drive_state = np.asarray(obs[OBS_DRIVE_STATE])
            ray_state = np.asarray(obs[OBS_RAYS])

            drive_state = drive_state[:, :-1]
            drive_state[:, 2] *= 10

            drive_state = np.squeeze(drive_state).reshape(-1)
            ray_state = np.squeeze(ray_state).reshape(-1)
            adapted_obs = np.concatenate((ray_state, drive_state), axis=0)

        return self._model.predict(adapted_obs, deterministic=deterministic)


def make_env(svg_map_path: str):
    """Create pedestrian simulation env configured for the differential-drive robot model."""
    map_definition = convert_map(svg_map_path)
    robot_model = LegacyRun023ObsAdapter(PPO.load("./model/run_023", env=None))
    ego_ped_lidar = LidarScannerSettings.ego_pedestrian_lidar()

    config = PedestrianSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
        sim_config=SimulationSettings(
            stack_steps=1,
            difficulty=0,
            ped_density_by_difficulty=[0.02],
            debug_without_robot_movement=False,
            peds_reset_follow_route_at_start=True,
        ),
        robot_config=DifferentialDriveSettings(radius=1.0, max_angular_speed=0.5),
        spawn_near_robot=False,
        ego_ped_lidar_config=ego_ped_lidar,
    )
    env = make_pedestrian_env(
        config=config,
        robot_model=robot_model,
        debug=True,
        recording_enabled=False,
        reward_func=stationary_collision_ped_reward,
    )

    return env


def get_latest_ped_model_file() -> Path:
    """Return the most recent pedestrian policy checkpoint under model_ped/."""
    model_dir = Path("model_ped")
    if not model_dir.exists():
        raise FileNotFoundError("Directory not found: model_ped")

    candidates = sorted(model_dir.glob("*.zip"), key=os.path.getctime)
    if not candidates:
        raise FileNotFoundError("No pedestrian model checkpoints found in model_ped")

    return candidates[-1]


def extract_info(meta: dict, reward: float) -> str:
    """Extract and format episode statistics from metadata."""
    meta = meta["meta"]
    eps_num = meta["episode"]
    steps = meta["step_of_episode"]
    done = [key for key, value in meta.items() if value is True]
    dis = meta["distance_to_robot"]
    angle = meta["collision_impact_angle_deg"]
    zone = meta["robot_ped_collision_zone"]
    speed = meta["ego_ped_speed"]
    return (
        f"Episode: {eps_num}, Steps: {steps}, Done: {done}, Reward: {reward}, "
        f"Distance: {dis}, Angle: {angle}, Zone: {zone}, speed: {speed}"
    )


def run():
    """Run the differential-drive pedestrian policy debugger."""
    env = make_env("maps/svg_maps/masterthesis/corner.svg")

    filename = get_latest_ped_model_file()
    filename = "./model_ped/ppo_2026-03-29_23-19-05.zip"
    filename = "./model_ped/ppo_2026-03-29_22-34-26.zip"
    logger.info(f"Loading pedestrian model from {filename}")
    model = PPO.load(str(filename), env=env)

    obs = env.reset()
    ep_rewards = 0.0

    for _ in range(10000):
        if isinstance(obs, tuple):
            obs = obs[0]
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, meta = env.step(action)
        robot_speed = env.simulator.robots[0].current_speed
        ego_speed = env.simulator.ego_ped.current_speed
        ep_rewards += float(reward)
        env.render()

        if done:
            logger.info(extract_info(meta, ep_rewards))
            logger.info(f"Robot speed {robot_speed}")
            logger.info(f"Ego speed {ego_speed}")
            ep_rewards = 0.0
            obs = env.reset()
            env.render()

    env.exit()


if __name__ == "__main__":
    run()
