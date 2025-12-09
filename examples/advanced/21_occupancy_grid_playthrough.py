"""
Occupancy grid visualization with a pretrained defensive policy.

Runs a short simulation using the same policy as `09_defensive_policy.py` and
renders the occupancy grid overlay in pygame.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from loguru import logger

from robot_sf.benchmark.helper_catalog import load_trained_policy
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.occupancy_grid import GridConfig
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS
from robot_sf.sim.sim_config import SimulationSettings

# Allow headless runs if the user sets SDL_VIDEODRIVER=dummy; otherwise pygame will open a window.
os.environ.setdefault("MPLBACKEND", "Agg")


def _obs_adapter(orig_obs: dict) -> np.ndarray:
    """Adapt dict observations to the flat array expected by the pretrained policy."""
    drive_state = orig_obs[OBS_DRIVE_STATE]
    ray_state = orig_obs[OBS_RAYS]
    drive_state = drive_state[:, :-1]
    drive_state[:, 2] *= 10
    drive_state = np.ravel(drive_state)
    ray_state = np.ravel(ray_state)
    return np.concatenate((ray_state, drive_state), axis=0)


def run_playthrough(steps: int = 600) -> None:
    """Run a policy-controlled playthrough while rendering the occupancy grid."""
    model_path = Path(__file__).resolve().parents[2] / "model" / "run_023.zip"
    model = load_trained_policy(str(model_path))

    config = RobotSimulationConfig(
        use_occupancy_grid=True,
        include_grid_in_observation=False,  # keep lidar-only observations for the policy
        show_occupancy_grid=True,
        grid_visualization_alpha=0.3,
        sim_config=SimulationSettings(
            stack_steps=1, difficulty=0, ped_density_by_difficulty=[0.06]
        ),
        grid_config=GridConfig(
            width=20.0,
            height=20.0,
            resolution=0.5,
            use_ego_frame=True,  # rotate grid axes with robot orientation
            center_on_robot=True,  # translate grid with robot without rotating it
        ),
    )
    env = make_robot_env(config=config, debug=True)
    if env.sim_ui:
        env.sim_ui.show_lidar = False
        env.sim_ui.grid_alpha = 0.7  # increase overlay clarity

    obs, _ = env.reset(seed=123)
    for _ in range(steps):
        policy_obs = _obs_adapter(obs)
        action, _ = model.predict(policy_obs, deterministic=True)
        obs, _reward, done, truncated, _info = env.step(action)
        env.render()
        if done or truncated:
            obs, _ = env.reset()

    env.close()
    logger.info("Occupancy grid playthrough complete.")


if __name__ == "__main__":
    run_playthrough()
