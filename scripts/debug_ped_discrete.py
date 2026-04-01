"""Debug runner for pedestrian policy with discrete hardcoded actions.

This script simulates a pedestrian using a simple hardcoded heuristic policy
with discrete actions (acceleration, left turn, right turn, no-op).
"""

from typing import Any

import loguru
from gymnasium import Env
from stable_baselines3 import PPO

from robot_sf.gym_env.environment_factory import make_pedestrian_env
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.ped_ego.unicycle_drive import UnicycleAction
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sensor.range_sensor import LidarScannerSettings
from robot_sf.sim.sim_config import SimulationSettings
from scripts.debug_ped_policy import extract_info

logger = loguru.logger

# Discrete actions for unicycle pedestrian
ACCELERATION: UnicycleAction = (1.0, 0.0)
LEFT_TURN: UnicycleAction = (0.5, 0.8)
RIGHT_TURN: UnicycleAction = (0.5, -0.8)
NO_OP: UnicycleAction = (0.0, 0.0)


def select_action(obs: dict) -> UnicycleAction:
    """Select a discrete action based on pedestrian observations.

    Uses a simple heuristic policy:
    - Accelerate if below target velocity
    - Turn left/right if target is off to the side
    - Otherwise maintain current motion

    Args:
        obs: Observation dictionary from environment containing drive_state.

    Returns:
        UnicycleAction: Tuple of (forward_velocity, angular_velocity).
    """
    target_obs = obs.get("target_sensor")
    if target_obs is None or len(target_obs) < 2:
        return NO_OP

    target_dist, target_angle = target_obs[0], target_obs[1]
    if target_dist <= 1.0:
        return NO_OP
    if abs(target_angle) > 0.3:  # ~17 degrees
        return LEFT_TURN if target_angle > 0 else RIGHT_TURN
    return ACCELERATION


def make_env(svg_map_path: str = "maps/svg_maps/debug_06.svg") -> Env[Any, Any]:
    """Create a pedestrian simulation environment for debugging.

    Parameters
    ----------
    svg_map_path : str, optional
        Path to the SVG map file, by default "maps/svg_maps/debug_06.svg"

    Returns
    -------
    gymnasium.Env
        Pedestrian simulation environment with loaded robot model and ego pedestrian.
    """
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 2

    map_definition = convert_map(svg_map_path)
    robot_model = PPO.load("./model/run_043", env=None)

    # Configure ego pedestrian lidar with longer range and 120 degree view
    ego_ped_lidar = LidarScannerSettings.ego_pedestrian_lidar()

    config = PedestrianSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
        sim_config=SimulationSettings(
            difficulty=difficulty,
            ped_density_by_difficulty=ped_densities,
            debug_without_robot_movement=False,
            peds_reset_follow_route_at_start=True,
        ),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
        spawn_near_robot=True,
        ego_ped_lidar_config=ego_ped_lidar,
    )

    env = make_pedestrian_env(
        config=config,
        robot_model=robot_model,
        debug=True,
        recording_enabled=False,
    )

    return env


def run() -> None:
    """Run the discrete action pedestrian debugger.

    Creates a pedestrian environment and runs the discrete heuristic policy
    for 10000 steps, collecting and logging episode statistics.
    """
    env = make_env("maps/svg_maps/debug_06.svg")
    try:
        obs = env.reset()
        ep_rewards = 0

        for _ in range(10000):
            if isinstance(obs, tuple):  # Handle env.reset() returning tuple
                obs = obs[0]

            # Select action using discrete heuristic policy
            action = select_action(obs)
            obs, reward, terminated, truncated, meta = env.step(action)
            done = bool(terminated or truncated)
            ep_rewards += reward
            env.render()

            if done:
                logger.info(extract_info(meta, ep_rewards))
                ep_rewards = 0
                obs = env.reset()
                env.render()
    finally:
        env.exit()


if __name__ == "__main__":
    run()
