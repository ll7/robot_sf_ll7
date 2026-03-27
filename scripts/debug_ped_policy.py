"""Debug runner for pedestrian policy models in the SocialForce simulator."""

import os
from pathlib import Path

import loguru
from stable_baselines3 import PPO

from robot_sf.gym_env.environment_factory import make_pedestrian_env
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sensor.range_sensor import LidarScannerSettings
from robot_sf.sim.sim_config import SimulationSettings

logger = loguru.logger


def make_env(svg_map_path):
    """Create a pedestrian simulation environment for debugging.

    Parameters
    ----------
    svg_map_path : str
        Path to the SVG map file.

    Returns
    -------
    gym.Env
        Pedestrian simulation environment with loaded robot model.
    """
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 0

    map_definition = convert_map(svg_map_path)
    robot_model = PPO.load("./model/run_043", env=None)  # 043, 023
    # robot_model = PPO.load("./model/ppo_model_retrained_10m_2024-09-17.zip", env=None)

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


def get_file():
    """Get the latest model file."""

    filename = max(
        os.listdir("model/pedestrian"),
        key=lambda x: os.path.getctime(os.path.join("model/pedestrian", x)),
    )
    return Path("model/pedestrian", filename)


def run():
    """Run the pedestrian policy debugger.

    Loads the latest pedestrian model from the model_ped directory,
    creates a pedestrian simulation environment, and runs the model
    for 10000 steps while collecting episode statistics.
    """
    # env = make_env("maps/svg_maps/debug_06.svg")
    env = make_env("maps/svg_maps/masterthesis/headon.svg")
    filename = get_file()
    # filename = "./model_ped/ppo_2024-09-06_23-52-17.zip"
    logger.info(f"Loading pedestrian model from {filename}")

    model = PPO.load(filename, env=env)

    obs = env.reset()
    ep_rewards = 0

    for _ in range(10000):
        if isinstance(obs, tuple):  # Check env.reset()
            obs = obs[0]
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, meta = env.step(action)
        ep_rewards += reward
        env.render()

        if done:
            logger.info(extract_info(meta, ep_rewards))
            ep_rewards = 0
            obs = env.reset()
            env.render()
    env.exit()


def extract_info(meta: dict, reward: float) -> str:
    """Extract and format episode statistics from metadata.

    Parameters
    ----------
    meta : dict
        Metadata dictionary containing episode information.
    reward : float
        Cumulative reward for the episode.

    Returns
    -------
    str
        Formatted string containing episode number, steps, done conditions,
        reward, and distance to robot.
    """
    meta = meta["meta"]
    eps_num = meta["episode"]
    steps = meta["step_of_episode"]
    done = [key for key, value in meta.items() if value is True]
    dis = meta["distance_to_robot"]
    return f"Episode: {eps_num}, Steps: {steps}, Done: {done}, Reward: {reward}, Distance: {dis}"


if __name__ == "__main__":
    run()
