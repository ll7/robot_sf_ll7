"""Debug runner for pedestrian policy models in the SocialForce simulator."""

from pathlib import Path

import loguru
from stable_baselines3 import PPO

from robot_sf.gym_env.environment_factory import make_pedestrian_env
from robot_sf.gym_env.reward import stationary_collision_ped_reward
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sensor.range_sensor import LidarScannerSettings
from robot_sf.sim.sim_config import SimulationSettings

logger = loguru.logger


def _extract_linear_speed(speed_like) -> float:
    """Return the translational component from a scalar/sequence speed value."""
    if isinstance(speed_like, (tuple, list)):
        return float(speed_like[0]) if speed_like else float("nan")
    return float(speed_like)


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


def get_file():
    """Get the latest model file."""
    model_dir = Path("model_ped")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    candidates = sorted(model_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No model checkpoints found in: {model_dir}")

    return candidates[-1]


def run():
    """Run the pedestrian policy debugger.

    Loads the latest pedestrian model from the model_ped directory,
    creates a pedestrian simulation environment, and runs the model
    for 10000 steps while collecting episode statistics.
    """
    # env = make_env("maps/svg_maps/debug_06.svg")
    env = make_env("maps/svg_maps/masterthesis/intersection.svg")
    filename = get_file()
    logger.info(f"Loading pedestrian model from {filename}")

    model = PPO.load(filename, env=env)

    obs = env.reset()
    ep_rewards = 0

    for _ in range(10000):
        if isinstance(obs, tuple):  # Check env.reset()
            obs = obs[0]
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, meta = env.step(action)
        done = bool(terminated or truncated)
        robot_speed = _extract_linear_speed(env.simulator.robots[0].current_speed)
        ego_speed = _extract_linear_speed(env.simulator.ego_ped.current_speed)
        ep_rewards += reward
        env.render()

        if done:
            logger.info(extract_info(meta, ep_rewards))
            logger.info(f"Robot speed {robot_speed}")
            logger.info(f"Ego speed {ego_speed}")
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
    angle = meta["collision_impact_angle_deg"]
    zone = meta["robot_ped_collision_zone"]
    speed = meta["ego_ped_speed"]
    return (
        f"Episode: {eps_num}, Steps: {steps}, Done: {done}, Reward: {reward}, "
        f"Distance: {dis}, Angle: {angle}, Zone: {zone}, speed: {speed}"
    )


if __name__ == "__main__":
    run()
