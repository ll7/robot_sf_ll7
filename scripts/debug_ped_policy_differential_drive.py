"""Debug runner for pedestrian policy models with a differential-drive robot policy.

This script mirrors ``scripts/debug_ped_policy.py`` while applying the
run_023 differential-drive robot profile and legacy observation adapter logic
used in ``scripts/debug_ped_apf.py``.
"""

from pathlib import Path

import loguru
from stable_baselines3 import PPO

from robot_sf.gym_env.environment_factory import make_pedestrian_env
from robot_sf.gym_env.reward import stationary_collision_ped_reward
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.sensor.range_sensor import LidarScannerSettings
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.training.observation_wrappers import LegacyRun023ObsAdapter

logger = loguru.logger


def _extract_linear_speed(speed_like) -> float:
    """Return the translational component from a scalar/sequence speed value."""
    if isinstance(speed_like, (tuple, list)):
        return float(speed_like[0]) if speed_like else float("nan")
    return float(speed_like)


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
            ped_density_by_difficulty=[0.04],
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

    candidates = sorted(model_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime)
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
    env = make_env("maps/svg_maps/masterthesis/headon.svg")
    try:
        filename = get_latest_ped_model_file()
        logger.info(f"Loading pedestrian model from {filename}")
        model = PPO.load(str(filename), env=env)

        obs = env.reset()
        ep_rewards = 0.0

        for _ in range(10000):
            if isinstance(obs, tuple):
                obs = obs[0]
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, meta = env.step(action)
            done = bool(terminated or truncated)
            robot_speed = _extract_linear_speed(env.simulator.robots[0].current_speed)
            ego_speed = _extract_linear_speed(env.simulator.ego_ped.current_speed)
            ep_rewards += float(reward)
            env.render()

            if done:
                logger.info(extract_info(meta, ep_rewards))
                logger.info(f"Robot speed {robot_speed}")
                logger.info(f"Ego speed {ego_speed}")
                ep_rewards = 0.0
                obs = env.reset()
                env.render()
    finally:
        env.exit()


if __name__ == "__main__":
    run()
