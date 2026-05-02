"""Demo for running a trained adversarial pedestrian policy.

Usage:
    uv run python examples/advanced/32_demo_adversarial_pedestrian.py
    uv run python examples/advanced/32_demo_adversarial_pedestrian.py \
        --ped-model model_ped/ppo_intersection.zip \
        --map maps/svg_maps/masterthesis/intersection.svg

Prerequisites:
    - maps/svg_maps/masterthesis/intersection.svg (default)
    - model/run_043.zip (default robot policy)
    - model_ped/ppo_intersection.zip (default pedestrian policy)

Expected Output:
    - Interactive pygame debug rollout and per-episode summary logs.

Limitations:
    - Requires an interactive display for rendering.

References:
    - docs/dev_guide.md#advanced-feature-demos
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from loguru import logger

from robot_sf.benchmark.helper_catalog import load_trained_policy
from robot_sf.gym_env._stub_robot_model import StubRobotModel
from robot_sf.gym_env.environment_factory import make_pedestrian_env
from robot_sf.gym_env.reward import stationary_collision_ped_reward
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sensor.range_sensor import LidarScannerSettings
from robot_sf.sim.sim_config import SimulationSettings


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the modern pedestrian debug demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        default="maps/svg_maps/masterthesis/intersection.svg",
        help="Path to SVG map used by the environment.",
    )
    parser.add_argument(
        "--robot-model",
        default="model/run_043.zip",
        help="Path to the trained robot policy (falls back to a stub if missing).",
    )
    parser.add_argument(
        "--ped-model",
        default="model_ped/ppo_intersection.zip",
        help="Path to the trained pedestrian PPO checkpoint.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Maximum number of rollout steps to execute.",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=0,
        help="Difficulty index into ped_density_by_difficulty.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions (recommended for debugging).",
    )
    return parser.parse_args()


def _load_robot_model_or_stub(robot_model_path: str) -> Any:
    """Load robot model checkpoint or fall back to a stub model when unavailable."""
    if Path(robot_model_path).exists():
        return load_trained_policy(robot_model_path)

    logger.warning(
        "Robot model not found at {}. Using StubRobotModel for debug run.",
        robot_model_path,
    )
    return StubRobotModel()


def _make_env(
    *,
    svg_map_path: str,
    robot_model_path: str,
    difficulty: int,
):
    """Build a pedestrian debug environment using unified factory APIs."""
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    map_definition = convert_map(svg_map_path)
    robot_model = _load_robot_model_or_stub(robot_model_path)

    config = PedestrianSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"debug_map": map_definition}),
        sim_config=SimulationSettings(
            difficulty=difficulty,
            ped_density_by_difficulty=ped_densities,
            debug_without_robot_movement=False,
            peds_reset_follow_route_at_start=True,
        ),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
        spawn_near_robot=False,
        ego_ped_lidar_config=LidarScannerSettings.ego_pedestrian_lidar(),
    )

    return make_pedestrian_env(
        config=config,
        robot_model=robot_model,
        debug=True,
        recording_enabled=False,
        reward_func=stationary_collision_ped_reward,
    )


def _extract_episode_info(info: dict[str, Any], episode_reward: float) -> str:
    """Format one concise episode summary line from environment info payload."""
    meta = info.get("meta", info)
    done_flags = [key for key, value in meta.items() if value is True]

    return (
        f"episode={meta.get('episode', '?')} "
        f"steps={meta.get('step_of_episode', '?')} "
        f"done={done_flags} "
        f"reward={episode_reward:.3f} "
        f"distance={meta.get('distance_to_robot', float('nan')):.3f} "
        f"impact_angle_deg={meta.get('collision_impact_angle_deg', 0.0):.3f} "
        f"collision_zone={meta.get('robot_ped_collision_zone', 'none')} "
        f"ego_speed={meta.get('ego_ped_speed', 0.0):.3f}"
    )


def run_debug_rollout(args: argparse.Namespace) -> None:
    """Execute a rendered pedestrian-policy rollout with episode logging."""
    if not Path(args.ped_model).exists():
        raise FileNotFoundError(
            "Pedestrian model not found at "
            f"{args.ped_model}. Provide --ped-model with a valid PPO checkpoint."
        )

    env = _make_env(
        svg_map_path=args.map,
        robot_model_path=args.robot_model,
        difficulty=args.difficulty,
    )
    logger.info("Loading pedestrian model from {}", args.ped_model)
    model = load_trained_policy(args.ped_model)

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    episode_reward = 0.0

    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=bool(args.deterministic))
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)
        env.render()

        done = bool(terminated) or bool(truncated)
        if done:
            logger.info(_extract_episode_info(info, episode_reward))
            episode_reward = 0.0
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            env.render()

    env.exit()


def main() -> None:
    """Run the modern pedestrian policy debug demo."""
    args = _parse_args()
    run_debug_rollout(args)


if __name__ == "__main__":
    main()
