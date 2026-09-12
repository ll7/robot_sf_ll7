"""Replay offensive PPO policy in the robot environment.

Usage:
    uv run python examples/advanced/10_offensive_policy.py
    uv run python examples/advanced/10_offensive_policy.py --check --format json

Prerequisites:
    - output/model_cache/legacy_ppo_run_043/legacy_ppo_run_043.zip

Expected Output:
    - Pygame window showing the offensive policy navigating the environment.

Limitations:
    - Uses direct `RobotEnv` instantiation and requires display access.

References:
    - docs/dev_guide.md#baseline-policies
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from robot_sf.examples.prerequisites import (
    add_prerequisite_check_arguments,
    run_prerequisite_check,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for the offensive-policy demo and its check-only mode."""

    parser = argparse.ArgumentParser(description="Replay the offensive PPO policy.")
    add_prerequisite_check_arguments(parser)
    return parser


def demo_offensive_policy() -> None:
    """Run the offensive policy rollout with an interactive renderer."""

    from robot_sf.benchmark.helper_catalog import load_trained_policy
    from robot_sf.gym_env.env_config import EnvSettings
    from robot_sf.gym_env.robot_env import RobotEnv
    from robot_sf.robot.bicycle_drive import BicycleDriveSettings
    from robot_sf.sim.sim_config import SimulationSettings

    env_config = EnvSettings(
        sim_config=SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.02]),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
    )
    env = RobotEnv(env_config, debug=True, recording_enabled=False)
    from robot_sf.models.registry import resolve_model_path

    model_path = resolve_model_path("legacy_ppo_run_043", allow_download=True)
    model = load_trained_policy(str(model_path))

    obs, _ = env.reset()
    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        env.render()

        if done:
            obs, _ = env.reset()
            env.render()
    env.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run check-only mode or the rendered demo."""

    args = build_parser().parse_args(argv)
    if args.check:
        return run_prerequisite_check(__file__, output_format=args.format)
    demo_offensive_policy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
