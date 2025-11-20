"""Playback and inspection tool for expert trajectory datasets.

Loads an NPZ trajectory dataset and replays it visually using the simulation view
to allow manual inspection of expert behavior, coverage, and anomalies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from robot_sf.common.artifact_paths import get_trajectory_dataset_path
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.render.sim_view import SimulationView, VisualizableSimState

if TYPE_CHECKING:
    from collections.abc import Sequence


def _load_dataset(dataset_path: Path) -> dict[str, Any]:
    """Load NPZ dataset and return arrays."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = np.load(str(dataset_path), allow_pickle=True)
    return {
        "positions": data["positions"],
        "actions": data["actions"],
        "observations": data["observations"],
        "episode_count": int(data.get("episode_count", len(data["positions"]))),
        "metadata": data.get("metadata", {}).item() if "metadata" in data else {},
    }


def _replay_episode(
    env: Any,
    positions: np.ndarray,
    actions: np.ndarray,
    observations: np.ndarray,
    *,
    view: SimulationView | None = None,
) -> None:
    """Replay a single episode with optional visualization."""
    env.reset()

    for step_idx, (pos, action, obs) in enumerate(
        zip(positions, actions, observations, strict=False)
    ):
        # Update environment state with recorded position
        if hasattr(env.state, "nav") and hasattr(env.state.nav, "pos"):
            env.state.nav.pos = tuple(pos)

        # Render current state if view is available
        if view is not None:
            state = VisualizableSimState(
                timestep=step_idx,
                robot_action=action,
                robot_pose=(tuple(pos), 0.0),  # Assuming orientation 0 for simplicity
                pedestrian_positions=[],
                ray_vecs=None,
                ped_actions=None,
                ego_ped_pose=None,
            )
            view.render(state)

        # Step environment (mainly to advance time)
        env.step(action)


def _inspect_dataset(dataset: dict[str, Any]) -> None:
    """Print dataset statistics and metadata."""
    episode_count = dataset["episode_count"]
    metadata = dataset["metadata"]

    logger.info("Dataset Statistics:")
    logger.info("  Episodes: {}", episode_count)
    logger.info("  Format: {}", metadata.get("format", "unknown"))

    if "scenario_label" in metadata:
        logger.info("  Scenario: {}", metadata["scenario_label"])

    if "coverage" in metadata:
        logger.info("  Coverage: {}", metadata["coverage"])

    if "random_seeds" in metadata:
        logger.info("  Seeds: {}", metadata["random_seeds"][:5])
        if len(metadata["random_seeds"]) > 5:
            logger.info("    ... and {} more", len(metadata["random_seeds"]) - 5)

    if "created_at" in metadata:
        logger.info("  Created: {}", metadata["created_at"])

    # Episode length statistics
    positions = dataset["positions"]
    lengths = [len(ep) for ep in positions]
    logger.info(
        "  Episode lengths: min={} max={} mean={:.1f}", min(lengths), max(lengths), np.mean(lengths)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for playback tool."""
    parser = argparse.ArgumentParser(description="Playback and inspect expert trajectory datasets.")
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Dataset identifier (will resolve to canonical path)",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Explicit dataset path (overrides --dataset-id resolution)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to replay (default: 0)",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only print statistics without visual playback",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without visualization (useful for CI validation)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for trajectory playback tool."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Resolve dataset path
    if args.dataset_path is not None:
        dataset_path = args.dataset_path
    else:
        dataset_path = get_trajectory_dataset_path(args.dataset_id)

    # Load dataset
    logger.info("Loading dataset from {}", dataset_path)
    dataset = _load_dataset(dataset_path)

    # Always print inspection info
    _inspect_dataset(dataset)

    # Exit early if inspect-only mode
    if args.inspect_only:
        return 0

    # Validate episode index
    episode_count = dataset["episode_count"]
    if args.episode < 0 or args.episode >= episode_count:
        logger.error(
            "Invalid episode index {} (dataset has {} episodes)", args.episode, episode_count
        )
        return 1

    # Prepare for playback
    positions = dataset["positions"][args.episode]
    actions = dataset["actions"][args.episode]
    observations = dataset["observations"][args.episode]

    logger.info("Replaying episode {} ({} steps)", args.episode, len(positions))

    # Create environment for playback
    env = make_robot_env(config=RobotSimulationConfig())

    # Create visualization if not headless
    view = None
    if not args.headless:
        try:
            view = SimulationView(
                width=800,
                height=800,
                caption=f"Trajectory Playback - Episode {args.episode}",
            )
        except Exception as exc:  # pragma: no cover - display dependency
            logger.warning("Could not initialize display: {}", exc)
            logger.info("Continuing in headless mode")

    try:
        _replay_episode(env, positions, actions, observations, view=view)
        logger.success("Playback complete")
    finally:
        env.close()
        if view is not None:
            # View cleanup happens automatically
            pass

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
