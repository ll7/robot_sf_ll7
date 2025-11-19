"""Behavioural cloning pre-training from expert trajectory datasets.

Trains a PPO policy using imitation learning (BC) on recorded expert trajectories,
producing a warm-start checkpoint for subsequent PPO fine-tuning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from loguru import logger

try:
    from gymnasium.spaces.utils import flatten as flatten_space
    from imitation.data import types as im_types
    from stable_baselines3 import PPO
except ImportError as exc:
    raise RuntimeError(
        "This script requires 'gymnasium', 'imitation', and 'stable-baselines3' packages."
    ) from exc

from robot_sf import common
from robot_sf.benchmark.imitation_manifest import write_training_run_manifest
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.training.imitation_config import BCPretrainingConfig
from robot_sf.training.observation_wrappers import maybe_flatten_env_observations

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymnasium.spaces import Space


def _load_trajectory_dataset(dataset_path: Path) -> dict[str, Any]:
    """Load NPZ trajectory dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = np.load(str(dataset_path), allow_pickle=True)
    return {
        "positions": data["positions"],
        "actions": data["actions"],
        "observations": data["observations"],
        "episode_count": int(data.get("episode_count", len(data["positions"]))),
    }


def _flatten_single_observation(obs: Any, observation_space: Space | None) -> np.ndarray:
    """Flatten an observation dict/object into a 1-D numpy array."""

    if observation_space is not None:
        try:
            return np.asarray(flatten_space(observation_space, obs), dtype=np.float32)
        except (AssertionError, ValueError, TypeError) as exc:
            logger.warning(
                "Failed to flatten observation via observation_space: {}. Falling back to simplified conversion.",
                exc,
            )

    if isinstance(obs, dict):
        flattened_parts: list[np.ndarray] = []
        for value in obs.values():
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            flattened_parts.append(arr)
        if not flattened_parts:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(flattened_parts).astype(np.float32)

    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _convert_to_transitions(
    dataset: dict[str, Any],
    observation_space: Space | None = None,
) -> list[im_types.Trajectory]:
    """Convert NPZ arrays to imitation-compatible trajectories."""
    trajectories = []

    for ep_idx in range(dataset["episode_count"]):
        obs_ep = dataset["observations"][ep_idx]
        acts_ep = dataset["actions"][ep_idx]

        obs_arrays = np.stack([_flatten_single_observation(o, observation_space) for o in obs_ep])

        acts_arrays = np.asarray(acts_ep, dtype=np.float32)
        obs_arrays = _ensure_observation_history(obs_arrays, acts_arrays, ep_idx)

        # Create trajectory for this episode
        traj = im_types.Trajectory(
            obs=obs_arrays,
            acts=acts_arrays,
            infos=None,
            terminal=True,
        )
        trajectories.append(traj)

    return trajectories


def _ensure_observation_history(
    obs_arrays: np.ndarray,
    acts_arrays: np.ndarray,
    episode_index: int,
) -> np.ndarray:
    """Ensure len(obs) == len(actions) + 1 as required by imitation Trajectory."""

    obs_steps = obs_arrays.shape[0]
    act_steps = acts_arrays.shape[0]

    if obs_steps == act_steps + 1:
        return obs_arrays

    if obs_steps == act_steps:
        logger.warning(
            "Episode {} missing terminal observation; duplicating last observation to satisfy len(obs) == len(actions) + 1.",
            episode_index,
        )
        padding = obs_arrays[-1][None, :]
        return np.vstack([obs_arrays, padding])

    raise ValueError(
        f"Episode {episode_index} has {obs_steps} observations and {act_steps} actions; "
        "expected len(obs) == len(actions) + 1."
    )


def _create_bc_trainer(
    env: Any,
    trajectories: list[im_types.Trajectory],
    config: BCPretrainingConfig,
) -> tuple[Any, PPO]:
    """Initialize BC trainer alongside the PPO container holding the policy."""
    from imitation.algorithms import bc

    policy_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        verbose=1,
    )

    trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=trajectories,
        policy=policy_model.policy,
        batch_size=config.batch_size,
        rng=np.random.default_rng(int(config.random_seeds[0])),
    )

    return trainer, policy_model


def run_bc_pretraining(
    config: BCPretrainingConfig,
    *,
    dry_run: bool = False,
) -> Path:
    """Execute behavioural cloning pre-training and save checkpoint."""
    logger.info("Starting BC pre-training run_id={}", config.run_id)

    # Load dataset
    dataset_path = common.get_trajectory_dataset_path(config.dataset_id)
    logger.info("Loading dataset from {}", dataset_path)
    dataset = _load_trajectory_dataset(dataset_path)

    # Create environment for BC
    env = make_robot_env(config=RobotSimulationConfig())
    raw_observation_space = env.observation_space
    env = maybe_flatten_env_observations(env, context="BC pre-training")

    # Prepare training
    if not dry_run:
        trajectories = _convert_to_transitions(dataset, raw_observation_space)
        trainer, policy_model = _create_bc_trainer(env, trajectories, config)

        # Train
        logger.info("Training BC for {} epochs", config.bc_epochs)
        trainer.train(n_epochs=config.bc_epochs)

        # Save policy
        policy_path = common.get_expert_policy_dir() / f"{config.policy_output_id}.zip"
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_model.save(str(policy_path))
    else:
        # Dry run - create placeholder
        policy_path = common.get_expert_policy_dir() / f"{config.policy_output_id}.zip"
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text("dry-run-bc-checkpoint", encoding="utf-8")

    env.close()

    logger.success("BC pre-training complete, policy saved to {}", policy_path)

    # Write training run manifest
    training_artifact = common.TrainingRunArtifact(
        run_id=config.run_id,
        run_type=common.TrainingRunType.BEHAVIOURAL_CLONING,
        input_artefacts=(config.dataset_id,),
        seeds=config.random_seeds,
        metrics={},  # BC metrics would be added here in production
        episode_log_path=Path(""),  # BC doesn't produce episode logs
        wall_clock_hours=0.0,  # Would track actual time in production
        status=common.TrainingRunStatus.COMPLETED,
        notes=[f"BC pre-training from dataset {config.dataset_id}"],
    )

    manifest_path = write_training_run_manifest(training_artifact)
    logger.info("Training run manifest written to {}", manifest_path)

    return policy_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for BC pre-training script."""
    parser = argparse.ArgumentParser(
        description="Pre-train PPO policy via behavioural cloning on expert trajectories."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to BC pre-training configuration YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate placeholder artifacts without actual training",
    )
    return parser


def load_bc_config(config_path: Path) -> BCPretrainingConfig:
    """Load and parse BC pre-training configuration from YAML."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    return BCPretrainingConfig.from_raw(
        run_id=raw["run_id"],
        dataset_id=raw["dataset_id"],
        policy_output_id=raw["policy_output_id"],
        bc_epochs=raw.get("bc_epochs", 10),
        batch_size=raw.get("batch_size", 32),
        learning_rate=raw.get("learning_rate", 0.0003),
        random_seeds=tuple(raw.get("random_seeds", [42])),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for BC pre-training script."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = load_bc_config(args.config)
    common.set_global_seed(int(config.random_seeds[0]))

    run_bc_pretraining(config, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
