"""Trajectory collection runner for PPO imitation workflows.

Records episodes using an approved expert policy, dumps observations/actions/positions
into an NPZ dataset, validates the artifact via ``TrajectoryDatasetValidator``, and
persists a manifest using the imitation helpers for downstream workflows.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from loguru import logger

try:  # pragma: no cover - module-level import used in production
    from stable_baselines3 import PPO
except ImportError as exc:
    raise RuntimeError("Stable-Baselines3 is required to load expert policies.") from exc

from robot_sf import common
from robot_sf.benchmark.imitation_manifest import write_trajectory_dataset_manifest
from robot_sf.benchmark.validation.trajectory_dataset import TrajectoryDatasetValidator
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.training.imitation_config import TrajectoryCollectionConfig

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
else:
    Mapping = Sequence = Any


def _resolve_scenario_config(
    scenario_arg: Path | None,
    training_config: Path | None,
) -> Path:
    if scenario_arg is not None:
        return scenario_arg.resolve()
    if training_config is None:
        return Path("configs/scenarios/classic_interactions.yaml").resolve()
    raw = yaml.safe_load(training_config.read_text(encoding="utf-8"))
    scenario_raw = raw.get("scenario_config")
    if scenario_raw is None:
        raise ValueError(
            "Training config must define scenario_config when scenario path is omitted."
        )
    scenario_path = Path(scenario_raw)
    if not scenario_path.is_absolute():
        scenario_path = (training_config.parent / scenario_path).resolve()
    return scenario_path


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - best effort
        return None


def _command_line() -> str:
    return " ".join(shlex.quote(arg) for arg in sys.argv)


def _zero_action(space: Any) -> np.ndarray:
    shape = getattr(space, "shape", ())
    dtype = float
    return np.zeros(shape, dtype=dtype)


def _record_episode(
    env: Any,
    *,
    policy: PPO | None,
    dry_run: bool,
) -> tuple[list[Any], list[Any], list[Any]]:
    obs, _ = env.reset()
    done = False
    positions: list[Any] = []
    actions: list[Any] = []
    observations: list[Any] = []
    steps = 0
    max_steps = env.state.max_sim_steps

    while not done and steps < max_steps:
        if dry_run or policy is None:
            action = _zero_action(env.action_space)
        else:
            action, _ = policy.predict(obs, deterministic=True)
        next_obs, _reward, terminated, truncated, _info = env.step(action)
        positions.append(tuple(env.state.nav.pos))
        actions.append(np.asarray(action, dtype=float))
        observations.append(next_obs)
        obs = next_obs
        done = bool(terminated or truncated)
        steps += 1

    return positions, actions, observations


def _record_dataset(
    config: TrajectoryCollectionConfig,
    *,
    policy: PPO | None,
    dry_run: bool,
    scenario_label: str,
) -> tuple[dict[str, list[np.ndarray]], dict[str, int]]:
    dataset: dict[str, list[np.ndarray]] = {
        "positions": [],
        "actions": [],
        "observations": [],
    }
    coverage: dict[str, int] = {}

    for episode in range(config.episodes):
        seed = int(config.random_seeds[episode % len(config.random_seeds)])
        env = make_robot_env(config=RobotSimulationConfig(), seed=seed)
        try:
            positions, actions, observations = _record_episode(env, policy=policy, dry_run=dry_run)
        finally:
            env.close()
        dataset["positions"].append(np.asarray(positions, dtype=object))
        dataset["actions"].append(np.asarray(actions, dtype=object))
        dataset["observations"].append(np.asarray(observations, dtype=object))
        coverage[scenario_label] = coverage.get(scenario_label, 0) + 1
    return dataset, coverage


def _write_dataset(
    path: Path,
    arrays: dict[str, list[np.ndarray]],
    episode_count: int,
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        positions=np.array(arrays["positions"], dtype=object),
        actions=np.array(arrays["actions"], dtype=object),
        observations=np.array(arrays["observations"], dtype=object),
        episode_count=np.array(episode_count),
        metadata=metadata,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect expert trajectory datasets using an expert PPO policy."
    )
    parser.add_argument("--dataset-id", required=True, help="Unique identifier for the dataset.")
    parser.add_argument("--policy-id", required=True, help="Approved expert policy identifier.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of episodes to record (per choreography).",
    )
    parser.add_argument(
        "--scenario-config",
        type=Path,
        default=None,
        help="Path to the scenario YAML file driving the episodes.",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=None,
        help="Expert training config to derive scenario metadata from.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional random seeds that rotate per episode.",
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        default=None,
        help="Label used for scenario coverage metadata (defaults to scenario file stem).",
    )
    parser.add_argument(
        "--output-format",
        choices=("npz",),
        default="npz",
        help="Storage format for the dataset (currently only npz).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate placeholder data without loading a policy for testing.",
    )
    parser.add_argument(
        "--override-seed",
        type=int,
        nargs=1,
        help="Deprecated alias for --seeds (kept for backwards compatibility).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    scenario_path = _resolve_scenario_config(args.scenario_config, args.training_config)
    scenario_label = args.scenario_id or scenario_path.stem
    seed_values = tuple(args.seeds) if args.seeds else tuple(range(max(1, args.episodes)))
    collection_config = TrajectoryCollectionConfig.from_raw(
        dataset_id=str(args.dataset_id),
        source_policy_id=str(args.policy_id),
        episodes=int(args.episodes),
        scenario_config=scenario_path,
        scenario_overrides=(),
        output_format=args.output_format,
        random_seeds=seed_values,
    )

    common.set_global_seed(int(collection_config.random_seeds[0]))

    policy = None
    if not args.dry_run:
        checkpoint = common.get_expert_policy_dir() / f"{collection_config.source_policy_id}.zip"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Expert policy checkpoint not found: {checkpoint}")
        policy = PPO.load(str(checkpoint))

    arrays, coverage = _record_dataset(
        collection_config,
        policy=policy,
        dry_run=args.dry_run,
        scenario_label=scenario_label,
    )

    dataset_path = common.get_trajectory_dataset_path(collection_config.dataset_id)
    timestamp = datetime.now(UTC)
    metadata = {
        "dataset_id": collection_config.dataset_id,
        "source_policy_id": collection_config.source_policy_id,
        "scenario_config": str(collection_config.scenario_config),
        "scenario_label": scenario_label,
        "scenario_overrides": list(collection_config.scenario_overrides),
        "random_seeds": list(collection_config.random_seeds),
        "coverage": coverage,
        "command": _command_line(),
        "git_commit": _git_commit_hash(),
        "dry_run": args.dry_run,
        "created_at": timestamp.isoformat(),
    }

    _write_dataset(dataset_path, arrays, len(arrays["positions"]), metadata)

    validator = TrajectoryDatasetValidator(dataset_path)
    validation_result = validator.validate(minimum_episodes=collection_config.episodes)

    dataset_artifact = common.TrajectoryDatasetArtifact(
        dataset_id=collection_config.dataset_id,
        source_policy_id=collection_config.source_policy_id,
        episode_count=validation_result.episode_count,
        storage_path=dataset_path,
        format=collection_config.output_format,
        scenario_coverage=validation_result.scenario_coverage,
        integrity_report=validation_result.integrity_report,
        metadata=metadata,
        quality_status=validation_result.quality_status,
        created_at=timestamp,
    )
    manifest_path = write_trajectory_dataset_manifest(dataset_artifact)

    logger.success(
        "Trajectory dataset stored={} manifest={} quality={} episodes={}",
        dataset_path,
        manifest_path,
        validation_result.quality_status.value,
        validation_result.episode_count,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
