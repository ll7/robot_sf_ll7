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
from collections.abc import Mapping as RuntimeMapping
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
from robot_sf.models import resolve_model_path
from robot_sf.training.imitation_config import TrajectoryCollectionConfig
from robot_sf.training.observation_wrappers import resolve_policy_obs_adapter
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
    select_scenario,
)
from scripts.training.imitation_env_contract import (
    observation_contract_from_space,
)
from scripts.training.train_ppo import _apply_env_overrides

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


def _resolve_scenario_config(
    scenario_arg: Path | None,
    training_config: Path | None,
) -> Path:
    """Resolve the scenario config used for trajectory collection.

    Args:
        scenario_arg: Explicit scenario config path from the CLI.
        training_config: Training config used as a fallback source for ``scenario_config``.

    Returns:
        Absolute path to the scenario config.
    """
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


def _load_env_contract(config_path: Path | None) -> tuple[dict[str, object], dict[str, object]]:
    """Load env overrides and factory kwargs from a training-style YAML file."""
    if config_path is None:
        return {}, {}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    env_overrides = raw.get("env_overrides") or {}
    env_factory_kwargs = raw.get("env_factory_kwargs") or {}
    if not isinstance(env_overrides, RuntimeMapping):
        raise TypeError(f"{config_path}: env_overrides must be a mapping")
    if not isinstance(env_factory_kwargs, RuntimeMapping):
        raise TypeError(f"{config_path}: env_factory_kwargs must be a mapping")
    return dict(env_overrides), dict(env_factory_kwargs)


def _resolve_expert_checkpoint(policy_id: str) -> Path:
    """Resolve an expert checkpoint from local output or the model registry."""
    checkpoint = common.get_expert_policy_dir() / f"{policy_id}.zip"
    if checkpoint.exists():
        return checkpoint
    try:
        return resolve_model_path(policy_id, allow_download=True)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise FileNotFoundError(
            f"Expert policy checkpoint not found at {checkpoint}, and registry lookup for "
            f"'{policy_id}' failed."
        ) from exc


def _git_commit_hash() -> str | None:
    """Return the current Git commit hash for manifest provenance when available.


    Returns:
        Commit hash string, or ``None`` when Git metadata cannot be read.
    """
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
    """Return the shell-escaped command line used for the current process.


    Returns:
        Command-line string suitable for manifest provenance.
    """
    return " ".join(shlex.quote(arg) for arg in sys.argv)


def _zero_action(space: Any) -> np.ndarray:
    """Create a zero-valued action compatible with an action space.

    Args:
        space: Gymnasium action space exposing a shape.

    Returns:
        Zero action array matching the space shape.
    """
    shape = getattr(space, "shape", ())
    dtype = float
    return np.zeros(shape, dtype=dtype)


def _record_episode(
    env: Any,
    *,
    policy: PPO | None,
    dry_run: bool,
    observation_adapter: Callable[[Mapping[str, Any]], Any] | None = None,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Record one episode using policy-compatible observations.

    Args:
        env: Environment instance to reset and step.
        policy: Optional PPO policy used for non-dry-run action selection.
        dry_run: Whether to use zero actions instead of policy inference.
        observation_adapter: Optional adapter that filters or reshapes raw env observations before
            policy inference and dataset storage.

    Returns:
        Episode positions, actions, and adapted observations.
    """
    raw_obs, _ = env.reset()
    obs = observation_adapter(raw_obs) if observation_adapter is not None else raw_obs
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
        raw_next_obs, _reward, terminated, truncated, _info = env.step(action)
        next_obs = (
            observation_adapter(raw_next_obs) if observation_adapter is not None else raw_next_obs
        )
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
    scenario: Mapping[str, Any],
    scenario_path: Path,
    observation_adapter: Callable[[Mapping[str, Any]], Any] | None = None,
) -> tuple[dict[str, list[np.ndarray]], dict[str, int]]:
    """Record all requested episodes for one scenario.

    Args:
        config: Trajectory collection settings, including dataset id, seeds, and episode count.
        policy: Optional PPO policy used for non-dry-run collection.
        dry_run: Whether to run deterministic zero-action placeholder collection.
        scenario_label: Scenario identifier used for coverage metadata.
        scenario: Scenario payload selected from the scenario manifest.
        scenario_path: Path to the scenario manifest used to build the env config.
        observation_adapter: Optional adapter matching raw env observations to the policy contract.

    Returns:
        Dataset arrays grouped by field and scenario coverage counts.
    """
    dataset: dict[str, list[np.ndarray]] = {
        "positions": [],
        "actions": [],
        "observations": [],
    }
    coverage: dict[str, int] = {}

    for episode in range(config.episodes):
        seed = int(config.random_seeds[episode % len(config.random_seeds)])
        env_config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
        _apply_env_overrides(env_config, config.env_overrides)
        env = make_robot_env(config=env_config, seed=seed, **config.env_factory_kwargs)
        try:
            positions, actions, observations = _record_episode(
                env,
                policy=policy,
                dry_run=dry_run,
                observation_adapter=observation_adapter,
            )
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
    """Write the collected trajectory arrays and metadata to an NPZ dataset.

    Args:
        path: Output NPZ path.
        arrays: Collected positions, actions, and observations by episode.
        episode_count: Number of recorded episodes.
        metadata: Dataset provenance and environment-contract metadata.
    """
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
    """Build the CLI argument parser for expert trajectory collection.


    Returns:
        Configured argument parser.
    """
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
        help="Training config to derive scenario metadata and env contract from.",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=None,
        help=(
            "Training-style YAML whose env_overrides/env_factory_kwargs should be used during "
            "collection. Defaults to --training-config when omitted."
        ),
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
        help="Scenario entry name to run (defaults to first entry in the file).",
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
    """Collect expert trajectories and write the dataset plus manifest.

    Args:
        argv: Optional CLI argument sequence for tests.

    Returns:
        Process exit code.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    scenario_path = _resolve_scenario_config(args.scenario_config, args.training_config)
    training_config_path = args.training_config.resolve() if args.training_config else None
    env_contract_path = args.env_config or args.training_config
    env_overrides, env_factory_kwargs = _load_env_contract(env_contract_path)
    seeds_arg: Sequence[int] | None = args.seeds
    if args.override_seed:
        override = tuple(int(value) for value in args.override_seed)
        if seeds_arg:
            logger.warning("Ignoring --override-seed because --seeds was provided explicitly")
        else:
            seeds_arg = override
    seed_values = tuple(seeds_arg) if seeds_arg else tuple(range(max(1, args.episodes)))

    scenario_definitions = load_scenarios(scenario_path)
    selected_scenario = select_scenario(scenario_definitions, args.scenario_id)
    scenario_label = (
        args.scenario_id
        or str(selected_scenario.get("name") or selected_scenario.get("scenario_id") or "")
        or scenario_path.stem
    )
    collection_config = TrajectoryCollectionConfig.from_raw(
        dataset_id=str(args.dataset_id),
        source_policy_id=str(args.policy_id),
        episodes=int(args.episodes),
        scenario_config=scenario_path,
        scenario_id=scenario_label,
        scenario_overrides=(),
        output_format=args.output_format,
        random_seeds=seed_values,
        env_overrides=env_overrides,
        env_factory_kwargs=env_factory_kwargs,
    )

    common.set_global_seed(int(collection_config.random_seeds[0]))

    policy = None
    observation_contract: dict[str, object] | None = None
    observation_adapter: Callable[[Mapping[str, Any]], Any] | None = None
    if not args.dry_run:
        checkpoint = _resolve_expert_checkpoint(collection_config.source_policy_id)
        policy = PPO.load(str(checkpoint))
        observation_contract = observation_contract_from_space(policy.observation_space)
        observation_adapter = resolve_policy_obs_adapter(policy)

    arrays, coverage = _record_dataset(
        collection_config,
        policy=policy,
        dry_run=args.dry_run,
        scenario_label=scenario_label,
        scenario=selected_scenario,
        scenario_path=scenario_path,
        observation_adapter=observation_adapter,
    )

    dataset_path = common.get_trajectory_dataset_path(collection_config.dataset_id)
    timestamp = datetime.now(UTC)
    metadata = {
        "dataset_id": collection_config.dataset_id,
        "source_policy_id": collection_config.source_policy_id,
        "scenario_config": str(collection_config.scenario_config),
        "scenario_label": scenario_label,
        "scenario_id": collection_config.scenario_id,
        "scenario_overrides": list(collection_config.scenario_overrides),
        "env_contract_config": str(env_contract_path) if env_contract_path is not None else None,
        "env_overrides": dict(collection_config.env_overrides),
        "env_factory_kwargs": dict(collection_config.env_factory_kwargs),
        "random_seeds": list(collection_config.random_seeds),
        "scenario_coverage": coverage,
        "training_config": str(training_config_path) if training_config_path else None,
        "observation_contract": observation_contract,
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
