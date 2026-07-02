"""RL trajectory dataset contract for offline reinforcement learning pipelines."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

RL_TRAJECTORY_DATASET_SCHEMA_VERSION = "RLTrajectoryDataset.v1"
RL_TRAJECTORY_EPISODE_SCHEMA_VERSION = "RLTrajectoryEpisode.v1"
RL_TRAJECTORY_DATASET_MANIFEST_SCHEMA_VERSION = "rl_trajectory_dataset_manifest.v1"
RL_TRAJECTORY_FORMAT = "jsonl"
REWARD_CONVENTION = "environment_step_reward"
RETURN_CONVENTION = "undiscounted_future_return_to_go"
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class RLTrajectoryEpisode:
    """One episode-major RL trajectory row loaded from JSONL."""

    dataset_id: str
    episode_id: str
    scenario_id: str
    seed: int
    source_policy_id: str
    split: str
    observations: tuple[Any, ...]
    actions: tuple[Any, ...]
    rewards: tuple[float, ...]
    return_to_go: tuple[float, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    pedestrians: tuple[Any, ...]
    robot_states: tuple[Any, ...]
    provenance: Mapping[str, Any]

    @property
    def step_count(self) -> int:
        """Return the number of environment steps in the episode."""
        return len(self.rewards)


def compute_return_to_go(rewards: Sequence[float]) -> list[float]:
    """Compute undiscounted future return-to-go for each reward.

    Returns:
        Future cumulative reward for each input step.
    """
    running = 0.0
    out: list[float] = []
    for reward in reversed(rewards):
        running += float(reward)
        out.append(running)
    return list(reversed(out))


def validate_rl_trajectory_episode(episode: RLTrajectoryEpisode) -> None:
    """Validate the stable offline RL episode contract."""
    if not episode.dataset_id:
        raise ValueError("dataset_id must be non-empty")
    if not episode.episode_id:
        raise ValueError("episode_id must be non-empty")
    if not episode.scenario_id:
        raise ValueError("scenario_id must be non-empty")
    if not isinstance(episode.seed, int):
        raise ValueError("seed must be an integer")
    if episode.split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {episode.split!r}")

    lengths = {
        "observations": len(episode.observations),
        "actions": len(episode.actions),
        "rewards": len(episode.rewards),
        "return_to_go": len(episode.return_to_go),
        "terminated": len(episode.terminated),
        "truncated": len(episode.truncated),
        "pedestrians": len(episode.pedestrians),
        "robot_states": len(episode.robot_states),
    }
    expected = lengths["rewards"]
    mismatched = {name: length for name, length in lengths.items() if length != expected}
    if mismatched:
        raise ValueError(
            f"RL trajectory field lengths must match rewards length {expected}: {mismatched}"
        )
    if expected == 0:
        raise ValueError("trajectory must contain at least one step")

    terminal_markers = [
        index
        for index, (terminated, truncated) in enumerate(
            zip(episode.terminated, episode.truncated, strict=True)
        )
        if bool(terminated) or bool(truncated)
    ]
    if len(terminal_markers) > 1:
        raise ValueError("trajectory may contain at most one terminal/truncated marker")
    if terminal_markers and terminal_markers[0] != expected - 1:
        raise ValueError("terminal/truncated marker must be on the final step")


def episode_from_jsonl_row(row: Mapping[str, Any]) -> RLTrajectoryEpisode:
    """Parse and validate an `RLTrajectoryEpisode.v1` JSONL row.

    Returns:
        Validated episode dataclass.
    """
    if row.get("schema_version") != RL_TRAJECTORY_EPISODE_SCHEMA_VERSION:
        raise ValueError(
            "expected schema_version "
            f"{RL_TRAJECTORY_EPISODE_SCHEMA_VERSION!r}, got {row.get('schema_version')!r}"
        )
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, Mapping):
        raise ValueError("trajectory must be an object")
    episode = RLTrajectoryEpisode(
        dataset_id=str(row.get("dataset_id", "")),
        episode_id=str(row.get("episode_id", "")),
        scenario_id=str(row.get("scenario_id", "")),
        seed=_parse_seed(row.get("seed")),
        source_policy_id=str(row.get("source_policy_id", "")),
        split=str(row.get("split", "")),
        observations=tuple(trajectory.get("observations", ())),
        actions=tuple(trajectory.get("actions", ())),
        rewards=tuple(float(value) for value in trajectory.get("rewards", ())),
        return_to_go=tuple(float(value) for value in trajectory.get("return_to_go", ())),
        terminated=tuple(bool(value) for value in trajectory.get("terminated", ())),
        truncated=tuple(bool(value) for value in trajectory.get("truncated", ())),
        pedestrians=tuple(trajectory.get("pedestrians", ())),
        robot_states=tuple(trajectory.get("robot_states", ())),
        provenance=_mapping_or_empty(row.get("provenance")),
    )
    validate_rl_trajectory_episode(episode)
    return episode


def load_rl_trajectory_dataset(dataset_path: Path | str) -> list[RLTrajectoryEpisode]:
    """Load an episode-major RL trajectory JSONL dataset.

    Returns:
        Validated episodes from the dataset.
    """
    path = Path(dataset_path)
    episodes: list[RLTrajectoryEpisode] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            episodes.append(episode_from_jsonl_row(row))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not episodes:
        raise ValueError(f"{path}: RL trajectory dataset contains no episodes")
    return episodes


def flatten_rl_trajectory_episodes(
    episodes: Sequence[RLTrajectoryEpisode],
) -> dict[str, Any]:
    """Flatten episode-major trajectories into framework-agnostic transition arrays.

    Returns:
        Dictionary of transition fields suitable for offline RL experiments.
    """
    observations: list[Any] = []
    actions: list[Any] = []
    rewards: list[float] = []
    return_to_go: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    episode_ids: list[str] = []
    scenario_ids: list[str] = []
    seeds: list[int] = []
    splits: list[str] = []

    for episode in episodes:
        validate_rl_trajectory_episode(episode)
        observations.extend(episode.observations)
        actions.extend(episode.actions)
        rewards.extend(episode.rewards)
        return_to_go.extend(episode.return_to_go)
        terminated.extend(episode.terminated)
        truncated.extend(episode.truncated)
        episode_ids.extend([episode.episode_id] * episode.step_count)
        scenario_ids.extend([episode.scenario_id] * episode.step_count)
        seeds.extend([episode.seed] * episode.step_count)
        splits.extend([episode.split] * episode.step_count)

    return {
        "observations": observations,
        "actions": actions,
        "rewards": np.asarray(rewards, dtype=float),
        "return_to_go": np.asarray(return_to_go, dtype=float),
        "terminated": np.asarray(terminated, dtype=bool),
        "truncated": np.asarray(truncated, dtype=bool),
        "episode_id": episode_ids,
        "scenario_id": scenario_ids,
        "seed": np.asarray(seeds, dtype=int),
        "split": splits,
    }


def row_from_episode(episode: RLTrajectoryEpisode) -> dict[str, Any]:
    """Serialize an RL trajectory episode to the JSONL row shape.

    Returns:
        JSON-serializable episode row.
    """
    validate_rl_trajectory_episode(episode)
    return {
        "schema_version": RL_TRAJECTORY_EPISODE_SCHEMA_VERSION,
        "dataset_id": episode.dataset_id,
        "episode_id": episode.episode_id,
        "scenario_id": episode.scenario_id,
        "seed": episode.seed,
        "source_policy_id": episode.source_policy_id,
        "split": episode.split,
        "trajectory": {
            "observations": list(episode.observations),
            "actions": list(episode.actions),
            "rewards": list(episode.rewards),
            "return_to_go": list(episode.return_to_go),
            "terminated": list(episode.terminated),
            "truncated": list(episode.truncated),
            "pedestrians": list(episode.pedestrians),
            "robot_states": list(episode.robot_states),
        },
        "provenance": dict(episode.provenance),
    }


def write_rl_trajectory_dataset(
    episodes: Sequence[RLTrajectoryEpisode],
    dataset_path: Path | str,
) -> None:
    """Write an episode-major RL trajectory JSONL dataset atomically."""
    path = Path(dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as stream:
        for episode in episodes:
            stream.write(json.dumps(row_from_episode(episode), sort_keys=True))
            stream.write("\n")
    tmp_path.replace(path)


def assign_deterministic_split(
    scenario_id: str,
    seed: int,
    *,
    train_cutoff: int = 80,
    validation_cutoff: int = 90,
) -> str:
    """Assign split by stable scenario-seed hash.

    Returns:
        Split name for the scenario-seed key.
    """
    key = f"{scenario_id}:{int(seed)}".encode()
    bucket = int(hashlib.sha256(key).hexdigest()[:8], 16) % 100
    if bucket < train_cutoff:
        return "train"
    if bucket < validation_cutoff:
        return "validation"
    return "test"


def sha256_file(path: Path | str) -> str:
    """Return SHA-256 hex digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_rl_trajectory_manifest(
    *,
    dataset_id: str,
    dataset_path: Path | str,
    episodes: Sequence[RLTrajectoryEpisode],
    created_at_utc: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and semantically validate an RL trajectory dataset manifest.

    Returns:
        Manifest dictionary ready for JSON serialization.
    """
    dataset_path_obj = Path(dataset_path)
    split_summaries = _split_summaries(episodes)
    manifest = {
        "schema_version": RL_TRAJECTORY_DATASET_MANIFEST_SCHEMA_VERSION,
        "dataset_schema_version": RL_TRAJECTORY_DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "created_at_utc": created_at_utc,
        "dataset_path": dataset_path_obj.name,
        "episode_count": len(episodes),
        "step_count": sum(episode.step_count for episode in episodes),
        "format": RL_TRAJECTORY_FORMAT,
        "reward_convention": REWARD_CONVENTION,
        "return_convention": RETURN_CONVENTION,
        "split_policy": {
            "strategy": "deterministic_scenario_seed_hash",
            "leakage_prevention": ["scenario_ids", "scenario_seed_keys"],
            "split_names": list(SPLIT_NAMES),
        },
        "splits": split_summaries,
        "provenance": dict(provenance),
        "dataset_sha256": sha256_file(dataset_path_obj),
    }
    validate_rl_trajectory_manifest_semantics(manifest)
    return manifest


def validate_rl_trajectory_manifest_semantics(manifest: Mapping[str, Any]) -> None:
    """Validate split leakage constraints not expressible in JSON Schema."""
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("manifest splits must be an object")
    missing = [name for name in SPLIT_NAMES if name not in splits]
    if missing:
        raise ValueError(f"manifest missing split summaries: {missing}")

    scenario_owners: dict[str, str] = {}
    scenario_seed_owners: dict[str, str] = {}
    for split_name, summary in splits.items():
        if not isinstance(summary, Mapping):
            raise ValueError(f"split {split_name!r} summary must be an object")
        for scenario_id in summary.get("scenario_ids", ()):
            owner = scenario_owners.setdefault(str(scenario_id), str(split_name))
            if owner != split_name:
                raise ValueError(
                    f"scenario_id {scenario_id!r} appears in both {owner!r} and {split_name!r}"
                )
        for scenario_seed_key in summary.get("scenario_seed_keys", ()):
            owner = scenario_seed_owners.setdefault(str(scenario_seed_key), str(split_name))
            if owner != split_name:
                raise ValueError(
                    "scenario_seed_key "
                    f"{scenario_seed_key!r} appears in both {owner!r} and {split_name!r}"
                )


def _split_summaries(
    episodes: Sequence[RLTrajectoryEpisode],
) -> dict[str, dict[str, Any]]:
    summaries = {
        name: {
            "episode_count": 0,
            "step_count": 0,
            "scenario_ids": [],
            "scenario_seed_keys": [],
            "episode_ids": [],
        }
        for name in SPLIT_NAMES
    }
    for episode in episodes:
        validate_rl_trajectory_episode(episode)
        summary = summaries[episode.split]
        summary["episode_count"] += 1
        summary["step_count"] += episode.step_count
        summary["episode_ids"].append(episode.episode_id)
        scenario_seed_key = f"{episode.scenario_id}:{episode.seed}"
        if episode.scenario_id not in summary["scenario_ids"]:
            summary["scenario_ids"].append(episode.scenario_id)
        if scenario_seed_key not in summary["scenario_seed_keys"]:
            summary["scenario_seed_keys"].append(scenario_seed_key)
    for summary in summaries.values():
        summary["scenario_ids"].sort()
        summary["scenario_seed_keys"].sort()
        summary["episode_ids"].sort()
    return summaries


def _parse_seed(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("seed must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be an integer") from exc


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("provenance must be an object")
    return value


__all__ = [
    "RL_TRAJECTORY_DATASET_MANIFEST_SCHEMA_VERSION",
    "RL_TRAJECTORY_DATASET_SCHEMA_VERSION",
    "RL_TRAJECTORY_EPISODE_SCHEMA_VERSION",
    "RLTrajectoryEpisode",
    "assign_deterministic_split",
    "build_rl_trajectory_manifest",
    "compute_return_to_go",
    "episode_from_jsonl_row",
    "flatten_rl_trajectory_episodes",
    "load_rl_trajectory_dataset",
    "row_from_episode",
    "sha256_file",
    "validate_rl_trajectory_episode",
    "validate_rl_trajectory_manifest_semantics",
    "write_rl_trajectory_dataset",
]
