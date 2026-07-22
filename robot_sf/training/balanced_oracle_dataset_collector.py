"""Collector for balanced non-learning oracle imitation datasets."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.errors import RobotSfError
from robot_sf.training.action_bin_accounting import compute_action_bin_accounting
from robot_sf.training.oracle_imitation_launch_packet import (
    load_launch_packet,
    validate_launch_packet,
)

EPISODE_ID_RE = re.compile(r"^(?P<split>[a-z_]+)__(?P<scenario>.+)__seed(?P<seed>\d+)$")
_SCHEMA_VERSION = "balanced-oracle-dataset-manifest.v1"
_PLAN_SCHEMA_VERSION = "balanced-oracle-collection-plan.v1"
_SPLITS = ("train", "validation", "evaluation")


class BalancedDatasetCollectionError(RobotSfError, ValueError):
    """Raised when balanced oracle dataset collection fails or violates gates."""


def _git_sha() -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_episode_id(episode_id: str, split: str | None = None) -> tuple[str, str, int]:
    """Parse launch-packet episode ID into (split, scenario_id, seed).

    Args:
        episode_id: Raw episode ID string.
        split: Expected split name.

    Returns:
        Tuple of (split, scenario_id, seed).
    """
    match = EPISODE_ID_RE.match(episode_id)
    if match is None:
        raise BalancedDatasetCollectionError(f"Invalid episode ID format: {episode_id!r}")
    parsed_split = match.group("split")
    if split is not None and parsed_split != split:
        raise BalancedDatasetCollectionError(
            f"Episode ID split mismatch for {episode_id!r}: expected {split!r}, got {parsed_split!r}"
        )
    return parsed_split, match.group("scenario"), int(match.group("seed"))


def validate_split_and_episode_invariants(packet: dict[str, Any]) -> None:
    """Validate split seed overlap and duplicate episode ID invariants."""
    seeds_by_split = packet.get("seeds_by_split", {})
    if isinstance(seeds_by_split, dict):
        for i, left in enumerate(_SPLITS):
            left_seeds = set(seeds_by_split.get(left, []))
            for right in _SPLITS[i + 1 :]:
                right_seeds = set(seeds_by_split.get(right, []))
                overlap = sorted(left_seeds & right_seeds)
                if overlap:
                    raise BalancedDatasetCollectionError(
                        f"Seed overlap detected between {left} and {right}: {overlap}"
                    )

    episodes_by_split = packet.get("episode_ids_by_split", {})
    if isinstance(episodes_by_split, dict):
        all_ids: list[str] = []
        for split in _SPLITS:
            split_ids = episodes_by_split.get(split, [])
            for ep_id in split_ids:
                if ep_id in all_ids:
                    raise BalancedDatasetCollectionError(
                        f"Duplicate episode ID detected in launch packet: {ep_id!r}"
                    )
                all_ids.append(ep_id)


class BalancedOracleCollector:
    """Orchestrates balanced oracle dataset collection, preflight planning, and validation."""

    def __init__(
        self,
        config_path: Path,
        *,
        output_root: Path,
        candidate_registry: Path | None = None,
        repo_root: Path | None = None,
        min_usable_transitions: int = 10000,
        min_episodes_per_stratum: int = 10,
    ) -> None:
        """Initialize BalancedOracleCollector.

        Args:
            config_path: Path to launch packet YAML.
            output_root: Output directory for dataset artifacts and manifests.
            candidate_registry: Path to candidate registry YAML.
            repo_root: Repository root path.
            min_usable_transitions: Minimum required usable training transitions.
            min_episodes_per_stratum: Minimum required usable episodes per stratum.
        """
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.config_path = (
            config_path.resolve()
            if config_path.is_absolute()
            else (self.repo_root / config_path).resolve()
        )
        self.output_root = (
            output_root.resolve()
            if output_root.is_absolute()
            else (self.repo_root / output_root).resolve()
        )
        self.candidate_registry = (
            candidate_registry.resolve()
            if candidate_registry and candidate_registry.is_absolute()
            else (
                (self.repo_root / candidate_registry).resolve()
                if candidate_registry
                else (self.repo_root / "docs/context/policy_search/candidate_registry.yaml")
            )
        )
        self.min_usable_transitions = min_usable_transitions
        self.min_episodes_per_stratum = min_episodes_per_stratum

        self.packet_validation = validate_launch_packet(self.config_path, repo_root=self.repo_root)
        self.packet = load_launch_packet(self.config_path)
        validate_split_and_episode_invariants(self.packet)

        self.dataset_id = str(self.packet["dataset_id"])
        self.source_candidate = str(self.packet["source_candidate"])
        self.scenario_ids = list(self.packet.get("scenario_ids", []))
        self.episodes_by_split = dict(self.packet.get("episode_ids_by_split", {}))
        self.seeds_by_split = dict(self.packet.get("seeds_by_split", {}))

    def build_preflight_plan(self) -> dict[str, Any]:
        """Build a deterministic launch plan without performing simulation.

        Returns:
            Dictionary containing the deterministic launch plan.
        """
        self.output_root.mkdir(parents=True, exist_ok=True)
        npz_filename = "expert_traj_v1.npz"
        output_npz_path = self.output_root / npz_filename
        manifest_destination = self.output_root / "balanced_oracle_dataset_manifest.json"

        plan = {
            "schema_version": _PLAN_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": self.packet.get("generating_commit") or _git_sha() or "unknown",
            "dataset_id": self.dataset_id,
            "source_candidate": self.source_candidate,
            "candidate_registry": str(self.candidate_registry),
            "config_path": str(self.config_path),
            "output_root": str(self.output_root),
            "output_npz_path": str(output_npz_path),
            "manifest_destination": str(manifest_destination),
            "scenarios": self.scenario_ids,
            "planned_strata": self.scenario_ids,
            "planned_episodes_by_split": {
                split: len(self.episodes_by_split.get(split, [])) for split in _SPLITS
            },
            "planned_seeds_by_split": {
                split: list(self.seeds_by_split.get(split, [])) for split in _SPLITS
            },
            "gates": {
                "min_usable_transitions": self.min_usable_transitions,
                "min_episodes_per_stratum": self.min_episodes_per_stratum,
            },
            "exclusion_rules": [
                "one-step trajectories (steps <= 1)",
                "failed or crashed trajectories",
                "fallback or degraded policy execution",
                "leakage invalid or seed overlap trajectories",
            ],
            "packet_validation_status": self.packet_validation.get("status", "valid"),
        }

        plan_path = self.output_root / "balanced_oracle_collection_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return plan

    def _filter_episodes(
        self, raw_episodes: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        usable: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []

        for ep in raw_episodes:
            ep_id = str(ep["episode_id"])
            sc_id = str(ep["scenario_id"])
            seed = int(ep["seed"])
            split = str(ep["split"])
            actions = ep.get("actions", np.zeros((0, 2), dtype=np.float32))
            steps = len(actions)

            reason: str | None = None
            if steps <= 1:
                reason = "one-step"
            elif bool(ep.get("failed", False)):
                reason = "failed"
            elif bool(ep.get("fallback", False) or ep.get("degraded", False)):
                reason = "fallback"
            elif bool(ep.get("leakage_invalid", False)):
                reason = "leakage_invalid"

            if reason is not None:
                exclusions.append(
                    {
                        "episode_id": ep_id,
                        "scenario_id": sc_id,
                        "seed": seed,
                        "split": split,
                        "reason": reason,
                        "steps": steps,
                    }
                )
            else:
                usable.append(ep)
        return usable, exclusions

    def collect_dataset(
        self,
        *,
        episodes_override: list[dict[str, Any]] | None = None,
        allow_insufficient_yield: bool = False,
        cli_command: str | None = None,
    ) -> dict[str, Any]:
        """Collect balanced dataset, materialize NPZ, and write manifest.

        Args:
            episodes_override: Optional list of episode dictionaries for testing.
            allow_insufficient_yield: Whether to bypass minimum yield gates.
            cli_command: Explicit CLI command string to record in manifest.

        Returns:
            Manifest dictionary.
        """
        self.output_root.mkdir(parents=True, exist_ok=True)
        raw_episodes = episodes_override if episodes_override is not None else []
        usable_episodes, exclusions = self._filter_episodes(raw_episodes)

        stratum_counts: dict[str, dict[str, int]] = {
            split: dict.fromkeys(self.scenario_ids, 0) for split in _SPLITS
        }
        stratum_transitions: dict[str, dict[str, int]] = {
            split: dict.fromkeys(self.scenario_ids, 0) for split in _SPLITS
        }

        usable_train_transitions = 0
        train_usable_episodes = [ep for ep in usable_episodes if ep["split"] == "train"]

        for ep in usable_episodes:
            split = ep["split"]
            sc_id = ep["scenario_id"]
            steps = len(ep["actions"])
            stratum_counts[split][sc_id] = stratum_counts[split].get(sc_id, 0) + 1
            stratum_transitions[split][sc_id] = stratum_transitions[split].get(sc_id, 0) + steps
            if split == "train":
                usable_train_transitions += steps

        if not allow_insufficient_yield:
            if usable_train_transitions < self.min_usable_transitions:
                raise BalancedDatasetCollectionError(
                    f"Insufficient yield: usable training transitions ({usable_train_transitions}) "
                    f"< required minimum ({self.min_usable_transitions})"
                )
            for sc_id in self.scenario_ids:
                cnt = stratum_counts["train"].get(sc_id, 0)
                if cnt < self.min_episodes_per_stratum:
                    raise BalancedDatasetCollectionError(
                        f"Insufficient yield for training stratum {sc_id!r}: "
                        f"usable episodes ({cnt}) < required minimum ({self.min_episodes_per_stratum})"
                    )

        npz_filename = "expert_traj_v1.npz"
        npz_path = self.output_root / npz_filename
        _write_expert_traj_npz(npz_path, usable_episodes, self.dataset_id, self.source_candidate)

        sha256_npz = _file_sha256(npz_path)
        train_actions_list = [ep["actions"] for ep in train_usable_episodes]
        _weights, bin_summary = compute_action_bin_accounting(train_actions_list)

        cmd_str = cli_command or " ".join(sys.argv)
        manifest_path = self.output_root / "balanced_oracle_dataset_manifest.json"
        bc_smoke_cmd = (
            f"uv run python scripts/validation/run_oracle_imitation_bc_smoke.py "
            f"--dataset-path {npz_path}"
        )

        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": self.packet.get("generating_commit") or _git_sha() or "unknown",
            "dataset_id": self.dataset_id,
            "source_candidate": self.source_candidate,
            "sha256_inventory": {
                npz_filename: sha256_npz,
            },
            "exact_public_sha": sha256_npz,
            "command": cmd_str,
            "exclusions": exclusions,
            "balance_summary": {
                "action_bin_accounting": bin_summary,
                "stratum_counts": stratum_counts,
                "stratum_transitions": stratum_transitions,
                "usable_train_episodes": len(train_usable_episodes),
                "usable_train_transitions": usable_train_transitions,
                "total_usable_episodes": len(usable_episodes),
                "total_excluded_episodes": len(exclusions),
            },
            "private_artifact_registry_candidate": {
                "dataset_id": self.dataset_id,
                "uri": f"private-artifact://oracle-imitation/{self.dataset_id}/{npz_filename}",
                "sha256": sha256_npz,
                "splits": {
                    split: {
                        "episode_ids": [
                            ep["episode_id"] for ep in usable_episodes if ep["split"] == split
                        ]
                    }
                    for split in _SPLITS
                },
            },
            "bc_loader_smoke_command": bc_smoke_cmd,
            "manifest_path": str(manifest_path),
            "npz_path": str(npz_path),
        }

        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest


def _write_expert_traj_npz(
    path: Path,
    episodes: list[dict[str, Any]],
    dataset_id: str,
    source_candidate: str,
) -> None:
    episode_count = len(episodes)
    max_steps = max((len(ep["actions"]) for ep in episodes), default=1)

    observations = np.empty((episode_count, max_steps), dtype=object)
    actions = np.empty((episode_count, max_steps, 2), dtype=object)
    positions = np.empty((episode_count, max_steps, 2), dtype=object)
    rewards = np.empty((episode_count, max_steps), dtype=object)
    return_to_go = np.empty((episode_count, max_steps), dtype=object)
    terminated = np.empty((episode_count, max_steps), dtype=object)
    truncated = np.empty((episode_count, max_steps), dtype=object)
    episode_ids = np.empty((episode_count, 1), dtype=object)
    scenario_ids = np.empty((episode_count, 1), dtype=object)
    seeds = np.empty((episode_count, 1), dtype=object)
    split_tags = np.empty((episode_count, 1), dtype=object)

    splits_mapping: dict[str, list[str]] = {split: [] for split in _SPLITS}

    for idx, ep in enumerate(episodes):
        ep_id = str(ep["episode_id"])
        sc_id = str(ep["scenario_id"])
        seed = int(ep["seed"])
        split = str(ep["split"])
        steps = len(ep["actions"])

        splits_mapping.setdefault(split, []).append(ep_id)

        ep_obs = ep.get("observations", [])
        ep_act = ep.get("actions", [])
        ep_rew = ep.get("rewards", [0.05] * steps)
        ep_pos = ep.get("positions", [np.zeros(2, dtype=np.float32)] * steps)

        running_rtg = 0.0
        rtg_vals: list[float] = []
        for r in reversed(ep_rew):
            running_rtg += float(r)
            rtg_vals.append(running_rtg)
        rtg_vals.reverse()

        for step in range(steps):
            observations[idx, step] = ep_obs[step] if step < len(ep_obs) else None
            actions[idx, step] = np.asarray(ep_act[step], dtype=np.float32)
            positions[idx, step] = np.asarray(ep_pos[step], dtype=np.float32)
            rewards[idx, step] = float(ep_rew[step])
            return_to_go[idx, step] = float(rtg_vals[step])
            terminated[idx, step] = step == steps - 1
            truncated[idx, step] = False

        for step in range(steps, max_steps):
            observations[idx, step] = None
            actions[idx, step] = np.zeros(2, dtype=np.float32)
            positions[idx, step] = np.zeros(2, dtype=np.float32)
            rewards[idx, step] = 0.0
            return_to_go[idx, step] = 0.0
            terminated[idx, step] = False
            truncated[idx, step] = False

        episode_ids[idx, 0] = ep_id
        scenario_ids[idx, 0] = sc_id
        seeds[idx, 0] = seed
        split_tags[idx, 0] = split

    metadata = {
        "dataset_id": dataset_id,
        "source_policy_id": source_candidate,
        "dataset_schema": "trajectory_dataset.v2.decision_transformer_preflight",
        "splits": {split: {"episode_ids": ids} for split, ids in splits_mapping.items()},
        "data_collection_only": True,
        "training_performed": False,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=observations,
        actions=actions,
        positions=positions,
        rewards=rewards,
        return_to_go=return_to_go,
        terminated=terminated,
        truncated=truncated,
        episode_ids=episode_ids,
        scenario_ids=scenario_ids,
        seeds=seeds,
        splits=split_tags,
        metadata=json.dumps(metadata),
    )


__all__ = [
    "BalancedDatasetCollectionError",
    "BalancedOracleCollector",
    "parse_episode_id",
    "validate_split_and_episode_invariants",
]
