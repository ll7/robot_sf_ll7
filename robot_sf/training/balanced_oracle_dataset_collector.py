"""Collector for balanced non-learning oracle imitation datasets."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark import map_runner
from robot_sf.benchmark.map_runner import _run_map_episode
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


def _git_sha(repo_root: Path) -> str | None:
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _contains_degraded_marker(value: Any) -> bool:
    """Return whether nested runtime metadata reports fallback/degraded execution."""
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key).lower()
            if key in {"fallback", "degraded", "fallback_or_degraded"} and bool(item):
                return True
            if key.endswith("fallback_count") and isinstance(item, (int, float)) and item > 0:
                return True
            if key == "readiness_status" and str(item).lower() in {"fallback", "degraded"}:
                return True
            if key == "availability_status" and str(item).lower() in {
                "failed",
                "not_available",
                "partial-failure",
            }:
                return True
            if _contains_degraded_marker(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_degraded_marker(item) for item in value)
    return False


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


def validate_split_and_episode_invariants(packet: dict[str, Any]) -> None:  # noqa: C901
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

    scenario_ids = packet.get("scenario_ids", [])
    allowed_scenarios = set(scenario_ids) if isinstance(scenario_ids, list) else set()
    episodes_by_split = packet.get("episode_ids_by_split", {})
    if isinstance(episodes_by_split, dict):
        all_ids: set[str] = set()
        for split in _SPLITS:
            split_ids = episodes_by_split.get(split, [])
            if not isinstance(split_ids, list) or not split_ids:
                raise BalancedDatasetCollectionError(
                    f"episode_ids_by_split.{split} must be a non-empty list"
                )
            for ep_id in split_ids:
                if ep_id in all_ids:
                    raise BalancedDatasetCollectionError(
                        f"Duplicate episode ID detected in launch packet: {ep_id!r}"
                    )
                parsed_split, scenario_id, seed = parse_episode_id(str(ep_id), split)
                if parsed_split != split:
                    raise BalancedDatasetCollectionError(
                        f"Episode ID {ep_id!r} does not belong to split {split!r}"
                    )
                if allowed_scenarios and scenario_id not in allowed_scenarios:
                    raise BalancedDatasetCollectionError(
                        f"Episode ID {ep_id!r} references undeclared scenario {scenario_id!r}"
                    )
                split_seeds = set(seeds_by_split.get(split, []))
                if split_seeds and seed not in split_seeds:
                    raise BalancedDatasetCollectionError(
                        f"Episode ID {ep_id!r} uses seed {seed} outside seeds_by_split.{split}"
                    )
                all_ids.add(ep_id)


class _CaptureEnv:
    """Transparent environment proxy that records the exact policy I/O trajectory."""

    def __init__(self, env: Any, sink: dict[str, Any]) -> None:
        self._env = env
        self._sink = sink

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        observation, info = self._env.reset(*args, **kwargs)
        self._sink["initial_observation"] = copy.deepcopy(observation)
        return observation, info

    def step(self, action: Any) -> Any:
        self._sink["actions"].append(np.asarray(action, dtype=np.float32).copy())
        observation, reward, terminated, truncated, info = self._env.step(action)
        self._sink["observations"].append(copy.deepcopy(observation))
        self._sink["rewards"].append(float(reward))
        self._sink["terminated"].append(bool(terminated))
        self._sink["truncated"].append(bool(truncated))
        return observation, reward, terminated, truncated, info


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

    def _public_git_sha(self) -> str:
        current_sha = _git_sha(self.repo_root)
        if current_sha is None:
            raise BalancedDatasetCollectionError("Cannot resolve the current public Git SHA")
        configured = self.packet.get("generating_commit")
        if configured in (None, "", "current"):
            return current_sha
        configured_sha = str(configured)
        if configured_sha != current_sha:
            raise BalancedDatasetCollectionError(
                "Launch packet generating_commit does not match the executing checkout: "
                f"packet={configured_sha}, checkout={current_sha}"
            )
        return current_sha

    def _repo_relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.repo_root).as_posix()
        except ValueError:
            return path.as_posix()

    def build_preflight_plan(self) -> dict[str, Any]:
        """Build a deterministic launch plan without performing simulation.

        Returns:
            Dictionary containing the deterministic launch plan.
        """
        self.output_root.mkdir(parents=True, exist_ok=True)
        npz_filename = "expert_traj_v1.npz"
        manifest_destination = self.output_root / "balanced_oracle_dataset_manifest.json"

        from scripts.validation.run_policy_search_candidate import (  # noqa: PLC0415
            load_candidate_definition,
        )

        public_sha = self._public_git_sha()
        candidate_entry, candidate_payload, _candidate_config, candidate_config_path = (
            load_candidate_definition(self.candidate_registry, self.source_candidate)
        )
        stratum_counts = dict.fromkeys(self.scenario_ids, 0)
        for episode_id in self.episodes_by_split.get("train", []):
            _split, scenario_id, _seed = parse_episode_id(str(episode_id), "train")
            stratum_counts[scenario_id] = stratum_counts.get(scenario_id, 0) + 1

        plan = {
            "schema_version": _PLAN_SCHEMA_VERSION,
            "git_commit": public_sha,
            "dataset_id": self.dataset_id,
            "source_candidate": self.source_candidate,
            "source_candidate_algorithm": str(candidate_payload.get("algo", "")),
            "source_candidate_config": self._repo_relative(candidate_config_path),
            "source_candidate_entry_sha256": _canonical_sha256(candidate_entry),
            "candidate_registry": self._repo_relative(self.candidate_registry),
            "candidate_registry_sha256": _file_sha256(self.candidate_registry),
            "config_path": self._repo_relative(self.config_path),
            "config_sha256": _file_sha256(self.config_path),
            "output_npz_path": npz_filename,
            "manifest_destination": manifest_destination.name,
            "scenarios": self.scenario_ids,
            "planned_strata": self.scenario_ids,
            "planned_train_episodes_per_stratum": stratum_counts,
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

        inadequate = {
            scenario_id: count
            for scenario_id, count in stratum_counts.items()
            if count < self.min_episodes_per_stratum
        }
        if inadequate:
            raise BalancedDatasetCollectionError(
                "Launch packet cannot satisfy the per-stratum episode gate: "
                f"{inadequate}; required={self.min_episodes_per_stratum}"
            )
        plan["plan_identity_sha256"] = _canonical_sha256(plan)

        plan_path = self.output_root / "balanced_oracle_collection_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return plan

    def _capture_episode(  # noqa: PLR0913
        self,
        scenario: dict[str, Any],
        *,
        seed: int,
        split: str,
        episode_id: str,
        algo: str,
        algo_config: dict[str, Any],
        scenario_path: Path,
        horizon: int,
        dt: float,
    ) -> dict[str, Any]:
        """Run one packet episode through the proven job-13520 capture seam.

        Returns:
            Captured episode fields plus exact benchmark-record provenance.
        """
        sink: dict[str, Any] = {
            "actions": [],
            "observations": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
        }
        original_factory = map_runner.make_robot_env

        def capture_factory(*args: Any, **kwargs: Any) -> _CaptureEnv:
            return _CaptureEnv(original_factory(*args, **kwargs), sink)

        map_runner.make_robot_env = capture_factory
        try:
            record = _run_map_episode(
                scenario,
                seed,
                horizon=horizon,
                dt=dt,
                record_forces=False,
                snqi_weights=None,
                snqi_baseline=None,
                algo=algo,
                scenario_path=scenario_path,
                algo_config=algo_config,
                benchmark_track=None,
                record_simulation_step_trace=True,
            )
        finally:
            map_runner.make_robot_env = original_factory

        trace_steps = (
            record.get("algorithm_metadata", {}).get("simulation_step_trace", {}).get("steps", [])
        )
        step_count = len(sink["actions"])
        if not (
            step_count
            == len(sink["observations"])
            == len(sink["rewards"])
            == len(sink["terminated"])
            == len(sink["truncated"])
            == len(trace_steps)
        ):
            raise BalancedDatasetCollectionError(
                f"Captured trajectory fields are misaligned for {episode_id!r}"
            )

        metadata = record.get("algorithm_metadata", {})
        planner_runtime = metadata.get("planner_runtime", {})
        kinematics = metadata.get("planner_kinematics", {})
        pedestrian_model = record.get("pedestrian_model", {})
        execution_mode = str(kinematics.get("execution_mode", "unknown"))
        fallback_count = int(planner_runtime.get("fallback_count", 0) or 0)
        pedestrian_status = str(pedestrian_model.get("fallback_degraded_status", "unknown"))
        fallback = (
            execution_mode not in {"native", "adapter"}
            or fallback_count > 0
            or _contains_degraded_marker(planner_runtime)
        )
        degraded = pedestrian_status != "native" or _contains_degraded_marker(metadata)
        failed = str(record.get("status", "failed")) != "success"
        actual_scenario = str(record.get("scenario_id", ""))
        actual_seed = int(record.get("seed", -1))
        _declared_split, declared_scenario, declared_seed = parse_episode_id(episode_id, split)
        leakage_invalid = actual_scenario != declared_scenario or actual_seed != declared_seed

        return {
            "episode_id": episode_id,
            "scenario_id": actual_scenario or declared_scenario,
            "seed": actual_seed if actual_seed >= 0 else declared_seed,
            "split": split,
            "actions": sink["actions"],
            "observations": sink["observations"],
            "positions": [step["robot"]["position"] for step in trace_steps],
            "rewards": sink["rewards"],
            "terminated": sink["terminated"],
            "truncated": sink["truncated"],
            "failed": failed,
            "fallback": fallback,
            "degraded": degraded,
            "leakage_invalid": leakage_invalid,
            "provenance": {
                "record": record,
                "execution_mode": execution_mode,
                "fallback_count": fallback_count,
                "pedestrian_fallback_degraded_status": pedestrian_status,
            },
        }

    def collect_source_episodes(
        self,
        *,
        horizon: int = 500,
        dt: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Collect every predeclared packet episode from the registered source candidate.

        Returns:
            One terminal captured or failed-provenance row per packet episode.
        """
        from scripts.training.collect_oracle_imitation_candidate_traces import (  # noqa: PLC0415
            build_split_scenarios,
        )
        from scripts.validation.run_policy_search_candidate import (  # noqa: PLC0415
            _group_scenarios_by_config_overrides,
            load_candidate_definition,
        )

        _entry, candidate_payload, candidate_config, candidate_config_path = (
            load_candidate_definition(self.candidate_registry, self.source_candidate)
        )
        default_algo = str(candidate_payload.get("algo", "")).strip().lower()
        if not default_algo:
            raise BalancedDatasetCollectionError(
                f"Registered source candidate {self.source_candidate!r} has no algorithm"
            )
        scenario_source = Path(str(self.packet["scenario_source"]))
        if not scenario_source.is_absolute():
            scenario_source = (self.repo_root / scenario_source).resolve()

        collected: list[dict[str, Any]] = []
        for split in _SPLITS:
            scenarios = build_split_scenarios(self.packet, split=split, repo_root=self.repo_root)
            groups = _group_scenarios_by_config_overrides(
                scenarios,
                candidate_payload=candidate_payload,
                candidate_config=candidate_config,
                default_algo=default_algo,
                config_anchor=candidate_config_path.parent,
            )
            entries: dict[str, dict[str, Any]] = {}
            for group in groups.values():
                for scenario in group["scenarios"]:
                    seeds = scenario.get("seeds")
                    if not isinstance(seeds, list) or len(seeds) != 1:
                        raise BalancedDatasetCollectionError(
                            f"Scenario in split {split!r} must declare exactly one seed"
                        )
                    episode_id = str(
                        scenario.get("metadata", {}).get("oracle_imitation_episode_id", "")
                    )
                    entries[episode_id] = {
                        "scenario": scenario,
                        "seed": int(seeds[0]),
                        "algo": str(group["algo"]),
                        "algo_config": dict(group["config"]),
                    }

            for episode_id in self.episodes_by_split[split]:
                entry = entries.get(str(episode_id))
                if entry is None:
                    raise BalancedDatasetCollectionError(
                        f"No runnable scenario entry for packet episode {episode_id!r}"
                    )
                try:
                    collected.append(
                        self._capture_episode(
                            entry["scenario"],
                            seed=entry["seed"],
                            split=split,
                            episode_id=str(episode_id),
                            algo=entry["algo"],
                            algo_config=entry["algo_config"],
                            scenario_path=scenario_source,
                            horizon=horizon,
                            dt=dt,
                        )
                    )
                except (
                    AssertionError,
                    KeyError,
                    OSError,
                    RobotSfError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:  # preserve a terminal row for fail-closed accounting
                    _parsed_split, scenario_id, seed = parse_episode_id(str(episode_id), split)
                    collected.append(
                        {
                            "episode_id": str(episode_id),
                            "scenario_id": scenario_id,
                            "seed": seed,
                            "split": split,
                            "actions": [],
                            "observations": [],
                            "positions": [],
                            "rewards": [],
                            "terminated": [],
                            "truncated": [],
                            "failed": True,
                            "fallback": False,
                            "degraded": False,
                            "leakage_invalid": False,
                            "provenance": {"collection_error": f"{type(exc).__name__}: {exc}"},
                        }
                    )
        return collected

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

    def _validate_collected_identities(self, raw_episodes: list[dict[str, Any]]) -> list[str]:
        expected: dict[str, tuple[str, str, int]] = {}
        for split in _SPLITS:
            for episode_id in self.episodes_by_split[split]:
                parsed_split, scenario_id, seed = parse_episode_id(str(episode_id), split)
                expected[str(episode_id)] = (parsed_split, scenario_id, seed)

        seen: set[str] = set()
        for episode in raw_episodes:
            episode_id = str(episode.get("episode_id", ""))
            if episode_id in seen:
                raise BalancedDatasetCollectionError(
                    f"Collected duplicate episode ID: {episode_id!r}"
                )
            seen.add(episode_id)
            identity = expected.get(episode_id)
            if identity is None:
                raise BalancedDatasetCollectionError(
                    f"Collected episode is not predeclared by the packet: {episode_id!r}"
                )
            split, scenario_id, seed = identity
            mismatches: list[str] = []
            if str(episode.get("split")) != split:
                mismatches.append("split")
            if str(episode.get("scenario_id")) != scenario_id:
                mismatches.append("scenario_id")
            if int(episode.get("seed", -1)) != seed:
                mismatches.append("seed")
            if mismatches:
                episode["leakage_invalid"] = True
                provenance = episode.setdefault("provenance", {})
                provenance["identity_mismatches"] = mismatches
                provenance["expected_identity"] = {
                    "split": split,
                    "scenario_id": scenario_id,
                    "seed": seed,
                }
        return sorted(set(expected) - seen)

    def _write_raw_provenance(self, raw_episodes: list[dict[str, Any]]) -> Path:
        path = self.output_root / "raw_episode_provenance.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for episode in raw_episodes:
                handle.write(json.dumps(_jsonable(episode), sort_keys=True) + "\n")
        return path

    def collect_dataset(
        self,
        *,
        episodes_override: list[dict[str, Any]] | None = None,
        allow_insufficient_yield: bool = False,
        cli_command: str | None = None,
        horizon: int = 500,
        dt: float = 0.1,
    ) -> dict[str, Any]:
        """Collect balanced dataset, materialize NPZ, and write manifest.

        Args:
            episodes_override: Optional list of episode dictionaries for testing.
            allow_insufficient_yield: Whether to bypass minimum yield gates.
            cli_command: Explicit CLI command string to record in manifest.
            horizon: Maximum simulation steps per episode.
            dt: Simulation step duration in seconds.

        Returns:
            Manifest dictionary.
        """
        self.output_root.mkdir(parents=True, exist_ok=True)
        raw_episodes = (
            episodes_override
            if episodes_override is not None
            else self.collect_source_episodes(horizon=horizon, dt=dt)
        )
        self._write_raw_provenance(raw_episodes)
        missing_episode_ids = self._validate_collected_identities(raw_episodes)
        raw_provenance_path = self._write_raw_provenance(raw_episodes)
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
            if missing_episode_ids:
                raise BalancedDatasetCollectionError(
                    "Insufficient yield: collection did not produce every predeclared "
                    "packet episode: "
                    f"{missing_episode_ids[:10]}"
                )
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

        train_actions_list = [
            np.asarray(ep["actions"], dtype=np.float32) for ep in train_usable_episodes
        ]
        train_weights, bin_summary = compute_action_bin_accounting(train_actions_list)
        weight_by_episode: dict[str, np.ndarray] = {}
        cursor = 0
        for episode in train_usable_episodes:
            steps = len(episode["actions"])
            weight_by_episode[str(episode["episode_id"])] = train_weights[cursor : cursor + steps]
            cursor += steps
        bin_summary["weights_sha256"] = hashlib.sha256(train_weights.tobytes()).hexdigest()
        bin_summary["weight_mean"] = float(np.mean(train_weights)) if len(train_weights) else 0.0
        bin_summary["weight_min"] = float(np.min(train_weights)) if len(train_weights) else 0.0
        bin_summary["weight_max"] = float(np.max(train_weights)) if len(train_weights) else 0.0

        npz_filename = "expert_traj_v1.npz"
        npz_path = self.output_root / npz_filename
        _write_expert_traj_npz(
            npz_path,
            usable_episodes,
            self.dataset_id,
            self.source_candidate,
            action_weights=weight_by_episode,
        )

        sha256_npz = _file_sha256(npz_path)
        sha256_raw_provenance = _file_sha256(raw_provenance_path)
        public_sha = self._public_git_sha()
        usable_split_counts = {
            split: sum(1 for episode in usable_episodes if episode["split"] == split)
            for split in _SPLITS
        }
        gates_passed = not missing_episode_ids and all(
            [
                usable_train_transitions >= self.min_usable_transitions,
                all(
                    stratum_counts["train"].get(sc_id, 0) >= self.min_episodes_per_stratum
                    for sc_id in self.scenario_ids
                ),
                all(usable_split_counts[split] > 0 for split in _SPLITS),
                not any(item["reason"] == "leakage_invalid" for item in exclusions),
            ]
        )

        cmd_str = cli_command or " ".join(sys.argv)
        manifest_path = self.output_root / "balanced_oracle_dataset_manifest.json"
        bc_smoke_cmd = (
            f"uv run python scripts/validation/run_oracle_imitation_bc_smoke.py "
            f"--dataset-path {npz_path}"
        )

        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": public_sha,
            "exact_public_sha": public_sha,
            "dataset_id": self.dataset_id,
            "source_candidate": self.source_candidate,
            "source_packet_sha256": _file_sha256(self.config_path),
            "candidate_registry_sha256": _file_sha256(self.candidate_registry),
            "sha256_inventory": {
                npz_filename: sha256_npz,
                raw_provenance_path.name: sha256_raw_provenance,
            },
            "dataset_sha256": sha256_npz,
            "command": cmd_str,
            "exclusions": exclusions,
            "missing_episode_ids": missing_episode_ids,
            "eligibility_status": (
                "training_ready" if gates_passed else "diagnostic_insufficient_yield"
            ),
            "yield_gates": {
                "status": "pass" if gates_passed else "fail",
                "min_usable_transitions": self.min_usable_transitions,
                "min_episodes_per_stratum": self.min_episodes_per_stratum,
            },
            "balance_summary": {
                "action_bin_accounting": bin_summary,
                "stratum_counts": stratum_counts,
                "stratum_transitions": stratum_transitions,
                "usable_train_episodes": len(train_usable_episodes),
                "usable_train_transitions": usable_train_transitions,
                "total_usable_episodes": len(usable_episodes),
                "total_excluded_episodes": len(exclusions),
            },
            "private_artifact_registry_candidate": (
                {
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
                }
                if gates_passed
                else None
            ),
            "bc_loader_smoke_command": bc_smoke_cmd,
            "manifest_path": str(manifest_path),
            "npz_path": str(npz_path),
            "raw_provenance_path": str(raw_provenance_path),
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
    *,
    action_weights: dict[str, np.ndarray],
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
    balance_weights = np.zeros((episode_count, max_steps), dtype=np.float32)

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
        ep_terminated = ep.get("terminated", [False] * steps)
        ep_truncated = ep.get("truncated", [False] * steps)
        if not all(
            len(values) == steps
            for values in (ep_obs, ep_act, ep_rew, ep_pos, ep_terminated, ep_truncated)
        ):
            raise BalancedDatasetCollectionError(
                f"Trajectory fields are misaligned for episode {ep_id!r}"
            )
        weights = action_weights.get(ep_id, np.ones(steps, dtype=np.float32))
        if len(weights) != steps:
            raise BalancedDatasetCollectionError(
                f"Action balance weights are misaligned for episode {ep_id!r}"
            )

        running_rtg = 0.0
        rtg_vals: list[float] = []
        for r in reversed(ep_rew):
            running_rtg += float(r)
            rtg_vals.append(running_rtg)
        rtg_vals.reverse()

        for step in range(steps):
            observations[idx, step] = ep_obs[step]
            actions[idx, step] = np.asarray(ep_act[step], dtype=np.float32)
            positions[idx, step] = np.asarray(ep_pos[step], dtype=np.float32)
            rewards[idx, step] = float(ep_rew[step])
            return_to_go[idx, step] = float(rtg_vals[step])
            terminated[idx, step] = bool(ep_terminated[step])
            truncated[idx, step] = bool(ep_truncated[step])
            balance_weights[idx, step] = float(weights[step])

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
        action_balance_weights=balance_weights,
        metadata=json.dumps(metadata),
    )


__all__ = [
    "BalancedDatasetCollectionError",
    "BalancedOracleCollector",
    "parse_episode_id",
    "validate_split_and_episode_invariants",
]
