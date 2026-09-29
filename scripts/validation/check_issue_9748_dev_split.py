"""Validate the issue #9748 development split and tuning-log boundary.

The checker validates the checked-in campaign through the same camera-ready
campaign and scenario loaders used by the benchmark. When a structured tuning
log is supplied, its frozen source/config provenance and only typed fields in
the ``issue_9748.tuning_log.v1`` schema are inspected for admission. Free-form
strings such as a rationale mentioning the held-out range are intentionally
not scanned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios, load_campaign_config
from robot_sf.training.scenario_loader import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/benchmarks/issue_9748_hybrid_v4_dev_split_v1.yaml"
RELEASE_CONFIG_RELATIVE_PATH = (
    "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate.yaml"
)
DEFAULT_RELEASE_MATRIX = (
    ROOT
    / "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_kernel_wrapped_v2.yaml"
)
RELEASE_PARITY_SIMULATION_OVERRIDES = {
    "goal_completion_policy": "goal_zone_entry_v1",
    "social_force_kernel_version": "wrapped_v2",
}
SIMULATOR_CAMPAIGN_FIELDS = (
    "horizon",
    "dt",
    "kinematics_matrix",
    "holonomic_command_mode",
    "observation_mode",
    "observation_noise",
    "synthetic_actuation_profile",
    "latency_stress_profile",
    "telemetry",
    "record_forces",
    "record_planner_decision_trace",
    "record_simulation_step_trace",
    "scenario_horizons_path",
    "radius_sweep",
)
EXPECTED_DEV_SEEDS = tuple(range(1001, 1031))
RELEASE_SEEDS = frozenset(range(111, 141))
EXPECTED_PLANNER_ALGO = "hybrid_rule_local_planner"
EXPECTED_VARIANTS = {
    "issue_9748_dev_classic_doorway_medium": {
        "source_file": "configs/scenarios/archetypes/classic_doorway.yaml",
        "source_id": "classic_doorway_medium",
        "simulation_overrides": {"ped_density": 0.065, "route_spawn_jitter_frac": 0.30},
        "development_variant": "density_and_route_spawn_jitter",
    },
    "issue_9748_dev_classic_group_crossing_medium": {
        "source_file": "configs/scenarios/archetypes/classic_group_crossing.yaml",
        "source_id": "classic_group_crossing_medium",
        "simulation_overrides": {"ped_density": 0.10},
        "development_variant": "density",
    },
    "issue_9748_dev_francis2023_perpendicular_traffic": {
        "source_file": "configs/scenarios/single/francis2023_perpendicular_traffic.yaml",
        "source_id": "francis2023_perpendicular_traffic",
        "simulation_overrides": {"ped_density": 0.12},
        "development_variant": "density",
    },
    "issue_9748_dev_francis2023_crowd_navigation": {
        "source_file": "configs/scenarios/single/francis2023_crowd_navigation.yaml",
        "source_id": "francis2023_crowd_navigation",
        "simulation_overrides": {"ped_density": 0.10},
        "development_variant": "density",
    },
}
EXPECTED_SCENARIO_IDS = frozenset(EXPECTED_VARIANTS)
DEV_METADATA = {
    "development_only": True,
    "evidence_tier": "development_tuning_only",
    "claim_boundary": "no release or paper-facing evidence",
}
EXPECTED_PLANNER_CONFIGS = {
    "hybrid_rule_v4_fast_progress_static_escape_s30_h600_release": (
        ROOT
        / "configs/policy_search/candidates/hybrid_rule_v4_fast_progress_static_escape_s30_h600_release.yaml"
    ),
    "hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release": (
        ROOT
        / "configs/policy_search/candidates/hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release.yaml"
    ),
}
TUNING_LOG_SCHEMA = "issue_9748.tuning_log.v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SOURCE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class ValidationError(ValueError):
    """Raised when a development-split admission contract is violated."""


def _repo_path(raw: str | Path, *, relative_to: Path | None = None) -> Path:
    """Resolve a repository-owned path from a config or the repository root."""
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()
    if relative_to is not None:
        from_config = (relative_to / candidate).resolve()
        if from_config.exists():
            return from_config
    return (ROOT / candidate).resolve()


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValidationError(f"could not read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"{label} must be a YAML mapping: {path}")
    return payload


def _sha256(path: Path, *, label: str) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValidationError(f"could not hash {label}: {path}: {exc}") from exc


def _require_working_hash(recorded_hash: str, path: Path, *, label: str) -> None:
    working_hash = _sha256(path, label=label)
    if recorded_hash != working_hash:
        raise ValidationError(f"{label} does not match the working tree file ({working_hash})")


def _commit_sha256(commit: str, path: Path, *, label: str) -> str:
    relative_path = _repo_relative_path(path, label=label)
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValidationError(f"could not read {label} from commit {commit}: {exc}") from exc
    return hashlib.sha256(result.stdout).hexdigest()


_TUNING_FILE_REFERENCE_KEYS = frozenset(
    {
        "algo_config",
        "base_config_path",
        "includes",
        "map_file",
        "map_svg",
        "scenario_matrix",
        "scenario_config",
        "scenario_config_path",
    }
)
_TUNING_LOADER_RELATIVE_PATHS = (
    "robot_sf/benchmark/camera_ready/_config.py",
    "robot_sf/benchmark/camera_ready/_config_types.py",
    "robot_sf/benchmark/camera_ready/_util.py",
    "robot_sf/training/scenario_loader.py",
    "robot_sf/benchmark/map_runner_policies/map_runner_policy_resolution.py",
    "robot_sf/benchmark/policy_search_manifest.py",
)


def _iter_tuning_file_references(value: Any, *, owner_path: Path) -> list[Path]:
    """Yield repository files referenced by tuning YAML inputs."""
    found: list[Path] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _TUNING_FILE_REFERENCE_KEYS:
                raw_values = (
                    nested if normalized == "includes" and isinstance(nested, list) else [nested]
                )
                for raw in raw_values:
                    if not isinstance(raw, str) or not raw.strip():
                        continue
                    path = _repo_path(raw, relative_to=owner_path.parent)
                    if not path.is_file():
                        raise ValidationError(
                            f"referenced tuning input does not exist: {raw!r} in {owner_path}"
                        )
                    found.append(path)
            found.extend(_iter_tuning_file_references(nested, owner_path=owner_path))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_iter_tuning_file_references(nested, owner_path=owner_path))
    return found


def _tuning_input_paths(
    *,
    config_path: Path,
    scenario_matrix_path: Path,
    candidate_paths: Mapping[str, Path],
) -> tuple[Path, ...]:
    """Resolve the transitive scenario, map, campaign, and candidate inputs."""
    pending = [config_path.resolve(), scenario_matrix_path.resolve()]
    pending.extend((ROOT / path).resolve() for path in _TUNING_LOADER_RELATIVE_PATHS)
    pending.extend(path.resolve() for path in candidate_paths.values())
    found: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        if not path.is_file():
            raise ValidationError(f"transitive tuning input does not exist: {path}")
        found.add(path)
        if path.suffix.lower() not in {".yaml", ".yml"}:
            continue
        payload = _load_mapping(path, label="transitive tuning input")
        pending.extend(_iter_tuning_file_references(payload, owner_path=path))
    # The candidate's release matrix is a comparator input for the dev split.
    # Freeze its manifest chain without pulling in unrelated release arms/maps;
    # the four selected map files are already in the dev input closure.
    found.update(_release_parity_input_paths())
    return tuple(sorted(found, key=lambda path: path.as_posix()))


def _release_parity_input_paths() -> set[Path]:
    campaign_path = ROOT / RELEASE_CONFIG_RELATIVE_PATH
    campaign = _load_mapping(campaign_path, label="0.0.8 candidate campaign")
    matrix_raw = campaign.get("scenario_matrix")
    if not isinstance(matrix_raw, str) or not matrix_raw.strip():
        raise ValidationError("0.0.8 candidate campaign must select a scenario matrix")
    pending = [_repo_path(matrix_raw, relative_to=campaign_path.parent)]
    found = {campaign_path}
    while pending:
        path = pending.pop()
        if path in found:
            continue
        payload = _load_mapping(path, label="0.0.8 candidate scenario matrix")
        found.add(path)
        includes = payload.get("includes", [])
        if not isinstance(includes, list) or any(not isinstance(raw, str) for raw in includes):
            raise ValidationError(f"invalid includes in 0.0.8 candidate matrix: {path}")
        pending.extend(_repo_path(raw, relative_to=path.parent) for raw in includes)
    return found


def _current_source_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValidationError(f"could not resolve the current source commit: {exc}") from exc
    commit = result.stdout.strip()
    if not SOURCE_COMMIT_PATTERN.fullmatch(commit):
        raise ValidationError("current source commit is not a 40-character hexadecimal SHA")
    return commit


def _require_sha256(raw: Any, *, label: str) -> str:
    if not isinstance(raw, str) or not SHA256_PATTERN.fullmatch(raw):
        raise ValidationError(f"{label} must be a lowercase 64-character SHA-256")
    return raw


def _require_source_commit(raw: Any) -> str:
    if not isinstance(raw, str) or not SOURCE_COMMIT_PATTERN.fullmatch(raw):
        raise ValidationError(
            "provenance.source_commit must be a lowercase 40-character source commit SHA"
        )
    current = _current_source_commit()
    if raw == current:
        raise ValidationError(
            "provenance.source_commit must be a strict ancestor of the validating checkout "
            f"commit {current}; record the tuning log in a descendant commit"
        )
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", raw, current],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ValidationError(f"could not verify frozen source commit ancestry: {exc}") from exc
    if result.returncode == 1:
        raise ValidationError(
            "provenance.source_commit must be an ancestor of the current checked-out source "
            f"commit {current}"
        )
    if result.returncode != 0:
        raise ValidationError(
            "provenance.source_commit must name a valid commit in the current repository "
            f"(git merge-base exited {result.returncode})"
        )
    return raw


def _require_committed_log(path: Path, *, source_commit: str) -> str:
    """Require the exact supplied log bytes in a commit descended from the freeze."""
    if path.is_symlink():
        raise ValidationError("structured tuning log must be a regular tracked file, not a symlink")
    relative_path = _repo_relative_path(path, label="structured tuning log")
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative_path],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise ValidationError(
                "structured tuning log must be tracked in the validating checkout"
            )
        committed_bytes = subprocess.run(
            ["git", "show", f"HEAD:{relative_path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        log_commit = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", relative_path],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValidationError(f"could not verify structured tuning-log commit: {exc}") from exc

    if path.read_bytes() != committed_bytes:
        raise ValidationError(
            "structured tuning log bytes differ from the tracked file at validating HEAD"
        )
    if not SOURCE_COMMIT_PATTERN.fullmatch(log_commit):
        raise ValidationError("could not identify the commit that recorded the tuning log")
    if log_commit == source_commit:
        raise ValidationError(
            "structured tuning log must be recorded in a strict descendant commit of "
            "provenance.source_commit"
        )
    try:
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, log_commit],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ValidationError(f"could not verify tuning-log commit ancestry: {exc}") from exc
    if ancestry.returncode == 1:
        raise ValidationError(
            "the commit recording the structured tuning log must descend from "
            "provenance.source_commit"
        )
    if ancestry.returncode != 0:
        raise ValidationError(
            "could not verify tuning-log commit ancestry "
            f"(git merge-base exited {ancestry.returncode})"
        )
    return log_commit


def _repo_relative_path(path: Path, *, label: str) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValidationError(f"{label} must be inside the repository: {path}") from exc


def _scenario_id(scenario: Mapping[str, Any], *, label: str) -> str:
    for key in ("name", "scenario_id", "id"):
        value = scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValidationError(f"{label} contains a scenario without a non-empty ID")


def _load_scenario_rows(path: Path, *, label: str) -> list[Mapping[str, Any]]:
    try:
        rows = load_scenarios(path, base_dir=path.parent)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValidationError(f"could not load {label} through scenario loader: {exc}") from exc
    if not rows:
        raise ValidationError(f"{label} resolved to no scenarios: {path}")
    return rows


def _require_exact_dev_seed_list(raw: Any, *, label: str) -> None:
    if not isinstance(raw, list) or any(isinstance(seed, bool) for seed in raw):
        raise ValidationError(f"{label} must be a list of integer seeds")
    if any(not isinstance(seed, int) for seed in raw):
        raise ValidationError(f"{label} must contain only integer seeds")
    if tuple(raw) != EXPECTED_DEV_SEEDS:
        raise ValidationError(f"{label} must be the ordered development seed list 1001–1030")


def _source_scenario_row(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    source_path = ROOT / spec["source_file"]
    source_id = spec["source_id"]
    source_rows = _load_scenario_rows(source_path, label=f"source scenario {source_id}")
    matches = [
        row for row in source_rows if _scenario_id(row, label="source scenario") == source_id
    ]
    if len(matches) != 1:
        raise ValidationError(
            f"source scenario {source_id} must resolve exactly once in {source_path}"
        )
    return matches[0]


def _validate_variant_source(
    scenario_id: str,
    actual: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> None:
    source_id = spec["source_id"]
    source = _source_scenario_row(spec)
    source_simulation = source.get("simulation_config")
    source_metadata = source.get("metadata")
    if not isinstance(source_simulation, Mapping) or not isinstance(source_metadata, Mapping):
        raise ValidationError(f"source scenario {source_id} lacks simulation or metadata")

    expected = dict(source)
    expected["name"] = scenario_id
    expected["seeds"] = list(EXPECTED_DEV_SEEDS)
    expected["simulation_config"] = {
        **source_simulation,
        **RELEASE_PARITY_SIMULATION_OVERRIDES,
        **spec["simulation_overrides"],
    }
    expected["metadata"] = {
        **source_metadata,
        **DEV_METADATA,
        "source_scenario": source_id,
        "development_variant": spec["development_variant"],
    }
    differences = sorted(
        key
        for key in expected.keys() | actual.keys()
        if key not in expected or key not in actual or expected[key] != actual[key]
    )
    if differences:
        raise ValidationError(
            f"{scenario_id} differs from its source outside approved development "
            f"overrides: {differences!r}"
        )


def _validate_scenario_parameters(rows: Sequence[Mapping[str, Any]]) -> None:
    ids = [_scenario_id(row, label="development scenario matrix") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValidationError("development scenario matrix contains duplicate scenario IDs")
    actual_ids = set(ids)
    if actual_ids != EXPECTED_SCENARIO_IDS:
        missing = sorted(EXPECTED_SCENARIO_IDS - actual_ids)
        unexpected = sorted(actual_ids - EXPECTED_SCENARIO_IDS)
        raise ValidationError(
            f"development scenario IDs drifted; missing={missing!r}, unexpected={unexpected!r}"
        )
    if len(rows) != len(EXPECTED_VARIANTS):
        raise ValidationError("development scenario matrix must contain exactly four variants")

    rows_by_id = dict(zip(ids, rows, strict=True))
    for scenario_id, spec in EXPECTED_VARIANTS.items():
        _validate_variant_source(scenario_id, rows_by_id[scenario_id], spec)


def _validate_planner_spec(
    planner: Any,
    *,
    config_path: Path,
    seen: set[str],
) -> None:
    if not isinstance(planner, Mapping):
        raise ValidationError("each campaign planner must be a mapping")
    key = planner.get("key")
    if not isinstance(key, str) or key not in EXPECTED_PLANNER_CONFIGS:
        raise ValidationError(f"planner key is not an approved v4 candidate: {key!r}")
    if key in seen:
        raise ValidationError(f"duplicate planner key: {key}")
    seen.add(key)
    if planner.get("algo") != EXPECTED_PLANNER_ALGO:
        raise ValidationError(f"planner {key} must declare algo: {EXPECTED_PLANNER_ALGO}")
    config_raw = planner.get("algo_config")
    if not isinstance(config_raw, str):
        raise ValidationError(f"planner {key} must declare algo_config")
    resolved = _repo_path(config_raw, relative_to=config_path.parent)
    expected = EXPECTED_PLANNER_CONFIGS[key].resolve()
    if resolved != expected:
        raise ValidationError(f"planner {key} must use existing candidate config {expected}")
    if not expected.is_file():
        raise ValidationError(f"approved v4 candidate config is missing: {expected}")
    _validate_planner_scenario_overrides(expected, key=key)


def _validate_planner_scenario_overrides(config_path: Path, *, key: str) -> None:
    planner_config = _load_mapping(config_path, label=f"planner config {key}")
    scenario_overrides = planner_config.get("scenario_overrides", {})
    if not isinstance(scenario_overrides, Mapping):
        raise ValidationError(f"planner config {key} scenario_overrides must be a mapping")
    overlap = EXPECTED_SCENARIO_IDS & scenario_overrides.keys()
    if overlap:
        raise ValidationError(
            f"planner config {key} applies scenario overrides to development IDs: "
            + ", ".join(sorted(overlap))
        )


def _validate_planner_specs(payload: Mapping[str, Any], *, config_path: Path) -> None:
    planners = payload.get("planners")
    if not isinstance(planners, list):
        raise ValidationError("campaign planners must be a list")
    if len(planners) != len(EXPECTED_PLANNER_CONFIGS):
        raise ValidationError(
            "development campaign must contain exactly the two approved v4 candidates"
        )
    seen: set[str] = set()
    for planner in planners:
        _validate_planner_spec(planner, config_path=config_path, seen=seen)
    if seen != set(EXPECTED_PLANNER_CONFIGS):
        raise ValidationError("development campaign planner roster is incomplete")


def _validate_config_metadata(payload: Mapping[str, Any]) -> None:
    if payload.get("issue") != 9748:
        raise ValidationError("development campaign config must declare issue: 9748")
    if (
        payload.get("development_only") is not True
        or payload.get("not_release_evidence") is not True
    ):
        raise ValidationError("development campaign must be explicitly marked not release evidence")
    if payload.get("evidence_class") != "development_tuning_only":
        raise ValidationError("development campaign evidence_class must be development_tuning_only")


def _validate_config_seed_policy(payload: Mapping[str, Any]) -> None:
    seed_policy = payload.get("seed_policy")
    if not isinstance(seed_policy, Mapping) or seed_policy.get("mode") != "fixed-list":
        raise ValidationError("development campaign must use a fixed-list seed policy")
    _require_exact_dev_seed_list(seed_policy.get("seeds"), label="seed_policy.seeds")
    typed_seed_values: list[int] = []
    for field, raw in _collect_typed_fields(payload, field_names=_SEED_FIELD_NAMES):
        values = _typed_values(raw)
        if not values:
            raise ValidationError(f"campaign seed field {field!r} has no typed integer seeds")
        typed_seed_values.extend(values)
    if set(typed_seed_values) & RELEASE_SEEDS:
        raise ValidationError("development seed policy admits a release seed")
    if set(typed_seed_values) - set(EXPECTED_DEV_SEEDS):
        raise ValidationError("development campaign admits a seed outside 1001–1030")
    _validate_typed_scenario_ids(payload, label="development campaign config")


def _load_development_rows(
    payload: Mapping[str, Any], *, config_path: Path
) -> tuple[Path, list[Mapping[str, Any]]]:
    scenario_raw = payload.get("scenario_matrix")
    if not isinstance(scenario_raw, str) or not scenario_raw.strip():
        raise ValidationError("development campaign must declare scenario_matrix")
    scenario_path = _repo_path(scenario_raw, relative_to=config_path.parent)
    rows = _load_scenario_rows(scenario_path, label="development scenario matrix")
    _validate_scenario_parameters(rows)
    for row in rows:
        scenario_id = _scenario_id(row, label="development scenario matrix")
        _require_exact_dev_seed_list(row.get("seeds"), label=f"{scenario_id}.seeds")
    return scenario_path, rows


def _load_release_candidate_rows(release_matrix_path: Path) -> tuple[Any, list[Mapping[str, Any]]]:
    campaign_path = ROOT / RELEASE_CONFIG_RELATIVE_PATH
    try:
        campaign = load_campaign_config(campaign_path, repository_root=ROOT)
        if campaign.scenario_matrix_path.resolve() != release_matrix_path.resolve():
            raise ValidationError(
                "0.0.8 candidate campaign scenario matrix differs from the parity comparator"
            )
        rows = _load_campaign_scenarios(campaign, repository_root=ROOT)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(f"could not resolve 0.0.8 candidate campaign: {exc}") from exc
    return campaign, rows


def _validate_release_disjointness(release_rows: Sequence[Mapping[str, Any]]) -> None:
    release_ids = {
        _scenario_id(row, label="frozen release scenario matrix") for row in release_rows
    }
    overlap = EXPECTED_SCENARIO_IDS & release_ids
    if overlap:
        raise ValidationError(
            "development scenario IDs overlap the frozen release matrix: "
            + ", ".join(sorted(overlap))
        )


def _validate_release_parity(
    *,
    development_campaign: Any,
    development_rows: Sequence[Mapping[str, Any]],
    release_campaign: Any,
    release_rows: Sequence[Mapping[str, Any]],
) -> None:
    campaign_differences = [
        field
        for field in SIMULATOR_CAMPAIGN_FIELDS
        if getattr(development_campaign, field) != getattr(release_campaign, field)
    ]
    if campaign_differences:
        raise ValidationError(
            "development simulator campaign settings differ from the 0.0.8 candidate: "
            f"{campaign_differences!r}"
        )

    release_by_id = {
        _scenario_id(row, label="0.0.8 candidate scenario matrix"): row for row in release_rows
    }
    development_by_id = {
        _scenario_id(row, label="development scenario matrix"): row for row in development_rows
    }
    for development_id, spec in EXPECTED_VARIANTS.items():
        source_id = spec["source_id"]
        release = release_by_id.get(source_id)
        if release is None:
            raise ValidationError(f"0.0.8 candidate is missing release source {source_id}")
        release_simulation = release.get("simulation_config")
        release_metadata = release.get("metadata")
        if not isinstance(release_simulation, Mapping) or not isinstance(release_metadata, Mapping):
            raise ValidationError(
                f"0.0.8 candidate source {source_id} lacks simulation or metadata"
            )
        expected = dict(release)
        expected["name"] = development_id
        expected["seeds"] = list(EXPECTED_DEV_SEEDS)
        expected["simulation_config"] = {
            **release_simulation,
            **spec["simulation_overrides"],
        }
        expected["metadata"] = {
            **release_metadata,
            **DEV_METADATA,
            "source_scenario": source_id,
            "development_variant": spec["development_variant"],
        }
        actual = development_by_id[development_id]
        differences = sorted(
            key
            for key in expected.keys() | actual.keys()
            if key not in expected or key not in actual or expected[key] != actual[key]
        )
        if differences:
            raise ValidationError(
                f"{development_id} differs from the 0.0.8 candidate outside approved "
                f"development overrides: {differences!r}"
            )


def _load_canonical_campaign_rows(
    config_path: Path,
) -> tuple[Any, list[Mapping[str, Any]]]:
    try:
        loaded = load_campaign_config(config_path, repository_root=ROOT)
        canonical_rows = _load_campaign_scenarios(loaded, repository_root=ROOT)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValidationError(f"campaign config failed canonical loader validation: {exc}") from exc
    return loaded, canonical_rows


def _validate_campaign_config(config_path: Path, release_matrix_path: Path) -> dict[str, Any]:
    payload = _load_mapping(config_path, label="development campaign config")
    _validate_config_metadata(payload)
    _validate_config_seed_policy(payload)
    scenario_path, rows = _load_development_rows(payload, config_path=config_path)
    _validate_planner_specs(payload, config_path=config_path)
    loaded, canonical_rows = _load_canonical_campaign_rows(config_path)
    release_campaign, release_rows = _load_release_candidate_rows(release_matrix_path)
    _validate_release_disjointness(release_rows)
    _validate_release_parity(
        development_campaign=loaded,
        development_rows=canonical_rows,
        release_campaign=release_campaign,
        release_rows=release_rows,
    )
    if tuple(loaded.seed_policy.seeds) != EXPECTED_DEV_SEEDS:
        raise ValidationError("canonical campaign loader did not preserve development seeds")
    resolved_ids = {
        _scenario_id(row, label="canonical campaign scenarios") for row in canonical_rows
    }
    if resolved_ids != EXPECTED_SCENARIO_IDS:
        raise ValidationError("canonical campaign loader resolved an unexpected scenario set")
    return {
        "config": str(config_path),
        "scenario_matrix": str(scenario_path),
        "scenario_count": len(rows),
        "scenario_ids": sorted(EXPECTED_SCENARIO_IDS),
        "seed_count": len(EXPECTED_DEV_SEEDS),
        "seed_range": [EXPECTED_DEV_SEEDS[0], EXPECTED_DEV_SEEDS[-1]],
        "planner_keys": sorted(EXPECTED_PLANNER_CONFIGS),
        "release_scenario_overlap": [],
    }


_SEED_FIELD_NAMES = frozenset(
    {
        "seed",
        "seeds",
        "scenario_seed",
        "scenario_seeds",
        "seed_range",
        "seed_ranges",
        "resolved_seeds",
        "tuning_seeds",
        "release_seed",
        "release_seeds",
        "evaluation_seed",
        "evaluation_seeds",
        "held_out_seed",
        "held_out_seeds",
        "release_seed_range",
        "evaluation_seed_range",
        "held_out_seed_range",
    }
)
_SCENARIO_FIELD_NAMES = frozenset(
    {
        "scenario_id",
        "scenario_ids",
        "scenario_name",
        "scenario_names",
        "scenario",
        "scenarios",
    }
)
_ENTRY_SCENARIO_FIELD_NAMES = frozenset({"scenario_id", "scenario_ids"})
_SCENARIO_NON_IDENTITY_FIELDS = frozenset({"scenario_manifest_sha256"})
_SCENARIO_LIST_FIELDS = frozenset({"scenario_ids", "scenario_names", "scenarios"})


def _typed_values(value: Any) -> list[int]:
    """Collect integer leaves from a typed field without parsing strings."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, Mapping):
        values: list[int] = []
        for nested in value.values():
            values.extend(_typed_values(nested))
        return values
    if isinstance(value, (list, tuple)):
        values: list[int] = []
        for nested in value:
            values.extend(_typed_values(nested))
        return values
    return []


def _require_typed_seed_values(value: Any, *, label: str) -> list[int]:
    """Return seed leaves while rejecting malformed typed seed fields."""
    if isinstance(value, bool):
        raise ValidationError(f"{label} must contain only typed integer seeds")
    if isinstance(value, int):
        return [value]
    if isinstance(value, Mapping):
        if not value:
            raise ValidationError(f"{label} must contain at least one typed integer seed")
        values: list[int] = []
        for nested in value.values():
            values.extend(_require_typed_seed_values(nested, label=label))
        return values
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValidationError(f"{label} must contain at least one typed integer seed")
        values = []
        for nested in value:
            values.extend(_require_typed_seed_values(nested, label=label))
        return values
    raise ValidationError(f"{label} must contain only typed integer seeds")


def _collect_typed_fields(value: Any, *, field_names: frozenset[str]) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in field_names:
                found.append((normalized, nested))
            found.extend(_collect_typed_fields(nested, field_names=field_names))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_collect_typed_fields(nested, field_names=field_names))
    return found


def _reject_unbound_seed_fields(value: Any, path: tuple[str, ...] = ()) -> None:
    """Reject seed-like keys except the canonical direct per-entry ``seeds`` list."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            child_path = (*path, normalized)
            allowed = len(path) == 2 and path[0] == "entries" and normalized == "seeds"
            if "seed" in normalized and not allowed:
                raise ValidationError(
                    "unsupported seed field "
                    f"{'.'.join(child_path)!r}; bind seeds only in entries[i].seeds"
                )
            _reject_unbound_seed_fields(nested, child_path)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_unbound_seed_fields(nested, (*path, str(index)))


def _entry_seed_values(entry: Any, *, index: int) -> list[int]:
    """Read one entry's canonical typed seed list."""
    if not isinstance(entry, Mapping):
        raise ValidationError(f"structured tuning-log entry {index} must be a mapping")
    if "seeds" not in entry:
        raise ValidationError(
            f"structured tuning-log entry {index} must contain a non-empty seeds list"
        )
    raw = entry["seeds"]
    if not isinstance(raw, list) or not raw:
        raise ValidationError(
            f"structured tuning-log entry {index} seeds must be a non-empty list of typed integer seeds"
        )
    if any(type(seed) is not int for seed in raw):
        raise ValidationError(
            f"structured tuning-log entry {index} seeds must contain only typed integer seeds"
        )
    return raw


def _validate_tuning_log_seeds(payload: Mapping[str, Any]) -> int:
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValidationError("structured tuning log entries must be a list")

    _reject_unbound_seed_fields(payload)
    seed_values = [
        seed
        for index, entry in enumerate(entries)
        for seed in _entry_seed_values(entry, index=index)
    ]
    if not seed_values:
        raise ValidationError("structured tuning log must contain at least one typed seed field")
    release_seed_values = sorted(set(seed_values) & RELEASE_SEEDS)
    if release_seed_values:
        raise ValidationError(
            "structured tuning log admits held-out release seeds: "
            + ", ".join(str(seed) for seed in release_seed_values)
        )
    invalid_seeds = sorted(set(seed_values) - set(EXPECTED_DEV_SEEDS))
    if invalid_seeds:
        raise ValidationError(
            "structured tuning log admits seeds outside the approved development set: "
            + ", ".join(str(seed) for seed in invalid_seeds)
        )
    return len(seed_values)


def _typed_scenario_values(field: str, raw: Any) -> list[str]:
    expected_type = list if field in _SCENARIO_LIST_FIELDS else str
    if not isinstance(raw, expected_type):
        type_name = "a list of scenario IDs" if expected_type is list else "a scenario ID string"
        raise ValidationError(f"structured field {field!r} must contain {type_name}")
    values = raw if isinstance(raw, list) else [raw]
    if not values:
        raise ValidationError(f"structured field {field!r} must contain at least one scenario ID")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValidationError(f"structured field {field!r} contains invalid scenario IDs")
    return [value.strip() for value in values]


def _validate_typed_scenario_ids(payload: Mapping[str, Any], *, label: str) -> int:
    scenario_ids = [
        scenario_id
        for field, raw in _collect_typed_fields(payload, field_names=_SCENARIO_FIELD_NAMES)
        for scenario_id in _typed_scenario_values(field, raw)
    ]
    unexpected = sorted(set(scenario_ids) - EXPECTED_SCENARIO_IDS)
    if unexpected:
        raise ValidationError(
            f"{label} admits scenario IDs outside the development set: {unexpected!r}"
        )
    return len(scenario_ids)


def _validate_tuning_log_scenarios(
    payload: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> int:
    for index, entry in enumerate(entries):
        direct_fields = [
            (field, entry[field]) for field in _ENTRY_SCENARIO_FIELD_NAMES if field in entry
        ]
        if len(direct_fields) > 1:
            raise ValidationError(
                f"structured tuning-log entry {index} must use exactly one direct scenario_id field"
            )
        entry_values = [
            scenario_id
            for field, raw in direct_fields
            for scenario_id in _typed_scenario_values(field, raw)
        ]
        if not entry_values:
            raise ValidationError(
                f"structured tuning-log entry {index} must contain a non-empty typed "
                "scenario_id or scenario_ids field"
            )
    _reject_unbound_scenario_fields(payload)
    return _validate_typed_scenario_ids(payload, label="structured tuning log")


def _reject_unbound_scenario_fields(value: Any, path: tuple[str, ...] = ()) -> None:
    """Reject scenario identity aliases outside the tuning-log schema."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            child_path = (*path, normalized)
            if (
                "scenario" in normalized
                and normalized not in _ENTRY_SCENARIO_FIELD_NAMES
                and normalized not in _SCENARIO_NON_IDENTITY_FIELDS
            ):
                raise ValidationError(
                    "unsupported scenario field "
                    f"{'.'.join(child_path)!r}; bind identities only as scenario_id or scenario_ids"
                )
            _reject_unbound_scenario_fields(nested, child_path)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_unbound_scenario_fields(nested, (*path, str(index)))


def _validate_candidate_provenance(raw: Any) -> dict[str, dict[str, str]]:
    if not isinstance(raw, Mapping):
        raise ValidationError("provenance.candidate_configs must be a mapping")
    expected_keys = set(EXPECTED_PLANNER_CONFIGS)
    actual_keys = set(raw)
    if actual_keys != expected_keys:
        missing_candidates = sorted(str(key) for key in expected_keys - actual_keys)
        unknown_candidates = sorted(str(key) for key in actual_keys - expected_keys)
        raise ValidationError(
            "provenance.candidate_configs must map exactly the approved candidates; "
            f"missing={missing_candidates!r}, unknown={unknown_candidates!r}"
        )

    normalized_candidates: dict[str, dict[str, str]] = {}
    for key, expected_path in EXPECTED_PLANNER_CONFIGS.items():
        record = raw[key]
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValidationError(
                f"provenance.candidate_configs[{key!r}] must contain exactly path and sha256"
            )
        expected_path_string = _repo_relative_path(expected_path, label=f"candidate {key}")
        if record["path"] != expected_path_string:
            raise ValidationError(f"provenance candidate {key} must name {expected_path_string!r}")
        recorded_hash = _require_sha256(
            record["sha256"],
            label=f"provenance.candidate_configs[{key!r}].sha256",
        )
        working_hash = _sha256(expected_path, label=f"candidate config {key}")
        if recorded_hash != working_hash:
            raise ValidationError(
                f"provenance candidate config hash for {key} does not match the working "
                f"tree file ({working_hash})"
            )
        expected_hash = _commit_sha256("HEAD", expected_path, label=f"candidate config {key}")
        if recorded_hash != expected_hash:
            raise ValidationError(
                f"provenance candidate config hash for {key} does not match the current HEAD "
                f"tracked file ({expected_hash})"
            )
        normalized_candidates[key] = {
            "path": expected_path_string,
            "sha256": recorded_hash,
        }
    return normalized_candidates


def _validate_frozen_tuning_inputs(
    source_commit: str,
    *,
    config_path: Path,
    scenario_matrix_path: Path,
) -> None:
    """Require every transitive campaign input to match its frozen source bytes."""
    input_paths = _tuning_input_paths(
        config_path=config_path,
        scenario_matrix_path=scenario_matrix_path,
        candidate_paths=EXPECTED_PLANNER_CONFIGS,
    )
    for input_path in input_paths:
        input_label = input_path.relative_to(ROOT).as_posix()
        working_input_hash = _sha256(input_path, label=f"transitive tuning input {input_label}")
        current_input_hash = _commit_sha256(
            "HEAD", input_path, label=f"transitive tuning input {input_label}"
        )
        frozen_input_hash = _commit_sha256(
            source_commit, input_path, label=f"transitive tuning input {input_label}"
        )
        if current_input_hash != frozen_input_hash:
            raise ValidationError(
                f"transitive tuning input {input_label} differs from frozen source "
                f"commit {source_commit}"
            )
        if working_input_hash != frozen_input_hash:
            raise ValidationError(
                f"transitive tuning input {input_label} differs from frozen source "
                f"commit {source_commit} in the working tree"
            )


def _validate_frozen_provenance(
    payload: Mapping[str, Any],
    *,
    config_path: Path,
    scenario_matrix_path: Path,
) -> dict[str, Any]:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValidationError("structured tuning log must declare a frozen provenance mapping")

    required = {
        "source_commit",
        "campaign_config_sha256",
        "scenario_manifest_sha256",
        "candidate_configs",
    }
    missing = sorted(required - provenance.keys())
    if missing:
        raise ValidationError(
            "structured tuning log is missing frozen provenance fields: " + ", ".join(missing)
        )

    source_commit = _require_source_commit(provenance["source_commit"])
    campaign_hash = _require_sha256(
        provenance["campaign_config_sha256"],
        label="provenance.campaign_config_sha256",
    )
    _require_working_hash(campaign_hash, config_path, label="provenance.campaign_config_sha256")
    expected_campaign_hash = _commit_sha256(
        "HEAD", config_path, label="development campaign config"
    )
    if campaign_hash != expected_campaign_hash:
        raise ValidationError(
            "provenance.campaign_config_sha256 does not match the current HEAD development "
            f"campaign config ({expected_campaign_hash})"
        )
    frozen_campaign_hash = _commit_sha256(
        source_commit, config_path, label="development campaign config"
    )
    if campaign_hash != frozen_campaign_hash:
        raise ValidationError(
            "provenance.campaign_config_sha256 does not match the development campaign config "
            f"at frozen source commit {source_commit} ({frozen_campaign_hash})"
        )

    scenario_hash = _require_sha256(
        provenance["scenario_manifest_sha256"],
        label="provenance.scenario_manifest_sha256",
    )
    _require_working_hash(
        scenario_hash, scenario_matrix_path, label="provenance.scenario_manifest_sha256"
    )
    expected_scenario_hash = _commit_sha256(
        "HEAD", scenario_matrix_path, label="development scenario manifest"
    )
    if scenario_hash != expected_scenario_hash:
        raise ValidationError(
            "provenance.scenario_manifest_sha256 does not match the current HEAD development "
            f"scenario manifest ({expected_scenario_hash})"
        )
    frozen_scenario_hash = _commit_sha256(
        source_commit, scenario_matrix_path, label="development scenario manifest"
    )
    if scenario_hash != frozen_scenario_hash:
        raise ValidationError(
            "provenance.scenario_manifest_sha256 does not match the development scenario "
            f"manifest at frozen source commit {source_commit} ({frozen_scenario_hash})"
        )

    normalized_candidates = _validate_candidate_provenance(provenance["candidate_configs"])
    for key, candidate_path in EXPECTED_PLANNER_CONFIGS.items():
        frozen_candidate_hash = _commit_sha256(
            source_commit, candidate_path, label=f"candidate config {key}"
        )
        if normalized_candidates[key]["sha256"] != frozen_candidate_hash:
            raise ValidationError(
                f"provenance candidate config hash for {key} does not match frozen source "
                f"commit {source_commit} ({frozen_candidate_hash})"
            )

    _validate_frozen_tuning_inputs(
        source_commit,
        config_path=config_path,
        scenario_matrix_path=scenario_matrix_path,
    )

    return {
        "source_commit": source_commit,
        "campaign_config_sha256": campaign_hash,
        "scenario_manifest_sha256": scenario_hash,
        "candidate_configs": normalized_candidates,
    }


def _validate_tuning_log_candidates(
    entries: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]
) -> None:
    candidate_configs = provenance["candidate_configs"]
    for index, entry in enumerate(entries):
        candidate = entry.get("candidate")
        if not isinstance(candidate, str) or candidate not in EXPECTED_PLANNER_CONFIGS:
            raise ValidationError(
                f"structured tuning-log entry {index} must name an approved candidate"
            )
        candidate_hash = entry.get("candidate_config_sha256")
        expected_hash = candidate_configs[candidate]["sha256"]
        if candidate_hash != expected_hash:
            raise ValidationError(
                f"structured tuning-log entry {index} has an unknown or mismatched "
                f"candidate config hash for {candidate}"
            )


def _validate_tuning_log(
    path: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    scenario_matrix_path: Path | None = None,
) -> dict[str, Any]:
    payload = _load_mapping(path, label="structured tuning log")
    if payload.get("schema_version") != TUNING_LOG_SCHEMA:
        raise ValidationError(
            f"structured tuning log must declare schema_version: {TUNING_LOG_SCHEMA}"
        )
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValidationError("structured tuning log entries must be a non-empty list")
    if any(not isinstance(entry, Mapping) for entry in entries):
        raise ValidationError("structured tuning log entries must be mappings")
    config_path = config_path.resolve()
    if scenario_matrix_path is None:
        config_payload = _load_mapping(config_path, label="development campaign config")
        scenario_raw = config_payload.get("scenario_matrix")
        if not isinstance(scenario_raw, str) or not scenario_raw.strip():
            raise ValidationError("development campaign must declare scenario_matrix")
        scenario_matrix_path = _repo_path(scenario_raw, relative_to=config_path.parent)
    provenance = _validate_frozen_provenance(
        payload,
        config_path=config_path,
        scenario_matrix_path=scenario_matrix_path.resolve(),
    )
    typed_seed_count = _validate_tuning_log_seeds(payload)
    typed_scenario_count = _validate_tuning_log_scenarios(payload, entries)
    _validate_tuning_log_candidates(entries, provenance)
    log_commit = _require_committed_log(path, source_commit=provenance["source_commit"])
    provenance["log_commit"] = log_commit

    return {
        "path": str(path),
        "schema_version": TUNING_LOG_SCHEMA,
        "entry_count": len(entries),
        "typed_seed_count": typed_seed_count,
        "typed_scenario_count": typed_scenario_count,
        "release_seed_overlap": [],
        "release_scenario_overlap": [],
        "provenance": provenance,
    }


def validate(
    config_path: Path = DEFAULT_CONFIG,
    *,
    release_matrix_path: Path = DEFAULT_RELEASE_MATRIX,
    tuning_logs: Sequence[Path] = (),
) -> dict[str, Any]:
    """Validate the checked-in development split and optional tuning logs."""
    config_path = config_path.resolve()
    summary = _validate_campaign_config(config_path, release_matrix_path.resolve())
    scenario_matrix_path = Path(summary["scenario_matrix"])
    summary["tuning_logs"] = [
        _validate_tuning_log(
            path.resolve(),
            config_path=config_path,
            scenario_matrix_path=scenario_matrix_path,
        )
        for path in tuning_logs
    ]
    summary["ok"] = True
    summary["status"] = "development_protocol_validated_not_benchmark_evidence"
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release-matrix", type=Path, default=DEFAULT_RELEASE_MATRIX)
    parser.add_argument(
        "--tuning-log",
        type=Path,
        action="append",
        default=[],
        help=f"Structured YAML/JSON log using schema {TUNING_LOG_SCHEMA}; repeatable.",
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the issue #9748 validator and return a process exit code."""
    args = _parser().parse_args(argv)
    try:
        summary = validate(
            args.config,
            release_matrix_path=args.release_matrix,
            tuning_logs=args.tuning_log,
        )
    except ValidationError as exc:
        error = {"ok": False, "status": "invalid", "error": str(exc)}
        print(json.dumps(error, sort_keys=True) if args.json else f"INVALID: {exc}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True) if args.json else "VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
