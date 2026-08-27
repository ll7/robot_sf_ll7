"""Benchmark release protocol helpers built on top of camera-ready campaigns."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.benchmark.camera_ready._preflight import _resolved_seed_inventory
from robot_sf.benchmark.camera_ready_campaign import CampaignConfig, load_campaign_config
from robot_sf.benchmark.effective_algorithm_branches import WITNESS_KINDS
from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256_file
from robot_sf.benchmark.release_tag_identity import (
    HISTORICAL_RELEASE_TAG,
)
from robot_sf.common.artifact_paths import get_repository_root

RELEASE_MANIFEST_SCHEMA_VERSION = "benchmark-release-manifest.v0.1"
RELEASE_MANIFEST_SCHEMA_VERSION_V0_2 = "benchmark-release-manifest.v0.2"
SUPPORTED_RELEASE_MANIFEST_SCHEMA_VERSIONS = frozenset(
    {RELEASE_MANIFEST_SCHEMA_VERSION, RELEASE_MANIFEST_SCHEMA_VERSION_V0_2}
)
BENCHMARK_PROTOCOL_VERSION = "0.1.0"
DIAGNOSTIC_RELEASE_MATURITY = "diagnostic"
# These are historical benchmark/software concepts.  A benchmark-data release
# must reserve a new concept rather than append another version to either one.
HISTORICAL_ZENODO_CONCEPT_DOIS = frozenset(
    {
        "10.5281/zenodo.19482025",
        "10.5281/zenodo.19563812",
    }
)
_SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DIAGNOSTIC_STRESS_RELEASE_KIND = "benchmark-stress-smoke"
STRESS_SMOKE_CONTRACT_SCHEMA_VERSION = "hybrid-release-stress-smoke.v1"
STRESS_SMOKE_SOURCE_POLICY = "exact-immutable-worktree-sha-required"
STRESS_SMOKE_EXPECTED_PLANNER_ARMS = 14
STRESS_SMOKE_EXPECTED_SCENARIO_COUNT = 5
STRESS_SMOKE_EXPECTED_SEED = 116
STRESS_SMOKE_EXPECTED_EPISODE_CELLS = 70
STRESS_SMOKE_EXPECTED_HORIZON_STEPS = 600
STRESS_SMOKE_EXPECTED_DT = 0.1
STRESS_SMOKE_EXPECTED_KINEMATICS = "differential_drive"
STRESS_SMOKE_EXPECTED_SCENARIO_IDS = (
    "classic_urban_crossing_medium",
    "classic_cross_trap_high",
    "classic_doorway_high",
    "francis2023_exiting_elevator",
    "francis2023_robot_crowding",
)


def _load_mapping(path: Path) -> dict[str, Any]:
    """Load a JSON or YAML mapping from disk.

    Returns:
        Parsed mapping payload.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping payload in {path}")
    return payload


def _repo_relative(path: Path, repository_root: Path | None = None) -> str:
    """Return a stable repository-relative path string when possible."""
    resolved = path.resolve()
    repo_root = (repository_root or get_repository_root()).resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_manifest_side_path(manifest_path: Path, value: Any) -> Path:
    """Build a manifest-relative path candidate without following links.

    Returns:
        Absolute or manifest-relative path candidate for later validation.
    """
    candidate = Path(str(value))
    return candidate if candidate.is_absolute() else manifest_path.parent / candidate


def _has_symlink_component(path: Path) -> bool:
    """Return whether a lexical path contains a symlink component."""
    lexical = Path(path.absolute())
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


@dataclass(frozen=True)
class StressSmokeAssetPin:
    """One source asset pinned by the diagnostic hybrid stress contract."""

    path: Path
    sha256: str
    planner_key: str | None = None


@dataclass(frozen=True)
class StressSmokeBranchWitness:
    """One manifest-declared witness for an effective algorithm branch."""

    kind: str
    arm: str
    scenario: str
    algorithm: str
    branch_key: str
    config_path: Path
    config_sha256: str


@dataclass(frozen=True)
class BenchmarkReleaseManifest:
    """Canonical release manifest for benchmark publication workflows."""

    path: Path
    schema_version: str
    benchmark_protocol_version: str
    release_id: str
    release_tag: str
    maturity: str
    canonical_campaign_config_path: Path
    expected_paper_profile_version: str | None
    expected_paper_interpretation_profile: str | None
    expected_kinematics_matrix: tuple[str, ...]
    expected_holonomic_command_mode: str | None
    scenario_matrix_path: Path
    scenario_matrix_sha256: str
    campaign_config_sha256: str
    seed_policy: dict[str, Any]
    snqi_weights_path: Path | None
    snqi_weights_sha256: str | None
    snqi_baseline_path: Path | None
    snqi_baseline_sha256: str | None
    planner_keys: tuple[str, ...]
    planner_groups: dict[str, str]
    required_artifact_paths: tuple[str, ...]
    repository_url: str
    doi: str
    citation_path: Path
    release_checklist_path: Path
    latest_main_base_commit: str | None = None
    expected_episode_cells: int | None = None
    expected_horizon_steps: int | None = None
    publication_channel: str | None = None
    suite_policy_path: Path | None = None
    suite_policy_sha256: str | None = None
    route_certification_path: Path | None = None
    route_certification_sha256: str | None = None
    seed_sets_sha256: str | None = None
    resolved_seeds: tuple[int, ...] = ()
    snqi_claim_policy: str | None = None
    concept_doi: str | None = None
    version_doi: str | None = None
    release_kind: str | None = None
    source_sha: str | None = None
    planning_base_sha: str | None = None
    metadata_path: Path | None = None
    metadata_sha256: str | None = None
    stress_smoke_review_base_commit: str | None = None
    stress_smoke_source_policy: str | None = None
    stress_smoke_expected_episode_cells: int | None = None
    stress_smoke_expected_horizon_steps: int | None = None
    stress_smoke_expected_dt: float | None = None
    stress_smoke_expected_kinematics: str | None = None
    stress_smoke_required_hybrid_arms: tuple[str, ...] = ()
    stress_smoke_suite_policy_path: Path | None = None
    stress_smoke_suite_policy_sha256: str | None = None
    stress_smoke_seed_sets_path: Path | None = None
    stress_smoke_seed_sets_sha256: str | None = None
    stress_smoke_route_certification_path: Path | None = None
    stress_smoke_route_certification_sha256: str | None = None
    stress_smoke_scenario_source_pins: tuple[StressSmokeAssetPin, ...] = ()
    stress_smoke_hybrid_config_pins: tuple[StressSmokeAssetPin, ...] = ()
    stress_smoke_branch_witnesses: tuple[StressSmokeBranchWitness, ...] = ()


def _resolve_required_file(
    manifest_path: Path,
    value: Any,
    field_name: str,
    *,
    repository_root: Path | None = None,
) -> Path:
    """Resolve and validate a required manifest-relative file path.

    Returns:
        Existing absolute file path.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty path string")
    candidate = _resolve_manifest_side_path(manifest_path, value.strip())
    if _has_symlink_component(candidate):
        raise ValueError(f"{field_name} must not contain symlink components: {candidate}")
    resolved = candidate.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{field_name} not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{field_name} must be a file path, got non-file path: {resolved}")
    root = (repository_root or get_repository_root()).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be contained by the repository checkout") from exc
    return resolved


def _resolve_stress_contract_file(
    manifest_path: Path,
    value: Any,
    field_name: str,
    *,
    repository_root: Path | None = None,
) -> Path:
    """Resolve one stress asset inside this checkout, rejecting symlink escapes.

    Returns:
        Existing repository-contained asset path.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty path string")
    candidate = Path(value.strip())
    candidate = candidate if candidate.is_absolute() else manifest_path.parent / candidate
    if _has_symlink_component(candidate):
        raise ValueError(f"{field_name} must not contain symlink components: {candidate}")
    resolved = candidate.resolve()
    repository_root = (repository_root or get_repository_root()).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{field_name} not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{field_name} must be a file path, got non-file path: {resolved}")
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be contained by the repository checkout") from exc

    return resolved


def _load_manifest_identity(payload: dict[str, Any]) -> dict[str, str]:
    """Load the core release identity fields from a manifest payload.

    Returns:
        Mapping with validated schema/version/identity values.
    """
    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version not in SUPPORTED_RELEASE_MANIFEST_SCHEMA_VERSIONS:
        raise ValueError(
            "schema_version must be one of "
            f"{sorted(SUPPORTED_RELEASE_MANIFEST_SCHEMA_VERSIONS)}, got {schema_version!r}"
        )

    protocol_version = str(payload.get("benchmark_protocol_version", "")).strip()
    if protocol_version != BENCHMARK_PROTOCOL_VERSION:
        raise ValueError(
            "benchmark_protocol_version must match the supported protocol "
            f"{BENCHMARK_PROTOCOL_VERSION}, got {protocol_version!r}"
        )
    if _SEMVER_RE.fullmatch(protocol_version) is None:
        raise ValueError("benchmark_protocol_version must be a semantic version string")

    release_id = str(payload.get("release_id", "")).strip()
    if not release_id:
        raise ValueError("release_id must be a non-empty string")

    release_tag = str(payload.get("release_tag", "")).strip()
    if not release_tag:
        raise ValueError("release_tag must be a non-empty string")

    return {
        "schema_version": schema_version,
        "benchmark_protocol_version": protocol_version,
        "release_id": release_id,
        "release_tag": release_tag,
        "maturity": str(payload.get("maturity", "pre-1.0")).strip() or "pre-1.0",
    }


def _load_manifest_scenario_section(
    manifest_path: Path,
    payload: dict[str, Any],
    *,
    repository_root: Path | None = None,
) -> tuple[Path, str]:
    """Load and validate the scenario section.

    Returns:
        Scenario matrix path and expected SHA-256.
    """
    scenario = payload.get("scenario")
    if not isinstance(scenario, dict):
        raise ValueError("scenario must be a mapping")
    scenario_matrix_path = _resolve_required_file(
        manifest_path,
        scenario.get("matrix_path"),
        "scenario.matrix_path",
        repository_root=repository_root,
    )
    scenario_matrix_sha256 = str(scenario.get("matrix_sha256", "")).strip()
    if not scenario_matrix_sha256:
        raise ValueError("scenario.matrix_sha256 must be a non-empty string")
    return scenario_matrix_path, scenario_matrix_sha256


def _load_manifest_metrics_section(
    manifest_path: Path,
    payload: dict[str, Any],
    *,
    repository_root: Path | None = None,
) -> dict[str, Path | str | None]:
    """Load and validate the metrics section.

    Returns:
        Metrics subsection payload with resolved paths and hashes.
    """
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    snqi_weights_path = (
        _resolve_required_file(
            manifest_path,
            metrics.get("snqi_weights_path"),
            "metrics.snqi_weights_path",
            repository_root=repository_root,
        )
        if metrics.get("snqi_weights_path") is not None
        else None
    )
    snqi_baseline_path = (
        _resolve_required_file(
            manifest_path,
            metrics.get("snqi_baseline_path"),
            "metrics.snqi_baseline_path",
            repository_root=repository_root,
        )
        if metrics.get("snqi_baseline_path") is not None
        else None
    )
    snqi_weights_sha256 = (
        str(metrics.get("snqi_weights_sha256", "")).strip()
        if snqi_weights_path is not None
        else None
    )
    snqi_baseline_sha256 = (
        str(metrics.get("snqi_baseline_sha256", "")).strip()
        if snqi_baseline_path is not None
        else None
    )
    if snqi_weights_path is not None and not snqi_weights_sha256:
        raise ValueError("metrics.snqi_weights_sha256 must be set when snqi_weights_path is set")
    if snqi_baseline_path is not None and not snqi_baseline_sha256:
        raise ValueError("metrics.snqi_baseline_sha256 must be set when snqi_baseline_path is set")
    return {
        "snqi_weights_path": snqi_weights_path,
        "snqi_weights_sha256": snqi_weights_sha256,
        "snqi_baseline_path": snqi_baseline_path,
        "snqi_baseline_sha256": snqi_baseline_sha256,
    }


def _load_manifest_planner_section(
    payload: dict[str, Any],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Load and validate the planners section.

    Returns:
        Planner keys and planner-group mapping.
    """
    planners = payload.get("planners")
    if not isinstance(planners, dict):
        raise ValueError("planners must be a mapping")
    keys_raw = planners.get("keys")
    if not isinstance(keys_raw, list) or not keys_raw:
        raise ValueError("planners.keys must be a non-empty list")
    planner_keys = tuple(str(item).strip() for item in keys_raw if str(item).strip())
    if len(planner_keys) != len(keys_raw):
        raise ValueError("planners.keys must not contain empty values")
    planner_groups_raw = planners.get("groups")
    if not isinstance(planner_groups_raw, dict):
        raise ValueError("planners.groups must be a mapping")
    return planner_keys, {str(key): str(value) for key, value in planner_groups_raw.items()}


def _load_manifest_kinematics_section(
    payload: dict[str, Any],
) -> tuple[tuple[str, ...], str | None]:
    """Load and validate the kinematics section.

    Returns:
        Expected kinematics matrix and optional holonomic command mode.
    """
    kinematics = payload.get("kinematics")
    if not isinstance(kinematics, dict):
        raise ValueError("kinematics must be a mapping")
    matrix_raw = kinematics.get("matrix")
    if not isinstance(matrix_raw, list) or not matrix_raw:
        raise ValueError("kinematics.matrix must be a non-empty list")
    expected_kinematics_matrix = tuple(
        str(item).strip() for item in matrix_raw if str(item).strip()
    )
    expected_holonomic_command_mode = kinematics.get("holonomic_command_mode")
    if expected_holonomic_command_mode is not None:
        expected_holonomic_command_mode = str(expected_holonomic_command_mode)
    return expected_kinematics_matrix, expected_holonomic_command_mode


def _load_manifest_artifacts_section(payload: dict[str, Any]) -> tuple[str, ...]:
    """Load and validate the artifacts section.

    Returns:
        Required artifact path tuple.
    """
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be a mapping")
    required_artifact_paths_raw = artifacts.get("required_paths")
    if not isinstance(required_artifact_paths_raw, list) or not required_artifact_paths_raw:
        raise ValueError("artifacts.required_paths must be a non-empty list")
    required_artifact_paths: list[str] = []
    for index, item in enumerate(required_artifact_paths_raw):
        if isinstance(item, str) and not item.strip():
            raise ValueError("artifacts.required_paths must not contain empty values")
        if not isinstance(item, str):
            raise ValueError("artifacts.required_paths must contain non-empty path strings")
        path = item.strip()
        candidate = Path(path)
        if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
            raise ValueError(
                f"artifacts.required_paths[{index}] must be campaign-relative without parent traversal"
            )
        required_artifact_paths.append(path)
    return tuple(required_artifact_paths)


def resolve_campaign_artifact_path(campaign_root: Path, raw_path: str) -> Path:
    """Resolve one required artifact as a contained, regular campaign file.

    Required artifact declarations are campaign-relative.  This check is used
    immediately before reading or writing release-owned artifacts so an
    absolute path, traversal, symlink component, outside resolution, directory,
    or missing file cannot be mistaken for a valid publication input.

    Returns:
        Existing regular file under ``campaign_root``.
    """
    root = Path(campaign_root).absolute()
    if _has_symlink_component(root):
        raise ValueError("campaign root must not contain symlink components")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"campaign root is not a directory: {root}")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("campaign artifact path must be a non-empty string")
    candidate_raw = Path(raw_path.strip())
    if candidate_raw.is_absolute():
        raise ValueError("campaign artifact path must be campaign-relative")
    if any(part == ".." for part in candidate_raw.parts):
        raise ValueError("campaign artifact path may not contain parent traversal")
    candidate = root / candidate_raw
    if _has_symlink_component(candidate):
        raise ValueError("campaign artifact path must not contain symlink components")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("campaign artifact path resolves outside campaign root") from exc
    if not resolved.is_file():
        raise ValueError(f"campaign artifact path is not a regular file: {resolved}")
    return resolved


def resolve_regular_directory_path(path: Path, *, field_name: str) -> Path:
    """Resolve one directory only after rejecting lexical symlink components.

    Returns:
        Existing regular directory with its resolved absolute path.
    """
    candidate = Path(path).absolute()
    if _has_symlink_component(candidate):
        raise ValueError(f"{field_name} must not contain symlink components")
    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{field_name} must be a regular directory: {resolved}")
    return resolved


def _load_manifest_provenance_section(payload: dict[str, Any]) -> tuple[str, str]:
    """Load and validate the provenance section.

    Returns:
        Repository URL and DOI placeholder/value.
    """
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a mapping")
    repository_url = str(provenance.get("repository_url", "")).strip()
    if not repository_url:
        raise ValueError("provenance.repository_url must be a non-empty string")
    doi = str(provenance.get("doi", "")).strip()
    if not doi:
        raise ValueError("provenance.doi must be a non-empty string")
    return repository_url, doi


def _load_manifest_release_metadata(payload: dict[str, Any]) -> dict[str, str | None]:
    """Load optional release metadata fields from the manifest payload.

    Returns:
        Mapping with normalized optional release metadata strings.
    """
    return {
        "expected_paper_profile_version": (
            str(payload.get("expected_paper_profile_version")).strip()
            if payload.get("expected_paper_profile_version") is not None
            else None
        ),
        "expected_paper_interpretation_profile": (
            str(payload.get("expected_paper_interpretation_profile")).strip()
            if payload.get("expected_paper_interpretation_profile") is not None
            else None
        ),
        "campaign_config_sha256": str(payload.get("campaign_config_sha256", "")).strip(),
        "release_kind": (
            str(payload.get("release_kind")).strip()
            if payload.get("release_kind") is not None
            else None
        ),
    }


def _load_stress_smoke_branch_witnesses(  # noqa: C901
    manifest_path: Path,
    raw_witnesses: Any,
    hybrid_config_pins: tuple[StressSmokeAssetPin, ...],
    *,
    repository_root: Path | None = None,
) -> tuple[StressSmokeBranchWitness, ...]:
    """Load exact branch witnesses and bind them to pinned candidate configs.

    Returns:
        Normalized witness records bound to the pinned candidate config assets.
    """
    if not isinstance(raw_witnesses, list) or not raw_witnesses:
        raise ValueError("stress_smoke_contract.branch_witnesses must be a non-empty list")
    pins_by_path = {pin.path.resolve(): pin for pin in hybrid_config_pins}
    witnesses: list[StressSmokeBranchWitness] = []
    seen_branch_keys: set[str] = set()
    for index, raw_witness in enumerate(raw_witnesses):
        if not isinstance(raw_witness, Mapping):
            raise ValueError(f"stress_smoke_contract.branch_witnesses[{index}] must be a mapping")
        kind = raw_witness.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError(
                f"stress_smoke_contract.branch_witnesses[{index}].kind must be a non-empty string"
            )
        kind = kind.strip()
        if kind not in WITNESS_KINDS:
            raise ValueError(
                f"stress_smoke_contract.branch_witnesses[{index}].kind has unsupported value "
                f"{kind!r}"
            )
        branch_key = raw_witness.get("branch_key")
        if not isinstance(branch_key, str) or not branch_key.strip():
            raise ValueError(
                f"stress_smoke_contract.branch_witnesses[{index}].branch_key must be a non-empty string"
            )
        branch_key = branch_key.strip()
        parts = tuple(part.strip() for part in branch_key.split("|"))
        if len(parts) != 3 or any(not part for part in parts):
            raise ValueError(
                f"stress_smoke_contract.branch_witnesses[{index}].branch_key must be arm|scenario|algorithm"
            )
        arm, scenario, algorithm = parts
        for field_name, expected in (
            ("arm", arm),
            ("scenario", scenario),
            ("algorithm", algorithm),
        ):
            value = raw_witness.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"stress_smoke_contract.branch_witnesses[{index}].{field_name} "
                    "must be a non-empty string"
                )
            if value.strip() != expected:
                raise ValueError(
                    f"stress_smoke_contract.branch_witnesses[{index}] {field_name} "
                    "does not match branch_key"
                )
        if branch_key in seen_branch_keys:
            raise ValueError(
                "stress_smoke_contract.branch_witnesses contains duplicate branch keys"
            )
        seen_branch_keys.add(branch_key)

        config_path = _resolve_stress_contract_file(
            manifest_path,
            raw_witness.get("config_path"),
            f"stress_smoke_contract.branch_witnesses[{index}].config_path",
            repository_root=repository_root,
        )
        config_sha256 = str(raw_witness.get("config_sha256", "")).strip().lower()
        if _SHA256_RE.fullmatch(config_sha256) is None:
            raise ValueError(
                f"stress_smoke_contract.branch_witnesses[{index}].config_sha256 "
                "must be a 64-character SHA-256"
            )
        pinned_config = pins_by_path.get(config_path.resolve())
        if pinned_config is None:
            raise ValueError(
                f"stress_smoke_contract.branch_witnesses[{index}].config_path "
                "must match a pinned hybrid config"
            )
        if config_sha256 != pinned_config.sha256:
            raise ValueError(
                f"stress_smoke_contract.branch_witnesses[{index}].config_sha256 "
                "does not match its pinned hybrid config"
            )
        witnesses.append(
            StressSmokeBranchWitness(
                kind=kind,
                arm=arm,
                scenario=scenario,
                algorithm=algorithm,
                branch_key=branch_key,
                config_path=config_path,
                config_sha256=config_sha256,
            )
        )
    return tuple(witnesses)


def _load_stress_smoke_contract(  # noqa: C901, PLR0912, PLR0915
    manifest_path: Path,
    payload: dict[str, Any],
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Load the diagnostic stress contract without pretending to pin its own commit.

    A tracked manifest cannot name the final commit that contains the manifest: doing so
    would create a self-referential hash loop.  The stress contract therefore records the
    review/base commit for audit context, while the runner and launch packet bind the exact
    checked-out runtime commit and campaign provenance.

    Returns:
        Normalized stress-contract fields, or compatibility defaults for other releases.
    """
    defaults: dict[str, Any] = {
        "stress_smoke_review_base_commit": None,
        "stress_smoke_source_policy": None,
        "stress_smoke_expected_episode_cells": None,
        "stress_smoke_expected_horizon_steps": None,
        "stress_smoke_expected_dt": None,
        "stress_smoke_expected_kinematics": None,
        "stress_smoke_required_hybrid_arms": (),
        "stress_smoke_suite_policy_path": None,
        "stress_smoke_suite_policy_sha256": None,
        "stress_smoke_seed_sets_path": None,
        "stress_smoke_seed_sets_sha256": None,
        "stress_smoke_route_certification_path": None,
        "stress_smoke_route_certification_sha256": None,
        "stress_smoke_scenario_source_pins": (),
        "stress_smoke_hybrid_config_pins": (),
        "stress_smoke_branch_witnesses": (),
    }
    if str(payload.get("release_kind", "")).strip() != DIAGNOSTIC_STRESS_RELEASE_KIND:
        return defaults

    contract = payload.get("stress_smoke_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("stress_smoke_contract must be a mapping for diagnostic stress smoke")
    if contract.get("schema_version") != STRESS_SMOKE_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"stress_smoke_contract.schema_version must be {STRESS_SMOKE_CONTRACT_SCHEMA_VERSION}"
        )
    review_base_commit = str(contract.get("review_base_commit", "")).strip().lower()
    if _GIT_SHA_RE.fullmatch(review_base_commit) is None:
        raise ValueError(
            "stress_smoke_contract.review_base_commit must be an exact 40-character Git SHA"
        )
    source_policy = str(contract.get("source_commit_policy", "")).strip()
    if source_policy != STRESS_SMOKE_SOURCE_POLICY:
        raise ValueError(
            f"stress_smoke_contract.source_commit_policy must be {STRESS_SMOKE_SOURCE_POLICY}"
        )

    def _required_int(field_name: str, expected: int) -> int:
        value = contract.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(
                f"stress_smoke_contract.{field_name} must be the fixed value {expected}"
            )
        return value

    def _required_float(field_name: str, expected: float) -> float:
        value = contract.get(field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"stress_smoke_contract.{field_name} must be the fixed value {expected}"
            )
        normalized = float(value)
        if not math.isfinite(normalized) or normalized != expected:
            raise ValueError(
                f"stress_smoke_contract.{field_name} must be the fixed value {expected}"
            )
        return normalized

    expected_episode_cells = _required_int(
        "expected_episode_cells", STRESS_SMOKE_EXPECTED_EPISODE_CELLS
    )
    expected_horizon_steps = _required_int(
        "expected_horizon_steps", STRESS_SMOKE_EXPECTED_HORIZON_STEPS
    )
    expected_dt = _required_float("expected_dt", STRESS_SMOKE_EXPECTED_DT)
    expected_kinematics = str(contract.get("expected_kinematics", "")).strip().lower()
    if expected_kinematics != STRESS_SMOKE_EXPECTED_KINEMATICS:
        raise ValueError(
            "stress_smoke_contract.expected_kinematics must be "
            f"{STRESS_SMOKE_EXPECTED_KINEMATICS!r}"
        )
    raw_hybrid_arms = contract.get("required_hybrid_arms")
    if not isinstance(raw_hybrid_arms, list) or not raw_hybrid_arms:
        raise ValueError("stress_smoke_contract.required_hybrid_arms must be a non-empty list")
    required_hybrid_arms = tuple(str(value).strip() for value in raw_hybrid_arms)
    if any(not value for value in required_hybrid_arms) or len(set(required_hybrid_arms)) != len(
        required_hybrid_arms
    ):
        raise ValueError("stress_smoke_contract.required_hybrid_arms must be unique non-empty keys")

    scenario_section = payload.get("scenario")
    if not isinstance(scenario_section, Mapping):
        raise ValueError("scenario must be a mapping for diagnostic stress smoke")
    seed_policy = payload.get("seed_policy")
    if not isinstance(seed_policy, Mapping):
        raise ValueError("seed_policy must be a mapping for diagnostic stress smoke")

    suite_policy_path = _resolve_stress_contract_file(
        manifest_path,
        scenario_section.get("suite_policy_path"),
        "scenario.suite_policy_path",
        repository_root=repository_root,
    )
    suite_policy_sha256 = str(scenario_section.get("suite_policy_sha256", "")).strip().lower()
    if _SHA256_RE.fullmatch(suite_policy_sha256) is None:
        raise ValueError("scenario.suite_policy_sha256 must be a 64-character SHA-256")
    seed_sets_path = _resolve_stress_contract_file(
        manifest_path,
        seed_policy.get("seed_sets_path"),
        "seed_policy.seed_sets_path",
        repository_root=repository_root,
    )
    seed_sets_sha256 = str(seed_policy.get("seed_sets_sha256", "")).strip().lower()
    if _SHA256_RE.fullmatch(seed_sets_sha256) is None:
        raise ValueError("seed_policy.seed_sets_sha256 must be a 64-character SHA-256")
    route_certification_path = _resolve_stress_contract_file(
        manifest_path,
        scenario_section.get("route_certification_path"),
        "scenario.route_certification_path",
        repository_root=repository_root,
    )
    route_certification_sha256 = (
        str(scenario_section.get("route_certification_sha256", "")).strip().lower()
    )
    if _SHA256_RE.fullmatch(route_certification_sha256) is None:
        raise ValueError("scenario.route_certification_sha256 must be a 64-character SHA-256")

    pinned_assets = contract.get("pinned_assets")
    if not isinstance(pinned_assets, Mapping):
        raise ValueError("stress_smoke_contract.pinned_assets must be a mapping")

    def _pinned_asset(name: str) -> tuple[Path, str]:
        raw_asset = pinned_assets.get(name)
        if not isinstance(raw_asset, Mapping):
            raise ValueError(f"stress_smoke_contract.pinned_assets.{name} must be a mapping")
        path = _resolve_stress_contract_file(
            manifest_path,
            raw_asset.get("path") or raw_asset.get(f"{name}_path"),
            f"stress_smoke_contract.pinned_assets.{name}_path",
            repository_root=repository_root,
        )
        sha256 = (
            str(raw_asset.get("sha256") or raw_asset.get(f"{name}_sha256") or "").strip().lower()
        )
        if _SHA256_RE.fullmatch(sha256) is None:
            raise ValueError(
                f"stress_smoke_contract.pinned_assets.{name}_sha256 must be a 64-character SHA-256"
            )
        return path, sha256

    # The checked-in contract currently uses flattened keys.  Accepting a
    # mapping form here would make the schema unnecessarily ambiguous, so
    # normalize both forms through one explicit path/hash pair and compare it
    # with the top-level campaign inputs below.
    seed_pin_raw = pinned_assets.get("seed_sets")
    route_pin_raw = pinned_assets.get("route_certification")
    if isinstance(seed_pin_raw, Mapping):
        seed_pin_path, seed_pin_sha256 = _pinned_asset("seed_sets")
    else:
        seed_pin_path = _resolve_stress_contract_file(
            manifest_path,
            pinned_assets.get("seed_sets_path"),
            "stress_smoke_contract.pinned_assets.seed_sets_path",
            repository_root=repository_root,
        )
        seed_pin_sha256 = str(pinned_assets.get("seed_sets_sha256", "")).strip().lower()
        if _SHA256_RE.fullmatch(seed_pin_sha256) is None:
            raise ValueError(
                "stress_smoke_contract.pinned_assets.seed_sets_sha256 must be a 64-character SHA-256"
            )
    if isinstance(route_pin_raw, Mapping):
        route_pin_path, route_pin_sha256 = _pinned_asset("route_certification")
    else:
        route_pin_path = _resolve_stress_contract_file(
            manifest_path,
            pinned_assets.get("route_certification_path"),
            "stress_smoke_contract.pinned_assets.route_certification_path",
            repository_root=repository_root,
        )
        route_pin_sha256 = str(pinned_assets.get("route_certification_sha256", "")).strip().lower()
        if _SHA256_RE.fullmatch(route_pin_sha256) is None:
            raise ValueError(
                "stress_smoke_contract.pinned_assets.route_certification_sha256 must be a 64-character SHA-256"
            )
    if (seed_pin_path, seed_pin_sha256) != (seed_sets_path, seed_sets_sha256):
        raise ValueError("stress_smoke_contract seed-set pin must match seed_policy")
    if (route_pin_path, route_pin_sha256) != (
        route_certification_path,
        route_certification_sha256,
    ):
        raise ValueError("stress_smoke_contract route-certification pin must match scenario")

    def _asset_pins(
        field_name: str, *, planner_key_required: bool
    ) -> tuple[StressSmokeAssetPin, ...]:
        raw_assets = contract.get(field_name)
        if not isinstance(raw_assets, list) or not raw_assets:
            raise ValueError(f"stress_smoke_contract.{field_name} must be a non-empty list")
        pins: list[StressSmokeAssetPin] = []
        seen_paths: set[Path] = set()
        seen_planners: set[str] = set()
        for index, raw_asset in enumerate(raw_assets):
            if not isinstance(raw_asset, Mapping):
                raise ValueError(f"stress_smoke_contract.{field_name}[{index}] must be a mapping")
            path = _resolve_stress_contract_file(
                manifest_path,
                raw_asset.get("path"),
                f"stress_smoke_contract.{field_name}[{index}].path",
                repository_root=repository_root,
            )
            if path in seen_paths:
                raise ValueError(
                    f"stress_smoke_contract.{field_name} contains duplicate asset paths"
                )
            seen_paths.add(path)
            sha256 = str(raw_asset.get("sha256", "")).strip().lower()
            if _SHA256_RE.fullmatch(sha256) is None:
                raise ValueError(
                    f"stress_smoke_contract.{field_name}[{index}].sha256 must be a 64-character SHA-256"
                )
            planner_key = raw_asset.get("planner_key")
            if planner_key_required:
                planner_key = str(planner_key or "").strip()
                if not planner_key:
                    raise ValueError(
                        f"stress_smoke_contract.{field_name}[{index}].planner_key is required"
                    )
                if planner_key in seen_planners:
                    raise ValueError(
                        f"stress_smoke_contract.{field_name} contains duplicate planner keys"
                    )
                seen_planners.add(planner_key)
            elif planner_key is not None:
                planner_key = str(planner_key).strip() or None
            pins.append(
                StressSmokeAssetPin(
                    path=path,
                    sha256=sha256,
                    planner_key=planner_key,
                )
            )
        return tuple(pins)

    hybrid_config_pins = _asset_pins("hybrid_configs", planner_key_required=True)
    branch_witnesses = _load_stress_smoke_branch_witnesses(
        manifest_path,
        contract.get("branch_witnesses"),
        hybrid_config_pins,
        repository_root=repository_root,
    )

    return {
        "stress_smoke_review_base_commit": review_base_commit,
        "stress_smoke_source_policy": source_policy,
        "stress_smoke_expected_episode_cells": expected_episode_cells,
        "stress_smoke_expected_horizon_steps": expected_horizon_steps,
        "stress_smoke_expected_dt": expected_dt,
        "stress_smoke_expected_kinematics": expected_kinematics,
        "stress_smoke_required_hybrid_arms": required_hybrid_arms,
        "stress_smoke_suite_policy_path": suite_policy_path,
        "stress_smoke_suite_policy_sha256": suite_policy_sha256,
        "stress_smoke_seed_sets_path": seed_sets_path,
        "stress_smoke_seed_sets_sha256": seed_sets_sha256,
        "stress_smoke_route_certification_path": route_certification_path,
        "stress_smoke_route_certification_sha256": route_certification_sha256,
        "stress_smoke_scenario_source_pins": _asset_pins(
            "scenario_sources", planner_key_required=False
        ),
        "stress_smoke_hybrid_config_pins": hybrid_config_pins,
        "stress_smoke_branch_witnesses": branch_witnesses,
    }


def _load_v02_contract(  # noqa: C901, PLR0912
    manifest_path: Path,
    payload: dict[str, Any],
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Load the stricter v0.2 release identity and publication contract.

    Returns:
        Normalized v0.2-only fields, or compatibility defaults for v0.1.
    """
    defaults = {
        "latest_main_base_commit": None,
        "source_sha": None,
        "planning_base_sha": None,
        "expected_episode_cells": None,
        "expected_horizon_steps": None,
        "publication_channel": None,
        "suite_policy_path": None,
        "suite_policy_sha256": None,
        "route_certification_path": None,
        "route_certification_sha256": None,
        "seed_sets_sha256": None,
        "resolved_seeds": (),
        "snqi_claim_policy": None,
        "concept_doi": None,
        "version_doi": None,
        "metadata_path": None,
        "metadata_sha256": None,
    }
    if payload.get("schema_version") != RELEASE_MANIFEST_SCHEMA_VERSION_V0_2:
        return defaults
    latest_main_base_commit = str(payload.get("latest_main_base_commit", "")).strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", latest_main_base_commit) is None:
        raise ValueError("latest_main_base_commit must be an exact 40-character Git SHA")
    source_sha_value = payload.get("source_sha")
    source_sha = str(source_sha_value).strip().lower() if source_sha_value is not None else None
    if source_sha_value is not None and re.fullmatch(r"[0-9a-f]{40}", source_sha or "") is None:
        raise ValueError("source_sha must be an exact 40-character Git SHA")
    planning_base_sha_value = payload.get("planning_base_sha")
    planning_base_sha = (
        str(planning_base_sha_value).strip().lower()
        if planning_base_sha_value is not None
        else None
    )
    if (
        planning_base_sha_value is not None
        and re.fullmatch(r"[0-9a-f]{40}", planning_base_sha or "") is None
    ):
        raise ValueError("planning_base_sha must be an exact 40-character Git SHA")
    release_kind = str(payload.get("release_kind", "")).strip()
    if release_kind == "benchmark-data" and payload.get("release_tag") != HISTORICAL_RELEASE_TAG:
        if source_sha is None:
            raise ValueError("source_sha is required for future benchmark-data v0.2 releases")
    matrix = payload.get("matrix")
    if not isinstance(matrix, dict) or not isinstance(matrix.get("expected_episode_cells"), int):
        raise ValueError("matrix.expected_episode_cells must be an integer")
    horizon_steps = matrix.get("horizon_steps")
    if not isinstance(horizon_steps, int) or isinstance(horizon_steps, bool) or horizon_steps <= 0:
        raise ValueError("matrix.horizon_steps must be a positive integer")
    publication = payload.get("publication")
    if not isinstance(publication, dict):
        raise ValueError("publication must be a mapping")
    if publication.get("channel") != "direct_zenodo_benchmark_dataset":
        raise ValueError("publication.channel must be direct_zenodo_benchmark_dataset")
    scenario = payload.get("scenario")
    if not isinstance(scenario, dict):
        raise ValueError("scenario must be a mapping")
    seed_policy = payload.get("seed_policy")
    if not isinstance(seed_policy, dict):
        raise ValueError("seed_policy must be a mapping")
    resolved_seeds = seed_policy.get("resolved_seeds")
    if not isinstance(resolved_seeds, list) or not resolved_seeds:
        raise ValueError("seed_policy.resolved_seeds must be a non-empty list")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("snqi_claim_policy") != "advisory_no_ranking":
        raise ValueError("metrics.snqi_claim_policy must be advisory_no_ranking")
    concept_doi = str(publication.get("concept_doi", "")).strip()
    version_doi = str(publication.get("version_doi", "")).strip()
    if (
        not concept_doi.startswith("10.5281/zenodo.")
        or concept_doi in HISTORICAL_ZENODO_CONCEPT_DOIS
    ):
        raise ValueError("publication.concept_doi must name a fresh Zenodo concept")
    if (
        not version_doi.startswith("10.5281/zenodo.")
        or version_doi in HISTORICAL_ZENODO_CONCEPT_DOIS
    ):
        raise ValueError("publication.version_doi must name the reserved Zenodo version")

    # The benchmark-data route must bind every direct Zenodo operation to the
    # exact metadata file that was reviewed.  Older synthetic v0.2 fixtures and
    # non-benchmark release manifests remain compatible when they do not opt
    # into the benchmark-data release kind.
    metadata_path: Path | None = None
    metadata_sha256: str | None = None
    metadata_declared = (
        payload.get("release_kind") == "benchmark-data"
        or "metadata_path" in publication
        or "metadata_sha256" in publication
    )
    if metadata_declared:
        metadata_path = _resolve_required_file(
            manifest_path,
            publication.get("metadata_path"),
            "publication.metadata_path",
            repository_root=repository_root,
        )
        metadata_sha256 = str(publication.get("metadata_sha256", "")).strip().lower()
        if _SHA256_RE.fullmatch(metadata_sha256) is None:
            raise ValueError("publication.metadata_sha256 must be a 64-character SHA-256")
        observed_sha256 = _sha256_file(metadata_path)
        if observed_sha256 != metadata_sha256:
            raise ValueError("publication.metadata_sha256 does not match publication.metadata_path")
    return {
        "latest_main_base_commit": latest_main_base_commit,
        "source_sha": source_sha,
        "planning_base_sha": planning_base_sha,
        "expected_episode_cells": int(matrix["expected_episode_cells"]),
        "expected_horizon_steps": horizon_steps,
        "publication_channel": str(publication["channel"]),
        "suite_policy_path": _resolve_required_file(
            manifest_path,
            scenario.get("suite_policy_path"),
            "scenario.suite_policy_path",
            repository_root=repository_root,
        ),
        "suite_policy_sha256": str(scenario.get("suite_policy_sha256", "")).strip(),
        "route_certification_path": _resolve_required_file(
            manifest_path,
            scenario.get("route_certification_path"),
            "scenario.route_certification_path",
            repository_root=repository_root,
        ),
        "route_certification_sha256": str(scenario.get("route_certification_sha256", "")).strip(),
        "seed_sets_sha256": str(seed_policy.get("seed_sets_sha256", "")).strip(),
        "resolved_seeds": tuple(int(seed) for seed in resolved_seeds),
        "snqi_claim_policy": "advisory_no_ranking",
        "concept_doi": concept_doi,
        "version_doi": version_doi,
        "metadata_path": metadata_path,
        "metadata_sha256": metadata_sha256,
    }


def _load_manifest_paths_section(
    manifest_path: Path,
    payload: dict[str, Any],
    *,
    repository_root: Path | None = None,
) -> dict[str, Path]:
    """Load required manifest-side file paths.

    Returns:
        Mapping of required resolved file paths.
    """
    return {
        "canonical_campaign_config_path": _resolve_required_file(
            manifest_path,
            payload.get("canonical_campaign_config"),
            "canonical_campaign_config",
            repository_root=repository_root,
        ),
        "citation_path": _resolve_required_file(
            manifest_path,
            payload.get("citation_path"),
            "citation_path",
            repository_root=repository_root,
        ),
        "release_checklist_path": _resolve_required_file(
            manifest_path,
            payload.get("release_checklist_path"),
            "release_checklist_path",
            repository_root=repository_root,
        ),
    }


def load_release_manifest(
    path: str | Path, *, repository_root: Path | None = None
) -> BenchmarkReleaseManifest:
    """Load, normalize, and validate a benchmark release manifest.

    Returns:
        Parsed benchmark release manifest.
    """
    manifest_path = Path(path).resolve()
    repository_root = (repository_root or get_repository_root()).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Benchmark release manifest not found: {manifest_path}")
    payload = _load_mapping(manifest_path)
    identity = _load_manifest_identity(payload)
    release_metadata = _load_manifest_release_metadata(payload)
    # Check top-level shapes before resolving any filesystem-backed side input.
    # This keeps malformed manifests diagnostic even when a synthetic fixture
    # lives outside the checkout; real side inputs are still resolved fail closed
    # below.
    scenario_payload = payload.get("scenario")
    if not isinstance(scenario_payload, dict):
        raise ValueError("scenario must be a mapping")
    if not str(scenario_payload.get("matrix_sha256", "")).strip():
        raise ValueError("scenario.matrix_sha256 must be a non-empty string")
    for field in ("seed_policy", "planners", "kinematics", "artifacts", "provenance"):
        if not isinstance(payload.get(field), dict):
            raise ValueError(f"{field} must be a mapping")
    v02_contract = _load_v02_contract(
        manifest_path,
        payload,
        repository_root=repository_root,
    )
    stress_contract = _load_stress_smoke_contract(
        manifest_path,
        payload,
        repository_root=repository_root,
    )

    config_sha256 = str(release_metadata["campaign_config_sha256"] or "").strip()
    if not config_sha256:
        raise ValueError("campaign_config_sha256 must be a non-empty string")

    seed_policy = payload.get("seed_policy")
    if not isinstance(seed_policy, dict):
        raise ValueError("seed_policy must be a mapping")
    metrics = _load_manifest_metrics_section(
        manifest_path,
        payload,
        repository_root=repository_root,
    )
    planner_keys, planner_groups = _load_manifest_planner_section(payload)
    expected_kinematics_matrix, expected_holonomic_command_mode = _load_manifest_kinematics_section(
        payload
    )
    required_artifact_paths = _load_manifest_artifacts_section(payload)
    repository_url, doi = _load_manifest_provenance_section(payload)
    scenario_matrix_path, scenario_matrix_sha256 = _load_manifest_scenario_section(
        manifest_path,
        payload,
        repository_root=repository_root,
    )
    path_section = _load_manifest_paths_section(
        manifest_path,
        payload,
        repository_root=repository_root,
    )

    return BenchmarkReleaseManifest(
        path=manifest_path,
        schema_version=identity["schema_version"],
        benchmark_protocol_version=identity["benchmark_protocol_version"],
        release_id=identity["release_id"],
        release_tag=identity["release_tag"],
        maturity=identity["maturity"],
        canonical_campaign_config_path=path_section["canonical_campaign_config_path"],
        expected_paper_profile_version=release_metadata["expected_paper_profile_version"],
        expected_paper_interpretation_profile=release_metadata[
            "expected_paper_interpretation_profile"
        ],
        expected_kinematics_matrix=expected_kinematics_matrix,
        expected_holonomic_command_mode=expected_holonomic_command_mode,
        scenario_matrix_path=scenario_matrix_path,
        scenario_matrix_sha256=scenario_matrix_sha256,
        campaign_config_sha256=config_sha256,
        seed_policy=dict(seed_policy),
        snqi_weights_path=metrics["snqi_weights_path"],
        snqi_weights_sha256=metrics["snqi_weights_sha256"],
        snqi_baseline_path=metrics["snqi_baseline_path"],
        snqi_baseline_sha256=metrics["snqi_baseline_sha256"],
        planner_keys=planner_keys,
        planner_groups=planner_groups,
        required_artifact_paths=required_artifact_paths,
        repository_url=repository_url,
        doi=doi,
        citation_path=path_section["citation_path"],
        release_checklist_path=path_section["release_checklist_path"],
        release_kind=release_metadata["release_kind"],
        **v02_contract,
        **stress_contract,
    )


def validate_release_manifest(
    manifest: BenchmarkReleaseManifest,
    *,
    campaign_config: CampaignConfig | None = None,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a release manifest against the referenced campaign config and files.

    Returns:
        Validation payload with status and problem list.
    """
    repository_root = (repository_root or get_repository_root()).resolve()
    cfg = campaign_config or load_campaign_config(
        manifest.canonical_campaign_config_path,
        repository_root=repository_root,
    )
    problems: list[str] = []
    _validate_release_hashes_and_assets(manifest, cfg, problems)
    _validate_stress_smoke_contract(
        manifest,
        cfg,
        problems,
        repository_root=repository_root,
    )
    if manifest.release_kind == DIAGNOSTIC_STRESS_RELEASE_KIND:
        # Keep the effective-branch admission check in the manifest validation path so
        # preflight and release-doctor callers reject incomplete witnesses before a campaign
        # can execute.  The local import avoids the release_protocol/release_acceptance cycle.
        from robot_sf.benchmark.release_acceptance import (  # noqa: PLC0415
            _stress_effective_branch_coverage,
        )

        branch_coverage = _stress_effective_branch_coverage(
            manifest=manifest,
            campaign_config=cfg,
            source_repository_root=repository_root,
        )
        for blocker in branch_coverage["blockers"]:
            problems.append(f"effective algorithm branches: {blocker}")
    _validate_release_campaign_contract(manifest, cfg, problems)
    _validate_release_seed_policy(manifest, cfg, problems)
    _validate_release_planners(manifest, cfg, problems)
    _validate_v02_contract(manifest, cfg, problems, repository_root=repository_root)
    _validate_release_metadata_contract(manifest, problems)

    return {
        "manifest_path": _repo_relative(manifest.path, repository_root),
        "status": "valid" if not problems else "invalid",
        "problem_count": len(problems),
        "problems": problems,
    }


def _scenario_matrix_include_paths(  # noqa: C901
    path: Path,
    *,
    visited: set[Path] | None = None,
    repository_root: Path | None = None,
) -> tuple[Path, ...]:
    """Resolve scenario matrix include files for stress-contract binding.

    Returns:
        Included scenario source paths in deterministic traversal order.
    """
    seen = visited if visited is not None else set()
    if _has_symlink_component(path):
        raise ValueError(f"scenario matrix include path contains a symlink: {path}")
    resolved = path.resolve()
    root = (repository_root or get_repository_root()).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"scenario matrix include escapes repository: {resolved}") from exc
    if resolved in seen:
        raise ValueError(f"scenario matrix include cycle detected at {resolved}")
    seen.add(resolved)
    try:
        payload = _load_mapping(resolved)
        raw_includes = (
            payload.get("includes") or payload.get("include") or payload.get("scenario_files")
        )
        if raw_includes is None:
            return ()
        if isinstance(raw_includes, (str, Path)):
            raw_includes = [raw_includes]
        if not isinstance(raw_includes, list):
            raise ValueError(f"scenario matrix includes must be a list: {resolved}")
        includes: list[Path] = []
        for raw_include in raw_includes:
            include = Path(str(raw_include))
            if not include.is_absolute():
                include = resolved.parent / include
            if _has_symlink_component(include):
                raise ValueError(f"scenario matrix include path contains a symlink: {include}")
            include = include.resolve()
            includes.append(include)
        nested: list[Path] = []
        for include in includes:
            nested.append(include)
            nested.extend(
                _scenario_matrix_include_paths(
                    include,
                    visited=seen,
                    repository_root=root,
                )
            )
        return tuple(nested)
    finally:
        seen.remove(resolved)


def _validate_stress_smoke_contract(  # noqa: C901, PLR0912, PLR0915
    manifest: BenchmarkReleaseManifest,
    cfg: CampaignConfig,
    problems: list[str],
    *,
    repository_root: Path | None = None,
) -> None:
    """Validate diagnostic stress source assets without pinning the final commit."""
    if manifest.release_kind != DIAGNOSTIC_STRESS_RELEASE_KIND:
        return
    if manifest.stress_smoke_review_base_commit is None:
        problems.append("stress_smoke_contract.review_base_commit is missing")
    if manifest.stress_smoke_expected_episode_cells != STRESS_SMOKE_EXPECTED_EPISODE_CELLS:
        problems.append("stress_smoke_contract.expected_episode_cells must be 70")
    if manifest.stress_smoke_expected_horizon_steps != STRESS_SMOKE_EXPECTED_HORIZON_STEPS:
        problems.append("stress_smoke_contract.expected_horizon_steps must be 600")
    if manifest.stress_smoke_expected_dt != STRESS_SMOKE_EXPECTED_DT:
        problems.append("stress_smoke_contract.expected_dt must be 0.1")
    if manifest.stress_smoke_expected_kinematics != STRESS_SMOKE_EXPECTED_KINEMATICS:
        problems.append("stress_smoke_contract.expected_kinematics must be 'differential_drive'")
    if not manifest.stress_smoke_branch_witnesses:
        problems.append("stress_smoke_contract.branch_witnesses must be non-empty")
    pinned_configs_by_path = {
        pin.path.resolve(): pin for pin in manifest.stress_smoke_hybrid_config_pins
    }
    for index, witness in enumerate(manifest.stress_smoke_branch_witnesses):
        pin = pinned_configs_by_path.get(witness.config_path.resolve())
        if pin is None:
            problems.append(
                "stress_smoke_contract.branch_witnesses config path is not a pinned hybrid config: "
                f"{witness.config_path}"
            )
        elif witness.config_sha256 != pin.sha256:
            problems.append(
                "stress_smoke_contract.branch_witnesses config hash does not match its pin: "
                f"index {index}"
            )

    enabled_planners = tuple(planner for planner in cfg.planners if planner.enabled)
    if len(enabled_planners) != STRESS_SMOKE_EXPECTED_PLANNER_ARMS:
        problems.append("stress smoke campaign must enable exactly 14 planner arms")
    if tuple(str(value).strip().lower() for value in cfg.kinematics_matrix) != (
        STRESS_SMOKE_EXPECTED_KINEMATICS,
    ):
        problems.append("stress smoke campaign kinematics must be differential_drive only")
    try:
        scenarios = _load_campaign_scenarios(cfg, repository_root=repository_root)
        scenario_ids = tuple(
            str(
                scenario.get("id") or scenario.get("scenario_id") or scenario.get("name") or ""
            ).strip()
            for scenario in scenarios
        )
        if scenario_ids != STRESS_SMOKE_EXPECTED_SCENARIO_IDS:
            problems.append(
                "stress smoke campaign scenarios do not match the fixed five-cell roster"
            )
        resolved_seeds = tuple(_resolved_seed_inventory(scenarios))
        if resolved_seeds != (STRESS_SMOKE_EXPECTED_SEED,):
            problems.append("stress smoke campaign must resolve exactly seed 116")
    except (OSError, TypeError, ValueError, KeyError, yaml.YAMLError) as exc:
        problems.append(f"stress smoke campaign axes cannot be resolved: {exc}")

    if manifest.stress_smoke_required_hybrid_arms != tuple(
        pin.planner_key for pin in manifest.stress_smoke_hybrid_config_pins
    ):
        problems.append("stress_smoke_contract.required_hybrid_arms does not match hybrid pins")
    for label, path, sha256 in (
        (
            "suite_policy",
            manifest.stress_smoke_suite_policy_path,
            manifest.stress_smoke_suite_policy_sha256,
        ),
        (
            "seed_sets",
            manifest.stress_smoke_seed_sets_path,
            manifest.stress_smoke_seed_sets_sha256,
        ),
        (
            "route_certification",
            manifest.stress_smoke_route_certification_path,
            manifest.stress_smoke_route_certification_sha256,
        ),
    ):
        if path is None or sha256 is None:
            problems.append(f"stress_smoke_contract.{label} pin is missing")
            continue
        if not path.is_file():
            problems.append(f"stress_smoke_contract.{label} asset is missing")
            continue
        try:
            observed = _sha256_file(path)
        except OSError:
            problems.append(f"stress_smoke_contract.{label} asset cannot be read")
        else:
            if observed != sha256:
                problems.append(f"stress_smoke_contract.{label} hash does not match pinned asset")

    if manifest.stress_smoke_seed_sets_path is not None:
        if (
            cfg.seed_policy.seed_sets_path.resolve()
            != manifest.stress_smoke_seed_sets_path.resolve()
        ):
            problems.append("stress smoke seed-set path does not match campaign config")
    if manifest.stress_smoke_route_certification_path is not None:
        if cfg.route_clearance_certifications_path is None:
            problems.append(
                "stress smoke route-certification asset is missing from campaign config"
            )
        elif (
            cfg.route_clearance_certifications_path.resolve()
            != manifest.stress_smoke_route_certification_path.resolve()
        ):
            problems.append("stress smoke route-certification path does not match campaign config")
    for field_name, pins in (
        ("scenario_sources", manifest.stress_smoke_scenario_source_pins),
        ("hybrid_configs", manifest.stress_smoke_hybrid_config_pins),
    ):
        for pin in pins:
            if not pin.path.is_file():
                problems.append(f"stress_smoke_contract.{field_name} asset is missing")
                continue
            try:
                observed = _sha256_file(pin.path)
            except OSError:
                problems.append(f"stress_smoke_contract.{field_name} asset cannot be read")
                continue
            if observed != pin.sha256:
                problems.append(
                    f"stress_smoke_contract.{field_name} hash does not match pinned asset: "
                    f"{_repo_relative(pin.path)}"
                )

    try:
        included_paths = set(
            _scenario_matrix_include_paths(
                manifest.scenario_matrix_path,
                repository_root=repository_root,
            )
        )
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        problems.append(f"stress smoke scenario includes cannot be resolved: {exc}")
    else:
        pinned_paths = {pin.path.resolve() for pin in manifest.stress_smoke_scenario_source_pins}
        if included_paths != pinned_paths:
            problems.append(
                "stress_smoke_contract.scenario_sources does not exactly match scenario matrix includes"
            )

    configured_hybrid_paths = {
        planner.key: planner.algo_config_path.resolve()
        for planner in cfg.planners
        if planner.enabled and planner.algo_config_path is not None
    }
    pinned_hybrid_keys = {pin.planner_key for pin in manifest.stress_smoke_hybrid_config_pins}
    for pin in manifest.stress_smoke_hybrid_config_pins:
        if pin.planner_key not in configured_hybrid_paths:
            problems.append(
                f"stress_smoke_contract.hybrid_configs names unknown planner: {pin.planner_key}"
            )
        elif configured_hybrid_paths[pin.planner_key] != pin.path.resolve():
            problems.append(
                "stress_smoke_contract.hybrid_configs path does not match campaign config for "
                f"{pin.planner_key}"
            )
    configured_hybrid_keys = {
        planner.key
        for planner in cfg.planners
        if planner.enabled and planner.algo == "hybrid_rule_local_planner"
    }
    if configured_hybrid_keys != pinned_hybrid_keys:
        problems.append(
            "stress_smoke_contract.hybrid_configs does not cover exactly the configured hybrid arms"
        )


def _validate_release_hashes_and_assets(
    manifest: BenchmarkReleaseManifest,
    cfg: CampaignConfig,
    problems: list[str],
) -> None:
    """Validate release hashes and asset/path alignment."""
    if _sha256_file(manifest.canonical_campaign_config_path) != manifest.campaign_config_sha256:
        problems.append("campaign_config_sha256 does not match canonical_campaign_config")
    if _sha256_file(manifest.scenario_matrix_path) != manifest.scenario_matrix_sha256:
        problems.append("scenario.matrix_sha256 does not match scenario.matrix_path")
    if cfg.scenario_matrix_path.resolve() != manifest.scenario_matrix_path.resolve():
        problems.append("canonical campaign config points at a different scenario_matrix")
    _validate_optional_metric_asset(
        label="metrics.snqi_weights_path",
        manifest_path=manifest.snqi_weights_path,
        manifest_sha256=manifest.snqi_weights_sha256,
        config_path=cfg.snqi_weights_path,
        digest_problem="metrics.snqi_weights_sha256 does not match snqi_weights_path",
        problems=problems,
    )
    _validate_optional_metric_asset(
        label="metrics.snqi_baseline_path",
        manifest_path=manifest.snqi_baseline_path,
        manifest_sha256=manifest.snqi_baseline_sha256,
        config_path=cfg.snqi_baseline_path,
        digest_problem="metrics.snqi_baseline_sha256 does not match snqi_baseline_path",
        problems=problems,
    )


def _validate_optional_metric_asset(
    *,
    label: str,
    manifest_path: Path | None,
    manifest_sha256: str | None,
    config_path: Path | None,
    digest_problem: str,
    problems: list[str],
) -> None:
    """Validate optional metric asset path alignment and digest correctness."""
    if manifest_path is not None and config_path is not None:
        if config_path.resolve() != manifest_path.resolve():
            problems.append(f"{label} does not match campaign config")
        if _sha256_file(manifest_path) != manifest_sha256:
            problems.append(digest_problem)
        return
    if manifest_path != config_path:
        problems.append(f"{label} presence does not match campaign config")


def _validate_release_campaign_contract(  # noqa: C901
    manifest: BenchmarkReleaseManifest,
    cfg: CampaignConfig,
    problems: list[str],
) -> None:
    """Validate paper-facing and kinematics contract alignment."""
    if manifest.maturity == DIAGNOSTIC_RELEASE_MATURITY:
        if cfg.paper_facing:
            problems.append("diagnostic release canonical config must be paper_facing: false")
    elif not cfg.paper_facing:
        problems.append("canonical campaign config must be paper_facing: true")
    if (
        manifest.expected_paper_profile_version is not None
        and cfg.paper_profile_version != manifest.expected_paper_profile_version
    ):
        problems.append("expected_paper_profile_version does not match campaign config")
    if (
        manifest.expected_paper_interpretation_profile is not None
        and cfg.paper_interpretation_profile != manifest.expected_paper_interpretation_profile
    ):
        problems.append("expected_paper_interpretation_profile does not match campaign config")
    if tuple(cfg.kinematics_matrix) != tuple(manifest.expected_kinematics_matrix):
        problems.append("kinematics.matrix does not match campaign config")
    if (
        manifest.expected_holonomic_command_mode is not None
        and cfg.holonomic_command_mode != manifest.expected_holonomic_command_mode
    ):
        problems.append("kinematics.holonomic_command_mode does not match campaign config")
    if manifest.release_kind == "benchmark-data":
        if cfg.checkpoint_provenance_enforcement != "error":
            problems.append(
                "benchmark-data release requires checkpoint_provenance_enforcement=error"
            )
        non_fail_fast = sorted(
            planner.key
            for planner in cfg.planners
            if planner.enabled and planner.socnav_missing_prereq_policy != "fail-fast"
        )
        if non_fail_fast:
            problems.append(
                "benchmark-data release requires fail-fast missing-prerequisite policy for: "
                + ", ".join(non_fail_fast)
            )
        if cfg.release_tag != manifest.release_tag:
            problems.append("campaign config release_tag does not match release manifest")
        if cfg.doi != manifest.doi:
            problems.append("campaign config doi does not match release manifest")


def _validate_release_seed_policy(
    manifest: BenchmarkReleaseManifest,
    cfg: CampaignConfig,
    problems: list[str],
) -> None:
    """Validate the configured and manifest seed-policy payloads."""
    cfg_seed_policy = {
        "mode": cfg.seed_policy.mode,
        "seed_set": cfg.seed_policy.seed_set,
        "seeds": list(cfg.seed_policy.seeds),
        "seed_sets_path": _repo_relative(cfg.seed_policy.seed_sets_path),
    }
    normalized_manifest_seed_policy = {
        "mode": manifest.seed_policy.get("mode"),
        "seed_set": manifest.seed_policy.get("seed_set"),
        "seeds": list(manifest.seed_policy.get("seeds", []) or []),
        "seed_sets_path": (
            _repo_relative(
                _resolve_manifest_side_path(manifest.path, manifest.seed_policy["seed_sets_path"])
            )
            if manifest.seed_policy.get("seed_sets_path") is not None
            else None
        ),
    }
    if cfg_seed_policy != normalized_manifest_seed_policy:
        problems.append("seed_policy does not match campaign config")


def _validate_release_planners(
    manifest: BenchmarkReleaseManifest,
    cfg: CampaignConfig,
    problems: list[str],
) -> None:
    """Validate planner keys and planner-group expectations."""
    problems.extend(validate_release_planner_roster(manifest, cfg)["blockers"])


def validate_release_planner_roster(
    manifest: BenchmarkReleaseManifest,
    cfg: CampaignConfig,
) -> dict[str, Any]:
    """Admit the complete enabled planner roster without creating campaign output.

    This is the pure planner-side admission shared by manifest validation and the
    no-campaign release rehearsal.  It deliberately checks the enabled planner
    keys, groups, algorithm names, and kinematics matrix before any campaign
    runner can allocate an output directory or episode worker.

    Returns:
        Structured, path-free planner roster admission evidence.
    """
    expected_keys = tuple(str(key).strip() for key in manifest.planner_keys)
    expected_groups = {
        str(key).strip(): str(group).strip() for key, group in manifest.planner_groups.items()
    }
    enabled_planners = tuple(
        planner for planner in getattr(cfg, "planners", ()) if getattr(planner, "enabled", True)
    )
    observed_keys = tuple(str(getattr(planner, "key", "")).strip() for planner in enabled_planners)
    observed_groups = {
        str(getattr(planner, "key", "")).strip(): str(getattr(planner, "planner_group", "")).strip()
        for planner in enabled_planners
    }
    observed_algorithms = {
        str(getattr(planner, "key", "")).strip(): str(getattr(planner, "algo", "")).strip()
        for planner in enabled_planners
    }
    blockers: list[str] = []
    if len(expected_keys) != len(set(expected_keys)):
        blockers.append("planners.keys contains duplicate planner arms")
    if len(observed_keys) != len(set(observed_keys)):
        blockers.append("enabled campaign config contains duplicate planner arms")
    if observed_keys != expected_keys:
        blockers.append("planners.keys does not match enabled planners in campaign config")
    if observed_groups != expected_groups:
        blockers.append("planners.groups does not match campaign config")
    if any(not key or not algorithm for key, algorithm in observed_algorithms.items()):
        blockers.append("enabled planner roster contains an empty key or algorithm")

    expected_kinematics = tuple(str(value).strip() for value in manifest.expected_kinematics_matrix)
    observed_kinematics = tuple(
        str(value).strip() for value in getattr(cfg, "kinematics_matrix", ())
    )
    if observed_kinematics != expected_kinematics:
        blockers.append("kinematics matrix does not match release manifest")

    arms = [
        {
            "key": key,
            "algo": observed_algorithms.get(key),
            "planner_group": observed_groups.get(key),
            "enabled": True,
        }
        for key in observed_keys
    ]
    return {
        "schema_version": "benchmark-release-planner-roster-admission.v1",
        "status": "valid" if not blockers else "invalid",
        "expected_planner_keys": list(expected_keys),
        "observed_planner_keys": list(observed_keys),
        "expected_planner_groups": expected_groups,
        "observed_planner_groups": observed_groups,
        "expected_kinematics": list(expected_kinematics),
        "observed_kinematics": list(observed_kinematics),
        "arms": arms,
        "blockers": blockers,
    }


def _validate_v02_contract(  # noqa: C901, PLR0912
    manifest: BenchmarkReleaseManifest,
    cfg: CampaignConfig,
    problems: list[str],
    *,
    repository_root: Path | None = None,
) -> None:
    """Validate stricter hashes, matrix size, seed inventory, and DOI separation for v0.2."""
    if manifest.schema_version != RELEASE_MANIFEST_SCHEMA_VERSION_V0_2:
        return
    if manifest.source_sha is not None and _GIT_SHA_RE.fullmatch(manifest.source_sha) is None:
        problems.append("source_sha must be an exact 40-character Git SHA")
    if (
        manifest.planning_base_sha is not None
        and _GIT_SHA_RE.fullmatch(manifest.planning_base_sha) is None
    ):
        problems.append("planning_base_sha must be an exact 40-character Git SHA")
    if manifest.planning_base_sha is not None and (
        manifest.latest_main_base_commit != manifest.planning_base_sha
    ):
        problems.append("planning_base_sha does not match latest_main_base_commit")
    if (
        manifest.release_kind == "benchmark-data"
        and manifest.source_sha is None
        and manifest.release_tag != HISTORICAL_RELEASE_TAG
    ):
        problems.append("source_sha is required for future benchmark-data v0.2 releases")
    path_hashes = (
        (manifest.suite_policy_path, manifest.suite_policy_sha256, "scenario.suite_policy_sha256"),
        (
            manifest.route_certification_path,
            manifest.route_certification_sha256,
            "scenario.route_certification_sha256",
        ),
    )
    for path, expected_sha, field in path_hashes:
        if path is None or not expected_sha or _sha256_file(path) != expected_sha:
            problems.append(f"{field} does not match its pinned asset")
    seed_sets_path = cfg.seed_policy.seed_sets_path
    if (
        seed_sets_path is None
        or not manifest.seed_sets_sha256
        or _sha256_file(seed_sets_path) != manifest.seed_sets_sha256
    ):
        problems.append("seed_policy.seed_sets_sha256 does not match campaign config")
    scenarios = _load_campaign_scenarios(cfg, repository_root=repository_root)
    resolved_seeds = tuple(_resolved_seed_inventory(scenarios))
    if resolved_seeds != manifest.resolved_seeds:
        problems.append("seed_policy.resolved_seeds does not match campaign config")
    enabled_planners = sum(1 for planner in cfg.planners if planner.enabled)
    cells = len(scenarios) * len(resolved_seeds) * enabled_planners
    if cells != manifest.expected_episode_cells:
        problems.append("matrix.expected_episode_cells does not match resolved matrix")
    if manifest.expected_horizon_steps is None:
        problems.append("matrix.horizon_steps is missing")
    elif cfg.horizon != manifest.expected_horizon_steps:
        problems.append("matrix.horizon_steps does not match campaign config")
    else:
        overridden_horizons = {
            planner.key
            for planner in cfg.planners
            if planner.enabled
            and planner.horizon_override is not None
            and planner.horizon_override != manifest.expected_horizon_steps
        }
        if overridden_horizons:
            problems.append(
                "planner horizon overrides do not match matrix.horizon_steps: "
                + ", ".join(sorted(overridden_horizons))
            )
    if manifest.doi != manifest.version_doi:
        problems.append("provenance.doi must match publication.version_doi")
    if manifest.concept_doi in HISTORICAL_ZENODO_CONCEPT_DOIS:
        problems.append("publication.concept_doi must name a fresh Zenodo concept")
    if manifest.version_doi in HISTORICAL_ZENODO_CONCEPT_DOIS:
        problems.append("publication.version_doi must name a fresh Zenodo version")
    if manifest.concept_doi == manifest.version_doi:
        problems.append("publication concept and version DOI must be distinct")


def _validate_release_metadata_contract(
    manifest: BenchmarkReleaseManifest,
    problems: list[str],
) -> None:
    """Recheck benchmark publication metadata path and bytes at validation time."""
    requires_metadata = (
        manifest.schema_version == RELEASE_MANIFEST_SCHEMA_VERSION_V0_2
        and manifest.release_kind == "benchmark-data"
    )
    if (
        not requires_metadata
        and manifest.metadata_path is None
        and manifest.metadata_sha256 is None
    ):
        return
    if manifest.metadata_path is None:
        problems.append("publication.metadata_path is missing")
        return
    if not manifest.metadata_path.is_file():
        problems.append("publication.metadata_path is missing or not a file")
        return
    if manifest.metadata_sha256 is None or _SHA256_RE.fullmatch(manifest.metadata_sha256) is None:
        problems.append("publication.metadata_sha256 is missing or invalid")
        return
    if _sha256_file(manifest.metadata_path) != manifest.metadata_sha256:
        problems.append("publication.metadata_sha256 does not match publication.metadata_path")


def is_diagnostic_stress_smoke(manifest: Any) -> bool:
    """Return whether a manifest opts into the bounded hybrid stress-smoke lane."""
    return (
        getattr(manifest, "schema_version", None) == RELEASE_MANIFEST_SCHEMA_VERSION
        and getattr(manifest, "release_kind", None) == DIAGNOSTIC_STRESS_RELEASE_KIND
        and getattr(manifest, "maturity", None) == DIAGNOSTIC_RELEASE_MATURITY
    )


def validate_stress_smoke_runtime_identity(
    manifest: Any,
    *,
    current_source_commit: str,
    launch_expected_source_commit: str | None = None,
    require_launch_pin: bool = False,
    worktree_clean: bool | None = None,
    require_clean_worktree: bool = False,
) -> dict[str, Any]:
    """Bind a diagnostic smoke to one exact runtime HEAD and optional launch pin.

    The tracked manifest carries ``review_base_commit`` for audit context.  The exact
    runtime commit is supplied by the checked-out worktree and, on a private SLURM
    launch, independently by ``SLURM_EXPECTED_PUBLIC_COMMIT``.

    Returns:
        JSON-safe runtime identity admission report.
    """
    if not is_diagnostic_stress_smoke(manifest):
        return {
            "schema_version": "benchmark-stress-smoke-runtime-identity.v1",
            "status": "not_applicable",
            "runtime_source_commit": None,
            "review_base_commit": None,
            "blockers": [],
        }

    blockers: list[str] = []
    runtime_commit = str(current_source_commit or "").strip().lower()
    if _GIT_SHA_RE.fullmatch(runtime_commit) is None:
        blockers.append("checked-out runtime source commit is not an exact 40-character SHA")
    launch_commit = None
    if require_launch_pin and launch_expected_source_commit is None:
        blockers.append("private/SLURM stress smoke requires SLURM_EXPECTED_PUBLIC_COMMIT")
    if launch_expected_source_commit is not None:
        launch_commit = str(launch_expected_source_commit).strip().lower()
        if _GIT_SHA_RE.fullmatch(launch_commit) is None:
            blockers.append("launch expected public commit is not an exact 40-character SHA")
        elif runtime_commit != launch_commit:
            blockers.append(
                "checked-out runtime source commit does not match launch expected commit"
            )
    if require_clean_worktree and worktree_clean is not True:
        blockers.append("private/SLURM stress smoke requires a clean source worktree")

    return {
        "schema_version": "benchmark-stress-smoke-runtime-identity.v1",
        "status": "valid" if not blockers else "invalid",
        "runtime_source_commit": runtime_commit or None,
        "launch_expected_source_commit": launch_commit,
        "worktree_clean": worktree_clean,
        "private_launch_contract": bool(require_launch_pin or require_clean_worktree),
        "review_base_commit": getattr(manifest, "stress_smoke_review_base_commit", None),
        "source_commit_policy": getattr(manifest, "stress_smoke_source_policy", None),
        "blockers": blockers,
    }


def _resolve_release_source_sha(
    manifest: BenchmarkReleaseManifest, source_commit: str | None
) -> str | None:
    """Resolve one validated final source SHA for emitted release artifacts.

    Returns:
        One normalized final source SHA, or ``None`` when no source is declared.
    """
    explicit = str(source_commit).strip().lower() if source_commit is not None else None
    declared = manifest.source_sha
    if explicit is not None and _GIT_SHA_RE.fullmatch(explicit) is None:
        raise ValueError("source_commit must be an exact 40-character Git SHA")
    if declared is not None and _GIT_SHA_RE.fullmatch(declared) is None:
        raise ValueError("manifest source_sha must be an exact 40-character Git SHA")
    if explicit is not None and declared is not None and explicit != declared:
        raise ValueError("source_commit does not match manifest source_sha")
    return explicit or declared


def build_release_provenance(
    manifest: BenchmarkReleaseManifest,
    *,
    campaign_root: Path,
    invoked_command: str,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Build stable release provenance metadata written into benchmark artifacts.

    Returns:
        Release provenance block for campaign artifacts and reports.
    """
    payload = {
        "schema_version": manifest.schema_version,
        "benchmark_protocol_version": manifest.benchmark_protocol_version,
        "release_id": manifest.release_id,
        "release_tag": manifest.release_tag,
        "release_kind": manifest.release_kind,
        "maturity": manifest.maturity,
        "manifest_path": _repo_relative(manifest.path),
        "manifest_sha256": _sha256_file(manifest.path),
        "canonical_campaign_config": _repo_relative(manifest.canonical_campaign_config_path),
        "canonical_campaign_config_sha256": manifest.campaign_config_sha256,
        "scenario_matrix": _repo_relative(manifest.scenario_matrix_path),
        "scenario_matrix_sha256": manifest.scenario_matrix_sha256,
        "campaign_root": _repo_relative(campaign_root),
        "repository_url": manifest.repository_url,
        "doi": manifest.doi,
        "citation_path": _repo_relative(manifest.citation_path),
        "release_checklist_path": _repo_relative(manifest.release_checklist_path),
        "invoked_release_command": invoked_command,
        "latest_main_base_commit": manifest.latest_main_base_commit,
        "publication_channel": manifest.publication_channel,
        "concept_doi": manifest.concept_doi,
        "version_doi": manifest.version_doi,
        "metadata_path": (
            _repo_relative(manifest.metadata_path) if manifest.metadata_path is not None else None
        ),
        "metadata_sha256": manifest.metadata_sha256,
    }
    source_sha = _resolve_release_source_sha(manifest, source_commit)
    if source_sha is not None:
        payload["source_sha"] = source_sha
        # Keep the historical source_commit key for consumers of v0.1 while
        # making source_sha the unambiguous release identity field.
        payload["source_commit"] = source_sha
    if manifest.planning_base_sha is not None:
        payload["planning_base_sha"] = manifest.planning_base_sha
    if is_diagnostic_stress_smoke(manifest):
        payload["stress_smoke_contract"] = {
            "review_base_commit": manifest.stress_smoke_review_base_commit,
            "source_commit_policy": manifest.stress_smoke_source_policy,
            "expected_episode_cells": manifest.stress_smoke_expected_episode_cells,
            "expected_horizon_steps": manifest.stress_smoke_expected_horizon_steps,
            "expected_dt": manifest.stress_smoke_expected_dt,
            "expected_kinematics": manifest.stress_smoke_expected_kinematics,
            "required_hybrid_arms": list(manifest.stress_smoke_required_hybrid_arms),
            "branch_witnesses": [
                {
                    "kind": witness.kind,
                    "arm": witness.arm,
                    "scenario": witness.scenario,
                    "algorithm": witness.algorithm,
                    "branch_key": witness.branch_key,
                    "config_path": _repo_relative(witness.config_path),
                    "config_sha256": witness.config_sha256,
                }
                for witness in manifest.stress_smoke_branch_witnesses
            ],
            "suite_policy": {
                "path": _repo_relative(manifest.stress_smoke_suite_policy_path)
                if manifest.stress_smoke_suite_policy_path is not None
                else None,
                "sha256": manifest.stress_smoke_suite_policy_sha256,
            },
            "seed_sets": {
                "path": _repo_relative(manifest.stress_smoke_seed_sets_path)
                if manifest.stress_smoke_seed_sets_path is not None
                else None,
                "sha256": manifest.stress_smoke_seed_sets_sha256,
            },
            "route_certification": {
                "path": _repo_relative(manifest.stress_smoke_route_certification_path)
                if manifest.stress_smoke_route_certification_path is not None
                else None,
                "sha256": manifest.stress_smoke_route_certification_sha256,
            },
            "pinned_assets": {
                "seed_sets_path": _repo_relative(manifest.stress_smoke_seed_sets_path)
                if manifest.stress_smoke_seed_sets_path is not None
                else None,
                "seed_sets_sha256": manifest.stress_smoke_seed_sets_sha256,
                "route_certification_path": _repo_relative(
                    manifest.stress_smoke_route_certification_path
                )
                if manifest.stress_smoke_route_certification_path is not None
                else None,
                "route_certification_sha256": manifest.stress_smoke_route_certification_sha256,
            },
            "scenario_sources": [
                {
                    "path": _repo_relative(pin.path),
                    "sha256": pin.sha256,
                }
                for pin in manifest.stress_smoke_scenario_source_pins
            ],
            "hybrid_configs": [
                {
                    "planner_key": pin.planner_key,
                    "path": _repo_relative(pin.path),
                    "sha256": pin.sha256,
                }
                for pin in manifest.stress_smoke_hybrid_config_pins
            ],
        }
    return payload


def build_resolved_release_manifest(
    manifest: BenchmarkReleaseManifest,
    *,
    campaign_config: CampaignConfig | None = None,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable resolved manifest payload for archival.

    Returns:
        Resolved release manifest payload with normalized repo-relative paths.
    """
    cfg = campaign_config or load_campaign_config(manifest.canonical_campaign_config_path)
    payload = {
        "schema_version": manifest.schema_version,
        "benchmark_protocol_version": manifest.benchmark_protocol_version,
        "release_id": manifest.release_id,
        "release_tag": manifest.release_tag,
        "maturity": manifest.maturity,
        "canonical_campaign_config": _repo_relative(manifest.canonical_campaign_config_path),
        "canonical_campaign_name": cfg.name,
        "paper_facing": cfg.paper_facing,
        "paper_profile_version": cfg.paper_profile_version,
        "paper_interpretation_profile": cfg.paper_interpretation_profile,
        "scenario": {
            "matrix_path": _repo_relative(manifest.scenario_matrix_path),
            "matrix_sha256": manifest.scenario_matrix_sha256,
        },
        "seed_policy": {
            **manifest.seed_policy,
            "seed_sets_path": (
                _repo_relative(
                    _resolve_manifest_side_path(
                        manifest.path, manifest.seed_policy["seed_sets_path"]
                    )
                )
                if manifest.seed_policy.get("seed_sets_path") is not None
                else None
            ),
        },
        "metrics": {
            "snqi_weights_path": (
                _repo_relative(manifest.snqi_weights_path) if manifest.snqi_weights_path else None
            ),
            "snqi_weights_sha256": manifest.snqi_weights_sha256,
            "snqi_baseline_path": (
                _repo_relative(manifest.snqi_baseline_path) if manifest.snqi_baseline_path else None
            ),
            "snqi_baseline_sha256": manifest.snqi_baseline_sha256,
        },
        "planners": {
            "keys": list(manifest.planner_keys),
            "groups": dict(manifest.planner_groups),
        },
        "kinematics": {
            "matrix": list(manifest.expected_kinematics_matrix),
            "holonomic_command_mode": manifest.expected_holonomic_command_mode,
        },
        "artifacts": {
            "required_paths": list(manifest.required_artifact_paths),
        },
        "provenance": {
            "repository_url": manifest.repository_url,
            "doi": manifest.doi,
            "citation_path": _repo_relative(manifest.citation_path),
            "release_checklist_path": _repo_relative(manifest.release_checklist_path),
            "latest_main_base_commit": manifest.latest_main_base_commit,
            "publication_channel": manifest.publication_channel,
            "concept_doi": manifest.concept_doi,
            "version_doi": manifest.version_doi,
            "metadata_path": (
                _repo_relative(manifest.metadata_path) if manifest.metadata_path else None
            ),
            "metadata_sha256": manifest.metadata_sha256,
        },
        "matrix": {
            "expected_episode_cells": manifest.expected_episode_cells,
            "horizon_steps": manifest.expected_horizon_steps,
        },
        "release_contract": {
            "suite_policy_path": (
                _repo_relative(manifest.suite_policy_path) if manifest.suite_policy_path else None
            ),
            "suite_policy_sha256": manifest.suite_policy_sha256,
            "route_certification_path": (
                _repo_relative(manifest.route_certification_path)
                if manifest.route_certification_path
                else None
            ),
            "route_certification_sha256": manifest.route_certification_sha256,
            "seed_sets_sha256": manifest.seed_sets_sha256,
            "resolved_seeds": list(manifest.resolved_seeds),
            "snqi_claim_policy": manifest.snqi_claim_policy,
        },
        "release_kind": manifest.release_kind,
    }
    source_sha = _resolve_release_source_sha(manifest, source_commit)
    if source_sha is not None:
        payload["provenance"]["source_sha"] = source_sha
        payload["provenance"]["source_commit"] = source_sha
    if manifest.planning_base_sha is not None:
        payload["provenance"]["planning_base_sha"] = manifest.planning_base_sha
    if is_diagnostic_stress_smoke(manifest):
        payload["provenance"]["stress_smoke_contract"] = {
            "review_base_commit": manifest.stress_smoke_review_base_commit,
            "source_commit_policy": manifest.stress_smoke_source_policy,
            "expected_episode_cells": manifest.stress_smoke_expected_episode_cells,
            "expected_horizon_steps": manifest.stress_smoke_expected_horizon_steps,
            "expected_dt": manifest.stress_smoke_expected_dt,
            "expected_kinematics": manifest.stress_smoke_expected_kinematics,
            "required_hybrid_arms": list(manifest.stress_smoke_required_hybrid_arms),
            "branch_witnesses": [
                {
                    "kind": witness.kind,
                    "arm": witness.arm,
                    "scenario": witness.scenario,
                    "algorithm": witness.algorithm,
                    "branch_key": witness.branch_key,
                    "config_path": _repo_relative(witness.config_path),
                    "config_sha256": witness.config_sha256,
                }
                for witness in manifest.stress_smoke_branch_witnesses
            ],
            "suite_policy": {
                "path": _repo_relative(manifest.stress_smoke_suite_policy_path)
                if manifest.stress_smoke_suite_policy_path is not None
                else None,
                "sha256": manifest.stress_smoke_suite_policy_sha256,
            },
            "seed_sets": {
                "path": _repo_relative(manifest.stress_smoke_seed_sets_path)
                if manifest.stress_smoke_seed_sets_path is not None
                else None,
                "sha256": manifest.stress_smoke_seed_sets_sha256,
            },
            "route_certification": {
                "path": _repo_relative(manifest.stress_smoke_route_certification_path)
                if manifest.stress_smoke_route_certification_path is not None
                else None,
                "sha256": manifest.stress_smoke_route_certification_sha256,
            },
            "pinned_assets": {
                "seed_sets_path": _repo_relative(manifest.stress_smoke_seed_sets_path)
                if manifest.stress_smoke_seed_sets_path is not None
                else None,
                "seed_sets_sha256": manifest.stress_smoke_seed_sets_sha256,
                "route_certification_path": _repo_relative(
                    manifest.stress_smoke_route_certification_path
                )
                if manifest.stress_smoke_route_certification_path is not None
                else None,
                "route_certification_sha256": manifest.stress_smoke_route_certification_sha256,
            },
            "scenario_sources": [
                {
                    "path": _repo_relative(pin.path),
                    "sha256": pin.sha256,
                }
                for pin in manifest.stress_smoke_scenario_source_pins
            ],
            "hybrid_configs": [
                {
                    "planner_key": pin.planner_key,
                    "path": _repo_relative(pin.path),
                    "sha256": pin.sha256,
                }
                for pin in manifest.stress_smoke_hybrid_config_pins
            ],
        }
    return payload


def parse_release_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Shared parser for release-entrypoint tests and CLI wrapper.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run a benchmark release workflow.")
    parser.add_argument("--manifest", type=Path, required=True, help="Benchmark release manifest.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional campaign output root. Defaults to output/benchmarks/camera_ready.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional release label suffix for the generated campaign_id.",
    )
    parser.add_argument(
        "--mode",
        choices=("run", "preflight", "rehearsal"),
        default="run",
        help=(
            "Preflight-only validation, no-campaign release rehearsal, or full release execution."
        ),
    )
    parser.add_argument(
        "--campaign-id",
        type=str,
        default=None,
        help=(
            "Optional exact campaign directory id (forwarded to the campaign runner). "
            "An existing fixed-id release campaign is rejected unless --resume-receipt "
            "proves an infrastructure-only interruption with unchanged inputs."
        ),
    )
    parser.add_argument(
        "--checkpoint-receipt",
        type=Path,
        default=None,
        help=(
            "Required for run mode: JSON receipt produced by "
            "preflight_campaign_checkpoints.py --stage."
        ),
    )
    parser.add_argument(
        "--checkpoint-receipt-max-age-hours",
        type=float,
        default=24.0,
        help="Maximum accepted staged-checkpoint receipt age (default: 24 hours).",
    )
    parser.add_argument(
        "--runtime-smoke-receipt",
        type=Path,
        default=None,
        help=(
            "Required for a v0.2 full release: release/release_result.json from a fresh "
            "canonical 14-arm runtime smoke at the exact release source commit."
        ),
    )
    parser.add_argument(
        "--runtime-smoke-receipt-max-age-hours",
        type=float,
        default=24.0,
        help="Maximum accepted runtime-smoke result age (default: 24 hours).",
    )
    parser.add_argument(
        "--resume-receipt",
        type=Path,
        default=None,
        help=(
            "Required only when resuming an existing fixed campaign: reviewed JSON receipt "
            "binding an infrastructure-only interruption to unchanged release inputs."
        ),
    )
    parser.add_argument(
        "--resume-receipt-max-age-hours",
        type=float,
        default=24.0,
        help="Maximum accepted infrastructure-resume receipt age (default: 24 hours).",
    )
    return parser.parse_args(argv)


__all__ = [
    "BENCHMARK_PROTOCOL_VERSION",
    "DIAGNOSTIC_STRESS_RELEASE_KIND",
    "HISTORICAL_ZENODO_CONCEPT_DOIS",
    "RELEASE_MANIFEST_SCHEMA_VERSION",
    "RELEASE_MANIFEST_SCHEMA_VERSION_V0_2",
    "STRESS_SMOKE_EXPECTED_DT",
    "STRESS_SMOKE_EXPECTED_EPISODE_CELLS",
    "STRESS_SMOKE_EXPECTED_HORIZON_STEPS",
    "STRESS_SMOKE_EXPECTED_KINEMATICS",
    "STRESS_SMOKE_EXPECTED_PLANNER_ARMS",
    "STRESS_SMOKE_EXPECTED_SCENARIO_COUNT",
    "STRESS_SMOKE_EXPECTED_SCENARIO_IDS",
    "STRESS_SMOKE_EXPECTED_SEED",
    "SUPPORTED_RELEASE_MANIFEST_SCHEMA_VERSIONS",
    "BenchmarkReleaseManifest",
    "StressSmokeAssetPin",
    "StressSmokeBranchWitness",
    "build_release_provenance",
    "build_resolved_release_manifest",
    "is_diagnostic_stress_smoke",
    "load_release_manifest",
    "parse_release_args",
    "validate_release_manifest",
    "validate_release_planner_roster",
    "validate_stress_smoke_runtime_identity",
]
