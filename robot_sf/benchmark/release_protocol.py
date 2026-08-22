"""Benchmark release protocol helpers built on top of camera-ready campaigns."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.benchmark.camera_ready._preflight import _resolved_seed_inventory
from robot_sf.benchmark.camera_ready_campaign import CampaignConfig, load_campaign_config
from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256_file
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


def _repo_relative(path: Path) -> str:
    """Return a stable repository-relative path string when possible."""
    resolved = path.resolve()
    repo_root = get_repository_root().resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_manifest_side_path(manifest_path: Path, value: Any) -> Path:
    """Resolve a manifest-relative path value to an absolute path.

    Returns:
        Absolute path resolved relative to the manifest location.
    """
    candidate = Path(str(value))
    return candidate if candidate.is_absolute() else (manifest_path.parent / candidate).resolve()


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


def _resolve_required_file(manifest_path: Path, value: Any, field_name: str) -> Path:
    """Resolve and validate a required manifest-relative file path.

    Returns:
        Existing absolute file path.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty path string")
    resolved = _resolve_manifest_side_path(manifest_path, value)
    if not resolved.exists():
        raise FileNotFoundError(f"{field_name} not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{field_name} must be a file path, got non-file path: {resolved}")
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
    manifest_path: Path, payload: dict[str, Any]
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
    )
    scenario_matrix_sha256 = str(scenario.get("matrix_sha256", "")).strip()
    if not scenario_matrix_sha256:
        raise ValueError("scenario.matrix_sha256 must be a non-empty string")
    return scenario_matrix_path, scenario_matrix_sha256


def _load_manifest_metrics_section(
    manifest_path: Path, payload: dict[str, Any]
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
            manifest_path, metrics.get("snqi_weights_path"), "metrics.snqi_weights_path"
        )
        if metrics.get("snqi_weights_path") is not None
        else None
    )
    snqi_baseline_path = (
        _resolve_required_file(
            manifest_path, metrics.get("snqi_baseline_path"), "metrics.snqi_baseline_path"
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
    required_artifact_paths = tuple(str(item).strip() for item in required_artifact_paths_raw)
    if any(not path for path in required_artifact_paths):
        raise ValueError("artifacts.required_paths must not contain empty values")
    return required_artifact_paths


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


def _load_v02_contract(  # noqa: C901
    manifest_path: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    """Load the stricter v0.2 release identity and publication contract.

    Returns:
        Normalized v0.2-only fields, or compatibility defaults for v0.1.
    """
    defaults = {
        "latest_main_base_commit": None,
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
    }
    if payload.get("schema_version") != RELEASE_MANIFEST_SCHEMA_VERSION_V0_2:
        return defaults
    latest_main_base_commit = str(payload.get("latest_main_base_commit", "")).strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", latest_main_base_commit) is None:
        raise ValueError("latest_main_base_commit must be an exact 40-character Git SHA")
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
    return {
        "latest_main_base_commit": latest_main_base_commit,
        "expected_episode_cells": int(matrix["expected_episode_cells"]),
        "expected_horizon_steps": horizon_steps,
        "publication_channel": str(publication["channel"]),
        "suite_policy_path": _resolve_required_file(
            manifest_path, scenario.get("suite_policy_path"), "scenario.suite_policy_path"
        ),
        "suite_policy_sha256": str(scenario.get("suite_policy_sha256", "")).strip(),
        "route_certification_path": _resolve_required_file(
            manifest_path,
            scenario.get("route_certification_path"),
            "scenario.route_certification_path",
        ),
        "route_certification_sha256": str(scenario.get("route_certification_sha256", "")).strip(),
        "seed_sets_sha256": str(seed_policy.get("seed_sets_sha256", "")).strip(),
        "resolved_seeds": tuple(int(seed) for seed in resolved_seeds),
        "snqi_claim_policy": "advisory_no_ranking",
        "concept_doi": concept_doi,
        "version_doi": version_doi,
    }


def _load_manifest_paths_section(manifest_path: Path, payload: dict[str, Any]) -> dict[str, Path]:
    """Load required manifest-side file paths.

    Returns:
        Mapping of required resolved file paths.
    """
    return {
        "canonical_campaign_config_path": _resolve_required_file(
            manifest_path,
            payload.get("canonical_campaign_config"),
            "canonical_campaign_config",
        ),
        "citation_path": _resolve_required_file(
            manifest_path,
            payload.get("citation_path"),
            "citation_path",
        ),
        "release_checklist_path": _resolve_required_file(
            manifest_path,
            payload.get("release_checklist_path"),
            "release_checklist_path",
        ),
    }


def load_release_manifest(path: str | Path) -> BenchmarkReleaseManifest:
    """Load, normalize, and validate a benchmark release manifest.

    Returns:
        Parsed benchmark release manifest.
    """
    manifest_path = Path(path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Benchmark release manifest not found: {manifest_path}")
    payload = _load_mapping(manifest_path)
    identity = _load_manifest_identity(payload)
    release_metadata = _load_manifest_release_metadata(payload)
    path_section = _load_manifest_paths_section(manifest_path, payload)
    v02_contract = _load_v02_contract(manifest_path, payload)

    scenario_matrix_path, scenario_matrix_sha256 = _load_manifest_scenario_section(
        manifest_path,
        payload,
    )

    config_sha256 = str(release_metadata["campaign_config_sha256"] or "").strip()
    if not config_sha256:
        raise ValueError("campaign_config_sha256 must be a non-empty string")

    seed_policy = payload.get("seed_policy")
    if not isinstance(seed_policy, dict):
        raise ValueError("seed_policy must be a mapping")
    metrics = _load_manifest_metrics_section(manifest_path, payload)
    planner_keys, planner_groups = _load_manifest_planner_section(payload)
    expected_kinematics_matrix, expected_holonomic_command_mode = _load_manifest_kinematics_section(
        payload
    )
    required_artifact_paths = _load_manifest_artifacts_section(payload)
    repository_url, doi = _load_manifest_provenance_section(payload)

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
    )


def validate_release_manifest(
    manifest: BenchmarkReleaseManifest,
    *,
    campaign_config: CampaignConfig | None = None,
) -> dict[str, Any]:
    """Validate a release manifest against the referenced campaign config and files.

    Returns:
        Validation payload with status and problem list.
    """
    cfg = campaign_config or load_campaign_config(manifest.canonical_campaign_config_path)
    problems: list[str] = []
    _validate_release_hashes_and_assets(manifest, cfg, problems)
    _validate_release_campaign_contract(manifest, cfg, problems)
    _validate_release_seed_policy(manifest, cfg, problems)
    _validate_release_planners(manifest, cfg, problems)
    _validate_v02_contract(manifest, cfg, problems)

    return {
        "manifest_path": _repo_relative(manifest.path),
        "status": "valid" if not problems else "invalid",
        "problem_count": len(problems),
        "problems": problems,
    }


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
    cfg_keys = tuple(planner.key for planner in cfg.planners if planner.enabled)
    if cfg_keys != manifest.planner_keys:
        problems.append("planners.keys does not match enabled planners in campaign config")
    cfg_groups = {planner.key: planner.planner_group for planner in cfg.planners if planner.enabled}
    if cfg_groups != manifest.planner_groups:
        problems.append("planners.groups does not match campaign config")


def _validate_v02_contract(  # noqa: C901
    manifest: BenchmarkReleaseManifest,
    cfg: CampaignConfig,
    problems: list[str],
) -> None:
    """Validate stricter hashes, matrix size, seed inventory, and DOI separation for v0.2."""
    if manifest.schema_version != RELEASE_MANIFEST_SCHEMA_VERSION_V0_2:
        return
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
    scenarios = _load_campaign_scenarios(cfg)
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


def build_release_provenance(
    manifest: BenchmarkReleaseManifest,
    *,
    campaign_root: Path,
    invoked_command: str,
) -> dict[str, Any]:
    """Build stable release provenance metadata written into benchmark artifacts.

    Returns:
        Release provenance block for campaign artifacts and reports.
    """
    return {
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
    }


def build_resolved_release_manifest(
    manifest: BenchmarkReleaseManifest,
    *,
    campaign_config: CampaignConfig | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable resolved manifest payload for archival.

    Returns:
        Resolved release manifest payload with normalized repo-relative paths.
    """
    cfg = campaign_config or load_campaign_config(manifest.canonical_campaign_config_path)
    return {
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
        choices=("run", "preflight"),
        default="run",
        help="Preflight-only validation or full release execution.",
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
    "HISTORICAL_ZENODO_CONCEPT_DOIS",
    "RELEASE_MANIFEST_SCHEMA_VERSION",
    "RELEASE_MANIFEST_SCHEMA_VERSION_V0_2",
    "SUPPORTED_RELEASE_MANIFEST_SCHEMA_VERSIONS",
    "BenchmarkReleaseManifest",
    "build_release_provenance",
    "build_resolved_release_manifest",
    "load_release_manifest",
    "parse_release_args",
    "validate_release_manifest",
]
