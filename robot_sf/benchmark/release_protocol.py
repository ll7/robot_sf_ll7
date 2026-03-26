"""Benchmark release protocol helpers built on top of camera-ready campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.camera_ready_campaign import CampaignConfig, load_campaign_config
from robot_sf.common.artifact_paths import get_repository_root

RELEASE_MANIFEST_SCHEMA_VERSION = "benchmark-release-manifest.v0.1"
BENCHMARK_PROTOCOL_VERSION = "0.1.0"
_SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_release_manifest(  # noqa: C901, PLR0912, PLR0915
    path: str | Path,
) -> BenchmarkReleaseManifest:
    """Load, normalize, and validate a benchmark release manifest.

    Returns:
        Parsed benchmark release manifest.
    """
    manifest_path = Path(path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Benchmark release manifest not found: {manifest_path}")
    payload = _load_mapping(manifest_path)

    def _resolve_required_file(value: Any, field_name: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty path string")
        resolved = _resolve_manifest_side_path(manifest_path, value)
        if not resolved.exists():
            raise FileNotFoundError(f"{field_name} not found: {resolved}")
        return resolved

    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != RELEASE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {RELEASE_MANIFEST_SCHEMA_VERSION}, got {schema_version!r}"
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

    maturity = str(payload.get("maturity", "pre-1.0")).strip() or "pre-1.0"

    campaign_config_path = _resolve_required_file(
        payload.get("canonical_campaign_config"),
        "canonical_campaign_config",
    )
    citation_path = _resolve_required_file(payload.get("citation_path"), "citation_path")
    release_checklist_path = _resolve_required_file(
        payload.get("release_checklist_path"),
        "release_checklist_path",
    )

    scenario = payload.get("scenario")
    if not isinstance(scenario, dict):
        raise ValueError("scenario must be a mapping")
    scenario_matrix_path = _resolve_required_file(
        scenario.get("matrix_path"), "scenario.matrix_path"
    )
    scenario_matrix_sha256 = str(scenario.get("matrix_sha256", "")).strip()
    if not scenario_matrix_sha256:
        raise ValueError("scenario.matrix_sha256 must be a non-empty string")

    config_sha256 = str(payload.get("campaign_config_sha256", "")).strip()
    if not config_sha256:
        raise ValueError("campaign_config_sha256 must be a non-empty string")

    seed_policy = payload.get("seed_policy")
    if not isinstance(seed_policy, dict):
        raise ValueError("seed_policy must be a mapping")

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    snqi_weights_path = (
        _resolve_required_file(metrics.get("snqi_weights_path"), "metrics.snqi_weights_path")
        if metrics.get("snqi_weights_path") is not None
        else None
    )
    snqi_baseline_path = (
        _resolve_required_file(metrics.get("snqi_baseline_path"), "metrics.snqi_baseline_path")
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
    planner_groups = {str(key): str(value) for key, value in planner_groups_raw.items()}

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

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be a mapping")
    required_artifact_paths_raw = artifacts.get("required_paths")
    if not isinstance(required_artifact_paths_raw, list) or not required_artifact_paths_raw:
        raise ValueError("artifacts.required_paths must be a non-empty list")
    required_artifact_paths = tuple(str(item).strip() for item in required_artifact_paths_raw)

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a mapping")
    repository_url = str(provenance.get("repository_url", "")).strip()
    if not repository_url:
        raise ValueError("provenance.repository_url must be a non-empty string")
    doi = str(provenance.get("doi", "")).strip()
    if not doi:
        raise ValueError("provenance.doi must be a non-empty string")

    return BenchmarkReleaseManifest(
        path=manifest_path,
        schema_version=schema_version,
        benchmark_protocol_version=protocol_version,
        release_id=release_id,
        release_tag=release_tag,
        maturity=maturity,
        canonical_campaign_config_path=campaign_config_path,
        expected_paper_profile_version=(
            str(payload.get("expected_paper_profile_version")).strip()
            if payload.get("expected_paper_profile_version") is not None
            else None
        ),
        expected_paper_interpretation_profile=(
            str(payload.get("expected_paper_interpretation_profile")).strip()
            if payload.get("expected_paper_interpretation_profile") is not None
            else None
        ),
        expected_kinematics_matrix=expected_kinematics_matrix,
        expected_holonomic_command_mode=expected_holonomic_command_mode,
        scenario_matrix_path=scenario_matrix_path,
        scenario_matrix_sha256=scenario_matrix_sha256,
        campaign_config_sha256=config_sha256,
        seed_policy=dict(seed_policy),
        snqi_weights_path=snqi_weights_path,
        snqi_weights_sha256=snqi_weights_sha256,
        snqi_baseline_path=snqi_baseline_path,
        snqi_baseline_sha256=snqi_baseline_sha256,
        planner_keys=planner_keys,
        planner_groups=planner_groups,
        required_artifact_paths=required_artifact_paths,
        repository_url=repository_url,
        doi=doi,
        citation_path=citation_path,
        release_checklist_path=release_checklist_path,
    )


def validate_release_manifest(  # noqa: C901, PLR0912
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

    if _sha256_file(manifest.canonical_campaign_config_path) != manifest.campaign_config_sha256:
        problems.append("campaign_config_sha256 does not match canonical_campaign_config")
    if _sha256_file(manifest.scenario_matrix_path) != manifest.scenario_matrix_sha256:
        problems.append("scenario.matrix_sha256 does not match scenario.matrix_path")
    if cfg.scenario_matrix_path.resolve() != manifest.scenario_matrix_path.resolve():
        problems.append("canonical campaign config points at a different scenario_matrix")
    if manifest.snqi_weights_path is not None and cfg.snqi_weights_path is not None:
        if cfg.snqi_weights_path.resolve() != manifest.snqi_weights_path.resolve():
            problems.append("metrics.snqi_weights_path does not match campaign config")
        if _sha256_file(manifest.snqi_weights_path) != manifest.snqi_weights_sha256:
            problems.append("metrics.snqi_weights_sha256 does not match snqi_weights_path")
    elif manifest.snqi_weights_path != cfg.snqi_weights_path:
        problems.append("metrics.snqi_weights_path presence does not match campaign config")
    if manifest.snqi_baseline_path is not None and cfg.snqi_baseline_path is not None:
        if cfg.snqi_baseline_path.resolve() != manifest.snqi_baseline_path.resolve():
            problems.append("metrics.snqi_baseline_path does not match campaign config")
        if _sha256_file(manifest.snqi_baseline_path) != manifest.snqi_baseline_sha256:
            problems.append("metrics.snqi_baseline_sha256 does not match snqi_baseline_path")
    elif manifest.snqi_baseline_path != cfg.snqi_baseline_path:
        problems.append("metrics.snqi_baseline_path presence does not match campaign config")

    if not cfg.paper_facing:
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
    if manifest.expected_holonomic_command_mode is not None:
        if cfg.holonomic_command_mode != manifest.expected_holonomic_command_mode:
            problems.append("kinematics.holonomic_command_mode does not match campaign config")

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

    cfg_keys = tuple(planner.key for planner in cfg.planners if planner.enabled)
    if cfg_keys != manifest.planner_keys:
        problems.append("planners.keys does not match enabled planners in campaign config")
    cfg_groups = {planner.key: planner.planner_group for planner in cfg.planners if planner.enabled}
    if cfg_groups != manifest.planner_groups:
        problems.append("planners.groups does not match campaign config")

    return {
        "manifest_path": _repo_relative(manifest.path),
        "status": "valid" if not problems else "invalid",
        "problem_count": len(problems),
        "problems": problems,
    }


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
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "benchmark_protocol_version": manifest.benchmark_protocol_version,
        "release_id": manifest.release_id,
        "release_tag": manifest.release_tag,
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
        },
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
    return parser.parse_args(argv)


__all__ = [
    "BENCHMARK_PROTOCOL_VERSION",
    "RELEASE_MANIFEST_SCHEMA_VERSION",
    "BenchmarkReleaseManifest",
    "build_release_provenance",
    "build_resolved_release_manifest",
    "load_release_manifest",
    "parse_release_args",
    "validate_release_manifest",
]
