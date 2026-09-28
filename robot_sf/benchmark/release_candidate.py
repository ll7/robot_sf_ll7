"""DOI-free, diagnostic-only input contract for benchmark release preflight.

This schema is deliberately separate from the publication manifest.  Nothing in
this module creates or reserves a tag, DOI, release bundle, or accepted result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.release_protocol import (
    BenchmarkReleaseManifest,
    _load_mapping,
    _run_git,
    _safe_identity_output,
    _safe_repository_file,
    _scenario_matrix_include_paths,
    load_release_manifest,
)
from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.training.scenario_loader import load_scenarios_for_validation

CANDIDATE_SCHEMA = "benchmark-release-prepublication-candidate.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_EXPECTED_SEEDS = tuple(range(111, 141))


@dataclass(frozen=True)
class PrepublicationCandidate:
    """Pinned setup input consumed only by the diagnostic preflight."""

    path: Path
    schema_version: str
    release_id: str
    release_kind: str
    source_sha: str
    canonical_campaign_config_path: Path
    campaign_config_sha256: str
    scenario_matrix_path: Path
    scenario_matrix_sha256: str
    scenario_identities: tuple[str, ...]
    seed_policy: dict[str, Any]
    resolved_seeds: tuple[int, ...]
    planner_keys: tuple[str, ...]
    expected_episode_cells: int
    expected_horizon_steps: int
    pinned_files: tuple[tuple[Path, str], ...]


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _root_file(root: Path, value: Any, field: str) -> Path:
    return _safe_repository_file(Path(_nonempty(value, field)), root, field_name=field)


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _verify_source_checkout(root: Path, source_sha: str) -> None:
    head = _run_git(root, "rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.decode().strip() != source_sha:
        raise ValueError("candidate source_commit must equal the exact checkout HEAD")
    status = _run_git(root, "status", "--porcelain", "--untracked-files=no")
    if status.returncode != 0 or status.stdout.strip():
        raise ValueError("candidate source checkout must have no tracked changes")


def _expected_input_paths(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    matrix_path: Path,
    scenarios: list[dict[str, Any]],
    seed_policy: dict[str, Any],
    planners: list[dict[str, Any]],
) -> set[Path]:
    paths = {config_path, matrix_path}
    paths.update(_scenario_matrix_include_paths(matrix_path, repository_root=root))
    for field in (
        "comparability_mapping",
        "route_clearance_certifications",
        "snqi_weights",
        "snqi_baseline",
    ):
        if field in config:
            paths.add(_root_file(root, config[field], f"campaign.{field}"))
    for field in ("suite_policy_path", "route_certification_path"):
        paths.add(_root_file(root, seed_policy[field], f"candidate.inputs.{field}"))
    seed_sets_path = _root_file(root, seed_policy["seed_sets_path"], "seed_policy.seed_sets_path")
    paths.add(seed_sets_path)
    for planner in planners:
        if planner.get("algo_config"):
            paths.add(_root_file(root, planner["algo_config"], "planner.algo_config"))
    for scenario in scenarios:
        if scenario.get("map_id"):
            raise ValueError("candidate scenarios must resolve to explicit map_file paths")
        paths.add(_root_file(root, scenario.get("map_file"), "scenario.map_file"))
    return paths


def _candidate_scenarios(
    root: Path, payload: dict[str, Any], config: dict[str, Any]
) -> tuple[Path, tuple[str, ...], list[dict[str, Any]]]:
    section = _require_mapping(payload.get("scenario"), "scenario")
    matrix_path = _root_file(root, section.get("matrix_path"), "scenario.matrix_path")
    if _root_file(root, config.get("scenario_matrix"), "campaign.scenario_matrix") != matrix_path:
        raise ValueError("campaign scenario_matrix differs from candidate scenario.matrix_path")
    result = load_scenarios_for_validation(matrix_path, base_dir=root)
    if result.load_error or result.load_issues or result.entry_issues:
        raise ValueError("candidate scenario matrix does not load without validation issues")
    scenarios = [dict(row) for row in result.scenarios]
    observed_ids = tuple(str(row.get("name") or "") for row in scenarios)
    declared_ids = section.get("identities")
    if (
        not isinstance(declared_ids, list)
        or len(declared_ids) != 48
        or tuple(declared_ids) != observed_ids
        or len(set(observed_ids)) != 48
    ):
        raise ValueError("scenario.identities must equal the 48 ordered scenario identities")
    return matrix_path, observed_ids, scenarios


def _candidate_planners(
    payload: dict[str, Any], config: dict[str, Any]
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    section = _require_mapping(payload.get("planners"), "planners")
    keys = section.get("keys")
    planner_rows = config.get("planners")
    if not isinstance(planner_rows, list) or any(not isinstance(row, dict) for row in planner_rows):
        raise ValueError("campaign planners must be a list of mappings")
    if any(
        not isinstance(row.get("key"), str)
        or not row["key"].strip()
        or type(row.get("enabled", True)) is not bool
        for row in planner_rows
    ):
        raise ValueError("campaign planner keys and enabled flags must be explicit and valid")
    enabled = [row for row in planner_rows if row.get("enabled", True)]
    observed_keys = tuple(str(row.get("key") or "") for row in enabled)
    if (
        not isinstance(keys, list)
        or len(keys) != 14
        or tuple(keys) != observed_keys
        or len(set(observed_keys)) != 14
    ):
        raise ValueError("planners.keys must equal the 14 ordered enabled campaign arms")
    return observed_keys, enabled


def _candidate_seed_policy(
    root: Path, payload: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    policy = _require_mapping(payload.get("seed_policy"), "seed_policy")
    seeds = policy.get("resolved_seeds")
    if not isinstance(seeds, list) or tuple(seeds) != _EXPECTED_SEEDS:
        raise ValueError("seed_policy.resolved_seeds must be seeds 111 through 140")
    if policy.get("mode") != "seed-set" or not isinstance(policy.get("seed_set"), str):
        raise ValueError("seed_policy must name a seed-set")
    _nonempty(policy["seed_set"], "seed_policy.seed_set")
    config_policy = _require_mapping(config.get("seed_policy"), "campaign.seed_policy")
    if config_policy.get("mode") != "seed-set":
        raise ValueError("campaign seed policy must use the named seed-set mode")
    if config_policy.get("seed_set") != policy["seed_set"]:
        raise ValueError("campaign seed set differs from candidate seed set")
    if config_policy.get("seed_sets_path") != policy.get("seed_sets_path"):
        raise ValueError("campaign seed-set file differs from candidate seed-set file")
    seed_sets_path = _root_file(root, policy.get("seed_sets_path"), "seed_policy.seed_sets_path")
    seed_sets = _load_mapping(seed_sets_path)
    if seed_sets.get(policy["seed_set"]) != seeds:
        raise ValueError("named seed set differs from candidate resolved seeds")
    return policy


def _candidate_pins(
    root: Path, payload: dict[str, Any], expected_paths: set[Path]
) -> dict[Path, str]:
    raw_pins = _require_mapping(payload.get("sha256_files"), "sha256_files")
    pinned: dict[Path, str] = {}
    for raw_path, raw_sha in raw_pins.items():
        file_path = _root_file(root, raw_path, "sha256_files path")
        digest = _nonempty(raw_sha, f"sha256_files[{raw_path}]").lower()
        if _SHA256.fullmatch(digest) is None or sha256_file(file_path) != digest:
            raise ValueError(f"sha256_files hash mismatch: {raw_path}")
        if file_path in pinned:
            raise ValueError(f"sha256_files contains duplicate resolved path: {raw_path}")
        pinned[file_path] = digest
    if set(pinned) != expected_paths:
        missing = sorted(path.relative_to(root).as_posix() for path in expected_paths - set(pinned))
        extra = sorted(path.relative_to(root).as_posix() for path in set(pinned) - expected_paths)
        raise ValueError(
            f"sha256_files must pin exactly the input closure; missing={missing}, extra={extra}"
        )
    tracked_result = _run_git(root, "ls-files", "-z", "--cached")
    if tracked_result.returncode != 0:
        raise ValueError("candidate tracked source inventory could not be read")
    tracked = {item.decode("utf-8") for item in tracked_result.stdout.split(b"\0") if item}
    untracked = sorted(
        path.relative_to(root).as_posix()
        for path in expected_paths
        if path.relative_to(root).as_posix() not in tracked
    )
    if untracked:
        raise ValueError(f"candidate input files are not in source_commit: {untracked}")
    return pinned


def load_prepublication_candidate(
    path: str | Path, *, repository_root: Path | None = None
) -> PrepublicationCandidate:
    """Validate one DOI-free candidate and every declared input byte.

    The strict publication loader rejects this distinct schema.  Callers must
    repeat this check after a preflight run to detect mid-run input drift.

    Returns:
        Validated, diagnostic-only candidate input.
    """
    root = (repository_root or get_repository_root()).resolve()
    candidate_path = _root_file(root, str(path), "candidate manifest")
    payload = _load_mapping(candidate_path)
    if payload.get("schema_version") != CANDIDATE_SCHEMA:
        raise ValueError(f"candidate schema_version must be {CANDIDATE_SCHEMA}")
    for forbidden in (
        "release_tag",
        "doi",
        "publication",
        "provenance",
        "concept_doi",
        "version_doi",
    ):
        if forbidden in payload:
            raise ValueError(f"DOI-free candidate must omit {forbidden}")
    source_sha = _nonempty(payload.get("source_commit"), "source_commit").lower()
    if _GIT_SHA.fullmatch(source_sha) is None:
        raise ValueError("source_commit must be an exact 40-character Git SHA")
    _verify_source_checkout(root, source_sha)

    release_id = _nonempty(payload.get("candidate_id"), "candidate_id")
    if payload.get("release_kind") != "benchmark-data-prepublication-candidate":
        raise ValueError("release_kind must be benchmark-data-prepublication-candidate")
    config_path = _root_file(
        root, payload.get("canonical_campaign_config"), "canonical_campaign_config"
    )
    config = _load_mapping(config_path)
    if config.get("release_tag") != "{{release_tag}}" or config.get("doi") != "{{version_doi}}":
        raise ValueError("candidate campaign config must retain unassigned publication slots")
    matrix_path, observed_ids, scenarios = _candidate_scenarios(root, payload, config)
    observed_keys, enabled = _candidate_planners(payload, config)
    seed_policy = _candidate_seed_policy(root, payload, config)

    matrix = _require_mapping(payload.get("matrix"), "matrix")
    if matrix.get("expected_episode_cells") != 20160 or matrix.get("horizon_steps") != 600:
        raise ValueError("candidate matrix must declare 20160 H600 episode cells")
    if config.get("horizon") != 600:
        raise ValueError("campaign horizon differs from candidate H600 contract")
    inputs = _require_mapping(payload.get("inputs"), "inputs")
    expected_paths = _expected_input_paths(
        root, config_path, config, matrix_path, scenarios, {**seed_policy, **inputs}, enabled
    )
    pinned = _candidate_pins(root, payload, expected_paths)
    return PrepublicationCandidate(
        path=candidate_path,
        schema_version=CANDIDATE_SCHEMA,
        release_id=release_id,
        release_kind="benchmark-data-prepublication-candidate",
        source_sha=source_sha,
        canonical_campaign_config_path=config_path,
        campaign_config_sha256=pinned[config_path],
        scenario_matrix_path=matrix_path,
        scenario_matrix_sha256=pinned[matrix_path],
        scenario_identities=observed_ids,
        seed_policy=seed_policy,
        resolved_seeds=_EXPECTED_SEEDS,
        planner_keys=observed_keys,
        expected_episode_cells=20160,
        expected_horizon_steps=600,
        pinned_files=tuple(sorted(pinned.items(), key=lambda item: str(item[0]))),
    )


def create_prepublication_candidate(
    *,
    campaign_config: Path,
    suite_policy: Path,
    route_certification: Path,
    candidate_id: str,
    output: Path,
    repository_root: Path | None = None,
) -> PrepublicationCandidate:
    """Write one ignored, exact-source candidate without publication coordinates.

    The output is created exclusively and removed on failed validation.  Copy
    the validated bytes to durable custody before treating its report as a
    review artifact.

    Returns:
        Validated candidate loaded from the newly created output file.
    """
    root = (repository_root or get_repository_root()).resolve()
    source = _run_git(root, "rev-parse", "HEAD")
    if source.returncode != 0:
        raise ValueError("candidate source commit could not be read")
    source_sha = source.stdout.decode().strip()
    _verify_source_checkout(root, source_sha)
    config_path = _root_file(root, str(campaign_config), "canonical_campaign_config")
    config = _load_mapping(config_path)
    matrix_path = _root_file(root, config.get("scenario_matrix"), "campaign.scenario_matrix")
    result = load_scenarios_for_validation(matrix_path, base_dir=root)
    if result.load_error or result.load_issues or result.entry_issues:
        raise ValueError("candidate scenario matrix does not load without validation issues")
    scenarios = [dict(row) for row in result.scenarios]
    planner_rows = config.get("planners")
    if not isinstance(planner_rows, list) or any(not isinstance(row, dict) for row in planner_rows):
        raise ValueError("campaign planners must be a list of mappings")
    enabled = [row for row in planner_rows if row.get("enabled", True)]
    config_policy = _require_mapping(config.get("seed_policy"), "campaign.seed_policy")
    seed_sets_path = _root_file(root, config_policy.get("seed_sets_path"), "seed_sets_path")
    seed_sets = _load_mapping(seed_sets_path)
    seed_set = _nonempty(config_policy.get("seed_set"), "seed_set")
    seed_policy = {
        "mode": "seed-set",
        "seed_set": seed_set,
        "seed_sets_path": seed_sets_path.relative_to(root).as_posix(),
        "resolved_seeds": seed_sets.get(seed_set),
    }
    inputs = {
        "suite_policy_path": _root_file(root, str(suite_policy), "suite_policy")
        .relative_to(root)
        .as_posix(),
        "route_certification_path": _root_file(
            root, str(route_certification), "route_certification"
        )
        .relative_to(root)
        .as_posix(),
    }
    paths = _expected_input_paths(
        root, config_path, config, matrix_path, scenarios, {**seed_policy, **inputs}, enabled
    )
    payload = {
        "schema_version": CANDIDATE_SCHEMA,
        "candidate_id": _nonempty(candidate_id, "candidate_id"),
        "release_kind": "benchmark-data-prepublication-candidate",
        "source_commit": source_sha,
        "canonical_campaign_config": config_path.relative_to(root).as_posix(),
        "scenario": {
            "matrix_path": matrix_path.relative_to(root).as_posix(),
            "identities": [str(row.get("name") or "") for row in scenarios],
        },
        "planners": {"keys": [str(row.get("key") or "") for row in enabled]},
        "seed_policy": seed_policy,
        "matrix": {"expected_episode_cells": 20160, "horizon_steps": 600},
        "inputs": inputs,
        "sha256_files": {
            path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(paths)
        },
    }
    output_path = _safe_identity_output(output, root, field_name="candidate output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    try:
        return load_prepublication_candidate(output_path, repository_root=root)
    except Exception:
        output_path.unlink()
        raise


def load_preflight_input(
    path: str | Path, *, repository_root: Path | None = None
) -> PrepublicationCandidate | BenchmarkReleaseManifest:
    """Dispatch setup preflight to the diagnostic or strict release loader.

    Returns:
        A validated candidate or canonical release manifest. Publication
        callers must continue using ``load_release_manifest`` directly.
    """
    manifest_path = Path(path).resolve()
    payload = _load_mapping(manifest_path)
    if payload.get("schema_version") == CANDIDATE_SCHEMA:
        return load_prepublication_candidate(manifest_path, repository_root=repository_root)
    return load_release_manifest(manifest_path, repository_root=repository_root)
