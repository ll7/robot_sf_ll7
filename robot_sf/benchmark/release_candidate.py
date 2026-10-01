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

from robot_sf.benchmark.camera_ready._config import (
    _apply_fixed_campaign_horizon,
    _apply_scenario_horizon_schedule,
)
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.policy_search_manifest import (
    is_candidate_manifest,
    resolve_candidate_manifest_runtime,
)
from robot_sf.benchmark.release_acceptance import _full_release_nested_config_path
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
_HISTORICAL_SCENARIO_MATRIX = (
    "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml"
)
_HISTORICAL_SCENARIO_MATRIX_SHA256 = (
    "03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c"
)
# The four v4 keys replace the historical keys at the same slot positions.
# Reconcile this explicit roster with #9751's final reviewed head before admitting
# a physical 0.0.8 candidate; arbitrary jointly edited config/candidate keys fail.
_APPROVED_008_PLANNER_KEYS = (
    "prediction_planner",
    "goal",
    "social_force",
    "orca",
    "ppo",
    "socnav_sampling",
    "sacadrl",
    "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4",
    "scenario_adaptive_hybrid_orca_v2_collision_guard_v4",
    "hybrid_rule_v4_fast_progress_static_escape",
    "hybrid_rule_v4_fast_progress_static_escape_continuous",
    "guarded_ppo",
    "predictive_mppi",
    "risk_dwa",
)
# Reviewed bindings from paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml.
# Keep these independent of candidate bytes: a jointly re-pinned campaign must
# not substitute a different algorithm under an approved arm key.
_APPROVED_008_PLANNER_ALGOS = {
    "prediction_planner": "prediction_planner",
    "goal": "goal",
    "social_force": "social_force",
    "orca": "orca",
    "ppo": "ppo",
    "socnav_sampling": "socnav_sampling",
    "sacadrl": "sacadrl",
    "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4": "hybrid_rule_local_planner",
    "scenario_adaptive_hybrid_orca_v2_collision_guard_v4": "hybrid_rule_local_planner",
    "hybrid_rule_v4_fast_progress_static_escape": "hybrid_rule_local_planner",
    "hybrid_rule_v4_fast_progress_static_escape_continuous": "hybrid_rule_local_planner",
    "guarded_ppo": "guarded_ppo",
    "predictive_mppi": "predictive_mppi",
    "risk_dwa": "risk_dwa",
}
_APPROVED_008_HYBRID_CONFIGS = {
    "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4": (
        "configs/policy_search/candidates/"
        "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4_s30_h600_release_0_0_8_frozen.yaml"
    ),
    "scenario_adaptive_hybrid_orca_v2_collision_guard_v4": (
        "configs/policy_search/candidates/"
        "scenario_adaptive_hybrid_orca_v2_collision_guard_v4_s30_h600_release_0_0_8_frozen.yaml"
    ),
    "hybrid_rule_v4_fast_progress_static_escape": (
        "configs/policy_search/candidates/"
        "hybrid_rule_v4_fast_progress_static_escape_s30_h600_release_0_0_8_frozen.yaml"
    ),
    "hybrid_rule_v4_fast_progress_static_escape_continuous": (
        "configs/policy_search/candidates/"
        "hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release_0_0_8_frozen.yaml"
    ),
}
_APPROVED_008_ACTIVE_WAYPOINT_CONFIGS = {
    "risk_dwa": "configs/algos/risk_dwa_release_v0_0_8.yaml",
    "predictive_mppi": "configs/algos/predictive_mppi_release_v0_0_8.yaml",
    "guarded_ppo": "configs/algos/guarded_ppo_release_v0_0_8.yaml",
}
_APPROVED_008_HYBRID_ALGO_OVERRIDES = {
    "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4": {
        "francis2023_leave_group": ("orca", "configs/algos/issue707_orca_tuned.yaml")
    },
    "scenario_adaptive_hybrid_orca_v2_collision_guard_v4": {
        "francis2023_leave_group": ("orca", "configs/algos/issue707_orca_tuned.yaml")
    },
}


@dataclass(frozen=True)
class PrepublicationCandidate:
    """Pinned setup input consumed only by the diagnostic preflight."""

    path: Path
    repository_root: Path
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
    expected_horizon_steps: int | None
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


def _planner_config_paths(root: Path, config_path: Path) -> set[Path]:
    """Collect a policy-search manifest and its runtime base configs.

    Returns:
        Source files whose bytes the candidate must pin.
    """
    paths = {config_path}
    manifest = _load_mapping(config_path)
    if not is_candidate_manifest(manifest):
        return paths

    if "base_config_path" in manifest:
        paths.add(
            _full_release_nested_config_path(
                manifest["base_config_path"],
                config_anchor=config_path.parent,
                source_repository_root=root,
                label="candidate planner base_config_path",
            )
        )
    overrides = manifest.get("scenario_algo_overrides")
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError("candidate planner scenario_algo_overrides must be a mapping")
        for scenario_id, override in overrides.items():
            if not isinstance(override, dict):
                raise ValueError(
                    f"candidate planner scenario_algo_overrides[{scenario_id!r}] must be a mapping"
                )
            if "base_config_path" in override:
                paths.add(
                    _full_release_nested_config_path(
                        override["base_config_path"],
                        config_anchor=config_path.parent,
                        source_repository_root=root,
                        label=f"candidate planner scenario_algo_overrides[{scenario_id!r}].base_config_path",
                    )
                )
    return paths


def _validate_v4_hybrid_runtime(
    planner_key: str,
    planner_algo: str,
    manifest: dict[str, Any],
    config_path: Path,
    root: Path,
    scenarios: list[dict[str, Any]],
) -> None:
    """Resolve every release scenario before admitting a v4-named slot."""

    def load_config(value: object) -> dict[str, Any]:
        path = _full_release_nested_config_path(
            value,
            config_anchor=config_path.parent,
            source_repository_root=root,
            label="v4 hybrid runtime base_config_path",
        )
        return _load_mapping(path)

    for scenario in scenarios:
        algo, effective = resolve_candidate_manifest_runtime(
            default_algo=planner_algo,
            manifest=manifest,
            scenario=scenario,
            load_config=load_config,
        )
        if algo == "orca":
            continue  # The exact reviewed ORCA hand-off is checked below.
        if algo != "hybrid_rule_local_planner" or (
            effective.get("planner_variant") != "hybrid_rule_v4_clearance_braking"
        ):
            raise ValueError(f"v4 hybrid slot {planner_key} resolves to a non-v4 planner variant")


def _validate_v4_hybrid_manifest(
    planner_key: str,
    planner_algo: str,
    config_path: Path,
    root: Path,
    scenarios: list[dict[str, Any]],
) -> None:
    """Reject a v4 slot that would execute an unreviewed algorithm/base pair."""
    if planner_key not in _APPROVED_008_HYBRID_CONFIGS:
        return
    manifest = _load_mapping(config_path)
    if manifest.get("name") != config_path.stem:
        raise ValueError(f"v4 hybrid slot {planner_key} has a mismatched config name")
    if manifest.get("base_config_path") != "configs/algos/hybrid_rule_v4_clearance_braking.yaml":
        raise ValueError(f"v4 hybrid slot {planner_key} must bind the v4 base config")
    for section_name in ("family_overrides", "scenario_overrides"):
        section = manifest.get(section_name) or {}
        if not isinstance(section, dict):
            raise ValueError(f"v4 hybrid {section_name} must be a mapping")
        for override in section.values():
            if not isinstance(override, dict):
                raise ValueError(f"v4 hybrid {section_name} entries must be mappings")
    overrides = manifest.get("scenario_algo_overrides") or {}
    if not isinstance(overrides, dict) or any(
        not isinstance(override, dict) for override in overrides.values()
    ):
        raise ValueError("v4 hybrid scenario algorithm overrides must be mappings")
    observed_overrides = {
        scenario_id: (override.get("algo"), override.get("base_config_path"))
        for scenario_id, override in overrides.items()
    }
    expected_overrides = _APPROVED_008_HYBRID_ALGO_OVERRIDES.get(planner_key, {})
    if observed_overrides != expected_overrides:
        raise ValueError(
            f"v4 hybrid slot {planner_key} has an unapproved scenario algorithm override"
        )
    _validate_v4_hybrid_runtime(planner_key, planner_algo, manifest, config_path, root, scenarios)


def _expected_input_paths(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    matrix_path: Path,
    scenarios: list[dict[str, Any]],
    seed_policy: dict[str, Any],
    planners: list[dict[str, Any]],
) -> set[Path]:
    paths = {
        config_path,
        matrix_path,
        _root_file(root, _HISTORICAL_SCENARIO_MATRIX, "historical scenario matrix"),
    }
    paths.update(_scenario_matrix_include_paths(matrix_path, repository_root=root))
    for field in (
        "comparability_mapping",
        "route_clearance_certifications",
        "snqi_weights",
        "snqi_baseline",
        "scenario_horizons",
    ):
        if field in config:
            paths.add(_root_file(root, config[field], f"campaign.{field}"))
    for field in ("suite_policy_path", "route_certification_path"):
        paths.add(_root_file(root, seed_policy[field], f"candidate.inputs.{field}"))
    seed_sets_path = _root_file(root, seed_policy["seed_sets_path"], "seed_policy.seed_sets_path")
    paths.add(seed_sets_path)
    for planner in planners:
        if planner.get("algo_config"):
            planner_config_path = _root_file(root, planner["algo_config"], "planner.algo_config")
            _validate_v4_hybrid_manifest(
                planner["key"], planner["algo"], planner_config_path, root, scenarios
            )
            paths.update(_planner_config_paths(root, planner_config_path))
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
    historical_path = _root_file(root, _HISTORICAL_SCENARIO_MATRIX, "historical scenario matrix")
    if sha256_file(historical_path) != _HISTORICAL_SCENARIO_MATRIX_SHA256:
        raise ValueError("frozen 0.0.7 scenario matrix bytes changed")
    historical = load_scenarios_for_validation(historical_path, base_dir=root)
    if historical.load_error or historical.load_issues or historical.entry_issues:
        raise ValueError("frozen 0.0.7 scenario matrix does not load without validation issues")
    historical_ids = tuple(str(row.get("name") or "") for row in historical.scenarios)
    if observed_ids != historical_ids:
        raise ValueError("candidate scenario identities differ from the 48 frozen 0.0.7 identities")
    declared_ids = section.get("identities")
    if (
        not isinstance(declared_ids, list)
        or len(declared_ids) != 48
        or tuple(declared_ids) != observed_ids
        or len(set(observed_ids)) != 48
    ):
        raise ValueError("scenario.identities must equal the 48 ordered scenario identities")
    return matrix_path, observed_ids, scenarios


def _validate_active_waypoint_binding(root: Path, row: dict[str, Any]) -> None:
    expected_config = _APPROVED_008_ACTIVE_WAYPOINT_CONFIGS.get(row["key"])
    if expected_config is None:
        return
    if row.get("algo_config") != expected_config:
        raise ValueError(f"{row['key']} must bind its active-waypoint v2 config path")
    planner_config = _load_mapping(_root_file(root, expected_config, "planner.algo_config"))
    selector_config = (
        _require_mapping(planner_config.get("fallback_risk_dwa"), "fallback_risk_dwa")
        if row["key"] == "guarded_ppo"
        else planner_config
    )
    if selector_config.get("goal_target_version") != "active_waypoint_v2":
        raise ValueError(f"{row['key']} must select active_waypoint_v2")


def _candidate_planners(
    root: Path, payload: dict[str, Any], config: dict[str, Any]
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
    if len({row["key"] for row in planner_rows}) != len(planner_rows):
        raise ValueError("campaign planner keys must be unique")
    enabled = [row for row in planner_rows if row.get("enabled", True)]
    observed_keys = tuple(str(row.get("key") or "") for row in enabled)
    if (
        not isinstance(keys, list)
        or len(keys) != 14
        or tuple(keys) != observed_keys
        or len(set(observed_keys)) != 14
    ):
        raise ValueError("planners.keys must equal the 14 ordered enabled campaign arms")
    if observed_keys != _APPROVED_008_PLANNER_KEYS:
        raise ValueError("campaign planner keys differ from the approved 0.0.8 14-slot roster")
    for row in enabled:
        expected_algo = _APPROVED_008_PLANNER_ALGOS[row["key"]]
        if row.get("algo") != expected_algo:
            raise ValueError(f"{row['key']} must bind its approved algorithm {expected_algo}")
        expected_config = _APPROVED_008_HYBRID_CONFIGS.get(row["key"])
        if expected_config is not None and row.get("algo_config") != expected_config:
            raise ValueError(f"v4 hybrid slot {row['key']} must bind its v4 config path")
        _validate_active_waypoint_binding(root, row)
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


def _candidate_horizon_contract(
    root: Path, config: dict[str, Any], scenarios: list[dict[str, Any]]
) -> dict[str, Any]:
    """Declare fixed or hash-pinned authored budgets without admitting a hidden minimum.

    Returns:
        Exact candidate matrix declaration for the campaign's budget mode.
    """

    if config.get("scenario_horizons") is None:
        if config.get("horizon") != 600:
            raise ValueError("campaign horizon differs from candidate H600 contract")
        _apply_fixed_campaign_horizon(
            scenarios, horizon=600, protocol_version=config.get("protocol_version")
        )
        return {"expected_episode_cells": 20160, "horizon_steps": 600}
    if config.get("horizon") is not None or any(
        p.get("horizon") is not None for p in config["planners"] if p.get("enabled", True)
    ):
        raise ValueError("scenario_horizons cannot be combined with fixed horizon")
    path = _root_file(root, config["scenario_horizons"], "scenario_horizons")
    digest = config.get("scenario_horizons_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("candidate requires scenario_horizons_sha256")
    patched = _apply_scenario_horizon_schedule(
        scenarios,
        schedule_path=path,
        expected_sha256=digest,
        protocol_version=config.get("protocol_version"),
    )
    for authored, effective in zip(scenarios, patched, strict=True):
        if effective["simulation_config"]["max_episode_steps"] != authored.get(
            "simulation_config", {}
        ).get("max_episode_steps"):
            raise ValueError("candidate scenario_horizons must preserve authored limits")
    return {
        "expected_episode_cells": 20160,
        "horizon_mode": "scenario_horizons",
        "scenario_horizons": config["scenario_horizons"],
        "scenario_horizons_sha256": digest,
    }


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
    observed_keys, enabled = _candidate_planners(root, payload, config)
    seed_policy = _candidate_seed_policy(root, payload, config)

    matrix = _require_mapping(payload.get("matrix"), "matrix")
    if matrix != _candidate_horizon_contract(root, config, scenarios):
        raise ValueError("candidate matrix differs from declared campaign budget contract")
    inputs = _require_mapping(payload.get("inputs"), "inputs")
    expected_paths = _expected_input_paths(
        root, config_path, config, matrix_path, scenarios, {**seed_policy, **inputs}, enabled
    )
    pinned = _candidate_pins(root, payload, expected_paths)
    return PrepublicationCandidate(
        path=candidate_path,
        repository_root=root,
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
        expected_horizon_steps=matrix.get("horizon_steps"),
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
        "matrix": _candidate_horizon_contract(root, config, scenarios),
        "inputs": inputs,
        "sha256_files": {
            path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(paths)
        },
    }
    output_path = _safe_identity_output(output, root, field_name="candidate output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    validated = False
    try:
        candidate = load_prepublication_candidate(output_path, repository_root=root)
        validated = True
        return candidate
    finally:
        if not validated:
            output_path.unlink(missing_ok=True)


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


def verify_prepublication_candidate_after_preflight(
    candidate: PrepublicationCandidate,
    *,
    manifest_sha256: str,
    repository_root: Path | None = None,
) -> None:
    """Reject any source or input drift observed after matrix preflight.

    The preflight report carries the original candidate digest. Re-loading the
    candidate rechecks the exact source HEAD, clean tree, all input hashes, and
    the complete scenario/config closure.
    """
    if sha256_file(candidate.path) != manifest_sha256:
        raise ValueError("prepublication candidate changed while preflight was running")
    current = load_prepublication_candidate(candidate.path, repository_root=repository_root)
    if current != candidate:
        raise ValueError("prepublication candidate inputs changed while preflight was running")
