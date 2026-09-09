"""User-facing ``robot-sf scenarios`` discovery and validation CLI (issue #8748).

Provides read-only discovery, inspection, and schema validation for bundled
scenarios without constructing an environment or running a simulation.

Plain-language summary:
- ``scenarios list``: Discover bundled scenarios across curated roots
  (single, archetypes, sets, top-level manifests).
- ``scenarios describe <id-or-name>``: Inspect declared scenario properties
  (map references, actor counts, kinematics, seeds, horizon, dt).
- ``scenarios validate <path>``: Validate a single scenario YAML or manifest
  against the benchmark scenario schema, map existence, and path safety.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    import argparse

# Schema versions
SCHEMA_LIST_VERSION = "scenario_list.v1"
SCHEMA_DESCRIBE_VERSION = "scenario_describe.v1"
SCHEMA_VALIDATE_VERSION = "scenario_validate.v1"

# Scenario validation and discovery statuses
STATUS_VALID = "valid"
STATUS_INVALID = "invalid"
STATUS_MISSING = "missing"
STATUS_DUPLICATE = "duplicate"
STATUS_UNSUPPORTED = "unsupported"
STATUS_EXTERNAL_ASSET_DEPENDENT = "external_asset_dependent"

# Reason codes for errors and diagnostics
REASON_OK = "OK"
REASON_PATH_TRAVERSAL = "PATH_TRAVERSAL"
REASON_FILE_NOT_FOUND = "FILE_NOT_FOUND"
REASON_IS_A_DIRECTORY = "IS_A_DIRECTORY"
REASON_EMPTY_FILE = "EMPTY_FILE"
REASON_MALFORMED_YAML = "MALFORMED_YAML"
REASON_LOAD_FAILURE = "LOAD_FAILURE"
REASON_SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
REASON_DUPLICATE_SCENARIO_ID = "DUPLICATE_SCENARIO_ID"
REASON_MAP_REFERENCE_MISSING = "MAP_REFERENCE_MISSING"
REASON_MAP_NOT_FOUND = "MAP_NOT_FOUND"
REASON_ROUTE_OVERRIDES_NOT_FOUND = "ROUTE_OVERRIDES_NOT_FOUND"
REASON_UNSUPPORTED_SCENARIO = "UNSUPPORTED_SCENARIO"
REASON_EXTERNAL_ASSET_DEPENDENT = "EXTERNAL_ASSET_DEPENDENT"

# Scenario rows intentionally use a stricter allow-list than the benchmark JSON
# Schema. The canonical schema keeps ``additionalProperties`` open for benchmark
# extensions; this CLI must still surface typos in the curated scenario contract.
_KNOWN_SCENARIO_FIELDS = frozenset(
    {
        "density",
        "flow",
        "generated_replay",
        "goal_topology",
        "groups",
        "amv",
        "id",
        "map_file",
        "map_geometry_contract",
        "map_id",
        "map_profile",
        "map_search_paths",
        "metadata",
        "multi_amv",
        "name",
        "observation_visibility",
        "obstacle",
        "platform_semantics",
        "repeats",
        "required_map_profile",
        "robot_config",
        "robot_context",
        "route_overrides_file",
        "scenario_family",
        "scenario_id",
        "seeds",
        "simulation_config",
        "single_pedestrians",
        "social_groups",
        "speed_var",
        "supported",
    }
)

# Documented curated exclusions
CURATED_EXCLUSIONS: list[dict[str, str]] = [
    {
        "path": "configs/scenarios/archetype_validation_waivers.yaml",
        "reason": "archetype validation waivers, not a scenario manifest",
    },
    {
        "path": "configs/scenarios/archetypes/classic_density_tier_index.yaml",
        "reason": "classic archetype density tier index documentation, not a scenario manifest",
    },
    {
        "path": "configs/scenarios/README.md",
        "reason": "documentation file, not a scenario manifest",
    },
    {
        "path": "configs/scenarios/debug_scenario.bash",
        "reason": "shell script entry point, not a scenario manifest",
    },
    {
        "path": "configs/scenarios/contracts",
        "reason": "contract specification directory, not a scenario manifest",
    },
    {
        "path": "configs/scenarios/maps",
        "reason": "map definition directory, not a scenario manifest",
    },
    {
        "path": "configs/scenarios/perturbations",
        "reason": "perturbation manifest directory, not a scenario manifest",
    },
    {
        "path": "configs/scenarios/results",
        "reason": "benchmark results directory, not a scenario manifest",
    },
    {
        "path": "configs/scenarios/route_overrides",
        "reason": "route overrides directory, not a scenario manifest",
    },
    {
        "path": "configs/scenarios/templates",
        "reason": "authoring template directory, not a scenario manifest",
    },
    {
        "path": "configs/scenarios/generated",
        "reason": "synthetic generated scenario directory, not a curated roster",
    },
    {
        "path": "configs/scenarios/sets/vulnerable_user_proxy_pack_v0_deferred_issue3654.yaml",
        "reason": "deferred vulnerable-user proxy scaffold; excluded from default roster",
    },
]


def _find_repo_root() -> Path:
    """Return the repository root by checking parents for pyproject.toml.

    Returns:
        Path: Resolved root directory of the repository.
    """
    current = Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent  # pragma: no cover - fallback


def _resolve_map_id_reference(
    scenario: Mapping[str, Any],
    source_file: Path,
) -> tuple[Path | None, bool]:
    """Resolve a declared map id through the canonical registry.

    Returns:
        tuple[Path | None, bool]: Resolved path and whether it is a file.
    """
    map_id = scenario.get("map_id")
    if not isinstance(map_id, str) or not map_id.strip():
        return None, False
    try:
        from robot_sf.training.scenario_loader import resolve_map_id  # noqa: PLC0415

        resolved_map = resolve_map_id(
            map_id,
            source=source_file,
            required_profile=str(
                scenario.get("required_map_profile")
                or scenario.get("map_profile")
                or "robot_runtime"
            ),
        )
    except (OSError, ValueError, TypeError):
        return None, False
    return resolved_map, resolved_map.is_file()


def _resolve_map_file_reference(
    map_file: Any,
    source_file: Path,
    repo_root: Path,
) -> tuple[Path | None, bool]:
    """Resolve a declared map file from its exact source or repository path.

    Returns:
        tuple[Path | None, bool]: Resolved path and whether it is a file.
    """
    if not isinstance(map_file, str) or not map_file.strip():
        return None, False
    candidate = Path(map_file)
    if candidate.is_absolute():
        resolved_map = candidate.resolve()
        return resolved_map, resolved_map.is_file()

    source_candidate = (source_file.parent / candidate).resolve()
    if source_candidate.is_file():
        return source_candidate, True
    repo_candidate = (repo_root / candidate).resolve()
    if repo_candidate.is_file():
        return repo_candidate, True
    return source_candidate, False


def _resolve_asset_path(value: Any, *, source_file: Path) -> Path | None:
    """Resolve a file-like asset reference relative to its manifest.

    Returns:
        Path | None: Resolved path, or ``None`` for an empty/non-string value.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = source_file.parent / candidate
    return candidate.resolve()


def _is_within_repo(path: Path, repo_root: Path) -> bool:
    """Return whether ``path`` is contained by the repository boundary."""
    try:
        path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    return True


def _resolve_map_reference(
    scenario: Mapping[str, Any],
    source_file: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Resolve map references and determine file existence on disk.

    Returns:
        dict[str, Any]: Map reference details and existence flags.
    """
    map_file = scenario.get("map_file")
    map_id = scenario.get("map_id")
    route_overrides = scenario.get("route_overrides_file")

    # A declared map_id owns resolution. Never fall back to map_file when the
    # id is empty, malformed, unknown, or points to a missing file.
    if map_id is not None:
        resolved_map, map_exists = _resolve_map_id_reference(scenario, source_file)
    else:
        resolved_map, map_exists = _resolve_map_file_reference(
            map_file,
            source_file,
            repo_root,
        )

    rel_resolved = None
    external_map_asset = False
    if resolved_map is not None:
        if _is_within_repo(resolved_map, repo_root):
            rel_resolved = resolved_map.relative_to(repo_root.resolve()).as_posix()
        else:
            rel_resolved = resolved_map.as_posix()
            external_map_asset = True

    route_path = _resolve_asset_path(route_overrides, source_file=source_file)
    external_route_asset = route_path is not None and not _is_within_repo(route_path, repo_root)

    return {
        "map_file": map_file,
        "map_id": map_id,
        "route_overrides_file": route_overrides,
        "resolved_map_path": rel_resolved,
        "map_exists": map_exists,
        "external_asset_dependent": external_map_asset or external_route_asset,
        "external_map_asset_dependent": external_map_asset,
        "external_route_asset_dependent": external_route_asset,
    }


def _extract_actor_counts(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Extract pedestrian, robot, and group counts.

    Returns:
        dict[str, Any]: Extracted actor counts and spawn density.
    """
    sim_cfg = scenario.get("simulation_config", {})
    meta = scenario.get("metadata", {})
    single_peds = scenario.get("single_pedestrians", [])
    multi_amv = scenario.get("multi_amv", {})

    ped_density = None
    if isinstance(sim_cfg, Mapping):
        ped_density = sim_cfg.get("ped_density")
    if ped_density is None and "density" in scenario:
        ped_density = scenario.get("density")

    groups = None
    if isinstance(sim_cfg, Mapping) and "groups" in sim_cfg:
        groups = sim_cfg.get("groups")
    elif "groups" in scenario:
        groups = scenario.get("groups")
    elif isinstance(meta, Mapping) and "groups" in meta:
        groups = meta.get("groups")

    num_robots = 1
    if isinstance(multi_amv, Mapping) and "num_robots" in multi_amv:
        try:
            num_robots = int(multi_amv["num_robots"])
        except (ValueError, TypeError):
            num_robots = 1

    return {
        "ped_density": float(ped_density) if isinstance(ped_density, (int, float)) else None,
        "single_pedestrians": len(single_peds) if isinstance(single_peds, list) else 0,
        "num_robots": num_robots,
        "groups": float(groups) if isinstance(groups, (int, float)) else None,
    }


def _extract_seed_policy(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Extract seed lists and repeat counts.

    Returns:
        dict[str, Any]: Seed policy properties.
    """
    seeds = scenario.get("seeds")
    repeats = scenario.get("repeats")
    clean_seeds = None
    if isinstance(seeds, list) and all(isinstance(x, int) for x in seeds):
        clean_seeds = list(seeds)

    count = 1
    if clean_seeds is not None:
        count = len(clean_seeds)
    elif isinstance(repeats, int) and repeats >= 1:
        count = repeats

    return {
        "seeds": clean_seeds,
        "repeats": repeats if isinstance(repeats, int) else None,
        "count": count,
    }


def _extract_horizon_and_dt(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Extract episode horizon and simulation timestep when declared.

    Returns:
        dict[str, Any]: Horizon and timing properties.
    """
    sim_cfg = scenario.get("simulation_config", {})
    max_steps = None
    time_step = None
    if isinstance(sim_cfg, Mapping):
        max_steps = sim_cfg.get("max_episode_steps") or sim_cfg.get("horizon")
        time_step = sim_cfg.get("time_step") or sim_cfg.get("dt")

    sim_time_s = None
    if isinstance(max_steps, (int, float)) and isinstance(time_step, (int, float)):
        sim_time_s = round(float(max_steps) * float(time_step), 3)

    return {
        "max_episode_steps": int(max_steps) if isinstance(max_steps, (int, float)) else None,
        "time_step": float(time_step) if isinstance(time_step, (int, float)) else None,
        "sim_time_s": sim_time_s,
    }


def _extract_kinematics_and_observation(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Extract robot drive type and observation sensor properties.

    Returns:
        dict[str, Any]: Kinematics and sensor properties.
    """
    robot_cfg = scenario.get("robot_config", {})
    obs_vis = scenario.get("observation_visibility", {})

    kinematics = None
    if isinstance(robot_cfg, Mapping):
        kinematics = robot_cfg.get("type") or robot_cfg.get("kinematics")

    obs_enabled = None
    fov = None
    max_range = None
    if isinstance(obs_vis, Mapping):
        obs_enabled = obs_vis.get("enabled")
        fov = obs_vis.get("fov_degrees")
        max_range = obs_vis.get("max_range_m")

    return {
        "kinematics": str(kinematics) if kinematics is not None else None,
        "observation_visibility_enabled": bool(obs_enabled) if obs_enabled is not None else None,
        "fov_degrees": float(fov) if isinstance(fov, (int, float)) else None,
        "max_range_m": float(max_range) if isinstance(max_range, (int, float)) else None,
    }


def _scenario_diagnostic_identity(
    scenario: Mapping[str, Any],
    *,
    source_file: Path,
) -> str | None:
    """Return an optional identity for a row-level validation diagnostic."""
    for key in ("name", "scenario_id", "id"):
        value = scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _scenario_asset_diagnostics(
    scenario: Mapping[str, Any],
    *,
    scenario_id: str,
    map_reference: Mapping[str, Any],
    source_file: Path,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], bool, bool]:
    """Return asset diagnostics, missing-asset state, and external-asset state."""
    errors: list[dict[str, Any]] = []
    has_missing_asset = False
    has_external_asset = False

    if not map_reference.get("map_file") and not map_reference.get("map_id"):
        errors.append(
            {
                "code": REASON_MAP_REFERENCE_MISSING,
                "message": f"Scenario '{scenario_id}' defines neither map_file nor map_id.",
                "scenario_id": scenario_id,
                "path": "/map_file",
            }
        )
    elif map_reference.get(
        "external_map_asset_dependent",
        map_reference.get("external_asset_dependent", False),
    ):
        has_external_asset = True
        reference_key = "map_id" if map_reference.get("map_id") else "map_file"
        reference_value = map_reference.get(reference_key)
        errors.append(
            {
                "code": REASON_EXTERNAL_ASSET_DEPENDENT,
                "message": (
                    f"Scenario '{scenario_id}' depends on an external map asset through "
                    f"{reference_key} '{reference_value}'."
                ),
                "scenario_id": scenario_id,
                "path": f"/{reference_key}",
            }
        )
    elif map_reference.get("map_id") and not map_reference.get("map_exists"):
        has_missing_asset = True
        map_id = map_reference["map_id"]
        errors.append(
            {
                "code": REASON_MAP_NOT_FOUND,
                "message": f"Scenario '{scenario_id}' references an unavailable map_id: '{map_id}'.",
                "scenario_id": scenario_id,
                "path": "/map_id",
            }
        )
    elif map_reference.get("map_file") and not map_reference.get("map_exists"):
        has_missing_asset = True
        map_file = map_reference["map_file"]
        errors.append(
            {
                "code": REASON_MAP_NOT_FOUND,
                "message": f"Scenario '{scenario_id}' references non-existent map: '{map_file}'.",
                "scenario_id": scenario_id,
                "path": "/map_file",
            }
        )

    route_overrides = map_reference.get("route_overrides_file")
    route_path = _resolve_asset_path(route_overrides, source_file=source_file)
    if route_path is not None:
        if not _is_within_repo(route_path, repo_root):
            has_external_asset = True
            errors.append(
                {
                    "code": REASON_EXTERNAL_ASSET_DEPENDENT,
                    "message": (
                        f"Scenario '{scenario_id}' depends on an external route override "
                        f"asset through route_overrides_file '{route_overrides}'."
                    ),
                    "scenario_id": scenario_id,
                    "path": "/route_overrides_file",
                }
            )
        elif not route_path.is_file():
            has_missing_asset = True
            errors.append(
                {
                    "code": REASON_ROUTE_OVERRIDES_NOT_FOUND,
                    "message": (
                        f"Scenario '{scenario_id}' references non-existent route overrides: "
                        f"'{route_overrides}'."
                    ),
                    "scenario_id": scenario_id,
                    "path": "/route_overrides_file",
                }
            )

    if scenario.get("supported") is False:
        errors.append(
            {
                "code": REASON_UNSUPPORTED_SCENARIO,
                "message": f"Scenario '{scenario_id}' is explicitly marked supported: false.",
                "scenario_id": scenario_id,
                "path": "/supported",
            }
        )

    return errors, has_missing_asset, has_external_asset


def _summary_status_and_reasons(
    scenario: Mapping[str, Any],
    *,
    map_reference: Mapping[str, Any],
    duplicate_sources: list[str] | None,
    diagnostic_errors: list[Mapping[str, Any]],
    has_missing_asset: bool,
    has_external_asset: bool,
) -> tuple[str, list[str]]:
    """Choose a summary status and human-readable reasons.

    Returns:
        tuple[str, list[str]]: Stable status followed by human-readable reasons.
    """
    duplicate_error = any(
        error.get("code") == REASON_DUPLICATE_SCENARIO_ID for error in diagnostic_errors
    )
    reasons: list[str] = []
    if has_external_asset:
        status = STATUS_EXTERNAL_ASSET_DEPENDENT
        reasons.append("Depends on an asset outside the repository")
    elif scenario.get("supported") is False:
        status = STATUS_UNSUPPORTED
        reasons.append("Explicitly marked supported: false")
    elif has_missing_asset:
        status = STATUS_MISSING
        if map_reference["map_id"]:
            reasons.append(
                f"Referenced map_id is not registered or its file is missing: "
                f"{map_reference['map_id']}"
            )
        elif map_reference["map_file"]:
            reasons.append(f"Referenced map_file not found on disk: {map_reference['map_file']}")
    elif duplicate_sources or duplicate_error:
        status = STATUS_DUPLICATE
        reasons.append(
            f"Conflicting catalog declarations: {', '.join(duplicate_sources)}"
            if duplicate_sources
            else "Duplicate scenario identity"
        )
    else:
        status = STATUS_INVALID if diagnostic_errors else STATUS_VALID

    for error in diagnostic_errors:
        code = error.get("code")
        if isinstance(code, str) and code not in reasons:
            reasons.append(code)
    return status, reasons


def _build_scenario_summary(
    scenario: Mapping[str, Any],
    source_file: Path,
    repo_root: Path,
    duplicate_sources: list[str] | None = None,
    validation_errors: list[Mapping[str, Any]] | None = None,
    validation_warnings: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a structured summary dictionary for one scenario.

    Returns:
        dict[str, Any]: Structured summary dictionary.
    """
    sid = str(
        scenario.get("name")
        or scenario.get("scenario_id")
        or scenario.get("id")
        or source_file.stem
    ).strip()

    meta = scenario.get("metadata", {})
    family = None
    if "scenario_family" in scenario:
        family = scenario.get("scenario_family")
    elif isinstance(meta, Mapping):
        family = meta.get("family") or meta.get("archetype")
    if family is None:
        family = scenario.get("obstacle") or scenario.get("flow")

    map_ref = _resolve_map_reference(scenario, source_file=source_file, repo_root=repo_root)
    actor_counts = _extract_actor_counts(scenario)
    seed_policy = _extract_seed_policy(scenario)
    horizon_dt = _extract_horizon_and_dt(scenario)
    kinematics_obs = _extract_kinematics_and_observation(scenario)

    asset_errors, has_missing_asset, has_external_asset = _scenario_asset_diagnostics(
        scenario,
        scenario_id=sid,
        map_reference=map_ref,
        source_file=source_file,
        repo_root=repo_root,
    )
    diagnostic_errors = [dict(error) for error in validation_errors or []]
    if duplicate_sources and not any(
        error.get("code") == REASON_DUPLICATE_SCENARIO_ID for error in diagnostic_errors
    ):
        diagnostic_errors.append(
            {
                "code": REASON_DUPLICATE_SCENARIO_ID,
                "message": (
                    f"Scenario '{sid}' has conflicting catalog declarations: "
                    f"{', '.join(duplicate_sources)}."
                ),
                "scenario_id": sid,
                "path": "/id",
            }
        )
    for asset_error in asset_errors:
        if asset_error not in diagnostic_errors:
            diagnostic_errors.append(asset_error)
    diagnostic_warnings = [dict(warning) for warning in validation_warnings or []]

    status, reasons = _summary_status_and_reasons(
        scenario,
        map_reference=map_ref,
        duplicate_sources=duplicate_sources,
        diagnostic_errors=diagnostic_errors,
        has_missing_asset=has_missing_asset,
        has_external_asset=has_external_asset,
    )

    try:
        rel_source = source_file.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        rel_source = source_file.as_posix()

    return {
        "identity": sid,
        "source_file": rel_source,
        "duplicate_sources": duplicate_sources or [],
        "family": str(family) if family is not None else None,
        "status": status,
        "map_reference": map_ref,
        "actor_counts": actor_counts,
        "seed_policy": seed_policy,
        "horizon_and_dt": horizon_dt,
        "kinematics_and_observation": kinematics_obs,
        "validation_status": {
            "status": status,
            "valid": status == STATUS_VALID,
            "reasons": reasons,
            "errors": diagnostic_errors,
            "warnings": diagnostic_warnings,
        },
    }


def _scenario_identity(scenario: Mapping[str, Any], *, source_file: Path) -> str:
    """Return the catalog identity for a canonical loader result."""
    raw_identity = scenario.get("name") or scenario.get("scenario_id") or scenario.get("id")
    if not isinstance(raw_identity, str) or not raw_identity.strip():
        raise ValueError(
            f"Scenario entry in '{source_file}' is missing a non-empty name, scenario_id, or id."
        )
    return raw_identity.strip()


def _scenario_definition_key(scenario: Mapping[str, Any], source_file: Path) -> str:
    """Return a stable key for equivalent canonical loader results."""

    def normalize(value: Any, *, key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            return {
                str(child_key): normalize(child_value, key=str(child_key))
                for child_key, child_value in value.items()
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if key in {"map_file", "route_overrides_file"} and isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                path = source_file.parent / path
            return path.resolve().as_posix()
        return value

    return json.dumps(normalize(scenario), sort_keys=True, separators=(",", ":"), default=str)


def _catalog_source_priority(manifest_path: Path, scenarios_root: Path) -> int:
    """Match the canonical API's source precedence for catalog ownership.

    Returns:
        int: Precedence value, where lower values are canonical.
    """
    try:
        relative_parts = manifest_path.resolve().relative_to(scenarios_root.resolve()).parts
    except ValueError:
        return 3
    if len(relative_parts) >= 2 and relative_parts[-2] in {"archetypes", "single"}:
        return 0
    if len(relative_parts) == 1:
        return 1
    if "sets" in relative_parts:
        return 2
    if "generated" in relative_parts:
        return 3
    return 1


def _manifest_metadata_errors(manifest_sources: list[Any]) -> list[dict[str, Any]]:
    """Validate metadata for every manifest read during canonical expansion.

    Returns:
        list[dict[str, Any]]: Canonical metadata errors in CLI form.
    """
    from robot_sf.benchmark.scenario.scenario_schema import (  # noqa: PLC0415
        validate_scenario_matrix_metadata,
    )

    errors: list[dict[str, Any]] = []
    for manifest in manifest_sources:
        errors.extend(
            _schema_errors_to_cli_errors(
                validate_scenario_matrix_metadata(manifest.data),
                source_file=manifest.path,
            )
        )
    return errors


def _load_catalog_declarations(  # noqa: C901
    *,
    scenarios_root: Path,
    exclusion_paths: set[str],
) -> dict[str, list[tuple[int, str, Path, dict[str, Any], list[dict[str, Any]]]]]:
    """Load catalog candidates through the canonical loader and retain provenance.

    Returns:
        dict[str, list[tuple[int, str, Path, dict[str, Any], list[dict[str, Any]]]]]:
            Declarations grouped by case-insensitive scenario identity. The last
            tuple item contains row-level schema and metadata diagnostics.
    """
    from robot_sf.training.scenario_loader import (  # noqa: PLC0415
        _SCENARIO_MANIFEST_KEYS,
        load_scenarios_for_validation,
    )

    repo_root = _find_repo_root()

    curated_roots = [
        scenarios_root / "single",
        scenarios_root / "archetypes",
        scenarios_root,
        scenarios_root / "sets",
    ]
    declarations: dict[str, list[tuple[int, str, Path, dict[str, Any], list[dict[str, Any]]]]] = {}
    for curated_root in curated_roots:
        if not curated_root.exists():
            continue
        for manifest_path in sorted(curated_root.glob("*.yaml")):
            rel_path = manifest_path.relative_to(scenarios_root.parent.parent).as_posix()
            if rel_path in exclusion_paths:
                continue
            if curated_root == scenarios_root and manifest_path.parent != scenarios_root:
                continue
            resolved_manifest = manifest_path.resolve()
            if not _is_within_repo(resolved_manifest, repo_root):
                raise ValueError(
                    f"{REASON_EXTERNAL_ASSET_DEPENDENT}: scenario manifest '{rel_path}' "
                    f"resolves outside the repository boundary: '{resolved_manifest}'."
                )
            raw_manifest = yaml.safe_load(resolved_manifest.read_text(encoding="utf-8"))
            if isinstance(raw_manifest, Mapping) and not _SCENARIO_MANIFEST_KEYS.intersection(
                raw_manifest
            ):
                continue
            if not isinstance(raw_manifest, (Mapping, list)):
                raise ValueError(
                    f"Scenario discovery candidate must contain a manifest: {resolved_manifest}"
                )
            external_errors = _manifest_external_asset_errors(
                raw_manifest,
                source_file=resolved_manifest,
                repo_root=repo_root,
            )
            if external_errors:
                raise ValueError(external_errors[0]["message"])
            report = load_scenarios_for_validation(resolved_manifest)
            if report.load_error:
                raise ValueError(
                    f"Failed to load scenario manifest '{rel_path}': {report.load_error}"
                )
            if report.entry_issues or report.load_issues:
                issue_messages = [
                    issue.message for issue in (*report.entry_issues, *report.load_issues)
                ]
                raise ValueError("; ".join(issue_messages))
            loaded = list(report.scenarios)
            metadata_errors = _manifest_metadata_errors(report.manifest_sources)
            row_diagnostics, orphan_errors = _scenario_validation_diagnostics(
                list(loaded),
                raw_manifest=raw_manifest,
                source_file=resolved_manifest,
                metadata_errors=metadata_errors,
            )
            if orphan_errors:
                raise ValueError("; ".join(error["message"] for error in orphan_errors))
            priority = _catalog_source_priority(resolved_manifest, scenarios_root)
            for index, scenario in enumerate(loaded):
                if not isinstance(scenario, Mapping):  # pragma: no cover - loader contract
                    raise ValueError(
                        f"Scenario loader returned a non-mapping for '{resolved_manifest}'."
                    )
                entry = dict(scenario)
                identity = _scenario_identity(entry, source_file=resolved_manifest)
                declarations.setdefault(identity.casefold(), []).append(
                    (priority, rel_path, resolved_manifest, entry, row_diagnostics[index])
                )
    return declarations


def _build_catalog() -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Build the complete catalog of curated bundled scenarios.

    Returns:
        tuple[dict[str, dict[str, Any]], list[dict[str, str]]]: Discovered catalog and exclusions.
    """
    repo_root = _find_repo_root()
    scenarios_root = repo_root / "configs" / "scenarios"
    exclusion_paths = {exc["path"] for exc in CURATED_EXCLUSIONS}

    declarations = _load_catalog_declarations(
        scenarios_root=scenarios_root,
        exclusion_paths=exclusion_paths,
    )

    catalog: dict[str, dict[str, Any]] = {}
    for sid, declarations_for_id in declarations.items():
        best_priority = min(declaration[0] for declaration in declarations_for_id)
        canonical_declarations = sorted(
            (declaration for declaration in declarations_for_id if declaration[0] == best_priority),
            key=lambda declaration: (declaration[1], declaration[2].as_posix()),
        )
        canonical = canonical_declarations[0]
        (
            _priority,
            _canonical_rel_path,
            canonical_path,
            canonical_entry,
            canonical_diagnostics,
        ) = canonical
        definition_keys = {
            _scenario_definition_key(declaration[3], declaration[2])
            for declaration in canonical_declarations
        }
        duplicate_sources: list[str] = []
        if len(definition_keys) > 1 or (
            len(canonical_declarations) > 1
            and len({declaration[1] for declaration in canonical_declarations}) == 1
        ):
            duplicate_sources = [declaration[1] for declaration in canonical_declarations[1:]]
        summary = _build_scenario_summary(
            canonical_entry,
            source_file=canonical_path,
            repo_root=repo_root,
            duplicate_sources=duplicate_sources,
            validation_errors=canonical_diagnostics,
        )
        catalog[sid] = summary

    return catalog, list(CURATED_EXCLUSIONS)


_CACHED_CATALOG: dict[str, dict[str, Any]] | None = None
_CACHED_EXCLUSIONS: list[dict[str, str]] | None = None


def _get_catalog() -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Return the lazily initialized scenario catalog.

    Returns:
        tuple[dict[str, dict[str, Any]], list[dict[str, str]]]: Cached catalog and exclusions.
    """
    global _CACHED_CATALOG, _CACHED_EXCLUSIONS
    if _CACHED_CATALOG is None or _CACHED_EXCLUSIONS is None:
        _CACHED_CATALOG, _CACHED_EXCLUSIONS = _build_catalog()
    return _CACHED_CATALOG, _CACHED_EXCLUSIONS


def _describe_validation_diagnostics(
    loaded: Mapping[str, Any],
    *,
    source_file: Path,
    catalog_summary: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return canonical diagnostics for a described row, including direct paths."""
    if catalog_summary is not None:
        validation = catalog_summary.get("validation_status", {})
        return (
            [dict(error) for error in validation.get("errors", [])],
            [dict(warning) for warning in validation.get("warnings", [])],
        )

    from robot_sf.training.scenario_loader import (  # noqa: PLC0415
        load_scenarios_for_validation,
    )

    try:
        raw_manifest = yaml.safe_load(source_file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        raw_manifest = None
    report = load_scenarios_for_validation(source_file)
    all_scenarios = list(report.scenarios) or [loaded]
    metadata_errors = _manifest_metadata_errors(report.manifest_sources)
    if not report.manifest_sources:
        metadata_errors = None

    diagnostics, orphan_errors = _scenario_validation_diagnostics(
        all_scenarios,
        raw_manifest=raw_manifest,
        source_file=source_file,
        metadata_errors=metadata_errors,
    )
    loaded_identity = _scenario_diagnostic_identity(loaded, source_file=source_file)
    selected_index = 0
    if loaded_identity is not None:
        for index, scenario in enumerate(all_scenarios):
            if not isinstance(scenario, Mapping):
                continue
            if _scenario_diagnostic_identity(scenario, source_file=source_file) == loaded_identity:
                selected_index = index
                break
    selected_errors = list(diagnostics[selected_index]) if diagnostics else []
    selected_errors.extend(error for error in orphan_errors if error not in selected_errors)
    selected_errors.extend(
        error
        for error in _scenario_load_issue_errors(report.entry_issues, report.load_issues)
        if error not in selected_errors
    )
    if report.load_error:
        selected_errors.append(
            {
                "code": REASON_LOAD_FAILURE,
                "message": f"Failed to load scenarios from '{source_file}': {report.load_error}",
                "scenario_id": None,
                "path": source_file.as_posix(),
                "index": None,
                "source_file": source_file.as_posix(),
            }
        )
    return selected_errors, []


def list_scenarios_payload() -> dict[str, Any]:
    """Return structured payload listing all discovered bundled scenarios.

    Returns:
        dict[str, Any]: Catalog list payload matching scenario_list.v1 schema.
    """
    catalog, exclusions = _get_catalog()
    sorted_scenarios = [catalog[k] for k in sorted(catalog.keys())]
    return {
        "schema_version": SCHEMA_LIST_VERSION,
        "count": len(sorted_scenarios),
        "scenarios": sorted_scenarios,
        "exclusions": exclusions,
    }


def describe_scenario_payload(query: str) -> dict[str, Any]:
    """Return structured description for a single scenario by id, name, or path.

    Args:
        query: Scenario identity, file stem, or relative file path.

    Returns:
        dict[str, Any]: Structured dictionary matching scenario_describe.v1 schema.

    Raises:
        KeyError: If the query cannot be resolved to any scenario.
        ValueError: If the query is ambiguous between multiple conflicting definitions.
    """
    from robot_sf.api import load_scenario  # noqa: PLC0415

    repo_root = _find_repo_root()
    q_str = str(query).strip()
    direct_candidate = Path(q_str).resolve()
    is_direct_path = direct_candidate.is_file()
    if is_direct_path and not _is_within_repo(direct_candidate, repo_root):
        raise ValueError(
            f"{REASON_EXTERNAL_ASSET_DEPENDENT}: scenario path '{query}' is outside "
            "the repository boundary."
        )
    if is_direct_path:
        try:
            raw_direct = yaml.safe_load(direct_candidate.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"{REASON_MALFORMED_YAML}: {exc}") from exc
        external_direct_errors = _manifest_external_asset_errors(
            raw_direct,
            source_file=direct_candidate,
            repo_root=repo_root,
        )
        if external_direct_errors:
            raise ValueError(external_direct_errors[0]["message"])

    catalog, _ = _get_catalog()
    try:
        loaded = load_scenario(q_str)
    except FileNotFoundError as exc:
        raise KeyError(
            f"Unknown scenario '{query}'. Available scenarios can be listed with 'robot-sf scenarios list'."
        ) from exc
    except ValueError:
        raise

    source_path_str = loaded.get("__scenario_path__")
    source_file = Path(source_path_str) if source_path_str else repo_root
    if not _is_within_repo(source_file, repo_root):
        raise ValueError(
            f"{REASON_EXTERNAL_ASSET_DEPENDENT}: scenario source '{source_file}' is outside "
            "the repository boundary."
        )
    sid = str(
        loaded.get("name") or loaded.get("scenario_id") or loaded.get("id") or source_file.stem
    ).strip()

    dup_sources: list[str] = []
    catalog_summary = next(
        (summary for key, summary in catalog.items() if key.casefold() == sid.casefold()),
        None,
    )
    if catalog_summary is not None:
        dup_sources = catalog_summary.get("duplicate_sources", [])

    validation_errors, validation_warnings = _describe_validation_diagnostics(
        loaded,
        source_file=source_file,
        catalog_summary=None if is_direct_path else catalog_summary,
    )
    summary = _build_scenario_summary(
        loaded,
        source_file=source_file,
        repo_root=repo_root,
        duplicate_sources=dup_sources,
        validation_errors=validation_errors,
        validation_warnings=validation_warnings,
    )
    payload = dict(summary)
    payload["schema_version"] = SCHEMA_DESCRIBE_VERSION
    payload["requested_query"] = query
    payload["raw_metadata"] = loaded.get("metadata", {})
    return payload


def _validate_path_and_syntax(
    path_str: str,
    repo_root: Path,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Check path escape, existence, and basic YAML readability.

    Returns:
        tuple[Path | None, dict[str, Any] | None]: Resolved path and error payload if invalid.
    """
    raw_path = Path(path_str)
    try:
        resolved = raw_path.resolve()
        resolved.relative_to(repo_root.resolve())
    except (ValueError, RuntimeError):
        return None, {
            "schema_version": SCHEMA_VALIDATE_VERSION,
            "target_path": str(path_str),
            "resolved_path": str(raw_path),
            "valid": False,
            "status": STATUS_INVALID,
            "num_scenarios": 0,
            "scenarios": [],
            "errors": [
                {
                    "code": REASON_PATH_TRAVERSAL,
                    "message": f"Path '{path_str}' traverses outside repository boundary.",
                    "scenario_id": None,
                    "path": str(path_str),
                }
            ],
            "warnings": [],
        }

    if not resolved.exists():
        return resolved, {
            "schema_version": SCHEMA_VALIDATE_VERSION,
            "target_path": str(path_str),
            "resolved_path": str(resolved),
            "valid": False,
            "status": STATUS_MISSING,
            "num_scenarios": 0,
            "scenarios": [],
            "errors": [
                {
                    "code": REASON_FILE_NOT_FOUND,
                    "message": f"Scenario file not found: '{path_str}'.",
                    "scenario_id": None,
                    "path": str(path_str),
                }
            ],
            "warnings": [],
        }

    if resolved.is_dir():
        return resolved, {
            "schema_version": SCHEMA_VALIDATE_VERSION,
            "target_path": str(path_str),
            "resolved_path": str(resolved),
            "valid": False,
            "status": STATUS_INVALID,
            "num_scenarios": 0,
            "scenarios": [],
            "errors": [
                {
                    "code": REASON_IS_A_DIRECTORY,
                    "message": f"Expected a scenario file, but '{path_str}' is a directory.",
                    "scenario_id": None,
                    "path": str(path_str),
                }
            ],
            "warnings": [],
        }

    try:
        with resolved.open("r", encoding="utf-8") as fh:
            raw_text = fh.read()
        if not raw_text.strip():
            return resolved, {
                "schema_version": SCHEMA_VALIDATE_VERSION,
                "target_path": str(path_str),
                "resolved_path": str(resolved),
                "valid": False,
                "status": STATUS_INVALID,
                "num_scenarios": 0,
                "scenarios": [],
                "errors": [
                    {
                        "code": REASON_EMPTY_FILE,
                        "message": f"Scenario file is empty: '{path_str}'.",
                        "scenario_id": None,
                        "path": str(path_str),
                    }
                ],
                "warnings": [],
            }
        yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        return resolved, {
            "schema_version": SCHEMA_VALIDATE_VERSION,
            "target_path": str(path_str),
            "resolved_path": str(resolved),
            "valid": False,
            "status": STATUS_INVALID,
            "num_scenarios": 0,
            "scenarios": [],
            "errors": [
                {
                    "code": REASON_MALFORMED_YAML,
                    "message": f"Malformed YAML in '{path_str}': {exc}",
                    "scenario_id": None,
                    "path": str(path_str),
                }
            ],
            "warnings": [],
        }

    return resolved, None


def _schema_errors_to_cli_errors(
    schema_errors: list[Mapping[str, Any]],
    *,
    scenarios: list[Mapping[str, Any]] | None = None,
    source_file: Path | None = None,
) -> list[dict[str, Any]]:
    """Convert canonical schema-validator errors to the CLI error envelope.

    Returns:
        list[dict[str, Any]]: Normalized CLI validation errors.
    """
    errors: list[dict[str, Any]] = []
    for schema_error in schema_errors:
        err_msg = schema_error.get("error", "Schema validation error")
        code = (
            REASON_DUPLICATE_SCENARIO_ID
            if "duplicate" in str(err_msg).lower()
            else REASON_SCHEMA_VALIDATION_ERROR
        )
        scenario_id = schema_error.get("id")
        index = schema_error.get("index")
        if scenario_id is None and scenarios is not None and isinstance(index, int):
            if 0 <= index < len(scenarios):
                scenario_id = _scenario_diagnostic_identity(
                    scenarios[index],
                    source_file=Path("scenario.yaml"),
                )
        error = {
            "code": code,
            "message": err_msg,
            "scenario_id": scenario_id,
            "path": schema_error.get("path"),
            "index": index,
        }
        if source_file is not None:
            error["source_file"] = source_file.as_posix()
        errors.append(error)
    return errors


def _scenario_load_issue_errors(
    entry_issues: list[Any],
    load_issues: list[Any],
) -> list[dict[str, Any]]:
    """Convert tolerant canonical-loader issues into stable CLI diagnostics.

    Returns:
        list[dict[str, Any]]: Stable row and manifest load diagnostics.
    """
    errors: list[dict[str, Any]] = []
    for issue in entry_issues:
        index = issue.index
        path = f"/scenarios/{index}" if index is not None else issue.source.as_posix()
        scenario_id = (
            _scenario_diagnostic_identity(issue.entry, source_file=issue.source)
            if isinstance(issue.entry, Mapping)
            else None
        )
        if index is not None and not isinstance(issue.entry, Mapping):
            errors.append(
                {
                    "code": REASON_SCHEMA_VALIDATION_ERROR,
                    "message": issue.message,
                    "scenario_id": scenario_id,
                    "path": path,
                    "index": index,
                    "source_file": issue.source.as_posix(),
                }
            )
        errors.append(
            {
                "code": REASON_LOAD_FAILURE,
                "message": (
                    f"Failed to load scenario entry {index} from '{issue.source}': {issue.message}"
                    if index is not None
                    else f"Failed to load scenarios from '{issue.source}': {issue.message}"
                ),
                "scenario_id": scenario_id,
                "path": path,
                "index": index,
                "source_file": issue.source.as_posix(),
            }
        )
    for issue in load_issues:
        errors.append(
            {
                "code": REASON_LOAD_FAILURE,
                "message": f"Failed to load scenarios from '{issue.source}': {issue.message}",
                "scenario_id": None,
                "path": issue.source.as_posix(),
                "index": None,
                "source_file": issue.source.as_posix(),
            }
        )
    return errors


def _scenario_validation_diagnostics(  # noqa: C901
    scenarios: list[Any],
    *,
    raw_manifest: Any,
    source_file: Path,
    metadata_errors: list[Mapping[str, Any]] | None = None,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Collect canonical and CLI contract diagnostics for each scenario row.

    The benchmark validator remains authoritative for schema and manifest metadata
    checks. The CLI adds a narrow unknown-field check because the canonical item
    schema intentionally allows extension keys. Non-mapping rows are retained as
    explicit diagnostics instead of being filtered out.

    Returns:
        tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]: Row diagnostics
        followed by diagnostics that cannot be associated with a loaded row.
    """
    from robot_sf.benchmark.scenario.scenario_schema import (  # noqa: PLC0415
        validate_scenario_list,
        validate_scenario_matrix_metadata,
    )

    diagnostics: list[list[dict[str, Any]]] = [[] for _ in scenarios]
    if metadata_errors is None:
        metadata_errors = _schema_errors_to_cli_errors(
            validate_scenario_matrix_metadata(raw_manifest),
            source_file=source_file,
        )
    else:
        metadata_errors = [dict(error) for error in metadata_errors]
    orphan_errors: list[dict[str, Any]] = []

    mapping_rows: list[Mapping[str, Any]] = []
    original_indices: list[int] = []
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, Mapping):
            malformed_error = {
                "code": REASON_SCHEMA_VALIDATION_ERROR,
                "message": (
                    f"Scenario entry {index} in '{source_file}' must be a mapping; "
                    f"got {type(scenario).__name__}."
                ),
                "scenario_id": None,
                "path": f"/scenarios/{index}",
                "index": index,
            }
            diagnostics[index].append(malformed_error)
            orphan_errors.append(malformed_error)
            continue
        mapping_rows.append(scenario)
        original_indices.append(index)
        unknown_fields = sorted(
            (str(key) for key in scenario if str(key) not in _KNOWN_SCENARIO_FIELDS),
            key=str,
        )
        for field in unknown_fields:
            diagnostics[index].append(
                {
                    "code": REASON_SCHEMA_VALIDATION_ERROR,
                    "message": f"Unknown scenario field '{field}'.",
                    "scenario_id": _scenario_diagnostic_identity(
                        scenario,
                        source_file=source_file,
                    ),
                    "path": f"/{field}",
                    "index": index,
                }
            )

    canonical_errors = validate_scenario_list([dict(row) for row in mapping_rows])
    for schema_error in canonical_errors:
        compact_index = schema_error.get("index")
        if not isinstance(compact_index, int) or not 0 <= compact_index < len(original_indices):
            orphan_errors.extend(_schema_errors_to_cli_errors([schema_error]))
            continue
        original_index = original_indices[compact_index]
        converted = _schema_errors_to_cli_errors(
            [schema_error],
            scenarios=mapping_rows,
        )[0]
        converted["index"] = original_index
        diagnostics[original_index].append(converted)
        if converted["code"] == REASON_DUPLICATE_SCENARIO_ID:
            details = schema_error.get("details", {})
            first_index = details.get("first_index") if isinstance(details, Mapping) else None
            if isinstance(first_index, int) and 0 <= first_index < len(original_indices):
                first_original_index = original_indices[first_index]
                first_error = dict(converted)
                first_error["index"] = first_original_index
                if first_error not in diagnostics[first_original_index]:
                    diagnostics[first_original_index].append(first_error)

    if metadata_errors:
        if diagnostics:
            for row_diagnostics in diagnostics:
                row_diagnostics.extend(dict(error) for error in metadata_errors)
        else:
            orphan_errors.extend(metadata_errors)

    return diagnostics, orphan_errors


def _manifest_external_asset_errors(
    raw_manifest: Any,
    *,
    source_file: Path,
    repo_root: Path,
    _visited: set[Path] | None = None,
) -> list[dict[str, Any]]:
    """Reject manifest include/search references that leave the repository.

    In-repository includes are inspected recursively before the canonical loader
    runs, so an external reference hidden in a nested manifest cannot be loaded
    as an apparently local scenario.

    Returns:
        list[dict[str, Any]]: External-asset diagnostics.
    """
    if not isinstance(raw_manifest, Mapping):
        return []

    visited = _visited if _visited is not None else set()
    resolved_source = source_file.resolve()
    if resolved_source in visited:
        return []
    visited.add(resolved_source)

    return [
        error
        for key in ("includes", "include", "scenario_files", "map_search_paths")
        for error in _manifest_reference_external_asset_errors(
            raw_manifest.get(key),
            key=key,
            source_file=source_file,
            repo_root=repo_root,
            visited=visited,
        )
    ]


def _manifest_reference_external_asset_errors(
    raw_values: Any,
    *,
    key: str,
    source_file: Path,
    repo_root: Path,
    visited: set[Path],
) -> list[dict[str, Any]]:
    """Inspect one manifest reference field for boundary violations.

    Returns:
        list[dict[str, Any]]: External-asset diagnostics for the field and its includes.
    """
    if raw_values is None:
        return []
    values = [raw_values] if isinstance(raw_values, (str, Path)) else raw_values
    if not isinstance(values, list):
        return []

    errors: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        resolved = _resolve_asset_path(value, source_file=source_file)
        if resolved is None:
            continue
        if not _is_within_repo(resolved, repo_root):
            errors.append(
                {
                    "code": REASON_EXTERNAL_ASSET_DEPENDENT,
                    "message": (
                        f"Manifest '{source_file}' references an external asset through "
                        f"{key}[{index}]: '{value}'."
                    ),
                    "scenario_id": None,
                    "path": f"/{key}/{index}",
                }
            )
            continue
        if key in {"includes", "include", "scenario_files"} and resolved.is_file():
            errors.extend(_nested_manifest_external_asset_errors(resolved, repo_root, visited))
    return errors


def _nested_manifest_external_asset_errors(
    source_file: Path,
    repo_root: Path,
    visited: set[Path],
) -> list[dict[str, Any]]:
    """Inspect one readable in-repository include for external references.

    Returns:
        list[dict[str, Any]]: External-asset diagnostics found below the include.
    """
    try:
        included_manifest = yaml.safe_load(source_file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        # The canonical loader will report this as a load failure. The
        # preflight only needs to prevent external assets from being read.
        return []
    return _manifest_external_asset_errors(
        included_manifest,
        source_file=source_file,
        repo_root=repo_root,
        _visited=visited,
    )


def _check_scenario_assets(
    scenarios: list[Any],
    resolved: Path,
    repo_root: Path,
    validation_errors_by_scenario: list[list[Mapping[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Build summaries and aggregate schema, metadata, and asset diagnostics.

    Returns:
        tuple: Summaries list, errors list, warnings list, and missing asset flag.
    """
    scenario_summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    has_missing_asset = False

    for index, s in enumerate(scenarios):
        if not isinstance(s, Mapping):
            raise ValueError(
                f"Scenario entry {index} in '{resolved}' must be a mapping; got {type(s).__name__}."
            )
        s_dict = dict(s)
        row_validation_errors = (
            validation_errors_by_scenario[index]
            if validation_errors_by_scenario is not None
            and index < len(validation_errors_by_scenario)
            else None
        )
        summary = _build_scenario_summary(
            s_dict,
            source_file=resolved,
            repo_root=repo_root,
            validation_errors=row_validation_errors,
        )
        scenario_summaries.append(summary)

        validation = summary["validation_status"]
        errors.extend(validation.get("errors", []))
        warnings.extend(validation.get("warnings", []))
        if any(
            error.get("code") in {REASON_MAP_NOT_FOUND, REASON_ROUTE_OVERRIDES_NOT_FOUND}
            for error in validation.get("errors", [])
        ):
            has_missing_asset = True

    return scenario_summaries, errors, warnings, has_missing_asset


def _manifest_scenario_entries(raw_manifest: Any) -> list[Any]:
    """Return direct scenario entries without dropping malformed rows."""
    if isinstance(raw_manifest, list):
        return list(raw_manifest)
    if isinstance(raw_manifest, Mapping):
        entries = raw_manifest.get("scenarios")
        if isinstance(entries, list):
            return list(entries)
    return []


def _deduplicate_diagnostics(errors: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Preserve diagnostic order while removing repeated aggregate errors.

    Returns:
        list[dict[str, Any]]: De-duplicated diagnostic dictionaries.
    """
    unique: list[dict[str, Any]] = []
    for error in errors:
        normalized = dict(error)
        if normalized not in unique:
            unique.append(normalized)
    return unique


def _status_from_diagnostics(
    *,
    errors: list[Mapping[str, Any]],
    has_missing_asset: bool,
) -> str:
    """Choose the stable validate status from the complete diagnostic set.

    Returns:
        str: Stable validation status.
    """
    if not errors:
        return STATUS_VALID
    if any(error.get("code") == REASON_EXTERNAL_ASSET_DEPENDENT for error in errors):
        return STATUS_EXTERNAL_ASSET_DEPENDENT
    non_status_errors = {
        error.get("code")
        for error in errors
        if error.get("code")
        not in {
            REASON_UNSUPPORTED_SCENARIO,
            REASON_MAP_NOT_FOUND,
            REASON_ROUTE_OVERRIDES_NOT_FOUND,
        }
    }
    if not non_status_errors and any(
        error.get("code") == REASON_UNSUPPORTED_SCENARIO for error in errors
    ):
        return STATUS_UNSUPPORTED
    if has_missing_asset and not non_status_errors:
        return STATUS_MISSING
    return STATUS_INVALID


def validate_scenario_payload(path_str: str) -> dict[str, Any]:  # noqa: C901
    """Validate a scenario YAML file or manifest against schema and filesystem rules.

    Args:
        path_str: Path to scenario file to validate.

    Returns:
        dict[str, Any]: Structured dictionary matching scenario_validate.v1 schema.
    """
    repo_root = _find_repo_root()
    resolved, early_error = _validate_path_and_syntax(path_str, repo_root)
    if early_error is not None or resolved is None:
        return early_error or {}

    from robot_sf.training.scenario_loader import (  # noqa: PLC0415
        load_scenarios_for_validation,
    )

    try:
        raw_manifest = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - guarded by the preflight above
        return {
            "schema_version": SCHEMA_VALIDATE_VERSION,
            "target_path": str(path_str),
            "resolved_path": str(resolved),
            "valid": False,
            "status": STATUS_INVALID,
            "num_scenarios": 0,
            "scenarios": [],
            "errors": [
                {
                    "code": REASON_MALFORMED_YAML,
                    "message": f"Could not read YAML in '{path_str}': {exc}",
                    "scenario_id": None,
                    "path": str(path_str),
                }
            ],
            "warnings": [],
        }

    raw_entries = _manifest_scenario_entries(raw_manifest)
    manifest_asset_errors = _manifest_external_asset_errors(
        raw_manifest,
        source_file=resolved,
        repo_root=repo_root,
    )

    def report_rows(
        entries: list[Any],
        diagnostics: list[list[dict[str, Any]]],
    ) -> tuple[list[Mapping[str, Any]], list[list[dict[str, Any]]]]:
        mappings: list[Mapping[str, Any]] = []
        mapping_diagnostics: list[list[dict[str, Any]]] = []
        for index, entry in enumerate(entries):
            if isinstance(entry, Mapping):
                mappings.append(entry)
                mapping_diagnostics.append(diagnostics[index])
        return mappings, mapping_diagnostics

    if manifest_asset_errors:
        diagnostics, orphan_errors = _scenario_validation_diagnostics(
            raw_entries,
            raw_manifest=raw_manifest,
            source_file=resolved,
        )
        rows, row_diagnostics = report_rows(raw_entries, diagnostics)
        summaries, asset_errors, warnings, has_missing = _check_scenario_assets(
            list(rows),
            resolved,
            repo_root,
            validation_errors_by_scenario=row_diagnostics,
        )
        errors = _deduplicate_diagnostics([*manifest_asset_errors, *orphan_errors, *asset_errors])
        return {
            "schema_version": SCHEMA_VALIDATE_VERSION,
            "target_path": str(path_str),
            "resolved_path": str(resolved),
            "valid": False,
            "status": _status_from_diagnostics(errors=errors, has_missing_asset=has_missing),
            "num_scenarios": len(raw_entries),
            "scenarios": summaries,
            "errors": errors,
            "warnings": warnings,
        }

    report = load_scenarios_for_validation(resolved)
    loaded_scenarios = list(report.scenarios)
    if report.manifest_sources:
        metadata_errors = _manifest_metadata_errors(report.manifest_sources)
    else:
        from robot_sf.benchmark.scenario.scenario_schema import (  # noqa: PLC0415
            validate_scenario_matrix_metadata,
        )

        metadata_errors = _schema_errors_to_cli_errors(
            validate_scenario_matrix_metadata(raw_manifest),
            source_file=resolved,
        )
    diagnostics, orphan_errors = _scenario_validation_diagnostics(
        loaded_scenarios,
        raw_manifest=raw_manifest,
        source_file=resolved,
        metadata_errors=metadata_errors,
    )
    summaries, asset_errors, warnings, _has_missing = _check_scenario_assets(
        loaded_scenarios,
        resolved,
        repo_root,
        validation_errors_by_scenario=diagnostics,
    )
    errors = [*orphan_errors, *asset_errors]
    errors.extend(_scenario_load_issue_errors(report.entry_issues, report.load_issues))
    if report.load_error:
        errors.append(
            {
                "code": REASON_LOAD_FAILURE,
                "message": f"Failed to load scenarios from '{path_str}': {report.load_error}",
                "scenario_id": None,
                "path": str(path_str),
                "index": None,
                "source_file": resolved.as_posix(),
            }
        )
    if manifest_asset_errors:
        errors.extend(manifest_asset_errors)
    errors = _deduplicate_diagnostics(errors)

    # A manifest can be valid as a loader input but invalid under the canonical
    # schema or the CLI's strict field contract. Keep the loader's expanded row
    # count while preserving all row-level diagnostics above.
    is_valid = not errors
    has_missing = any(
        error.get("code") in {REASON_MAP_NOT_FOUND, REASON_ROUTE_OVERRIDES_NOT_FOUND}
        for error in errors
    )
    malformed_row_count = sum(issue.index is not None for issue in report.entry_issues)
    if report.entry_issues or report.load_issues or report.load_error:
        num_scenarios = max(
            report.raw_entry_count,
            len(loaded_scenarios) + malformed_row_count,
            len(raw_entries) if report.load_error else 0,
        )
    else:
        num_scenarios = len(loaded_scenarios)
    return {
        "schema_version": SCHEMA_VALIDATE_VERSION,
        "target_path": str(path_str),
        "resolved_path": str(resolved),
        "valid": is_valid,
        "status": _status_from_diagnostics(errors=errors, has_missing_asset=has_missing),
        "num_scenarios": num_scenarios,
        "scenarios": summaries,
        "errors": errors,
        "warnings": warnings,
    }


def _format_list_scenarios(payload: dict[str, Any]) -> str:
    """Format scenarios list for human-friendly terminal display.

    Returns:
        str: Formatted terminal text.
    """
    lines: list[str] = [
        f"Curated Scenarios ({payload['count']}):",
        f"schema_version: {payload.get('schema_version')}",
        f"count: {payload.get('count')}",
        "",
    ]
    for s in payload["scenarios"]:
        fam = f" ({s['family']})" if s.get("family") else ""
        lines.append(f"- {s['identity']}  [{s['status']}]{fam}")
        lines.append(f"    source: {s['source_file']}")
        map_ref = s.get("map_reference", {})
        m_str = map_ref.get("map_file") or map_ref.get("map_id") or "none"
        exists_str = "exists" if map_ref.get("map_exists") else "not found"
        lines.append(f"    map: {m_str} [{exists_str}]")
        lines.append(f"    map_file: {map_ref.get('map_file')}")
        lines.append(f"    map_id: {map_ref.get('map_id')}")
        lines.append(f"    route_overrides_file: {map_ref.get('route_overrides_file')}")
        lines.append(f"    resolved_map_path: {map_ref.get('resolved_map_path')}")
        lines.append(
            f"    external_asset_dependent: {map_ref.get('external_asset_dependent', False)}"
        )
        lines.append(
            "    external_map_asset_dependent: "
            f"{map_ref.get('external_map_asset_dependent', False)}"
        )
        lines.append(
            "    external_route_asset_dependent: "
            f"{map_ref.get('external_route_asset_dependent', False)}"
        )
        ac = s.get("actor_counts", {})
        lines.append(
            f"    actors: ped_density={ac.get('ped_density')}, single_peds={ac.get('single_pedestrians')}, robots={ac.get('num_robots')}"
        )
        lines.append(f"    groups: {ac.get('groups')}")
        sp = s.get("seed_policy", {})
        lines.append(f"    seeds: count={sp.get('count')}, seeds={sp.get('seeds')}")
        lines.append(f"    repeats: {sp.get('repeats')}")
        hd = s.get("horizon_and_dt", {})
        lines.append(
            f"    horizon: {hd.get('max_episode_steps')} steps "
            f"(dt={hd.get('time_step')}, sim_time={hd.get('sim_time_s')}s)"
        )
        ko = s.get("kinematics_and_observation", {})
        lines.append(f"    kinematics: {ko.get('kinematics')}")
        lines.append(
            f"    observation_visibility_enabled: {ko.get('observation_visibility_enabled')}"
        )
        lines.append(f"    fov_degrees: {ko.get('fov_degrees')}")
        lines.append(f"    max_range_m: {ko.get('max_range_m')}")
        lines.append(f"    duplicate_sources: {s.get('duplicate_sources', [])}")
        validation = s.get("validation_status", {})
        lines.append(
            "    validation: "
            f"{validation.get('status', s.get('status'))} "
            f"valid={validation.get('valid', False)} "
            f"({len(validation.get('errors', []))} error(s))"
        )
        lines.append(f"    validation reasons: {validation.get('reasons', [])}")
        for error in validation.get("errors", []):
            lines.append(f"      [{error.get('code')}] {error.get('message')}")
        for warning in validation.get("warnings", []):
            lines.append(f"      warning [{warning.get('code')}] {warning.get('message')}")
        lines.append("")

    if payload.get("exclusions"):
        lines.append(f"Exclusions ({len(payload['exclusions'])}):")
        for exc in payload["exclusions"]:
            lines.append(f"- {exc['path']}: {exc['reason']}")
        lines.append("")

    lines.append("Contract facts:")
    _append_contract_facts(lines, payload, indent="  ")

    return "\n".join(lines)


def _append_contract_facts(lines: list[str], value: Any, *, indent: str) -> None:
    """Append every JSON contract value in deterministic, readable form."""
    if isinstance(value, Mapping):
        if not value:
            lines.append(f"{indent}{{}}")
            return
        for key in sorted(value, key=str):
            child = value[key]
            if isinstance(child, (Mapping, list)):
                lines.append(f"{indent}{key}:")
                _append_contract_facts(lines, child, indent=f"{indent}  ")
            else:
                lines.append(f"{indent}{key}: {json.dumps(child, sort_keys=True, default=str)}")
        return
    if isinstance(value, list):
        if not value:
            lines.append(f"{indent}[]")
            return
        for child in value:
            if isinstance(child, (Mapping, list)):
                lines.append(f"{indent}-")
                _append_contract_facts(lines, child, indent=f"{indent}  ")
            else:
                lines.append(f"{indent}- {json.dumps(child, sort_keys=True, default=str)}")
        return
    lines.append(f"{indent}{json.dumps(value, sort_keys=True, default=str)}")


def _format_describe_scenario(payload: dict[str, Any]) -> str:
    """Format scenario description for human-friendly terminal display.

    Returns:
        str: Formatted terminal text.
    """
    lines: list[str] = [f"Scenario: {payload['identity']}\n"]
    lines.append(f"  Schema Version: {payload.get('schema_version')}")
    lines.append(f"  Requested Query: {payload.get('requested_query')}")
    lines.append(f"  Source File: {payload['source_file']}")
    if payload.get("duplicate_sources"):
        lines.append(f"  Duplicate Sources: {', '.join(payload['duplicate_sources'])}")
    lines.append(f"  Family: {payload.get('family') or 'none'}")
    lines.append(f"  Status: {payload['status']}")

    map_ref = payload.get("map_reference", {})
    lines.append("  Map Reference:")
    lines.append(f"    map_file: {map_ref.get('map_file')}")
    lines.append(f"    map_id: {map_ref.get('map_id')}")
    lines.append(f"    route_overrides: {map_ref.get('route_overrides_file')}")
    lines.append(f"    resolved_path: {map_ref.get('resolved_map_path')}")
    lines.append(f"    map_exists: {map_ref.get('map_exists')}")
    lines.append(f"    external_asset_dependent: {map_ref.get('external_asset_dependent', False)}")
    lines.append(
        f"    external_map_asset_dependent: {map_ref.get('external_map_asset_dependent', False)}"
    )
    lines.append(
        "    external_route_asset_dependent: "
        f"{map_ref.get('external_route_asset_dependent', False)}"
    )

    ac = payload.get("actor_counts", {})
    lines.append("  Actor Counts:")
    lines.append(f"    ped_density: {ac.get('ped_density')}")
    lines.append(f"    single_pedestrians: {ac.get('single_pedestrians')}")
    lines.append(f"    num_robots: {ac.get('num_robots')}")
    lines.append(f"    groups: {ac.get('groups')}")

    sp = payload.get("seed_policy", {})
    lines.append("  Seed Policy:")
    lines.append(f"    count: {sp.get('count')}")
    lines.append(f"    seeds: {sp.get('seeds')}")
    lines.append(f"    repeats: {sp.get('repeats')}")

    hd = payload.get("horizon_and_dt", {})
    lines.append("  Horizon / Timing:")
    lines.append(f"    max_episode_steps: {hd.get('max_episode_steps')}")
    lines.append(f"    time_step: {hd.get('time_step')}")
    lines.append(f"    sim_time_s: {hd.get('sim_time_s')}")

    ko = payload.get("kinematics_and_observation", {})
    lines.append("  Kinematics / Observation:")
    lines.append(f"    kinematics: {ko.get('kinematics')}")
    lines.append(f"    observation_visibility: {ko.get('observation_visibility_enabled')}")
    lines.append(f"    fov_degrees: {ko.get('fov_degrees')}")
    lines.append(f"    max_range_m: {ko.get('max_range_m')}")

    vs = payload.get("validation_status", {})
    lines.append("  Validation:")
    lines.append(f"    valid: {vs.get('valid')}")
    lines.append(f"    status: {vs.get('status', payload.get('status'))}")
    lines.append(f"    reasons: {', '.join(vs.get('reasons', [])) or 'none'}")
    for error in vs.get("errors", []):
        lines.append(f"    [{error.get('code')}] {error.get('message')}")
    for warning in vs.get("warnings", []):
        lines.append(f"    warning [{warning.get('code')}] {warning.get('message')}")
    lines.append(
        "  Raw Metadata: "
        f"{json.dumps(payload.get('raw_metadata', {}), sort_keys=True, default=str)}"
    )
    lines.append("\nContract facts:")
    _append_contract_facts(lines, payload, indent="  ")

    return "\n".join(lines)


def _format_validate_scenario(payload: dict[str, Any]) -> str:
    """Format validation report for human-friendly terminal display.

    Returns:
        str: Formatted terminal text.
    """
    status_str = "PASS" if payload["valid"] else "FAIL"
    lines: list[str] = [
        f"Scenario validation: {status_str}",
        f"Schema version: {payload.get('schema_version')}",
        f"Target: {payload['target_path']} ({payload['num_scenarios']} scenario(s))",
        f"Resolved path: {payload.get('resolved_path')}",
        f"Valid: {payload.get('valid')}",
        f"Status: {payload['status']}",
    ]

    if payload["errors"]:
        lines.append("\nErrors:")
        for err in payload["errors"]:
            sid_info = f" [{err['scenario_id']}]" if err.get("scenario_id") else ""
            path_info = f" ({err['path']})" if err.get("path") else ""
            lines.append(f"- [{err['code']}]{sid_info}{path_info}: {err['message']}")

    if payload["warnings"]:
        lines.append("\nWarnings:")
        for warn in payload["warnings"]:
            sid_info = f" [{warn['scenario_id']}]" if warn.get("scenario_id") else ""
            path_info = f" ({warn['path']})" if warn.get("path") else ""
            lines.append(f"- [{warn['code']}]{sid_info}{path_info}: {warn['message']}")

    lines.append("\nContract facts:")
    _append_contract_facts(lines, payload, indent="  ")

    return "\n".join(lines)


def _handle_scenarios_list(args: argparse.Namespace) -> int:
    """Handle ``robot-sf scenarios list`` command.

    Returns:
        int: Process exit code (0 for success).
    """
    try:
        payload = list_scenarios_payload()
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        if getattr(args, "format", "friendly") == "json":
            error_payload = {
                "schema_version": SCHEMA_LIST_VERSION,
                "status": "error",
                "count": 0,
                "scenarios": [],
                "exclusions": list(CURATED_EXCLUSIONS),
                "error": str(exc),
            }
            sys.stdout.write(json.dumps(error_payload, indent=2, sort_keys=True) + "\n")
        else:
            sys.stderr.write(f"Error: could not discover scenarios: {exc}\n")
        return 2
    if getattr(args, "format", "friendly") == "json":
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(_format_list_scenarios(payload))
    return 0


def _handle_scenarios_describe(args: argparse.Namespace) -> int:
    """Handle ``robot-sf scenarios describe <id-or-name>`` command.

    Returns:
        int: Process exit code (0 for success, 2 for error).
    """
    query = getattr(args, "id_or_name", "")
    try:
        payload = describe_scenario_payload(query)
    except (KeyError, ValueError) as exc:
        if getattr(args, "format", "friendly") == "json":
            err_payload = {
                "schema_version": SCHEMA_DESCRIBE_VERSION,
                "status": "error",
                "requested_query": query,
                "error": str(exc),
            }
            sys.stdout.write(json.dumps(err_payload, indent=2, sort_keys=True) + "\n")
        else:
            sys.stderr.write(f"Error: {exc}\n")
        return 2

    if getattr(args, "format", "friendly") == "json":
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(_format_describe_scenario(payload) + "\n")
    return 0


def _handle_scenarios_validate(args: argparse.Namespace) -> int:
    """Handle ``robot-sf scenarios validate <path>`` command.

    Returns:
        int: Process exit code (0 for valid, 2 for error/invalid).
    """
    path_str = getattr(args, "path", "")
    payload = validate_scenario_payload(path_str)

    if getattr(args, "format", "friendly") == "json":
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(_format_validate_scenario(payload) + "\n")

    return 0 if payload["valid"] else 2


def _handle_scenarios(args: argparse.Namespace) -> int:
    """Dispatch the ``robot-sf scenarios`` subcommand.

    Returns:
        int: Process-style exit code (0 success, 2 on failure/error).
    """
    cmd = getattr(args, "scenarios_cmd", None)
    if cmd == "list":
        return _handle_scenarios_list(args)
    if cmd == "describe":
        return _handle_scenarios_describe(args)
    if cmd == "validate":
        return _handle_scenarios_validate(args)
    sys.stderr.write(f"Unknown scenarios command: {cmd}\n")
    return 2


def _add_scenarios_subparser(sub: Any) -> None:
    """Register the ``robot-sf scenarios`` subcommand tree."""
    scenarios = sub.add_parser(
        "scenarios",
        help="Discover, inspect, and validate scenarios (uv run robot-sf scenarios ...)",
    )
    scenarios_sub = scenarios.add_subparsers(dest="scenarios_cmd", required=True)

    slist = scenarios_sub.add_parser("list", help="List discovered bundled scenarios.")
    slist.add_argument(
        "--format",
        choices=("friendly", "json"),
        default="friendly",
        help="Output format (default: friendly).",
    )

    sdescribe = scenarios_sub.add_parser(
        "describe", help="Describe one scenario by id, name, or path."
    )
    sdescribe.add_argument("id_or_name", help="Scenario id, name, file stem, or path.")
    sdescribe.add_argument(
        "--format",
        choices=("friendly", "json"),
        default="friendly",
        help="Output format (default: friendly).",
    )

    svalidate = scenarios_sub.add_parser(
        "validate", help="Validate a scenario file against schema and map rules."
    )
    svalidate.add_argument("path", help="Path to scenario YAML file to validate.")
    svalidate.add_argument(
        "--format",
        choices=("friendly", "json"),
        default="friendly",
        help="Output format (default: friendly).",
    )
