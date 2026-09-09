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
from collections import OrderedDict
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

    resolved_map: Path | None = None
    map_exists = False

    if isinstance(map_file, str) and map_file.strip():
        cand = Path(map_file)
        if not cand.is_absolute():
            p1 = (source_file.parent / cand).resolve()
            p2 = (repo_root / cand).resolve()
            p3 = (repo_root / "maps" / "svg_maps" / cand.name).resolve()
            if p1.exists() and p1.is_file():
                resolved_map = p1
                map_exists = True
            elif p2.exists() and p2.is_file():
                resolved_map = p2
                map_exists = True
            elif p3.exists() and p3.is_file():
                resolved_map = p3
                map_exists = True
            else:
                resolved_map = p1
        else:
            resolved_map = cand
            map_exists = cand.exists() and cand.is_file()

    rel_resolved = None
    if resolved_map is not None:
        try:
            rel_resolved = resolved_map.relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            rel_resolved = resolved_map.as_posix()

    return {
        "map_file": map_file,
        "map_id": map_id,
        "route_overrides_file": route_overrides,
        "resolved_map_path": rel_resolved,
        "map_exists": map_exists if map_file is not None else (map_id is not None),
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


def _build_scenario_summary(
    scenario: Mapping[str, Any],
    source_file: Path,
    repo_root: Path,
    duplicate_sources: list[str] | None = None,
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

    reasons: list[str] = []
    status = STATUS_VALID

    if scenario.get("supported") is False:
        status = STATUS_UNSUPPORTED
        reasons.append("Explicitly marked supported: false")
    elif map_ref["map_file"] and not map_ref["map_exists"]:
        status = STATUS_MISSING
        reasons.append(f"Referenced map_file not found on disk: {map_ref['map_file']}")
    elif duplicate_sources:
        status = STATUS_DUPLICATE
        reasons.append(f"Declared in multiple source files: {', '.join(duplicate_sources)}")

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
        },
    }


def _scan_yaml_scenarios(
    yf: Path,
    rel_path: str,
    declarations: dict[str, list[tuple[str, Path, dict[str, Any]]]],
) -> None:
    """Read a YAML scenario file and append scenario declarations."""
    try:
        with yf.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError, ValueError):
        return

    entries: list[Any] = []
    if isinstance(data, dict):
        if "scenarios" in data and isinstance(data["scenarios"], list):
            entries = data["scenarios"]
    elif isinstance(data, list):
        entries = data

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        sid = entry.get("name") or entry.get("scenario_id") or entry.get("id")
        if not sid:
            continue
        sid_str = str(sid).strip()
        declarations.setdefault(sid_str, []).append((rel_path, yf, entry))


def _build_catalog() -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Build the complete catalog of curated bundled scenarios.

    Returns:
        tuple[dict[str, dict[str, Any]], list[dict[str, str]]]: Discovered catalog and exclusions.
    """
    repo_root = _find_repo_root()
    scenarios_root = repo_root / "configs" / "scenarios"
    exclusion_paths = {exc["path"] for exc in CURATED_EXCLUSIONS}

    curated_roots = [
        scenarios_root / "single",
        scenarios_root / "archetypes",
        scenarios_root,
        scenarios_root / "sets",
    ]

    declarations: dict[str, list[tuple[str, Path, dict[str, Any]]]] = {}
    for c_dir in curated_roots:
        if not c_dir.exists():
            continue
        for yf in sorted(c_dir.glob("*.yaml")):
            rel_path = yf.relative_to(repo_root).as_posix()
            if rel_path in exclusion_paths:
                continue
            if c_dir == scenarios_root and yf.parent != scenarios_root:
                continue
            _scan_yaml_scenarios(yf, rel_path, declarations)

    catalog: dict[str, dict[str, Any]] = OrderedDict()
    for sid, decls in declarations.items():
        canonical_rel_path, canonical_path, canonical_entry = decls[0]
        dup_sources = [rel for rel, _path, _entry in decls[1:] if rel != canonical_rel_path]
        summary = _build_scenario_summary(
            canonical_entry,
            source_file=canonical_path,
            repo_root=repo_root,
            duplicate_sources=dup_sources,
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
    catalog, _ = _get_catalog()

    q_str = str(query).strip()
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
    sid = str(
        loaded.get("name") or loaded.get("scenario_id") or loaded.get("id") or source_file.stem
    ).strip()

    dup_sources: list[str] = []
    if sid in catalog:
        dup_sources = catalog[sid].get("duplicate_sources", [])

    summary = _build_scenario_summary(
        loaded,
        source_file=source_file,
        repo_root=repo_root,
        duplicate_sources=dup_sources,
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


def _check_scenario_assets(
    scenarios: list[Any],
    resolved: Path,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Inspect map references and support flags for loaded scenarios.

    Returns:
        tuple: Summaries list, errors list, warnings list, and missing asset flag.
    """
    scenario_summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    has_missing_asset = False

    for s in scenarios:
        s_dict = dict(s)
        summary = _build_scenario_summary(s_dict, source_file=resolved, repo_root=repo_root)
        scenario_summaries.append(summary)

        sid = summary["identity"]
        map_ref = summary["map_reference"]

        if not map_ref.get("map_file") and not map_ref.get("map_id"):
            errors.append(
                {
                    "code": REASON_MAP_REFERENCE_MISSING,
                    "message": f"Scenario '{sid}' defines neither map_file nor map_id.",
                    "scenario_id": sid,
                    "path": "/map_file",
                }
            )
        elif map_ref.get("map_file") and not map_ref.get("map_exists"):
            has_missing_asset = True
            errors.append(
                {
                    "code": REASON_MAP_NOT_FOUND,
                    "message": f"Scenario '{sid}' references non-existent map: '{map_ref['map_file']}'.",
                    "scenario_id": sid,
                    "path": "/map_file",
                }
            )

        if map_ref.get("route_overrides_file"):
            ro_path = Path(map_ref["route_overrides_file"])
            if not ro_path.is_absolute():
                ro_path = (resolved.parent / ro_path).resolve()
            if not ro_path.exists():
                has_missing_asset = True
                errors.append(
                    {
                        "code": REASON_ROUTE_OVERRIDES_NOT_FOUND,
                        "message": f"Scenario '{sid}' references non-existent route overrides: '{map_ref['route_overrides_file']}'.",
                        "scenario_id": sid,
                        "path": "/route_overrides_file",
                    }
                )

        if s_dict.get("supported") is False:
            warnings.append(
                {
                    "code": REASON_UNSUPPORTED_SCENARIO,
                    "message": f"Scenario '{sid}' is explicitly marked supported: false.",
                    "scenario_id": sid,
                    "path": "/supported",
                }
            )

    return scenario_summaries, errors, warnings, has_missing_asset


def validate_scenario_payload(path_str: str) -> dict[str, Any]:
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

    from robot_sf.benchmark.scenario.scenario_schema import (  # noqa: PLC0415
        validate_scenario_list,
    )
    from robot_sf.training.scenario_loader import load_scenarios  # noqa: PLC0415

    try:
        scenarios = load_scenarios(resolved)
    except (OSError, ValueError, yaml.YAMLError, RuntimeError) as exc:
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
                    "code": REASON_LOAD_FAILURE,
                    "message": f"Failed to load scenarios from '{path_str}': {exc}",
                    "scenario_id": None,
                    "path": str(path_str),
                }
            ],
            "warnings": [],
        }

    schema_errors = validate_scenario_list([dict(s) for s in scenarios])
    errors: list[dict[str, Any]] = []
    for se in schema_errors:
        err_msg = se.get("error", "Schema validation error")
        code = (
            REASON_DUPLICATE_SCENARIO_ID
            if "duplicate" in err_msg.lower()
            else REASON_SCHEMA_VALIDATION_ERROR
        )
        errors.append(
            {
                "code": code,
                "message": err_msg,
                "scenario_id": se.get("id"),
                "path": se.get("path"),
            }
        )

    summaries, asset_errors, warnings, has_missing = _check_scenario_assets(
        list(scenarios), resolved, repo_root
    )
    all_errors = errors + asset_errors
    is_valid = len(all_errors) == 0

    if is_valid:
        status = (
            STATUS_UNSUPPORTED
            if any(w["code"] == REASON_UNSUPPORTED_SCENARIO for w in warnings)
            else STATUS_VALID
        )
    elif has_missing and not errors:
        status = STATUS_MISSING
    else:
        status = STATUS_INVALID

    return {
        "schema_version": SCHEMA_VALIDATE_VERSION,
        "target_path": str(path_str),
        "resolved_path": str(resolved),
        "valid": is_valid,
        "status": status,
        "num_scenarios": len(scenarios),
        "scenarios": summaries,
        "errors": all_errors,
        "warnings": warnings,
    }


def _format_list_scenarios(payload: dict[str, Any]) -> str:
    """Format scenarios list for human-friendly terminal display.

    Returns:
        str: Formatted terminal text.
    """
    lines: list[str] = [f"Curated Scenarios ({payload['count']}):\n"]
    for s in payload["scenarios"]:
        fam = f" ({s['family']})" if s.get("family") else ""
        lines.append(f"- {s['identity']}  [{s['status']}]{fam}")
        lines.append(f"    source: {s['source_file']}")
        map_ref = s.get("map_reference", {})
        m_str = map_ref.get("map_file") or map_ref.get("map_id") or "none"
        exists_str = "exists" if map_ref.get("map_exists") else "not found"
        lines.append(f"    map: {m_str} [{exists_str}]")
        ac = s.get("actor_counts", {})
        lines.append(
            f"    actors: ped_density={ac.get('ped_density')}, single_peds={ac.get('single_pedestrians')}, robots={ac.get('num_robots')}"
        )
        sp = s.get("seed_policy", {})
        lines.append(f"    seeds: count={sp.get('count')}, seeds={sp.get('seeds')}")
        hd = s.get("horizon_and_dt", {})
        if hd.get("max_episode_steps"):
            lines.append(
                f"    horizon: {hd.get('max_episode_steps')} steps (dt={hd.get('time_step')}, sim_time={hd.get('sim_time_s')}s)"
            )
        ko = s.get("kinematics_and_observation", {})
        if ko.get("kinematics"):
            lines.append(f"    kinematics: {ko.get('kinematics')}")
        if s.get("duplicate_sources"):
            lines.append(f"    duplicate sources: {', '.join(s['duplicate_sources'])}")
        lines.append("")

    if payload.get("exclusions"):
        lines.append(f"Exclusions ({len(payload['exclusions'])}):")
        for exc in payload["exclusions"]:
            lines.append(f"- {exc['path']}: {exc['reason']}")
        lines.append("")

    return "\n".join(lines)


def _format_describe_scenario(payload: dict[str, Any]) -> str:
    """Format scenario description for human-friendly terminal display.

    Returns:
        str: Formatted terminal text.
    """
    lines: list[str] = [f"Scenario: {payload['identity']}\n"]
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
    lines.append(f"    reasons: {', '.join(vs.get('reasons', [])) or 'none'}")

    return "\n".join(lines)


def _format_validate_scenario(payload: dict[str, Any]) -> str:
    """Format validation report for human-friendly terminal display.

    Returns:
        str: Formatted terminal text.
    """
    status_str = "PASS" if payload["valid"] else "FAIL"
    lines: list[str] = [
        f"Scenario validation: {status_str}",
        f"Target: {payload['target_path']} ({payload['num_scenarios']} scenario(s))",
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

    return "\n".join(lines)


def _handle_scenarios_list(args: argparse.Namespace) -> int:
    """Handle ``robot-sf scenarios list`` command.

    Returns:
        int: Process exit code (0 for success).
    """
    payload = list_scenarios_payload()
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
