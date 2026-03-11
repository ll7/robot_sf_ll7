"""Helpers for loading scenario definitions into environment configs."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import (
    MapDefinition,
    MapDefinitionPool,
    PedestrianWaitRule,
    SinglePedestrianDefinition,
    serialize_map,
)
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.robot.holonomic_drive import HolonomicDriveSettings

_MAP_REGISTRY_ENV = "ROBOT_SF_MAP_REGISTRY"
_MAP_REGISTRY_PATH = Path("maps/registry.yaml")


def _load_yaml_documents(path: Path) -> Any:
    """Load YAML content from disk.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        Any: Parsed YAML content.
    """
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_scenarios(path: Path, *, base_dir: Path | None = None) -> list[Mapping[str, Any]]:
    """Load scenario definitions from a YAML file.

    Supports a list of scenarios, a mapping with ``scenarios``, and optional
    include lists (``includes``, ``include``, or ``scenario_files``) for
    composing per-scenario and per-archetype files into a single list.

    Returns:
        list[Mapping[str, Any]]: Parsed scenario entries from the file(s).
    """
    resolved = path.resolve()
    if base_dir is None:
        root = resolved
    else:
        root = base_dir.resolve()
        if not root.exists():
            raise ValueError(f"Scenario base_dir does not exist: {root}")
    return _load_scenarios_recursive(resolved, visited=set(), root=root)


def _load_scenarios_recursive(
    path: Path,
    *,
    visited: set[Path],
    root: Path,
    map_search_paths: list[Path] | None = None,
) -> list[Mapping[str, Any]]:
    """Load scenarios from path, expanding any include references.

    Returns:
        list[Mapping[str, Any]]: Combined scenario entries.
    """
    resolved = path.resolve()
    if resolved in visited:
        raise ValueError(f"Scenario include cycle detected at '{resolved}'.")
    visited.add(resolved)
    try:
        data = _load_yaml_documents(resolved)
        scenarios: list[Any] = []
        includes: list[Path] = []
        local_map_search_paths = (
            _resolve_map_search_paths(data, root=root) if isinstance(data, Mapping) else []
        )
        if local_map_search_paths:
            logger.info(
                "Scenario manifest '{}' configured map_search_paths: {}",
                resolved,
                ", ".join(str(path) for path in local_map_search_paths),
            )
        if isinstance(data, Mapping):
            includes = _resolve_includes(data, source=resolved)
            if "scenarios" in data:
                scenarios = data["scenarios"]
            elif not includes:
                raise ValueError(f"Scenario config must contain a 'scenarios' list: {resolved}")
        elif isinstance(data, list):
            scenarios = data
        else:  # pragma: no cover - malformed input handled by caller
            raise ValueError(f"Scenario config must contain a 'scenarios' list: {resolved}")

        if scenarios and not isinstance(scenarios, list):
            raise ValueError(f"Scenario config 'scenarios' must be a list: {resolved}")

        combined: list[Mapping[str, Any]] = []
        inherited_search_paths = map_search_paths or []
        effective_search_paths = local_map_search_paths or inherited_search_paths
        for include_path in includes:
            combined.extend(
                _load_scenarios_recursive(
                    include_path,
                    visited=visited,
                    root=root,
                    map_search_paths=effective_search_paths,
                )
            )
        combined.extend(
            _normalize_scenarios(
                scenarios,
                source=resolved,
                root=root,
                map_search_paths=effective_search_paths,
            )
        )
        if not combined:
            raise ValueError(f"Scenario config missing scenarios: {resolved}")
        return combined
    finally:
        visited.remove(resolved)


def _resolve_includes(data: Mapping[str, Any], *, source: Path) -> list[Path]:
    """Resolve include entries into absolute paths.

    Returns:
        list[Path]: Absolute include paths.
    """
    raw = data.get("includes") or data.get("include") or data.get("scenario_files")
    if raw is None:
        return []
    if isinstance(raw, (str, Path)):
        entries = [raw]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError(f"Include list must be a list or string in '{source}'.")
    includes: list[Path] = []
    for entry in entries:
        if not isinstance(entry, (str, Path)):
            raise ValueError(f"Include entry '{entry}' must be a string in '{source}'.")
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = (source.parent / candidate).resolve()
        if candidate.is_dir():
            raise ValueError(
                f"Include entry '{entry}' resolves to a directory; "
                f"list scenario files explicitly in '{source}'."
            )
        includes.append(candidate)
    return includes


def _resolve_map_search_paths(data: Mapping[str, Any], *, root: Path) -> list[Path]:
    """Resolve optional map search paths for the scenario root.

    Returns:
        list[Path]: Resolved search paths.
    """
    raw = data.get("map_search_paths")
    if raw is None:
        return []
    if isinstance(raw, (str, Path)):
        entries = [raw]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError(f"map_search_paths must be a list or string in '{root}'.")
    resolved: list[Path] = []
    base_root = root if root.is_dir() else root.parent
    for entry in entries:
        if not isinstance(entry, (str, Path)):
            raise ValueError(f"map_search_paths entry '{entry}' must be a string in '{root}'.")
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = (base_root / candidate).resolve()
        if not candidate.exists():
            logger.warning("map_search_paths entry does not exist: {}", candidate)
            continue
        resolved.append(candidate)
    return resolved


def _resolve_map_registry_path() -> Path | None:
    """Resolve the map registry path from the environment or repo default.

    Returns:
        Path | None: Path to the registry file, or None if unavailable.
    """
    override = os.getenv(_MAP_REGISTRY_ENV)
    if override:
        return Path(override).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / _MAP_REGISTRY_PATH


def _iter_registry_mapping(entries: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    """Yield map_id/path pairs from a mapping-form registry.

    Yields:
        tuple[str, str]: Map id and path entries.
    """
    for map_id, map_path in entries.items():
        if map_id == "version":
            continue
        if not isinstance(map_path, str):
            logger.warning("Skipping map registry entry '{}' with non-string path.", map_id)
            continue
        yield str(map_id), map_path


def _iter_registry_list(entries: list[Any]) -> Iterable[tuple[str, str]]:
    """Yield map_id/path pairs from a list-form registry.

    Yields:
        tuple[str, str]: Map id and path entries.
    """
    for entry in entries:
        if not isinstance(entry, Mapping):
            logger.warning("Skipping non-mapping map registry entry: {}", entry)
            continue
        map_id = entry.get("map_id") or entry.get("id")
        map_path = entry.get("path") or entry.get("map_file")
        if not isinstance(map_id, str) or not isinstance(map_path, str):
            logger.warning("Skipping invalid map registry entry: {}", entry)
            continue
        yield map_id, map_path


def _iter_map_registry_entries(
    data: Mapping[str, Any],
    *,
    registry_path: Path,
) -> Iterable[tuple[str, str]]:
    """Yield map registry entries from the loaded registry data.

    Yields:
        tuple[str, str]: Map id and path entries.
    """
    entries = data.get("maps", data if "maps" not in data else None)
    if isinstance(entries, Mapping):
        yield from _iter_registry_mapping(entries)
        return
    if isinstance(entries, list):
        yield from _iter_registry_list(entries)
        return
    raise ValueError(f"Map registry '{registry_path}' has invalid format.")


def _register_map_entry(
    registry: dict[str, Path],
    *,
    map_id: str,
    map_path: str,
    registry_path: Path,
) -> None:
    """Insert a map registry entry, resolving relative paths."""
    key = map_id.strip()
    if not key:
        raise ValueError(f"Map registry entry in '{registry_path}' has empty map_id.")
    if key in registry:
        raise ValueError(f"Duplicate map_id '{key}' in map registry '{registry_path}'.")
    candidate = Path(map_path)
    if not candidate.is_absolute():
        candidate = (registry_path.parent / candidate).resolve()
    registry[key] = candidate


@lru_cache(maxsize=4)
def _load_map_registry(path: Path | None = None) -> dict[str, Path]:
    """Load map registry entries, returning map_id -> absolute path.

    Returns:
        dict[str, Path]: Map registry map_id to absolute map file path.
    """
    registry_path = path or _resolve_map_registry_path()
    if registry_path is None or not registry_path.exists():
        return {}
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Map registry '{registry_path}' must contain a mapping.")
    registry: dict[str, Path] = {}
    for map_id, map_path in _iter_map_registry_entries(data, registry_path=registry_path):
        _register_map_entry(
            registry,
            map_id=map_id,
            map_path=map_path,
            registry_path=registry_path,
        )
    return registry


def _resolve_map_id(
    map_id: str,
    *,
    map_registry: Mapping[str, Path],
    source: Path,
) -> Path:
    """Resolve a map_id to a filesystem path using the registry.

    Returns:
        Path: Absolute path to the map file.
    """
    if not map_registry:
        raise ValueError(
            f"Scenario in '{source}' references map_id '{map_id}', but the map registry is empty."
        )
    if map_id not in map_registry:
        raise ValueError(f"Unknown map_id '{map_id}' in '{source}'.")
    resolved = map_registry[map_id]
    if not resolved.exists():
        raise ValueError(f"Map registry entry '{map_id}' points to missing file: {resolved}")
    return resolved


def _normalize_scenarios(
    scenarios: list[Any],
    *,
    source: Path,
    root: Path,
    map_search_paths: list[Path],
) -> list[Mapping[str, Any]]:
    """Filter and validate scenario mappings while preserving order.

    Returns:
        list[Mapping[str, Any]]: Normalized scenario entries.
    """
    map_registry: dict[str, Path] = {}
    if any(isinstance(entry, Mapping) and entry.get("map_id") for entry in scenarios):
        map_registry = _load_map_registry()
    normalized: list[Mapping[str, Any]] = []
    for idx, scenario in enumerate(scenarios):
        if not isinstance(scenario, Mapping):
            logger.warning("Scenario entry {} in '{}' is not a mapping; skipping.", idx, source)
            continue
        _validate_scenario_entry(scenario, source=source, index=idx)
        normalized.append(
            _rebase_scenario_paths(
                scenario,
                source=source,
                root=root,
                map_search_paths=map_search_paths,
                map_registry=map_registry,
            )
        )
    return normalized


def _validate_scenario_entry(
    scenario: Mapping[str, Any],
    *,
    source: Path,
    index: int,
) -> None:
    """Emit helpful errors/warnings for common schema expectations."""
    name = scenario.get("name") or scenario.get("scenario_id")
    if name is None:
        logger.warning("Scenario entry {} in '{}' is missing a name or scenario_id.", index, source)
    elif not isinstance(name, str):
        raise ValueError(f"Scenario name must be a string in '{source}' at index {index}.")

    _validate_map_reference(scenario, name=name, source=source, index=index)
    _validate_optional_mapping(scenario, key="simulation_config", source=source, index=index)
    _validate_optional_mapping(scenario, key="robot_config", source=source, index=index)
    _validate_optional_mapping(scenario, key="metadata", source=source, index=index)
    _validate_seed_list(scenario, source=source, index=index)


def _validate_map_reference(
    scenario: Mapping[str, Any],
    *,
    name: str | None,
    source: Path,
    index: int,
) -> None:
    map_id = scenario.get("map_id")
    map_file = scenario.get("map_file")
    if map_id is None and map_file is None:
        logger.warning("Scenario '{}' in '{}' has no map_file or map_id.", name or index, source)
        return
    if map_id is not None:
        if not isinstance(map_id, str) or not map_id.strip():
            raise ValueError(f"map_id must be a non-empty string in '{source}' at index {index}.")
    if map_file is not None and not isinstance(map_file, str):
        raise ValueError(f"map_file must be a string in '{source}' at index {index}.")
    if map_id is not None and map_file is not None:
        logger.warning(
            "Scenario '{}' in '{}' defines both map_id and map_file; map_id will be used.",
            name or index,
            source,
        )


def _rebase_scenario_paths(
    scenario: Mapping[str, Any],
    *,
    source: Path,
    root: Path,
    map_search_paths: list[Path],
    map_registry: Mapping[str, Path],
) -> Mapping[str, Any]:
    """Rewrite relative map paths to be relative to the root scenario file.

    Returns:
        Mapping[str, Any]: Scenario entry with rebased paths when applicable.
    """
    search_root = root if root.is_dir() else root.parent
    map_id = scenario.get("map_id")
    if isinstance(map_id, str) and map_id.strip():
        resolved = _resolve_map_id(map_id, map_registry=map_registry, source=source)
        rel = os.path.relpath(resolved, search_root)
        updated = dict(scenario)
        updated["map_file"] = Path(rel).as_posix()
        return updated

    map_file = scenario.get("map_file")
    if not isinstance(map_file, str):
        return scenario
    candidate = Path(map_file)
    if candidate.is_absolute():
        return scenario
    probe = (search_root / candidate).resolve()
    if probe.exists():
        return scenario
    if source.parent != search_root:
        abs_target = (source.parent / candidate).resolve()
        if abs_target.exists():
            rel = os.path.relpath(abs_target, search_root)
            updated = dict(scenario)
            updated["map_file"] = Path(rel).as_posix()
            return updated
    resolved = _resolve_map_with_search_paths(
        map_file,
        map_search_paths=map_search_paths,
        root=search_root,
        source=source,
    )
    if resolved is None:
        _emit_map_resolution_error(
            map_file,
            map_search_paths=map_search_paths,
            root=search_root,
            source=source,
        )
        return scenario
    rel = os.path.relpath(resolved, search_root)
    updated = dict(scenario)
    updated["map_file"] = Path(rel).as_posix()
    return updated


def _resolve_map_with_search_paths(
    map_file: str,
    *,
    map_search_paths: list[Path],
    root: Path,
    source: Path,
) -> Path | None:
    candidate = Path(map_file)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    probe = (root / candidate).resolve()
    if probe.exists():
        return probe
    for base in map_search_paths:
        probe = (base / candidate).resolve()
        if probe.exists():
            return probe
    logger.debug(
        "Map '{}' not resolved relative to '{}' or search paths from '{}'.",
        map_file,
        root,
        source,
    )
    return None


def _emit_map_resolution_error(
    map_file: str,
    *,
    map_search_paths: list[Path],
    root: Path,
    source: Path,
) -> None:
    search_desc = ", ".join(str(path) for path in map_search_paths) if map_search_paths else "-"
    logger.warning(
        "Could not resolve map_file '{}' from '{}'. Tried root '{}' and map_search_paths [{}]. "
        "Consider fixing the map_file path or setting map_search_paths in the manifest.",
        map_file,
        source,
        root,
        search_desc,
    )


def _validate_optional_mapping(
    scenario: Mapping[str, Any],
    *,
    key: str,
    source: Path,
    index: int,
) -> None:
    value = scenario.get(key)
    if value is not None and not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping in '{source}' at index {index}.")


def _validate_seed_list(
    scenario: Mapping[str, Any],
    *,
    source: Path,
    index: int,
) -> None:
    seeds = scenario.get("seeds")
    if seeds is None:
        return
    if not isinstance(seeds, list):
        raise ValueError(f"seeds must be a list in '{source}' at index {index}.")
    if not all(isinstance(seed, int) for seed in seeds):
        raise ValueError(f"seeds must contain integers in '{source}' at index {index}.")


def select_scenario(
    scenarios: list[Mapping[str, Any]],
    scenario_id: str | None,
) -> Mapping[str, Any]:
    """Return the scenario matching ``scenario_id`` or the first entry.

    Returns:
        Mapping[str, Any]: Selected scenario entry.
    """

    if scenario_id:
        for sc in scenarios:
            name = str(sc.get("name") or sc.get("scenario_id") or "").strip()
            if name.lower() == scenario_id.lower():
                return sc
        raise ValueError(f"Scenario id '{scenario_id}' not found in scenario config")
    return scenarios[0]


@lru_cache(maxsize=8)
def _load_map_definition(map_path: str) -> MapDefinition | None:
    """Load and convert a map definition, caching by absolute path.

    Returns:
        MapDefinition | None: Parsed map definition for supported formats, else ``None``.
    """

    path = Path(map_path)
    if not path.exists():
        logger.warning("Scenario map file not found: {}", path)
        return None
    if path.suffix.lower() == ".svg":
        return convert_map(str(path))
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            logger.error("Invalid JSON map '{}': {}", path, exc)
            return None
        return serialize_map(data)
    logger.warning("Unsupported map extension '{}' for scenario maps", path.suffix)
    return None


def build_robot_config_from_scenario(
    scenario: Mapping[str, Any],
    *,
    scenario_path: Path,
) -> RobotSimulationConfig:
    """Create a ``RobotSimulationConfig`` derived from a scenario definition.

    Returns:
        RobotSimulationConfig: Config populated with overrides and map pool.
    """

    config = RobotSimulationConfig()
    _apply_simulation_overrides(config, scenario.get("simulation_config", {}))
    _apply_robot_overrides(config, scenario.get("robot_config", {}))
    _apply_map_pool(config, scenario, scenario_path)
    _apply_route_overrides(config, scenario.get("route_overrides_file"), scenario_path)
    _apply_single_pedestrian_overrides(config, scenario.get("single_pedestrians"))
    return config


def _coerce_finite_float(value: Any, *, field_name: str) -> float:
    """Parse and validate a finite float value for scenario robot_config fields.

    Returns:
        float: Parsed finite float value.
    """
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"robot_config.{field_name} must be finite.")
    return parsed


def _coerce_non_negative_float(value: Any, *, field_name: str) -> float:
    """Parse and validate a non-negative finite float for scenario robot_config fields.

    Returns:
        float: Parsed finite float value that is ``>= 0``.
    """
    parsed = _coerce_finite_float(value, field_name=field_name)
    if parsed < 0.0:
        raise ValueError(f"robot_config.{field_name} must be >= 0.")
    return parsed


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    """Coerce boolean-like override values with strict validation.

    Returns:
        bool: Parsed boolean value.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"robot_config.{field_name} must be a boolean.")


def _robot_type_alias(raw: str) -> str:
    """Normalize scenario robot type aliases to canonical labels.

    Returns:
        str: Canonical robot type label.
    """
    robot_type = raw.lower()
    if robot_type in {"diff_drive", "diff", "differential"}:
        return "differential_drive"
    if robot_type in {"bicycle", "bike"}:
        return "bicycle_drive"
    if robot_type in {"omni", "omnidirectional"}:
        return "holonomic"
    return robot_type


def _differential_robot_settings(overrides: Mapping[str, Any]) -> DifferentialDriveSettings:
    """Build differential-drive settings from scenario overrides.

    Returns:
        DifferentialDriveSettings: Parsed settings object.
    """
    kwargs: dict[str, Any] = {}
    if "radius" in overrides:
        kwargs["radius"] = _coerce_non_negative_float(overrides["radius"], field_name="radius")
    if "max_linear_speed" in overrides:
        kwargs["max_linear_speed"] = _coerce_finite_float(
            overrides["max_linear_speed"], field_name="max_linear_speed"
        )
    if "max_angular_speed" in overrides:
        kwargs["max_angular_speed"] = _coerce_finite_float(
            overrides["max_angular_speed"], field_name="max_angular_speed"
        )
    if "wheel_radius" in overrides:
        kwargs["wheel_radius"] = _coerce_finite_float(
            overrides["wheel_radius"], field_name="wheel_radius"
        )
    if "interaxis_length" in overrides:
        kwargs["interaxis_length"] = _coerce_finite_float(
            overrides["interaxis_length"], field_name="interaxis_length"
        )
    if "allow_backwards" in overrides:
        kwargs["allow_backwards"] = _coerce_bool(
            overrides["allow_backwards"],
            field_name="allow_backwards",
        )
    return DifferentialDriveSettings(**kwargs)


def _bicycle_robot_settings(overrides: Mapping[str, Any]) -> BicycleDriveSettings:
    """Build bicycle-drive settings from scenario overrides.

    Returns:
        BicycleDriveSettings: Parsed settings object.
    """
    kwargs: dict[str, Any] = {}
    if "radius" in overrides:
        kwargs["radius"] = _coerce_non_negative_float(overrides["radius"], field_name="radius")
    if "wheelbase" in overrides:
        kwargs["wheelbase"] = _coerce_finite_float(overrides["wheelbase"], field_name="wheelbase")
    if "max_steer" in overrides:
        kwargs["max_steer"] = _coerce_finite_float(overrides["max_steer"], field_name="max_steer")
    if "max_velocity" in overrides:
        kwargs["max_velocity"] = _coerce_finite_float(
            overrides["max_velocity"], field_name="max_velocity"
        )
    if "max_accel" in overrides:
        kwargs["max_accel"] = _coerce_finite_float(overrides["max_accel"], field_name="max_accel")
    if "allow_backwards" in overrides:
        kwargs["allow_backwards"] = _coerce_bool(
            overrides["allow_backwards"],
            field_name="allow_backwards",
        )
    return BicycleDriveSettings(**kwargs)


def _holonomic_robot_settings(overrides: Mapping[str, Any]) -> HolonomicDriveSettings:
    """Build holonomic-drive settings from scenario overrides.

    Returns:
        HolonomicDriveSettings: Parsed settings object.
    """
    kwargs: dict[str, Any] = {}
    if "radius" in overrides:
        kwargs["radius"] = _coerce_non_negative_float(overrides["radius"], field_name="radius")
    if "max_speed" in overrides:
        kwargs["max_speed"] = _coerce_finite_float(overrides["max_speed"], field_name="max_speed")
    if "max_angular_speed" in overrides:
        kwargs["max_angular_speed"] = _coerce_finite_float(
            overrides["max_angular_speed"], field_name="max_angular_speed"
        )
    if "command_mode" in overrides:
        kwargs["command_mode"] = str(overrides["command_mode"]).strip().lower()
    return HolonomicDriveSettings(**kwargs)


def _apply_robot_overrides(
    config: RobotSimulationConfig,
    overrides: Mapping[str, Any] | None,
) -> None:
    """Apply optional scenario-level robot kinematics overrides."""
    if not isinstance(overrides, Mapping) or not overrides:
        return
    raw_type = str(overrides.get("type", overrides.get("model", "differential_drive"))).strip()
    robot_type = _robot_type_alias(raw_type)
    if robot_type == "differential_drive":
        config.robot_config = _differential_robot_settings(overrides)
        return
    if robot_type == "bicycle_drive":
        config.robot_config = _bicycle_robot_settings(overrides)
        return
    if robot_type == "holonomic":
        config.robot_config = _holonomic_robot_settings(overrides)
        return
    raise ValueError(
        "robot_config.type must be one of 'differential_drive', 'bicycle_drive', or 'holonomic'."
    )


def resolve_map_definition(map_file: str | None, *, scenario_path: Path) -> MapDefinition | None:
    """Resolve and load a map definition from a scenario map file reference.

    Returns:
        MapDefinition | None: Loaded map definition when the file exists, otherwise ``None``.
    """
    if not map_file:
        return None
    candidate = Path(map_file)
    if not candidate.is_absolute():
        candidate = (scenario_path.parent / candidate).resolve()
    if not candidate.exists():
        logger.error(
            "Scenario map file '{}' resolved from '{}' does not exist. "
            "Check map_file or manifest map_search_paths.",
            map_file,
            scenario_path,
        )
    return _load_map_definition(str(candidate))


def apply_single_pedestrian_overrides(
    map_def: MapDefinition,
    overrides: list[Mapping[str, Any]] | None,
) -> None:
    """Apply single-pedestrian overrides from scenario YAML onto a map definition."""
    if not overrides:
        return
    if not map_def.single_pedestrians:
        raise ValueError("single_pedestrians overrides provided but map has no single pedestrians")

    overrides_by_id: dict[str, Mapping[str, Any]] = {}
    for idx, entry in enumerate(overrides):
        if not isinstance(entry, Mapping):
            raise ValueError(f"single_pedestrians[{idx}] must be a mapping")
        ped_id = str(entry.get("id") or "").strip()
        if not ped_id:
            raise ValueError(f"single_pedestrians[{idx}] missing non-empty id")
        if ped_id in overrides_by_id:
            raise ValueError(f"single_pedestrians contains duplicate id '{ped_id}'")
        overrides_by_id[ped_id] = entry

    updated: list[SinglePedestrianDefinition] = []
    for ped in map_def.single_pedestrians:
        entry = overrides_by_id.pop(ped.id, None)
        if entry is None:
            updated.append(ped)
            continue
        updated.append(_apply_single_pedestrian_override(ped, entry, map_def))

    if overrides_by_id:
        unknown = ", ".join(sorted(overrides_by_id.keys()))
        raise ValueError(f"Unknown single_pedestrians ids in overrides: {unknown}")

    map_def.single_pedestrians = updated


def _apply_single_pedestrian_overrides(
    config: RobotSimulationConfig,
    overrides: list[Mapping[str, Any]] | None,
) -> None:
    """Apply single-pedestrian overrides to the loaded map pool."""
    if not overrides:
        return
    if not getattr(config, "map_pool", None) or not config.map_pool.map_defs:
        logger.warning("single_pedestrians overrides provided but no map_pool is loaded")
        return
    map_name, map_def = next(iter(config.map_pool.map_defs.items()))
    # Avoid mutating cached map definitions shared across scenarios.
    map_def = deepcopy(map_def)
    config.map_pool.map_defs[map_name] = map_def
    apply_single_pedestrian_overrides(map_def, overrides)


_MISSING = object()


def _resolve_point_override(
    entry: Mapping[str, Any],
    *,
    key: str,
    poi_key: str,
    map_def: MapDefinition,
) -> tuple[object, bool]:
    """Resolve a point override from coordinates or POI label.

    Returns:
        tuple[object, bool]: Resolved point (or sentinel) and whether the key was specified.
    """
    if key in entry and poi_key in entry:
        raise ValueError(f"Specify only one of '{key}' or '{poi_key}'")
    if poi_key in entry:
        label = entry.get(poi_key)
        if label is None:
            return None, True
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"'{poi_key}' must be a non-empty string")
        try:
            return map_def.get_poi_by_label(label), True
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
    if key in entry:
        point = entry.get(key)
        if point is None:
            return None, True
        return _coerce_point(point, key), True
    return _MISSING, False


def _resolve_trajectory_override(
    entry: Mapping[str, Any],
    *,
    map_def: MapDefinition,
) -> tuple[object, list[str] | None, bool]:
    """Resolve a trajectory override from coordinates or POI labels.

    Returns:
        tuple[object, list[str] | None, bool]: Resolved trajectory (or sentinel),
            optional POI labels, and whether a trajectory was specified.
    """
    if "trajectory" in entry and "trajectory_poi" in entry:
        raise ValueError("Specify only one of 'trajectory' or 'trajectory_poi'")
    if "trajectory_poi" in entry:
        points, labels = _resolve_trajectory_poi_override(
            entry.get("trajectory_poi"),
            map_def=map_def,
        )
        return points, labels, True
    if "trajectory" in entry:
        points = _resolve_trajectory_points_override(entry.get("trajectory"))
        return points, None, True
    return _MISSING, None, False


def _resolve_trajectory_poi_override(
    labels: Any,
    *,
    map_def: MapDefinition,
) -> tuple[list[tuple[float, float]] | None, list[str] | None]:
    """Resolve a POI trajectory override into coordinates and labels.

    Args:
        labels: Raw value from the override entry.
        map_def: Map definition providing POI lookups.

    Returns:
        tuple[list[tuple[float, float]] | None, list[str] | None]: Resolved coordinates
            and the labels used (or ``None`` if labels are unset).
    """
    if labels is None:
        return None, None
    if not isinstance(labels, list):
        raise ValueError("'trajectory_poi' must be a list of POI labels")
    points: list[tuple[float, float]] = []
    for idx, label in enumerate(labels):
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"'trajectory_poi' entry {idx} must be a non-empty string")
        try:
            points.append(map_def.get_poi_by_label(label))
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
    return points, [str(label_value) for label_value in labels]


def _resolve_trajectory_points_override(traj: Any) -> list[tuple[float, float]] | None:
    """Resolve a coordinate trajectory override into points.

    Args:
        traj: Raw trajectory list from the override entry.

    Returns:
        list[tuple[float, float]] | None: Normalized points or ``None`` when unset.
    """
    if traj is None:
        return None
    if not isinstance(traj, list):
        raise ValueError("'trajectory' must be a list of [x, y] points")
    return [_coerce_point(point, "trajectory") for point in traj]


def _coerce_point(value: Any, label: str) -> tuple[float, float]:
    """Coerce a list/tuple into a (x, y) pair.

    Returns:
        tuple[float, float]: Normalized coordinate pair.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"'{label}' must be a 2-item list or tuple, got: {value!r}")
    return (float(value[0]), float(value[1]))


def _parse_wait_overrides(
    wait_entries: list[Mapping[str, Any]] | None,
    *,
    trajectory: list[tuple[float, float]] | None,
    trajectory_labels: list[str] | None,
) -> list[PedestrianWaitRule] | None:
    """Parse wait rules from overrides.

    Returns:
        list[PedestrianWaitRule] | None: Parsed wait rules or ``None`` when unset.
    """
    if wait_entries is None:
        return None
    if trajectory is None:
        raise ValueError("wait_at requires a trajectory to be set")
    if not isinstance(wait_entries, list):
        raise ValueError("wait_at must be a list of wait rules")

    rules: list[PedestrianWaitRule] = []
    for idx, entry in enumerate(wait_entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"wait_at[{idx}] must be a mapping")
        wait_s = entry.get("wait_s")
        if wait_s is None:
            raise ValueError(f"wait_at[{idx}] missing wait_s")
        waypoint_index = _resolve_wait_waypoint_index(
            entry,
            idx,
            trajectory_labels=trajectory_labels,
            trajectory_len=len(trajectory),
        )
        note = entry.get("note")
        rules.append(
            PedestrianWaitRule(
                waypoint_index=waypoint_index,
                wait_s=float(wait_s),
                note=str(note) if note is not None else None,
            )
        )
    return rules


def _resolve_wait_waypoint_index(
    entry: Mapping[str, Any],
    idx: int,
    *,
    trajectory_labels: list[str] | None,
    trajectory_len: int,
) -> int:
    """Resolve a wait rule to a trajectory waypoint index.

    Returns:
        int: Waypoint index into the trajectory.
    """
    if "poi" in entry:
        if not trajectory_labels:
            raise ValueError("wait_at.poi requires trajectory_poi to be set")
        poi_label = entry.get("poi")
        if not isinstance(poi_label, str) or not poi_label.strip():
            raise ValueError(f"wait_at[{idx}].poi must be a non-empty string")
        try:
            waypoint_index = trajectory_labels.index(poi_label)
        except ValueError as exc:
            raise ValueError(f"wait_at[{idx}].poi '{poi_label}' not in trajectory_poi") from exc
    elif "waypoint_index" in entry:
        waypoint_index = int(entry["waypoint_index"])
    else:
        raise ValueError(f"wait_at[{idx}] must define 'poi' or 'waypoint_index'")
    if waypoint_index < 0 or waypoint_index >= trajectory_len:
        raise ValueError(f"wait_at[{idx}] waypoint_index out of range")
    return waypoint_index


def _resolve_goal_trajectory_override(
    ped: SinglePedestrianDefinition,
    entry: Mapping[str, Any],
    map_def: MapDefinition,
) -> tuple[tuple[float, float] | None, list[tuple[float, float]] | None, list[str] | None]:
    """Resolve goal/trajectory overrides while enforcing mutual exclusivity.

    Returns:
        tuple[Vec2D | None, list[Vec2D] | None, list[str] | None]:
            Updated goal, trajectory, and optional trajectory label list.
    """
    goal_override, goal_specified = _resolve_point_override(
        entry,
        key="goal",
        poi_key="goal_poi",
        map_def=map_def,
    )
    trajectory_override, trajectory_labels, trajectory_specified = _resolve_trajectory_override(
        entry,
        map_def=map_def,
    )

    if goal_specified and trajectory_specified:
        if goal_override is not None and trajectory_override is not None:
            raise ValueError(f"single_pedestrians '{ped.id}' cannot set both goal and trajectory")

    goal = ped.goal if not goal_specified else goal_override
    trajectory = ped.trajectory if not trajectory_specified else trajectory_override

    if goal is not None and trajectory is not None:
        raise ValueError(f"single_pedestrians '{ped.id}' cannot define both goal and trajectory")

    return goal, trajectory, trajectory_labels


def _resolve_speed_override(
    ped: SinglePedestrianDefinition,
    entry: Mapping[str, Any],
) -> float | None:
    """Resolve speed override for a pedestrian definition.

    Returns:
        float | None: Resolved speed override or existing speed.
    """
    if "speed_m_s" not in entry:
        return ped.speed_m_s
    value = entry.get("speed_m_s")
    return float(value) if value is not None else None


def _resolve_wait_override(
    ped: SinglePedestrianDefinition,
    entry: Mapping[str, Any],
    *,
    trajectory: list[tuple[float, float]] | None,
    trajectory_labels: list[str] | None,
) -> list[PedestrianWaitRule] | None:
    """Resolve wait overrides for a pedestrian definition.

    Returns:
        list[PedestrianWaitRule] | None: Resolved wait rules or existing waits.
    """
    if "wait_at" not in entry:
        return ped.wait_at
    return _parse_wait_overrides(
        entry.get("wait_at"),
        trajectory=trajectory,
        trajectory_labels=trajectory_labels,
    )


def _resolve_note_override(
    ped: SinglePedestrianDefinition,
    entry: Mapping[str, Any],
) -> str | None:
    """Resolve an optional note override for a pedestrian definition.

    Returns:
        str | None: Resolved note text or existing note.
    """
    if "note" not in entry:
        return ped.note
    value = entry.get("note")
    return str(value) if value is not None else None


def _resolve_role_overrides(
    ped: SinglePedestrianDefinition,
    entry: Mapping[str, Any],
) -> tuple[str | None, str | None, tuple[float, float] | None]:
    """Resolve role, target, and offset overrides for a pedestrian definition.

    Returns:
        tuple[str | None, str | None, tuple[float, float] | None]: Role, target id,
        and role offset tuple.
    """
    if "role" in entry:
        role_value = entry.get("role")
        role = str(role_value) if role_value is not None else None
    else:
        role = ped.role

    if "role_target_id" in entry:
        target_value = entry.get("role_target_id")
        role_target_id = str(target_value) if target_value is not None else None
    else:
        role_target_id = ped.role_target_id

    if "role_offset" in entry:
        offset_value = entry.get("role_offset")
        role_offset = (
            _coerce_point(offset_value, "role_offset") if offset_value is not None else None
        )
    else:
        role_offset = ped.role_offset

    return role, role_target_id, role_offset


def _apply_single_pedestrian_override(
    ped: SinglePedestrianDefinition,
    entry: Mapping[str, Any],
    map_def: MapDefinition,
) -> SinglePedestrianDefinition:
    """Apply overrides to a single pedestrian definition.

    Returns:
        SinglePedestrianDefinition: Updated pedestrian definition.
    """
    goal, trajectory, trajectory_labels = _resolve_goal_trajectory_override(ped, entry, map_def)
    speed = _resolve_speed_override(ped, entry)
    wait_at = _resolve_wait_override(
        ped,
        entry,
        trajectory=trajectory,
        trajectory_labels=trajectory_labels,
    )
    note = _resolve_note_override(ped, entry)
    role, role_target_id, role_offset = _resolve_role_overrides(ped, entry)

    return SinglePedestrianDefinition(
        id=ped.id,
        start=ped.start,
        goal=goal,
        trajectory=trajectory,
        speed_m_s=speed,
        wait_at=wait_at,
        note=note,
        role=role,
        role_target_id=role_target_id,
        role_offset=role_offset,
    )


def _apply_simulation_overrides(
    config: RobotSimulationConfig,
    overrides: Mapping[str, Any] | None,
) -> None:
    """Apply scenario-level simulation overrides to a config instance."""
    if not isinstance(overrides, Mapping):
        return
    if "max_episode_steps" in overrides:
        steps = max(1, int(overrides["max_episode_steps"]))
        config.sim_config.sim_time_in_secs = steps * config.sim_config.time_per_step_in_secs
    # Apply difficulty first so ped_density uses the correct index
    if "difficulty" in overrides:
        config.sim_config.difficulty = overrides["difficulty"]
    if "ped_density" in overrides:
        density = float(overrides["ped_density"])
        difficulty = min(
            max(config.sim_config.difficulty, 0),
            len(config.sim_config.ped_density_by_difficulty) - 1,
        )
        config.sim_config.ped_density_by_difficulty[difficulty] = density
    if "max_peds_per_group" in overrides:
        config.sim_config.max_peds_per_group = int(overrides["max_peds_per_group"])
    for attr in (
        "peds_speed_mult",
        "ped_radius",
        "goal_radius",
        "route_spawn_distribution",
        "route_spawn_jitter_frac",
        "route_spawn_seed",
    ):
        if attr in overrides:
            setattr(config.sim_config, attr, overrides[attr])


def _apply_map_pool(
    config: RobotSimulationConfig,
    scenario: Mapping[str, Any],
    scenario_path: Path,
) -> None:
    """Load a scenario map file into the config map pool."""
    map_file = scenario.get("map_file")
    map_def = resolve_map_definition(map_file, scenario_path=scenario_path)
    if map_def is None:
        return
    map_id = scenario.get("map_id")
    map_name = str(map_id) if isinstance(map_id, str) and map_id.strip() else None
    if not map_name:
        map_name = Path(map_file).stem if map_file else "scenario_map"
    config.map_pool = MapDefinitionPool(map_defs={map_name: map_def})
    config.map_id = map_name


def _resolve_route_overrides_path(path_value: str, *, scenario_path: Path) -> Path:
    """Resolve a route override file path relative to scenario YAML.

    Returns:
        Path: Absolute route override file path.
    """
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = (scenario_path.parent / candidate).resolve()
    return candidate


def _route_zone_from_map(
    map_def: MapDefinition,
    *,
    is_robot: bool,
    spawn_id: int,
    goal_id: int,
    waypoints: list[tuple[float, float]],
) -> tuple[Any, Any]:
    """Resolve spawn/goal zone geometry for a route payload entry.

    Returns:
        tuple[Any, Any]: Spawn and goal zone geometries.
    """
    spawn_zones = map_def.robot_spawn_zones if is_robot else map_def.ped_spawn_zones
    goal_zones = map_def.robot_goal_zones if is_robot else map_def.ped_goal_zones
    routes = map_def.robot_routes if is_robot else map_def.ped_routes

    if 0 <= spawn_id < len(spawn_zones):
        spawn_zone = spawn_zones[spawn_id]
    else:
        spawn_zone = next(
            (route.spawn_zone for route in routes if route.spawn_id == spawn_id),
            (
                (waypoints[0][0], waypoints[0][1]),
                (waypoints[0][0] + 0.1, waypoints[0][1]),
                (waypoints[0][0], waypoints[0][1] + 0.1),
            ),
        )
    if 0 <= goal_id < len(goal_zones):
        goal_zone = goal_zones[goal_id]
    else:
        goal_zone = next(
            (route.goal_zone for route in routes if route.goal_id == goal_id),
            (
                (waypoints[-1][0], waypoints[-1][1]),
                (waypoints[-1][0] + 0.1, waypoints[-1][1]),
                (waypoints[-1][0], waypoints[-1][1] + 0.1),
            ),
        )
    return spawn_zone, goal_zone


def _coerce_route_payload(
    map_def: MapDefinition,
    route_entries: list[Any],
    *,
    is_robot: bool,
) -> list[GlobalRoute]:
    """Coerce a list of route payload dictionaries into GlobalRoute objects.

    Returns:
        list[GlobalRoute]: Parsed route objects for the selected entity class.
    """
    coerced: list[GlobalRoute] = []
    entity_name = "robot_routes" if is_robot else "ped_routes"
    for idx, entry in enumerate(route_entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{entity_name}[{idx}] must be a mapping")
        if "spawn_id" not in entry or "goal_id" not in entry:
            raise ValueError(f"{entity_name}[{idx}] missing spawn_id/goal_id")
        if "waypoints" not in entry:
            raise ValueError(f"{entity_name}[{idx}] missing waypoints")
        spawn_id = int(entry["spawn_id"])
        goal_id = int(entry["goal_id"])
        waypoints_raw = entry["waypoints"]
        if not isinstance(waypoints_raw, list) or len(waypoints_raw) < 2:
            raise ValueError(f"{entity_name}[{idx}].waypoints must be a list with >=2 points")
        waypoints = [
            _coerce_point(point, f"{entity_name}[{idx}].waypoints") for point in waypoints_raw
        ]
        spawn_zone, goal_zone = _route_zone_from_map(
            map_def,
            is_robot=is_robot,
            spawn_id=spawn_id,
            goal_id=goal_id,
            waypoints=waypoints,
        )
        coerced.append(
            GlobalRoute(
                spawn_id=spawn_id,
                goal_id=goal_id,
                waypoints=waypoints,
                spawn_zone=spawn_zone,
                goal_zone=goal_zone,
                source_label="override_adversarial",
            )
        )
    return coerced


def apply_route_overrides(
    map_def: MapDefinition,
    route_payload: Mapping[str, Any],
) -> None:
    """Apply route payload overrides to a map definition in-place."""
    robot_entries = route_payload.get("robot_routes", [])
    ped_entries = route_payload.get("ped_routes", [])
    if not isinstance(robot_entries, list):
        raise ValueError("route_payload.robot_routes must be a list")
    if not isinstance(ped_entries, list):
        raise ValueError("route_payload.ped_routes must be a list")
    if robot_entries:
        map_def.robot_routes = _coerce_route_payload(map_def, robot_entries, is_robot=True)
    if ped_entries:
        map_def.ped_routes = _coerce_route_payload(map_def, ped_entries, is_robot=False)
    map_def.__post_init__()


def _load_route_override_payload(route_overrides_path: Path) -> Mapping[str, Any]:
    """Load route override payload from YAML artifact file.

    Returns:
        Mapping[str, Any]: Payload containing robot_routes/ped_routes lists.
    """
    data = yaml.safe_load(route_overrides_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Route override file must contain a mapping: {route_overrides_path}")
    if "route_payload" in data:
        payload = data["route_payload"]
        if not isinstance(payload, Mapping):
            raise ValueError("route_payload must be a mapping")
        return payload
    return data


def _apply_route_overrides(
    config: RobotSimulationConfig,
    route_overrides_file: Any,
    scenario_path: Path,
) -> None:
    """Apply route overrides artifact to the active scenario map."""
    if route_overrides_file is None:
        return
    if not isinstance(route_overrides_file, str) or not route_overrides_file.strip():
        raise ValueError("route_overrides_file must be a non-empty string path")
    if not getattr(config, "map_pool", None) or not config.map_pool.map_defs:
        raise ValueError("route_overrides_file provided but no map_pool is loaded")
    route_overrides_path = _resolve_route_overrides_path(
        route_overrides_file, scenario_path=scenario_path
    )
    if not route_overrides_path.exists():
        raise ValueError(f"route_overrides_file does not exist: {route_overrides_path}")
    map_name, map_def = next(iter(config.map_pool.map_defs.items()))
    map_copy = deepcopy(map_def)
    payload = _load_route_override_payload(route_overrides_path)
    apply_route_overrides(map_copy, payload)
    config.map_pool.map_defs[map_name] = map_copy


__all__ = [
    "apply_route_overrides",
    "apply_single_pedestrian_overrides",
    "build_robot_config_from_scenario",
    "load_scenarios",
    "resolve_map_definition",
    "select_scenario",
]
