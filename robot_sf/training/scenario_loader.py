"""Helpers for loading scenario definitions into environment configs."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from robot_sf.gym_env.unified_config import (
    ObservationVisibilitySettings,
    RobotSimulationConfig,
)
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import (
    MapDefinition,
    MapDefinitionPool,
    PedestrianWaitRule,
    SinglePedestrianDefinition,
    parse_social_group_definitions,
    serialize_map,
)
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.robot.holonomic_drive import HolonomicDriveSettings
from robot_sf.sim.pedestrian_model_variants import (
    HSFM_ALIGNMENT_TORQUE_V1,
    HSFM_ANISOTROPIC_FOV_V1,
    HSFM_TTC_PREDICTIVE_V1,
)
from robot_sf.sim.sim_config import (
    AlignmentTorqueConfig,
    AnisotropicFovConfig,
    TtcPredictiveForceConfig,
)

_MAP_REGISTRY_ENV = "ROBOT_SF_MAP_REGISTRY"
_MAP_REGISTRY_PATH = Path("maps/registry.yaml")
_MAP_CATALOG_SCHEMA = "robot_sf.map_catalog.v2"
_MAP_CATALOG_PARSER_VERSION = "parser-capability-metadata.v1"
_DEFAULT_MAP_PROFILE = "robot_runtime"
_MAP_PROFILE_CAPABILITIES = {
    "robot_runtime": "robot_runtime",
    "pedestrian_runtime": "pedestrian_runtime",
    "route_only": "route_only",
    "obstacle_source": "obstacle_source",
    "benchmark_candidate": "benchmark_candidate",
}
_PLATFORM_SEMANTIC_STATUSES = {"metadata_only", "require_consumers"}
_PLATFORM_SEMANTIC_KINDS = {"hazard", "keep_clear"}
_PLATFORM_SEMANTIC_SHAPES = {"polygon", "bbox"}


@dataclass(frozen=True)
class _MapRegistryEntry:
    """One map registry row resolved to local filesystem state."""

    map_id: str
    path: Path
    capabilities: Mapping[str, bool] | None = None
    source_sha256: str | None = None
    role: str | None = None
    profile: str | None = None
    limitations: tuple[str, ...] = ()
    validation_status: str | None = None


def _load_yaml_documents(path: Path) -> Any:
    """Load YAML content from disk.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        Any: Parsed YAML content.
    """
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_scenario_manifest(
    data: Any,
    *,
    source: Path,
) -> tuple[list[Any], list[Path], list[Path]]:
    """Extract scenario entries, includes, and search paths from loaded YAML.

    Returns:
        tuple[list[Any], list[Path], list[Path]]: Scenarios, include paths, and
        resolved map search paths.
    """
    scenarios: list[Any] = []
    includes: list[Path] = []
    local_map_search_paths = (
        _resolve_map_search_paths(data, source=source) if isinstance(data, Mapping) else []
    )
    if local_map_search_paths:
        logger.info(
            "Scenario manifest '{}' configured map_search_paths: {}",
            source,
            ", ".join(str(path) for path in local_map_search_paths),
        )
    if isinstance(data, Mapping):
        includes = _resolve_includes(data, source=source)
        if "scenarios" in data:
            scenarios = data["scenarios"]
            if not isinstance(scenarios, list):
                raise ValueError(f"Scenario config 'scenarios' must be a list: {source}")
        elif not includes:
            raise ValueError(f"Scenario config must contain a 'scenarios' list: {source}")
    elif isinstance(data, list):
        scenarios = data
    else:  # pragma: no cover - malformed input handled by caller
        raise ValueError(f"Scenario config must contain a 'scenarios' list: {source}")
    return scenarios, includes, local_map_search_paths


def load_scenarios(path: str | Path, *, base_dir: Path | None = None) -> list[Mapping[str, Any]]:
    """Load scenario definitions from a YAML file.

    Supports a list of scenarios, a mapping with ``scenarios``, and optional
    include lists (``includes``, ``include``, or ``scenario_files``) for
    composing per-scenario and per-archetype files into a single list.
    Manifests can also provide ``select_scenarios`` to keep only an explicit,
    deterministic subset of the expanded scenarios by name, ``scenario_overrides``
    to apply the same nested override block to every expanded scenario, and
    ``scenario_overrides_by_name`` to apply nested overrides to specific named
    scenarios after expansion.

    Returns:
        list[Mapping[str, Any]]: Parsed scenario entries from the file(s).
    """
    from robot_sf.training.task_bundles import (  # noqa: PLC0415
        is_task_bundle_reference,
        load_task_bundle_scenarios,
    )

    if is_task_bundle_reference(path):
        return load_task_bundle_scenarios(path)

    path = Path(path)
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
        scenarios, includes, local_map_search_paths = _load_scenario_manifest(
            data,
            source=resolved,
        )
        combined: list[Mapping[str, Any]] = []
        inherited_search_paths = map_search_paths or []
        effective_search_paths = _merge_map_search_paths(
            inherited_search_paths,
            local_map_search_paths,
        )
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
        if isinstance(data, Mapping):
            combined = _apply_scenario_selection(combined, data=data, source=resolved)
            combined = _apply_scenario_overrides(
                combined,
                data=data,
                source=resolved,
                root=root,
                map_search_paths=effective_search_paths,
            )
            combined = _apply_scenario_overrides_by_name(
                combined,
                data=data,
                source=resolved,
                root=root,
                map_search_paths=effective_search_paths,
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


def _resolve_scenario_selection(data: Mapping[str, Any], *, source: Path) -> list[str]:
    """Resolve explicit scenario selection names from a manifest.

    Returns:
        list[str]: Ordered scenario names to keep.
    """
    raw = data.get("select_scenarios")
    if raw is None:
        return []
    if isinstance(raw, (str, Path)):
        entries = [raw]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError(f"select_scenarios must be a list or string in '{source}'.")
    selected: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, (str, Path)):
            raise ValueError(f"select_scenarios entry '{entry}' must be a string in '{source}'.")
        name = str(entry).strip()
        if not name:
            raise ValueError(f"select_scenarios entry must not be empty in '{source}'.")
        key = name.lower()
        if key in seen:
            raise ValueError(f"Duplicate select_scenarios entry '{name}' in '{source}'.")
        seen.add(key)
        selected.append(name)
    if not selected:
        raise ValueError(f"select_scenarios must not be empty in '{source}'.")
    return selected


def _apply_scenario_selection(
    scenarios: list[Mapping[str, Any]],
    *,
    data: Mapping[str, Any],
    source: Path,
) -> list[Mapping[str, Any]]:
    """Apply explicit scenario selection after manifest expansion.

    Returns:
        list[Mapping[str, Any]]: Filtered scenarios in selector order.
    """
    selected_names = _resolve_scenario_selection(data, source=source)
    if not selected_names:
        return scenarios

    scenario_map: dict[str, Mapping[str, Any]] = {}
    for idx, scenario in enumerate(scenarios):
        name = _scenario_identifier(scenario, source=source, index=idx)
        key = name.lower()
        if key in scenario_map:
            raise ValueError(
                f"Duplicate scenario name '{name}' in '{source}' prevents select_scenarios."
            )
        scenario_map[key] = scenario

    selected: list[Mapping[str, Any]] = []
    for name in selected_names:
        key = name.lower()
        if key not in scenario_map:
            raise ValueError(f"Unknown select_scenarios entry '{name}' in '{source}'.")
        selected.append(scenario_map[key])
    return selected


def _resolve_scenario_overrides(
    data: Mapping[str, Any],
    *,
    source: Path,
) -> Mapping[str, Any]:
    """Resolve manifest-wide scenario overrides.

    Returns:
        Mapping[str, Any]: Nested override mapping applied to each expanded
            scenario.
    """
    raw = data.get("scenario_overrides")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"scenario_overrides must be a mapping in '{source}'.")
    return raw


def _deep_merge_mapping(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Iteratively merge a nested override mapping into a scenario mapping.

    Iterative implementation avoids stack exhaustion on deeply nested override
    payloads from external manifests.

    Returns:
        dict[str, Any]: Deep-merged mapping with override values taking
            precedence.
    """
    merged: dict[str, Any] = deepcopy(dict(base))
    stack: list[tuple[dict[str, Any], Mapping[str, Any]]] = [(merged, overrides)]
    while stack:
        target, source = stack.pop()
        for key, value in source.items():
            current = target.get(key)
            if isinstance(current, Mapping) and isinstance(value, Mapping):
                nested = dict(current)
                target[key] = nested
                stack.append((nested, value))
            else:
                target[key] = deepcopy(value)
    return merged


def _apply_scenario_overrides(
    scenarios: list[Mapping[str, Any]],
    *,
    data: Mapping[str, Any],
    source: Path,
    root: Path,
    map_search_paths: list[Path],
) -> list[Mapping[str, Any]]:
    """Apply manifest-wide nested overrides after include expansion.

    Returns:
        list[Mapping[str, Any]]: Scenario list with overrides applied to each
            scenario.
    """
    overrides = _resolve_scenario_overrides(data, source=source)
    if not overrides:
        return scenarios
    return _normalize_scenarios(
        [_deep_merge_mapping(scenario, overrides) for scenario in scenarios],
        source=source,
        root=root,
        map_search_paths=map_search_paths,
    )


def _resolve_scenario_overrides_by_name(
    data: Mapping[str, Any],
    *,
    source: Path,
) -> Mapping[str, Mapping[str, Any]]:
    """Resolve per-scenario override blocks keyed by scenario name.

    Returns:
        Mapping[str, Mapping[str, Any]]: Nested override mappings to apply to
            matching expanded scenario names.
    """
    raw = data.get("scenario_overrides_by_name")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"scenario_overrides_by_name must be a mapping in '{source}'.")

    overrides: dict[str, Mapping[str, Any]] = {}
    seen_names: dict[str, str] = {}
    for name, payload in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"scenario_overrides_by_name keys must be non-empty strings in '{source}'."
            )
        resolved_name = name.strip()
        name_key = resolved_name.lower()
        if name_key in seen_names:
            raise ValueError(
                "Duplicate case-insensitive scenario_overrides_by_name entries "
                f"in '{source}': {seen_names[name_key]}, {resolved_name}"
            )
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"scenario_overrides_by_name entry '{name}' must be a mapping in '{source}'."
            )
        seen_names[name_key] = resolved_name
        overrides[resolved_name] = payload
    return overrides


def _apply_scenario_overrides_by_name(
    scenarios: list[Mapping[str, Any]],
    *,
    data: Mapping[str, Any],
    source: Path,
    root: Path,
    map_search_paths: list[Path],
) -> list[Mapping[str, Any]]:
    """Apply nested overrides to specific scenario names after expansion.

    Returns:
        list[Mapping[str, Any]]: Scenario list with matching named overrides
            applied.
    """
    overrides_by_name = _resolve_scenario_overrides_by_name(data, source=source)
    if not overrides_by_name:
        return scenarios

    unused = {name.lower(): name for name in overrides_by_name}
    target_keys = set(unused)
    merged: list[Mapping[str, Any]] = []
    applied_targets: set[str] = set()
    for index, scenario in enumerate(scenarios):
        name = _scenario_identifier(scenario, source=source, index=index)
        key = name.lower()
        if key in target_keys and key in applied_targets:
            raise ValueError(
                f"Duplicate scenario name '{name}' in '{source}' prevents "
                "scenario_overrides_by_name."
            )
        if key in target_keys:
            applied_targets.add(key)
        override_name = unused.pop(key, None)
        if override_name is None:
            merged.append(scenario)
            continue
        merged.append(_deep_merge_mapping(scenario, overrides_by_name[override_name]))

    if unused:
        unknown = ", ".join(sorted(unused.values()))
        raise ValueError(
            f"Unknown scenario_overrides_by_name entr{'y' if len(unused) == 1 else 'ies'} "
            f"in '{source}': {unknown}"
        )

    return _normalize_scenarios(
        merged,
        source=source,
        root=root,
        map_search_paths=map_search_paths,
    )


def _scenario_identifier(
    scenario: Mapping[str, Any],
    *,
    source: Path,
    index: int,
) -> str:
    """Return the stable scenario identifier used for selection and deduping."""
    name = scenario.get("name") or scenario.get("scenario_id")
    if name is None:
        raise ValueError(f"Scenario entry {index} in '{source}' is missing a name or scenario_id.")
    if not isinstance(name, str):
        raise ValueError(f"Scenario name must be a string in '{source}' at index {index}.")
    normalized = name.strip()
    if not normalized:
        raise ValueError(
            f"Scenario name must be a non-empty string in '{source}' at index {index}."
        )
    return normalized


def _resolve_map_search_paths(data: Mapping[str, Any], *, source: Path) -> list[Path]:
    """Resolve optional map search paths for the scenario manifest.

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
        raise ValueError(f"map_search_paths must be a list or string in '{source}'.")
    resolved: list[Path] = []
    base_root = source.parent
    for entry in entries:
        if not isinstance(entry, (str, Path)):
            raise ValueError(f"map_search_paths entry '{entry}' must be a string in '{source}'.")
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = (base_root / candidate).resolve()
        if not candidate.exists():
            logger.warning("map_search_paths entry does not exist: {}", candidate)
            continue
        resolved.append(candidate)
    return resolved


def _merge_map_search_paths(*path_groups: list[Path]) -> list[Path]:
    """Combine map search paths while preserving order and removing duplicates.

    Returns:
        list[Path]: Deduplicated search paths in first-seen order.
    """
    merged: list[Path] = []
    seen: set[Path] = set()
    for group in path_groups:
        for path in group:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            merged.append(resolved)
    return merged


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


def _iter_registry_mapping(entries: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Yield map_id/path pairs from a mapping-form registry.

    Yields:
        tuple[str, Mapping[str, Any]]: Map id and normalized row data.
    """
    for map_id, map_path in entries.items():
        if map_id == "version":
            continue
        if not isinstance(map_path, str):
            logger.warning("Skipping map registry entry '{}' with non-string path.", map_id)
            continue
        yield str(map_id), {"map_id": str(map_id), "path": map_path}


def _iter_registry_list(entries: list[Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Yield map_id/path pairs from a list-form registry.

    Yields:
        tuple[str, Mapping[str, Any]]: Map id and normalized row data.
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
        yield map_id, entry


def _iter_map_registry_entries(
    data: Mapping[str, Any],
    *,
    registry_path: Path,
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Yield map registry entries from the loaded registry data.

    Yields:
        tuple[str, Mapping[str, Any]]: Map id and row entries.
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
    registry: dict[str, _MapRegistryEntry],
    *,
    map_id: str,
    row: Mapping[str, Any],
    registry_path: Path,
) -> None:
    """Insert a map registry entry, resolving relative paths."""
    key = map_id.strip()
    if not key:
        raise ValueError(f"Map registry entry in '{registry_path}' has empty map_id.")
    if key in registry:
        raise ValueError(f"Duplicate map_id '{key}' in map registry '{registry_path}'.")
    map_path = row.get("path") or row.get("map_file")
    if not isinstance(map_path, str):
        raise ValueError(f"Map registry entry '{key}' in '{registry_path}' has invalid path.")
    candidate = Path(map_path)
    if not candidate.is_absolute():
        candidate = (registry_path.parent / candidate).resolve()
    registry[key] = _MapRegistryEntry(
        map_id=key,
        path=candidate,
        capabilities=_coerce_capabilities(row.get("capabilities")),
        source_sha256=row.get("source_sha256")
        if isinstance(row.get("source_sha256"), str)
        else None,
        role=row.get("role") if isinstance(row.get("role"), str) else None,
        profile=row.get("profile") if isinstance(row.get("profile"), str) else None,
        limitations=_coerce_limitations(row.get("limitations")),
        validation_status=_coerce_validation_status(row.get("validation")),
    )


def _coerce_capabilities(raw: Any) -> Mapping[str, bool] | None:
    """Return boolean capability fields from a registry row, if present."""
    if not isinstance(raw, Mapping):
        return None
    return {
        key: value for key, value in raw.items() if isinstance(key, str) and isinstance(value, bool)
    }


def _coerce_limitations(raw: Any) -> tuple[str, ...]:
    """Return stable limitation labels from a registry row."""
    if not isinstance(raw, list):
        return ()
    return tuple(str(item) for item in raw if isinstance(item, str))


def _coerce_validation_status(raw: Any) -> str | None:
    """Return the catalog validation status string, if present."""
    if not isinstance(raw, Mapping):
        return None
    status = raw.get("status")
    return status if isinstance(status, str) else None


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_catalog_header(data: Mapping[str, Any], *, registry_path: Path) -> None:
    """Fail closed when a v2 catalog has stale schema/parser headers."""
    version = data.get("version")
    if version != 2:
        return
    if data.get("schema") != _MAP_CATALOG_SCHEMA:
        raise ValueError(
            f"Map registry '{registry_path}' has stale schema "
            f"{data.get('schema')!r}; expected {_MAP_CATALOG_SCHEMA}."
        )
    if data.get("parser_version") != _MAP_CATALOG_PARSER_VERSION:
        raise ValueError(
            f"Map registry '{registry_path}' has stale parser_version "
            f"{data.get('parser_version')!r}; expected {_MAP_CATALOG_PARSER_VERSION}."
        )


@lru_cache(maxsize=4)
def _load_map_registry(path: Path | None = None) -> dict[str, _MapRegistryEntry]:
    """Load map registry entries, returning map_id -> catalog entry.

    Returns:
        dict[str, _MapRegistryEntry]: Map registry map_id to resolved catalog entries.
    """
    registry_path = path or _resolve_map_registry_path()
    if registry_path is None or not registry_path.exists():
        return {}
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Map registry '{registry_path}' must contain a mapping.")
    _validate_catalog_header(data, registry_path=registry_path)
    registry: dict[str, _MapRegistryEntry] = {}
    for map_id, row in _iter_map_registry_entries(data, registry_path=registry_path):
        _register_map_entry(
            registry,
            map_id=map_id,
            row=row,
            registry_path=registry_path,
        )
    return registry


def _resolve_map_id(
    map_id: str,
    *,
    map_registry: Mapping[str, _MapRegistryEntry],
    source: Path,
    required_profile: str = _DEFAULT_MAP_PROFILE,
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
    entry = map_registry[map_id]
    if not entry.path.exists():
        raise ValueError(f"Map registry entry '{map_id}' points to missing file: {entry.path}")
    _validate_map_catalog_entry(
        entry,
        source=source,
        required_profile=required_profile,
    )
    return entry.path


def _validate_map_catalog_entry(
    entry: _MapRegistryEntry,
    *,
    source: Path,
    required_profile: str,
) -> None:
    """Validate a resolved catalog entry for a requested map profile."""
    if entry.source_sha256 is not None and _sha256_file(entry.path) != entry.source_sha256:
        raise ValueError(
            f"Scenario in '{source}' requested profile '{required_profile}' for map_id "
            f"'{entry.map_id}', but registry source_sha256 is stale for {entry.path}."
        )
    if entry.capabilities is None:
        return
    required_capability = _MAP_PROFILE_CAPABILITIES.get(required_profile)
    if required_capability is None:
        raise ValueError(
            f"Scenario in '{source}' requested unknown map profile '{required_profile}' "
            f"for map_id '{entry.map_id}'."
        )
    if entry.capabilities.get(required_capability):
        return
    raise ValueError(
        f"Scenario in '{source}' requested profile '{required_profile}' for map_id "
        f"'{entry.map_id}', but missing capability '{required_capability}' for {entry.path}. "
        f"catalog_status={entry.validation_status or 'unknown'}; "
        f"role={entry.role or 'unknown'}; profile={entry.profile or 'unknown'}; "
        f"limitations={list(entry.limitations)}"
    )


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
    map_registry: dict[str, _MapRegistryEntry] = {}
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
    _resolve_required_map_profile(scenario, source=source)
    _validate_optional_mapping(scenario, key="simulation_config", source=source, index=index)
    _validate_optional_mapping(scenario, key="robot_config", source=source, index=index)
    _validate_optional_mapping(scenario, key="observation_visibility", source=source, index=index)
    _validate_optional_mapping(scenario, key="metadata", source=source, index=index)
    _validate_platform_semantics(scenario, source=source, index=index)
    _validate_seed_list(scenario, source=source, index=index)


def _validate_map_reference(
    scenario: Mapping[str, Any],
    *,
    name: str | None,
    source: Path,
    index: int,
) -> None:
    """Validate that a scenario has a usable map reference shape."""
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


def _resolve_required_map_profile(
    scenario: Mapping[str, Any],
    *,
    source: Path,
) -> str:
    """Resolve the capability profile a scenario requires from a map_id row.

    Returns:
        str: Required map capability profile.
    """
    raw = scenario.get("required_map_profile") or scenario.get("map_profile")
    if raw is None:
        return _DEFAULT_MAP_PROFILE
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"required_map_profile must be a non-empty string in '{source}'.")
    profile = raw.strip()
    if profile not in _MAP_PROFILE_CAPABILITIES:
        raise ValueError(f"Unknown required_map_profile '{profile}' in '{source}'.")
    return profile


def _rebase_scenario_paths(
    scenario: Mapping[str, Any],
    *,
    source: Path,
    root: Path,
    map_search_paths: list[Path],
    map_registry: Mapping[str, _MapRegistryEntry],
) -> Mapping[str, Any]:
    """Rewrite relative map paths to be relative to the root scenario file.

    Returns:
        Mapping[str, Any]: Scenario entry with rebased paths when applicable.
    """
    search_root = root if root.is_dir() else root.parent
    map_id = scenario.get("map_id")
    if isinstance(map_id, str) and map_id.strip():
        resolved = _resolve_map_id(
            map_id,
            map_registry=map_registry,
            source=source,
            required_profile=_resolve_required_map_profile(scenario, source=source),
        )
        rel = os.path.relpath(resolved, search_root)
        updated = dict(scenario)
        updated["map_file"] = Path(rel).as_posix()
        return _rebase_route_override_path(updated, source=source)

    map_file = scenario.get("map_file")
    if not isinstance(map_file, str):
        return _rebase_route_override_path(scenario, source=source)
    candidate = Path(map_file)
    if candidate.is_absolute():
        return _rebase_route_override_path(scenario, source=source)
    probe = (search_root / candidate).resolve()
    if probe.exists():
        return _rebase_route_override_path(scenario, source=source)
    if source.parent != search_root:
        abs_target = (source.parent / candidate).resolve()
        if abs_target.exists():
            rel = os.path.relpath(abs_target, search_root)
            updated = dict(scenario)
            updated["map_file"] = Path(rel).as_posix()
            return _rebase_route_override_path(updated, source=source)
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
        return _rebase_route_override_path(scenario, source=source)
    rel = os.path.relpath(resolved, search_root)
    updated = dict(scenario)
    updated["map_file"] = Path(rel).as_posix()
    return _rebase_route_override_path(updated, source=source)


def _rebase_route_override_path(
    scenario: Mapping[str, Any],
    *,
    source: Path,
) -> Mapping[str, Any]:
    """Rewrite relative route override paths to the root scenario manifest base.

    Returns:
        Mapping[str, Any]: Scenario entry with a rebased ``route_overrides_file`` when needed.
    """
    route_file = scenario.get("route_overrides_file")
    if not isinstance(route_file, str):
        return scenario
    candidate = Path(route_file)
    if candidate.is_absolute():
        return scenario
    abs_target = (source.parent / candidate).resolve()
    if not abs_target.exists():
        return scenario
    updated = dict(scenario)
    updated["route_overrides_file"] = abs_target.as_posix()
    return updated


def _resolve_map_with_search_paths(
    map_file: str,
    *,
    map_search_paths: list[Path],
    root: Path,
    source: Path,
) -> Path | None:
    """Resolve a map path relative to the manifest root or configured search paths.

    Returns:
        Path | None: Existing map path, or ``None`` when no candidate resolves.
    """
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
    """Log an actionable warning for a map path that could not be resolved."""
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
    """Validate that an optional scenario section is a mapping when present."""
    value = scenario.get(key)
    if value is not None and not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping in '{source}' at index {index}.")


def _validate_seed_list(
    scenario: Mapping[str, Any],
    *,
    source: Path,
    index: int,
) -> None:
    """Validate optional scenario seed overrides."""
    seeds = scenario.get("seeds")
    if seeds is None:
        return
    if not isinstance(seeds, list):
        raise ValueError(f"seeds must be a list in '{source}' at index {index}.")
    if not all(isinstance(seed, int) for seed in seeds):
        raise ValueError(f"seeds must contain integers in '{source}' at index {index}.")


def _validate_platform_semantics(
    scenario: Mapping[str, Any],
    *,
    source: Path,
    index: int,
) -> None:
    """Validate optional platform hazard and keep-clear semantic metadata."""
    semantics = scenario.get("platform_semantics")
    if semantics is None:
        return
    if not isinstance(semantics, Mapping):
        raise ValueError(f"platform_semantics must be a mapping in '{source}' at index {index}.")

    status = semantics.get("status", "metadata_only")
    if status not in _PLATFORM_SEMANTIC_STATUSES:
        raise ValueError(
            f"platform_semantics.status must be one of "
            f"{sorted(_PLATFORM_SEMANTIC_STATUSES)} in '{source}' at index {index}."
        )

    regions = semantics.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError(
            f"platform_semantics.regions must be a non-empty list in '{source}' at index {index}."
        )
    for region_idx, region in enumerate(regions):
        _validate_platform_semantic_region(
            region,
            source=source,
            index=index,
            region_idx=region_idx,
        )


def _validate_platform_semantic_region(
    region: Any,
    *,
    source: Path,
    index: int,
    region_idx: int,
) -> None:
    """Validate one platform semantic region."""
    prefix = f"platform_semantics.regions[{region_idx}]"
    if not isinstance(region, Mapping):
        raise ValueError(f"{prefix} must be a mapping in '{source}' at index {index}.")
    region_id = region.get("id")
    if not isinstance(region_id, str) or not region_id.strip():
        raise ValueError(f"{prefix}.id must be a non-empty string in '{source}' at index {index}.")
    kind = region.get("kind")
    if kind not in _PLATFORM_SEMANTIC_KINDS:
        raise ValueError(
            f"{prefix}.kind must be one of {sorted(_PLATFORM_SEMANTIC_KINDS)} "
            f"in '{source}' at index {index}."
        )
    shape = region.get("shape")
    if shape not in _PLATFORM_SEMANTIC_SHAPES:
        raise ValueError(
            f"{prefix}.shape must be one of {sorted(_PLATFORM_SEMANTIC_SHAPES)} "
            f"in '{source}' at index {index}."
        )
    if shape == "polygon":
        _validate_platform_semantic_polygon(region, prefix=prefix, source=source, index=index)
    else:
        _validate_platform_semantic_bbox(region, prefix=prefix, source=source, index=index)


def _validate_platform_semantic_polygon(
    region: Mapping[str, Any],
    *,
    prefix: str,
    source: Path,
    index: int,
) -> None:
    """Validate polygon platform semantics and point coordinates."""
    points = region.get("points")
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError(
            f"{prefix}.points must contain at least 3 points in '{source}' at index {index}."
        )
    for point_idx, point in enumerate(points):
        try:
            _coerce_point(point, f"{prefix}.points[{point_idx}]")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{prefix}.points[{point_idx}] must be an [x, y] point.") from exc


def _validate_platform_semantic_bbox(
    region: Mapping[str, Any],
    *,
    prefix: str,
    source: Path,
    index: int,
) -> None:
    """Validate bounding-box platform semantics and coordinate ordering."""
    bounds = region.get("bounds")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        raise ValueError(f"{prefix}.bounds must be [min_x, min_y, max_x, max_y].")
    min_x, min_y, max_x, max_y = (float(value) for value in bounds)
    if min_x >= max_x or min_y >= max_y:
        raise ValueError(f"{prefix}.bounds must have min values below max values.")


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


@lru_cache(maxsize=256)
def _load_map_definition(map_path: str) -> MapDefinition | None:
    """Load and convert a map definition, caching by absolute path.

    The cache size is set to 256 to accommodate all unique maps across typical
    multi-scenario SAC training runs. ``classic_interactions.yaml`` alone
    references 12 distinct SVG maps; the previous ``maxsize=8`` caused
    repeated cache evictions and redundant SVG parsing whenever more than
    8 unique maps were active in the same training session.

    Returns:
        MapDefinition | None: Parsed map definition for SVG maps, else ``None``.
    """

    path = Path(map_path)
    if not path.exists():
        logger.warning("Scenario map file not found: {}", path)
        return None
    if path.suffix.lower() == ".svg":
        return convert_map(str(path))
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        data = _load_yaml_documents(path)
        if not isinstance(data, dict):
            logger.warning("Map definition '{}' must contain a mapping.", path)
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

    _reject_required_platform_semantic_consumers(scenario)

    config = RobotSimulationConfig()
    _apply_simulation_overrides(config, scenario.get("simulation_config", {}))
    _apply_robot_overrides(config, scenario.get("robot_config", {}))
    _apply_observation_visibility_overrides(
        config,
        scenario.get("observation_visibility"),
    )
    _apply_map_pool(config, scenario, scenario_path)
    _apply_route_overrides(config, scenario.get("route_overrides_file"), scenario_path)
    _apply_single_pedestrian_overrides(
        config,
        scenario.get("single_pedestrians"),
        default_hold_ref_point=_scenario_conflict_point(scenario),
    )
    _apply_social_group_overrides(config, scenario.get("social_groups"))
    return config


def _reject_required_platform_semantic_consumers(scenario: Mapping[str, Any]) -> None:
    """Fail closed when scenario semantics require consumers that do not exist yet."""
    semantics = scenario.get("platform_semantics")
    if not isinstance(semantics, Mapping):
        return
    if semantics.get("status", "metadata_only") == "require_consumers":
        raise NotImplementedError(
            "platform_semantics consumers are not implemented; use status='metadata_only' "
            "for provenance-only regions or add explicit planner/metric support."
        )


def _coerce_finite_float(value: Any, *, field_name: str) -> float:
    """Parse and validate a finite float value for scenario robot_config fields.

    Returns:
        float: Parsed finite float value.
    """
    parsed = float(value)
    if not math.isfinite(parsed):
        prefix = "" if "." in field_name else "robot_config."
        raise ValueError(f"{prefix}{field_name} must be finite.")
    return parsed


def _coerce_non_negative_float(value: Any, *, field_name: str) -> float:
    """Parse and validate a non-negative finite float for scenario robot_config fields.

    Returns:
        float: Parsed finite float value that is ``>= 0``.
    """
    parsed = _coerce_finite_float(value, field_name=field_name)
    if parsed < 0.0:
        prefix = "" if "." in field_name else "robot_config."
        raise ValueError(f"{prefix}{field_name} must be >= 0.")
    return parsed


def _coerce_positive_float(value: Any, *, field_name: str) -> float:
    """Parse and validate a positive finite float for scenario fields.

    Returns:
        float: Parsed finite float that is ``> 0``.
    """
    parsed = _coerce_finite_float(value, field_name=field_name)
    if parsed <= 0.0:
        prefix = "" if "." in field_name else "robot_config."
        raise ValueError(f"{prefix}{field_name} must be > 0.")
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
    prefix = "" if "." in field_name else "robot_config."
    raise ValueError(f"{prefix}{field_name} must be a boolean.")


def _apply_observation_visibility_overrides(
    config: RobotSimulationConfig,
    overrides: Mapping[str, Any] | None,
) -> None:
    """Apply optional planner-facing visibility limits from scenario YAML."""
    if overrides is None:
        return
    if not isinstance(overrides, Mapping):
        raise ValueError("observation_visibility must be a mapping.")
    allowed = {"enabled", "fov_degrees", "max_range_m", "static_occlusion", "dynamic_occlusion"}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"observation_visibility contains unknown keys: {', '.join(unknown)}.")

    enabled = (
        _coerce_bool(overrides["enabled"], field_name="observation_visibility.enabled")
        if "enabled" in overrides
        else True
    )
    fov_degrees = (
        _coerce_positive_float(
            overrides["fov_degrees"],
            field_name="observation_visibility.fov_degrees",
        )
        if "fov_degrees" in overrides
        else 360.0
    )
    if fov_degrees > 360.0:
        raise ValueError("observation_visibility.fov_degrees must be <= 360.")
    max_range_m = None
    if "max_range_m" in overrides and overrides["max_range_m"] is not None:
        max_range_m = _coerce_positive_float(
            overrides["max_range_m"],
            field_name="observation_visibility.max_range_m",
        )
    static_occlusion = (
        _coerce_bool(
            overrides["static_occlusion"],
            field_name="observation_visibility.static_occlusion",
        )
        if "static_occlusion" in overrides
        else False
    )
    dynamic_occlusion = (
        _coerce_bool(
            overrides["dynamic_occlusion"],
            field_name="observation_visibility.dynamic_occlusion",
        )
        if "dynamic_occlusion" in overrides
        else False
    )
    config.observation_visibility = ObservationVisibilitySettings(
        enabled=enabled,
        fov_degrees=fov_degrees,
        max_range_m=max_range_m,
        static_occlusion=static_occlusion,
        dynamic_occlusion=dynamic_occlusion,
    )


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
    *,
    default_hold_ref_point: tuple[float, float] | None = None,
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
        updated.append(
            _apply_single_pedestrian_override(
                ped,
                entry,
                map_def,
                default_hold_ref_point=default_hold_ref_point,
            )
        )

    if overrides_by_id:
        unknown = ", ".join(sorted(overrides_by_id.keys()))
        raise ValueError(f"Unknown single_pedestrians ids in overrides: {unknown}")

    map_def.single_pedestrians = updated


def _apply_single_pedestrian_overrides(
    config: RobotSimulationConfig,
    overrides: list[Mapping[str, Any]] | None,
    *,
    default_hold_ref_point: tuple[float, float] | None = None,
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
    apply_single_pedestrian_overrides(
        map_def,
        overrides,
        default_hold_ref_point=default_hold_ref_point,
    )


def _apply_social_group_overrides(
    config: RobotSimulationConfig,
    overrides: list[Mapping[str, Any]] | None,
) -> None:
    """Apply scenario-level social group overrides to cloned map definitions."""
    if not overrides:
        return
    if config.map_pool is None:
        raise ValueError("social_groups overrides provided but config has no map pool")

    cloned_maps = dict(config.map_pool.map_defs)
    for map_id, map_def in cloned_maps.items():
        updated = deepcopy(map_def)
        updated.social_groups = parse_social_group_definitions(overrides)
        updated._validate_social_groups()
        cloned_maps[map_id] = updated
    config.map_pool = MapDefinitionPool(map_defs=cloned_maps)


def _scenario_conflict_point(scenario: Mapping[str, Any]) -> tuple[float, float] | None:
    """Extract the public-requirement event-contract conflict point, if present.

    Returns:
        tuple[float, float] | None: The conflict point used as the default proximity-hold
        reference, or ``None`` when the scenario declares no such contract.
    """
    metadata = scenario.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    public_requirement = metadata.get("public_requirement")
    if not isinstance(public_requirement, Mapping):
        return None
    event_contract = public_requirement.get("event_contract")
    if not isinstance(event_contract, Mapping):
        return None
    conflict_point = event_contract.get("conflict_point")
    if conflict_point is None:
        return None
    return _coerce_point(conflict_point, "event_contract.conflict_point")


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
    point = (float(value[0]), float(value[1]))
    if not math.isfinite(point[0]) or not math.isfinite(point[1]):
        raise ValueError(f"'{label}' coordinates must be finite, got: {value!r}")
    return point


def _coerce_positive_finite_float(value: object, label: str) -> float:
    """Coerce a scenario scalar that must be finite and strictly positive.

    Returns:
        float: Validated scenario scalar.
    """
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"'{label}' must be finite, got: {value!r}")
    if parsed <= 0.0:
        raise ValueError(f"'{label}' must be > 0, got: {value!r}")
    return parsed


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


def _resolve_start_delay_override(
    ped: SinglePedestrianDefinition,
    entry: Mapping[str, Any],
) -> float:
    """Resolve bounded start-delay dwell overrides for a pedestrian definition.

    Returns:
        float: Existing or overridden start-delay duration in seconds.
    """
    if "start_delay_s" not in entry:
        return float(ped.start_delay_s)
    value = entry.get("start_delay_s")
    return float(value) if value is not None else 0.0


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


def _resolve_hold_overrides(
    ped: SinglePedestrianDefinition,
    entry: Mapping[str, Any],
    *,
    default_hold_ref_point: tuple[float, float] | None,
) -> tuple[float | None, tuple[float, float] | None, float | None]:
    """Resolve proximity-hold overrides for a pedestrian definition.

    When ``hold_until_robot_within_m`` is set without an explicit ``hold_ref_point``, the
    reference point defaults to the scenario event-contract ``conflict_point`` so authors do
    not have to repeat the coordinate.

    Returns:
        tuple[float | None, tuple[float, float] | None, float | None]: Hold radius, reference
        point, and timeout in seconds.
    """
    if "hold_until_robot_within_m" in entry:
        value = entry.get("hold_until_robot_within_m")
        hold_within = (
            _coerce_positive_finite_float(value, "hold_until_robot_within_m")
            if value is not None
            else None
        )
    else:
        hold_within = ped.hold_until_robot_within_m

    if "hold_ref_point" in entry:
        ref_value = entry.get("hold_ref_point")
        hold_ref_point = (
            _coerce_point(ref_value, "hold_ref_point") if ref_value is not None else None
        )
    else:
        hold_ref_point = ped.hold_ref_point

    if "hold_timeout_s" in entry:
        timeout_value = entry.get("hold_timeout_s")
        hold_timeout_s = (
            _coerce_positive_finite_float(timeout_value, "hold_timeout_s")
            if timeout_value is not None
            else None
        )
    else:
        hold_timeout_s = ped.hold_timeout_s

    if hold_within is not None and hold_ref_point is None and default_hold_ref_point is not None:
        hold_ref_point = default_hold_ref_point
    if hold_within is not None and hold_ref_point is None:
        raise ValueError(
            f"Pedestrian '{ped.id}': hold_until_robot_within_m requires hold_ref_point "
            "or scenario public_requirement.event_contract.conflict_point"
        )

    return hold_within, hold_ref_point, hold_timeout_s


def _resolve_metadata_override(
    ped: SinglePedestrianDefinition,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve additive per-pedestrian metadata from a scenario override.

    Returns:
        dict[str, Any]: Existing metadata merged with override metadata when present.
    """
    if "metadata" not in entry:
        return dict(ped.metadata)
    raw_metadata = entry.get("metadata")
    if raw_metadata is None:
        return {}
    if not isinstance(raw_metadata, Mapping):
        raise ValueError(f"single_pedestrians '{ped.id}' metadata must be a mapping")
    metadata = dict(ped.metadata)
    metadata.update(dict(raw_metadata))
    return metadata


def _apply_single_pedestrian_override(
    ped: SinglePedestrianDefinition,
    entry: Mapping[str, Any],
    map_def: MapDefinition,
    *,
    default_hold_ref_point: tuple[float, float] | None = None,
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
    start_delay_s = _resolve_start_delay_override(ped, entry)
    note = _resolve_note_override(ped, entry)
    role, role_target_id, role_offset = _resolve_role_overrides(ped, entry)
    hold_within, hold_ref_point, hold_timeout_s = _resolve_hold_overrides(
        ped,
        entry,
        default_hold_ref_point=default_hold_ref_point,
    )
    metadata = _resolve_metadata_override(ped, entry)

    return SinglePedestrianDefinition(
        id=ped.id,
        start=ped.start,
        goal=goal,
        trajectory=trajectory,
        speed_m_s=speed,
        wait_at=wait_at,
        start_delay_s=start_delay_s,
        note=note,
        role=role,
        role_target_id=role_target_id,
        role_offset=role_offset,
        hold_until_robot_within_m=hold_within,
        hold_ref_point=hold_ref_point,
        hold_timeout_s=hold_timeout_s,
        metadata=metadata,
    )


# Opt-in pedestrian-model config attribute -> (config dataclass, activating model selector).
# Each entry drives both the nested-mapping override path and the ``pedestrian_model``
# selector path below, so adding a new opt-in force model is a single-line change here.
_OPT_IN_PEDESTRIAN_MODEL_CONFIGS: dict[str, tuple[type, str]] = {
    "ttc_predictive_force": (TtcPredictiveForceConfig, HSFM_TTC_PREDICTIVE_V1),
    "anisotropic_fov": (AnisotropicFovConfig, HSFM_ANISOTROPIC_FOV_V1),
    "alignment_torque": (AlignmentTorqueConfig, HSFM_ALIGNMENT_TORQUE_V1),
}
# Reverse lookup: activating model selector -> its opt-in config attribute name.
_PEDESTRIAN_MODEL_ENABLE_ATTR: dict[str, str] = {
    selector: attr for attr, (_, selector) in _OPT_IN_PEDESTRIAN_MODEL_CONFIGS.items()
}


def _set_simulation_override_attr(
    config: RobotSimulationConfig,
    attr: str,
    overrides: Mapping[str, Any],
) -> None:
    """Apply one scenario-level simulation override attribute."""
    config_spec = _OPT_IN_PEDESTRIAN_MODEL_CONFIGS.get(attr)
    if config_spec is not None and isinstance(overrides[attr], Mapping):
        # Nested opt-in force config given directly; auto-enable when its selector is active.
        config_cls, selector_model = config_spec
        sub_overrides = dict(overrides[attr])
        # Auto-enable when the selector model is active either via this scenario's overrides or
        # via the already-applied base config; checking only the overrides would silently reset
        # ``enabled`` to False when the base config selects the model but the scenario omits
        # ``pedestrian_model`` (provenance-loss regression flagged in review).
        if (
            overrides.get("pedestrian_model") == selector_model
            or config.sim_config.pedestrian_model == selector_model
        ):
            sub_overrides["enabled"] = True
        setattr(config.sim_config, attr, config_cls(**sub_overrides))
    elif attr == "pedestrian_model" and overrides[attr] in _PEDESTRIAN_MODEL_ENABLE_ATTR:
        # Selecting an opt-in model marks its companion config enabled for provenance.
        setattr(config.sim_config, attr, overrides[attr])
        enable_attr = _PEDESTRIAN_MODEL_ENABLE_ATTR[overrides[attr]]
        setattr(
            config.sim_config,
            enable_attr,
            replace(getattr(config.sim_config, enable_attr), enabled=True),
        )
    elif attr == "pedestrian_uncertainty_envelope_enabled":
        setattr(
            config.sim_config,
            attr,
            _coerce_bool(overrides[attr], field_name=f"simulation_config.{attr}"),
        )
    elif attr == "pedestrian_uncertainty_alpha_mps":
        alpha = _coerce_finite_float(overrides[attr], field_name=f"simulation_config.{attr}")
        if alpha < 0.0:
            raise ValueError("simulation_config.pedestrian_uncertainty_alpha_mps must be >= 0.")
        setattr(config.sim_config, attr, alpha)
    elif attr in {"action_latency_steps", "action_latency_ms"}:
        setattr(config.sim_config, attr, overrides[attr])
        config.sim_config._validate_action_latency_config()
    else:
        setattr(config.sim_config, attr, overrides[attr])


_PRF_CONFIG_OVERRIDE_FIELDS = (
    "is_active",
    "robot_radius",
    "activation_threshold",
    "force_multiplier",
)


def _apply_prf_config_override(
    config: RobotSimulationConfig,
    overrides: Mapping[str, Any] | None,
) -> None:
    """Apply scenario-level pedestrian-robot force calibration overrides (issue #4974).

    Exposes the ped-robot force's coefficient (``force_multiplier``), effective
    radius (``robot_radius``), activation distance (``activation_threshold``), and
    active flag (``is_active``) as a per-scenario calibration surface. Defaults are
    preserved when a field is omitted, so existing scenarios are unchanged.
    """
    if overrides is None:
        return
    if not isinstance(overrides, Mapping):
        raise ValueError("simulation_config.prf_config must be a mapping.")
    unknown = sorted(set(overrides) - set(_PRF_CONFIG_OVERRIDE_FIELDS))
    if unknown:
        raise ValueError(
            f"simulation_config.prf_config contains unknown keys: {', '.join(unknown)}."
        )
    base = config.sim_config.prf_config
    kwargs: dict[str, Any] = {
        "is_active": base.is_active,
        "robot_radius": base.robot_radius,
        "activation_threshold": base.activation_threshold,
        "force_multiplier": base.force_multiplier,
    }
    if "is_active" in overrides:
        kwargs["is_active"] = _coerce_bool(
            overrides["is_active"], field_name="prf_config.is_active"
        )
    if "robot_radius" in overrides:
        kwargs["robot_radius"] = _coerce_positive_float(
            overrides["robot_radius"], field_name="prf_config.robot_radius"
        )
    if "activation_threshold" in overrides:
        kwargs["activation_threshold"] = _coerce_positive_float(
            overrides["activation_threshold"], field_name="prf_config.activation_threshold"
        )
    if "force_multiplier" in overrides:
        kwargs["force_multiplier"] = _coerce_finite_float(
            overrides["force_multiplier"], field_name="prf_config.force_multiplier"
        )
    config.sim_config.prf_config = PedRobotForceConfig(**kwargs)


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
        "action_latency_steps",
        "action_latency_ms",
        "pedestrian_integration_scheme",
        "ped_radius",
        "pedestrian_uncertainty_envelope_enabled",
        "pedestrian_uncertainty_alpha_mps",
        "goal_radius",
        "pedestrian_model",
        "ttc_predictive_force",
        "anisotropic_fov",
        "alignment_torque",
        "route_spawn_distribution",
        "route_spawn_jitter_frac",
        "route_spawn_seed",
        "archetype_composition",
        "archetype_speed_factors",
        "archetype_seed",
        "response_law_composition",
        "response_law_seed",
        "population_size",
        "non_reactive_response_multiplier",
        "hesitating_response_multiplier",
    ):
        if attr in overrides:
            _set_simulation_override_attr(config, attr, overrides)
    # Expose the pedestrian-robot force as a calibration surface (issue #4974):
    # coefficient (force_multiplier), effective radius (robot_radius), activation
    # distance, and active flag can be tuned per-scenario without touching defaults.
    if "prf_config" in overrides:
        _apply_prf_config_override(config, overrides["prf_config"])


def _apply_map_pool(
    config: RobotSimulationConfig,
    scenario: Mapping[str, Any],
    scenario_path: Path,
) -> None:
    """Load a scenario map file into the config map pool.

    Raises:
        ValueError: When ``map_file`` is explicitly specified in the scenario but
            the file cannot be resolved or loaded.  Failing early here prevents
            the silent fallback to the default map pool — which would either use
            an unrelated map (``uni_campus_big``) or raise a confusing
            ``"Map pool is empty!"`` error much later during the first scenario
            reset (the original issue #830 failure mode on long SLURM runs).
    """
    map_file = scenario.get("map_file")
    map_def = resolve_map_definition(map_file, scenario_path=scenario_path)
    if map_def is None:
        if map_file:
            scenario_name = scenario.get("name") or scenario.get("scenario_id") or "unknown"
            raise ValueError(
                f"Scenario '{scenario_name}': map_file '{map_file}' could not be "
                f"resolved or loaded from scenario_path='{scenario_path}'. "
                "Check the map_file path or manifest map_search_paths."
            )
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


def map_cache_info() -> dict[str, int]:
    """Return hit/miss/eviction statistics for the map-definition cache.

    Useful for diagnosing cache churn during multi-scenario training runs.
    Example::

        from robot_sf.training.scenario_loader import map_cache_info

        info = map_cache_info()
        # {'hits': 42, 'misses': 12, 'maxsize': 64, 'currsize': 12}

    Returns:
        dict[str, int]: Mapping of ``hits``, ``misses``, ``maxsize``, and
        ``currsize`` from the underlying LRU cache.
    """
    ci = _load_map_definition.cache_info()
    return {
        "hits": ci.hits,
        "misses": ci.misses,
        "maxsize": ci.maxsize,
        "currsize": ci.currsize,
    }


__all__ = [
    "_apply_social_group_overrides",
    "apply_route_overrides",
    "apply_single_pedestrian_overrides",
    "build_robot_config_from_scenario",
    "load_scenarios",
    "map_cache_info",
    "resolve_map_definition",
    "select_scenario",
]
