"""Helpers for loading scenario definitions into environment configs."""

from __future__ import annotations

import json
from copy import deepcopy
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import (
    MapDefinition,
    MapDefinitionPool,
    PedestrianWaitRule,
    SinglePedestrianDefinition,
    serialize_map,
)
from robot_sf.nav.svg_map_parser import convert_map


def _load_yaml_documents(path: Path) -> Any:
    """Load YAML content from disk.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        Any: Parsed YAML content.
    """
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_scenarios(path: Path) -> list[Mapping[str, Any]]:
    """Load raw scenario definitions from a YAML file.

    Returns:
        list[Mapping[str, Any]]: Parsed scenario entries from the file.
    """

    data = _load_yaml_documents(path)
    if isinstance(data, dict) and "scenarios" in data:
        scenarios = data["scenarios"]
    elif isinstance(data, list):
        scenarios = data
    else:  # pragma: no cover - malformed input handled by caller
        raise ValueError(f"Scenario config must contain a 'scenarios' list: {path}")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"Scenario config missing scenarios: {path}")
    return [sc for sc in scenarios if isinstance(sc, Mapping)]


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
    _apply_map_pool(config, scenario.get("map_file"), scenario_path)
    _apply_single_pedestrian_overrides(config, scenario.get("single_pedestrians"))
    return config


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
    for attr in ("peds_speed_mult", "ped_radius", "goal_radius"):
        if attr in overrides:
            setattr(config.sim_config, attr, overrides[attr])


def _apply_map_pool(
    config: RobotSimulationConfig,
    map_file: str | None,
    scenario_path: Path,
) -> None:
    """Load a scenario map file into the config map pool."""
    map_def = resolve_map_definition(map_file, scenario_path=scenario_path)
    if map_def is None:
        return
    map_name = Path(map_file).stem if map_file else "scenario_map"
    config.map_pool = MapDefinitionPool(map_defs={map_name: map_def})


__all__ = [
    "apply_single_pedestrian_overrides",
    "build_robot_config_from_scenario",
    "load_scenarios",
    "resolve_map_definition",
    "select_scenario",
]
