"""Helpers for loading scenario definitions into environment configs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool, serialize_map
from robot_sf.nav.svg_map_parser import convert_map


def _load_yaml_documents(path: Path) -> Any:
    """Load yaml documents.

    Args:
        path: Filesystem path to the resource.

    Returns:
        Any: Arbitrary value passed through unchanged.
    """
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_scenarios(path: Path) -> list[Mapping[str, Any]]:
    """Load raw scenario definitions from a YAML file."""

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
    """Return the scenario matching ``scenario_id`` or the first entry."""

    if scenario_id:
        for sc in scenarios:
            name = str(sc.get("name") or sc.get("scenario_id") or "").strip()
            if name.lower() == scenario_id.lower():
                return sc
        raise ValueError(f"Scenario id '{scenario_id}' not found in scenario config")
    return scenarios[0]


@lru_cache(maxsize=8)
def _load_map_definition(map_path: str) -> MapDefinition | None:
    """Load and convert a map definition, caching by absolute path."""

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
    """Create a ``RobotSimulationConfig`` derived from a scenario definition."""

    config = RobotSimulationConfig()
    _apply_simulation_overrides(config, scenario.get("simulation_config", {}))
    _apply_map_pool(config, scenario.get("map_file"), scenario_path)
    return config


def _apply_simulation_overrides(
    config: RobotSimulationConfig,
    overrides: Mapping[str, Any] | None,
) -> None:
    """Apply simulation overrides.

    Args:
        config: Configuration object controlling the component.
        overrides: Override dictionary.

    Returns:
        None: none.
    """
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
    """Apply map pool.

    Args:
        config: Configuration object controlling the component.
        map_file: map file.
        scenario_path: filesystem path for the scenario.

    Returns:
        None: none.
    """
    if not map_file:
        return
    candidate = Path(map_file)
    if not candidate.is_absolute():
        candidate = (scenario_path.parent / candidate).resolve()
    map_def = _load_map_definition(str(candidate))
    if map_def is None:
        return
    config.map_pool = MapDefinitionPool(map_defs={candidate.stem: map_def})


__all__ = [
    "build_robot_config_from_scenario",
    "load_scenarios",
    "select_scenario",
]
