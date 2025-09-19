"""Planning utilities for the Full Classic Interaction Benchmark.

Currently placeholder functions; implemented across tasks T022-T024.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml  # type: ignore

# ---- Internal helpers (kept simple to keep public function complexity low) ----


def _parse_yaml_file(p: Path) -> dict:
    try:
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to parse YAML: {exc}") from exc


def _extract_scenarios(root: dict) -> list[dict]:
    scenarios = root.get("scenarios") if isinstance(root, dict) else None
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Scenario matrix missing non-empty 'scenarios' list")
    return scenarios


_REQUIRED_SC_KEYS = {"name", "map_file", "simulation_config", "metadata"}


def _validate_scenario_dicts(scenarios: list[dict]) -> None:
    for idx, sc in enumerate(scenarios):
        if not isinstance(sc, dict):
            raise ValueError(f"Scenario index {idx} not a mapping")
        missing = _REQUIRED_SC_KEYS - sc.keys()
        if missing:
            raise ValueError(f"Scenario '{sc.get('name', idx)}' missing keys: {missing}")
        sim_cfg = sc.get("simulation_config", {})
        meta = sc.get("metadata", {})
        if not isinstance(sim_cfg, dict) or not isinstance(meta, dict):
            raise ValueError(f"Scenario '{sc.get('name')}' has invalid nested structures")
        if "max_episode_steps" not in sim_cfg:
            raise ValueError(f"Scenario '{sc.get('name')}' missing max_episode_steps")
        if "archetype" not in meta or "density" not in meta:
            raise ValueError(f"Scenario '{sc.get('name')}' metadata missing archetype/density")


# Minimal placeholder dataclasses matching data-model (subset for scaffolding)
@dataclass
class ScenarioDescriptor:  # duplicated light form; real version will live centrally later
    scenario_id: str
    archetype: str
    density: str
    map_path: str
    params: Dict[str, object]
    planned_seeds: List[int]
    max_episode_steps: int
    hash_fragment: str


def load_scenario_matrix(path: str) -> list[dict]:  # T022
    """Load scenario matrix YAML returning the raw scenario dictionaries.

    Public function kept intentionally thin for low cyclomatic complexity.
    """
    p = Path(path)
    if not p.exists():  # contract: FileNotFoundError path missing
        raise FileNotFoundError(path)
    data = _parse_yaml_file(p)
    scenarios = _extract_scenarios(data)
    _validate_scenario_dicts(scenarios)
    return scenarios


def plan_scenarios(raw: list[dict], cfg, *, rng) -> list[ScenarioDescriptor]:  # T023
    raise NotImplementedError("Implemented in task T023")


def expand_episode_jobs(scenarios: list[ScenarioDescriptor], cfg) -> list:  # T024
    raise NotImplementedError("Implemented in task T024")
