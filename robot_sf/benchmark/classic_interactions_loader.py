"""Utilities to load and iterate classic interaction scenarios for visualization.

Business-level intent (Spec 128): Provide a thin, programmatic interface to
select a scenario and yield (scenario_dict, seed) pairs in deterministic order
without embedding rendering or policy logic here.

Public entry points:
  - load_classic_matrix(path: str) -> list[dict]
  - select_scenario(scenarios, name: str|None) -> dict
  - iter_episode_seeds(scenario: dict) -> list[int]

This module intentionally avoids dependencies on RL libraries or rendering
components; higher-level scripts (examples) compose these utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml  # type: ignore

if TYPE_CHECKING:
    from collections.abc import Sequence


def load_classic_matrix(path: str) -> list[dict]:
    """Load classic matrix.

    Args:
        path: Filesystem path to the resource.

    Returns:
        list[dict]: list of dict.
    """
    p = Path(path)
    if not p.exists():  # explicit fast failure
        raise FileNotFoundError(f"Scenario matrix not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    scenarios = data.get("scenarios") if isinstance(data, dict) else None
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Scenario matrix missing non-empty 'scenarios' list")
    return scenarios


def select_scenario(scenarios: Sequence[dict], name: str | None) -> dict:
    """Select scenario.

    Args:
        scenarios: Scenario list.
        name: Human-friendly name.

    Returns:
        dict: Dictionary value.
    """
    if not scenarios:
        raise ValueError("No scenarios available")
    if name is None:
        return scenarios[0]
    for sc in scenarios:
        if sc.get("name") == name:
            return sc
    available = ", ".join(str(sc.get("name")) for sc in scenarios)
    raise ValueError(f"Scenario '{name}' not found. Available: {available}")


def iter_episode_seeds(scenario: dict) -> list[int]:
    """Iter episode seeds.

    Args:
        scenario: Scenario definition dictionary.

    Returns:
        list[int]: list of int.
    """
    seeds = scenario.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"Scenario '{scenario.get('name')}' missing non-empty seeds list")
    # Deterministic order as listed
    return [int(s) for s in seeds]


__all__ = [
    "iter_episode_seeds",
    "load_classic_matrix",
    "select_scenario",
]
