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

from robot_sf.training.scenario_loader import load_scenarios

if TYPE_CHECKING:
    from collections.abc import Sequence


def load_classic_matrix(path: str) -> list[dict]:
    """Load classic interactions scenario matrix from YAML.

    Returns:
        List of scenario dictionaries.
    """
    p = Path(path)
    if not p.exists():  # explicit fast failure
        raise FileNotFoundError(f"Scenario matrix not found: {path}")
    scenarios = load_scenarios(p, base_dir=p)
    if not scenarios:
        raise ValueError("Scenario matrix missing non-empty scenarios list")
    return list(scenarios)


def select_scenario(scenarios: Sequence[dict], name: str | None) -> dict:
    """Select a scenario by name or return the first.

    Returns:
        Selected scenario dictionary.
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
    """Return the deterministic seed list for a scenario.

    Returns:
        List of episode seeds.
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
