"""Planning utilities for the Full Classic Interaction Benchmark.

Currently placeholder functions; implemented across tasks T022-T024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


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
    raise NotImplementedError("Implemented in task T022")


def plan_scenarios(raw: list[dict], cfg, *, rng) -> list[ScenarioDescriptor]:  # T023
    raise NotImplementedError("Implemented in task T023")


def expand_episode_jobs(scenarios: list[ScenarioDescriptor], cfg) -> list:  # T024
    raise NotImplementedError("Implemented in task T024")
