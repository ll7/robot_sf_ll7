"""Development copies of real scenario inputs; retained release bytes stay untouched."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import yaml

from robot_sf.evidence.writers import write_text
from robot_sf.training.scenario_loader import load_scenarios

if TYPE_CHECKING:
    from pathlib import Path


def development_scenarios(path: Path) -> list[dict]:
    """Keep authored geometry and ordering, with explicit development episode seeds."""
    scenarios = [deepcopy(dict(scenario)) for scenario in load_scenarios(path)]
    for scenario in scenarios:
        seeds = scenario.get("seeds") or [1001]
        scenario["seeds"] = [1001 + index % 30 for index in range(len(seeds))]
    return scenarios


def write_development_matrix(source: Path, destination: Path) -> Path:
    """Serialize resolved scenarios with the shared marked writer."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_text(
        destination,
        "# AI-GENERATED NEEDS-REVIEW\n" + yaml.safe_dump(development_scenarios(source)),
    )
    return destination
