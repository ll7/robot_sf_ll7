"""Shared scenario-matrix fixture writers for CLI tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml  # type: ignore

if TYPE_CHECKING:
    from pathlib import Path


def write_scenario_matrix(path: Path, scenario_id: str, repeats: int = 3) -> None:
    """Write a minimal one-scenario matrix YAML file for CLI tests.

    Args:
        path: Destination filesystem path for the matrix YAML.
        scenario_id: Scenario id recorded in the matrix row.
        repeats: Number of episode repetitions per scenario.
    """
    scenarios = [
        {
            "id": scenario_id,
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": repeats,
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)
