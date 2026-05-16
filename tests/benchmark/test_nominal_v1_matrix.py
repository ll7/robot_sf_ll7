"""Tests for the nominal_v1 shared-space scenario matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from robot_sf.training.scenario_loader import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
NOMINAL_MATRIX = ROOT / "configs" / "scenarios" / "nominal_v1.yaml"


@pytest.fixture(scope="module")
def scenarios() -> list[dict[str, Any]]:
    """Load the nominal matrix scenarios once for the module."""
    return load_scenarios(NOMINAL_MATRIX)


def test_nominal_v1_matrix_loads_expected_calibration_scenarios(
    scenarios: list[dict[str, Any]],
) -> None:
    """Nominal v1 should expose the four low-stress calibration archetypes."""
    names = [scenario["name"] for scenario in scenarios]
    assert names == [
        "empty_map_8_directions_east",
        "single_ped_crossing_orthogonal",
        "classic_doorway_low",
        "classic_bottleneck_low",
    ]


def test_nominal_v1_matrix_has_low_stress_metadata(
    scenarios: list[dict[str, Any]],
) -> None:
    """Each nominal scenario should stay low-density and interpretation-bounded."""
    archetypes = {scenario["metadata"]["archetype"] for scenario in scenarios}
    assert archetypes == {
        "empty_map_8_directions",
        "single_ped_crossing_orthogonal",
        "doorway",
        "bottleneck",
    }

    for scenario in scenarios:
        assert len(scenario.get("seeds", [])) >= 3
        metadata = scenario["metadata"]
        assert (
            metadata.get("density") in {"low", "none"}
            or metadata.get("density_advisory") == "zero_baseline_route_spawn"
        )
        assert scenario["simulation_config"]["max_episode_steps"] <= 500
