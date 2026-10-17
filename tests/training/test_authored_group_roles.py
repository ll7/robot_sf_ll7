"""Reject malformed authored runtime group settings through the real loader."""

from pathlib import Path

import pytest

from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "override",
    [
        {"initial_group_id": ""},
        {"initial_group_id": 1},
        {"join_radius_m": -0.1},
        {"join_radius_m": float("nan")},
        {"join_radius_m": True},
    ],
)
def test_authored_membership_overrides_reject_invalid_settings(override):
    """Bad authoring must fail at the real loader rather than silently disable grouping."""
    path = ROOT / "configs/scenarios/single/francis2023_join_group.yaml"
    scenario = load_scenarios(path)[0]
    scenario["single_pedestrians"][2].update(override)
    with pytest.raises(ValueError, match="initial_group_id|join_radius_m"):
        build_robot_config_from_scenario(scenario, scenario_path=path)
