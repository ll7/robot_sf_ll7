"""Authored historical scenario packets retain their wall-law identity."""

from copy import deepcopy
from pathlib import Path

import pytest
from pysocialforce.config import (
    BODY_EDGE_EXPONENTIAL_V3,
    BODY_EDGE_EXPONENTIAL_V3_CONTACT_STIFF,
    BODY_EDGE_EXPONENTIAL_V3_MULTI_SEGMENT,
    BODY_EDGE_EXPONENTIAL_V3_PHYSICAL_MARGIN,
    BODY_EDGE_EXPONENTIAL_V3_RANGE_ONLY,
)

from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"


def test_authored_scenario_missing_wall_selector_preserves_legacy_identity():
    """Loading the frozen packet must retain its historical wall-law identity."""
    scenario = load_scenarios(MANIFEST)[0]
    assert "obstacle_force_law" not in scenario.get("simulation_config", {})
    config = build_robot_config_from_scenario(scenario, scenario_path=MANIFEST)
    assert config.sim_config.obstacle_force_law == "legacy_shifted_gradient_v1"


@pytest.mark.parametrize(
    "law",
    [
        BODY_EDGE_EXPONENTIAL_V3,
        BODY_EDGE_EXPONENTIAL_V3_RANGE_ONLY,
        BODY_EDGE_EXPONENTIAL_V3_PHYSICAL_MARGIN,
        BODY_EDGE_EXPONENTIAL_V3_CONTACT_STIFF,
        BODY_EDGE_EXPONENTIAL_V3_MULTI_SEGMENT,
    ],
)
def test_authored_scenario_can_explicitly_select_new_wall_law(law):
    """The legacy packet boundary must not suppress an explicitly requested v3 law."""
    scenario = deepcopy(load_scenarios(MANIFEST)[0])
    scenario.setdefault("simulation_config", {})["obstacle_force_law"] = law
    config = build_robot_config_from_scenario(scenario, scenario_path=MANIFEST)
    assert config.sim_config.obstacle_force_law == law
    assert config.sim_config.obstacle_force_law_resolution_mode == "explicit"
