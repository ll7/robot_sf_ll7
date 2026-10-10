"""Authored historical scenario packets retain their wall-law identity."""

from copy import deepcopy
from pathlib import Path

from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"


def test_authored_scenario_missing_wall_selector_preserves_legacy_identity():
    """Loading the frozen packet must not silently inherit the 0.1.0 default."""
    scenario = load_scenarios(MANIFEST)[0]
    assert "obstacle_force_law" not in scenario.get("simulation_config", {})
    config = build_robot_config_from_scenario(scenario, scenario_path=MANIFEST)
    assert config.sim_config.obstacle_force_law == "legacy_shifted_gradient_v1"


def test_authored_scenario_can_explicitly_select_new_wall_law():
    """The legacy packet boundary must not suppress an explicitly requested v3 law."""
    scenario = deepcopy(load_scenarios(MANIFEST)[0])
    scenario.setdefault("simulation_config", {})["obstacle_force_law"] = "body_edge_exponential_v3"
    config = build_robot_config_from_scenario(scenario, scenario_path=MANIFEST)
    assert config.sim_config.obstacle_force_law == "body_edge_exponential_v3"
