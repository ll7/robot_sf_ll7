"""0.1.0 defaults and immutable historical wall-law inputs."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pysocialforce.config import ObstacleForceConfig
from pysocialforce.forces import ObstacleForce

from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

ROOT = Path(__file__).resolve().parents[1]
NEW_LAW = "body_edge_exponential_v3_physical_margin10x"
LEGACY = "legacy_shifted_gradient_v1"


def test_new_default_matches_physical_margin_x10():
    """Default force is exactly the diagnostic candidate, including its amplitude."""
    assert SimulationSettings().obstacle_force_law == NEW_LAW
    default = ObstacleForceConfig()
    assert default.law_version == NEW_LAW
    simulation = SimpleNamespace(
        peds=SimpleNamespace(agent_radius=0.35, pos=lambda: np.array([[0.0, 0.45]])),
        get_raw_obstacles=lambda: np.array([[-2.0, 0.0, 2.0, 0.0, 0.0, 1.0]]),
    )
    reference = ObstacleForceConfig(
        factor=100.0, law_version="body_edge_exponential_v3_physical_margin"
    )
    actual = ObstacleForce(default, simulation)
    np.testing.assert_allclose(actual(), ObstacleForce(reference, simulation)(), rtol=1e-12)
    assert np.linalg.norm(actual()) > 0
    assert actual.law_metadata()["parameters"]["amplitude_m_s2"] == pytest.approx(30.0)


def test_legacy_still_selectable_with_original_strength():
    """Selecting legacy by name keeps its factor and historical mapping semantics."""
    assert ObstacleForceConfig(law_version=LEGACY).factor == 10.0
    assert ObstacleForceConfig(law_version={}).law_version == LEGACY
    assert SimulationSettings(obstacle_force_law=LEGACY).obstacle_force_law == LEGACY


def test_new_scenario_defaults_to_physical_margin_x10():
    """New authored inputs inherit the same default as direct configuration."""
    manifest = ROOT / "configs/scenarios/issue_10304_wall_law_bottleneck_successor_v1.yaml"
    scenario = load_scenarios(manifest)[0]
    config = build_robot_config_from_scenario(scenario, scenario_path=manifest)
    assert config.sim_config.obstacle_force_law == NEW_LAW


def test_released_and_frozen_config_bytes_unchanged():
    """Every archived config stays byte-identical to the pre-decision main snapshot."""
    snapshot = json.loads(
        (ROOT / "tests/fixtures/wall_law_released_config_sha256.json").read_text()
    )
    assert len(snapshot) > 10
    for relative, expected in snapshot.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative


@pytest.mark.parametrize(
    "filename",
    [
        "classic_interactions_francis2023_release_0_0_8_v1.yaml",
        "classic_interactions_francis2023_frozen.yaml",
        "release_0_0_5/scenario.yaml",
        "classic_interactions_francis2023.yaml",
        "francis2023_blind_corner.yaml",
        "francis2023_blind_corner_goal_zone_entry_v1.yaml",
        "paper_matrix_v2_h600_hybrid_stress_smoke.yaml",
    ],
)
def test_released_scenario_retains_legacy(filename):
    """Release and frozen scenario contexts retain the historical law without edits."""
    manifest = ROOT / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
    scenario = load_scenarios(manifest)[0]
    scenario["map_file"] = str((manifest.parent / scenario["map_file"]).resolve())
    config = build_robot_config_from_scenario(scenario, scenario_path=manifest.parent / filename)
    assert config.sim_config.obstacle_force_law == LEGACY


@pytest.mark.parametrize(
    "relative",
    [
        "configs/scenarios/classic_interactions_francis2023.yaml",
        "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_kernel_wrapped_v2.yaml",
        "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_route_width_v4.yaml",
        "configs/scenarios/issue_9860_corrected_diagnostic_v1.yaml",
        "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml",
        "configs/scenarios/francis2023_narrow_doorway_three_width_release_0_0_8_v1.yaml",
    ],
)
def test_actual_released_scenarios_all_resolve_to_legacy(relative):
    """All scenarios in released matrices retain their effective law without simulation."""
    manifest = ROOT / relative
    scenarios = load_scenarios(manifest)
    assert scenarios
    for scenario in scenarios:
        config = build_robot_config_from_scenario(scenario, scenario_path=manifest)
        assert config.sim_config.obstacle_force_law == LEGACY, scenario["name"]
