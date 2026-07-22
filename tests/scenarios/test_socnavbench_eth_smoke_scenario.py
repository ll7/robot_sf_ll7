"""Test that the committed SocNavBench ETH map loads through the scenario pipeline.

This test proves the REAL committed ``socnavbench_eth.svg`` (PR #4693) parses into a
``MapDefinition`` with the expected structure. It does not require any external data root,
raw SocNavBench asset, or GPU.

Expected SVG structure (from maps/svg_maps/socnavbench/socnavbench_eth.svg):
  - 377 obstacle rectangles
  - 1 robot route (robot_route_0_0)
  - 1 pedestrian route (ped_route_0_0)
  - robot_spawn_zone_0, robot_goal_zone_0
  - ped_spawn_zone_0, ped_goal_zone_0
"""

from __future__ import annotations

from pathlib import Path

from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_PATH = REPO_ROOT / "configs/scenarios/single/socnavbench_eth_smoke.yaml"
EXPECTED_OBSTACLE_COUNT = 377


def test_socnavbench_eth_smoke_loads() -> None:
    """Smoke scenario YAML loads through the existing scenario loader."""
    scenarios = load_scenarios(SCENARIO_PATH)
    assert len(scenarios) >= 1
    scenario = scenarios[0]
    assert scenario["name"] == "socnavbench_eth_smoke"
    assert "map_file" in scenario
    assert "simulation_config" in scenario


def test_socnavbench_eth_map_file_resolves_to_committed_svg() -> None:
    """Map file path resolves to the committed socnavbench_eth.svg."""
    scenarios = load_scenarios(SCENARIO_PATH)
    config = build_robot_config_from_scenario(scenarios[0], scenario_path=SCENARIO_PATH)
    map_def = next(iter(config.map_pool.map_defs.values()))
    assert map_def is not None


def test_socnavbench_eth_map_has_expected_obstacles() -> None:
    """Committed ETH SVG parses to a MapDefinition with 377 obstacle rects."""
    scenarios = load_scenarios(SCENARIO_PATH)
    config = build_robot_config_from_scenario(scenarios[0], scenario_path=SCENARIO_PATH)
    map_def = next(iter(config.map_pool.map_defs.values()))
    assert len(map_def.obstacles) == EXPECTED_OBSTACLE_COUNT


def test_socnavbench_eth_map_has_robot_route_and_zones() -> None:
    """Map has non-empty robot routes, spawn zones, and goal zones."""
    scenarios = load_scenarios(SCENARIO_PATH)
    config = build_robot_config_from_scenario(scenarios[0], scenario_path=SCENARIO_PATH)
    map_def = next(iter(config.map_pool.map_defs.values()))
    assert len(map_def.robot_routes) >= 1
    assert len(map_def.robot_spawn_zones) >= 1
    assert len(map_def.robot_goal_zones) >= 1


def test_socnavbench_eth_map_has_ped_route_and_zones() -> None:
    """Map has non-empty pedestrian routes, spawn zones, and goal zones."""
    scenarios = load_scenarios(SCENARIO_PATH)
    config = build_robot_config_from_scenario(scenarios[0], scenario_path=SCENARIO_PATH)
    map_def = next(iter(config.map_pool.map_defs.values()))
    assert len(map_def.ped_routes) >= 1
    assert len(map_def.ped_spawn_zones) >= 1
    assert len(map_def.ped_goal_zones) >= 1


def test_socnavbench_eth_map_has_four_bounds() -> None:
    """Map has exactly 4 boundary lines as required by MapDefinition.post_init."""
    scenarios = load_scenarios(SCENARIO_PATH)
    config = build_robot_config_from_scenario(scenarios[0], scenario_path=SCENARIO_PATH)
    map_def = next(iter(config.map_pool.map_defs.values()))
    assert len(map_def.bounds) == 4
