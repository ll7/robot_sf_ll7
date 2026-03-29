"""Structural validation tests for the station-platform map pack entry."""

from __future__ import annotations

from pathlib import Path

from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
    select_scenario,
)

MAP_PATH = (
    Path(__file__).resolve().parents[2] / "maps" / "svg_maps" / "classic_station_platform.svg"
)
SCENARIO_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "scenarios"
    / "archetypes"
    / "classic_station_platform.yaml"
)


def test_station_platform_map_parses_with_bidirectional_platform_flow() -> None:
    """The map should parse and expose station-like route, zone, and obstacle primitives."""
    map_def = convert_map(str(MAP_PATH))

    assert map_def.obstacles, "Expected obstacles defining train edge, benches, and columns"
    assert len(map_def.obstacles) >= 9, "Expected walls plus platform furniture obstacles"
    assert map_def.robot_spawn_zones, "Expected at least one robot spawn zone"
    assert map_def.robot_goal_zones, "Expected at least one robot goal zone"
    assert map_def.robot_routes, "Expected a robot route through the concourse edge"

    assert len(map_def.ped_spawn_zones) >= 2, "Expected bidirectional pedestrian spawn zones"
    assert len(map_def.ped_goal_zones) >= 2, "Expected bidirectional pedestrian goal zones"
    assert len(map_def.ped_routes) >= 2, "Expected bidirectional pedestrian routes"
    assert len(map_def.single_pedestrians) >= 4, "Expected explicit single-ped platform markers"


def test_station_platform_scenario_overrides_add_waiting_passengers() -> None:
    """Scenario wiring should convert selected single pedestrians into deterministic waiters."""
    scenarios = load_scenarios(SCENARIO_PATH, base_dir=SCENARIO_PATH)
    scenario = select_scenario(scenarios, "classic_station_platform_medium")
    config = build_robot_config_from_scenario(scenario, scenario_path=SCENARIO_PATH)

    map_def = next(iter(config.map_pool.map_defs.values()))
    by_id = {ped.id: ped for ped in map_def.single_pedestrians}

    assert by_id["p1"].trajectory is not None
    assert by_id["p1"].wait_at is not None
    assert by_id["p1"].wait_at[0].wait_s == 6.0

    assert by_id["p3"].trajectory is not None
    assert by_id["p3"].wait_at is not None
    assert by_id["p3"].wait_at[0].wait_s == 5.0
