"""Structural validation tests for the station-platform map pack entry."""

from __future__ import annotations

from pathlib import Path

from robot_sf.nav.svg_map_parser import convert_map

MAP_PATH = (
    Path(__file__).resolve().parents[2] / "maps" / "svg_maps" / "classic_station_platform.svg"
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
