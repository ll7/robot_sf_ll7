"""Structural validation tests for the real-world bottleneck map pack entry."""

from __future__ import annotations

from pathlib import Path

from robot_sf.nav.svg_map_parser import convert_map

MAP_PATH = (
    Path(__file__).resolve().parents[2] / "maps" / "svg_maps" / "classic_realworld_bottleneck.svg"
)


def test_realworld_bottleneck_map_parses_with_sequential_bottlenecks() -> None:
    """The map should parse and expose bidirectional dense-flow routing primitives."""
    map_def = convert_map(str(MAP_PATH))

    assert map_def.obstacles, "Expected obstacles defining corridor and bottlenecks"
    assert len(map_def.obstacles) >= 8, "Expected outer walls + sequential bottleneck obstacles"
    assert map_def.robot_spawn_zones, "Expected at least one robot spawn zone"
    assert map_def.robot_goal_zones, "Expected at least one robot goal zone"
    assert map_def.robot_routes, "Expected at least one robot route"

    assert len(map_def.ped_spawn_zones) >= 2, "Expected bidirectional pedestrian spawn zones"
    assert len(map_def.ped_goal_zones) >= 2, "Expected bidirectional pedestrian goal zones"
    assert len(map_def.ped_routes) >= 2, "Expected bidirectional pedestrian routes"
    assert len(map_def.single_pedestrians) >= 8, "Expected dense single-ped trajectory seeds"
