"""Structural validation tests for the real-world bottleneck map pack entry."""

from __future__ import annotations

from pathlib import Path

from robot_sf.nav.svg_map_parser import convert_map

MAP_PATH = (
    Path(__file__).resolve().parents[2] / "maps" / "svg_maps" / "classic_realworld_bottleneck.svg"
)


def test_realworld_bottleneck_map_parses_with_routes_and_zones() -> None:
    """The new real-world bottleneck map should parse with robot routes and zones."""
    map_def = convert_map(str(MAP_PATH))

    assert map_def.robot_spawn_zones, "Expected at least one robot spawn zone"
    assert map_def.robot_goal_zones, "Expected at least one robot goal zone"
    assert map_def.robot_routes, "Expected at least one robot route"
    assert map_def.bounds, "Expected non-empty map bounds"
