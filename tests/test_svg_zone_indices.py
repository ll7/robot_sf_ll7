"""Tests for numbered SVG spawn/goal zones."""

from __future__ import annotations

from typing import TYPE_CHECKING

from robot_sf.nav.svg_map_parser import convert_map

if TYPE_CHECKING:
    from pathlib import Path


def test_numbered_spawn_zones_are_ordered(tmp_path: Path) -> None:
    """Preserve explicit numbering so scenario routes stay aligned after edits."""
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg"
         xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
         width="10" height="10">
      <rect id="robot_spawn_zone_1" inkscape:label="robot_spawn_zone_1" x="1" y="1" width="1" height="1" />
      <rect id="robot_spawn_zone" inkscape:label="robot_spawn_zone" x="2" y="1" width="1" height="1" />
      <rect id="robot_goal_zone_0" inkscape:label="robot_goal_zone_0" x="8" y="8" width="1" height="1" />
      <path id="robot_route_1_0" inkscape:label="robot_route_1_0" d="M 1 1 L 8 8" />
    </svg>
    """
    svg_path = tmp_path / "map.svg"
    svg_path.write_text(svg.strip(), encoding="utf-8")

    map_def = convert_map(str(svg_path))
    assert map_def is not None
    assert len(map_def.robot_spawn_zones) == 2

    first_zone = map_def.robot_spawn_zones[0]
    second_zone = map_def.robot_spawn_zones[1]
    assert first_zone[0] == (2.0, 1.0)
    assert second_zone[0] == (1.0, 1.0)
    assert map_def.robot_routes[0].spawn_zone == second_zone
