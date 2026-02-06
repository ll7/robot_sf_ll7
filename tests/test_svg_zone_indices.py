"""Tests for numbered SVG spawn/goal zones."""

from __future__ import annotations

from typing import TYPE_CHECKING

from robot_sf.nav.svg_map_parser import SvgMapConverter, convert_map

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


def test_zone_index_helpers_handle_duplicates_and_gaps() -> None:
    """Honor explicit indices, fill gaps with unindexed zones, and keep first duplicate."""
    zone_type = "robot_spawn_zone"
    indexed = {zone_type: {}}
    unindexed = {zone_type: []}
    first_zone = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    second_zone = ((2.0, 0.0), (3.0, 0.0), (3.0, 1.0))
    gap_zone = ((4.0, 0.0), (5.0, 0.0), (5.0, 1.0))

    assert SvgMapConverter._parse_zone_index("robot_spawn_zone_1", None) == (zone_type, 1)
    assert SvgMapConverter._parse_zone_index("robot_spawn_zone", None) == (zone_type, None)
    assert SvgMapConverter._parse_zone_index(None, "robot_goal_zone_2") == ("robot_goal_zone", 2)
    assert SvgMapConverter._parse_zone_index("other", None) == (None, None)

    SvgMapConverter._assign_zone(
        zone_type,
        0,
        first_zone,
        indexed,
        unindexed,
    )
    SvgMapConverter._assign_zone(
        zone_type,
        0,
        second_zone,
        indexed,
        unindexed,
    )
    SvgMapConverter._assign_zone(
        zone_type,
        1,
        second_zone,
        indexed,
        unindexed,
    )
    SvgMapConverter._assign_zone(
        zone_type,
        None,
        gap_zone,
        indexed,
        unindexed,
    )

    assembled = SvgMapConverter._assemble_zones(zone_type, indexed, unindexed)
    assert assembled[0] == first_zone
    assert assembled[1] == second_zone


def test_process_rects_supports_bounds_obstacles_and_fallback_routes(tmp_path: Path) -> None:
    """Ensure rect parsing handles bounds/obstacles and id-based zone labels."""
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg"
         xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
         width="10" height="10">
      <rect id="robot_spawn_zone_0" x="1" y="1" width="1" height="1" />
      <rect id="robot_goal_zone_0" x="8" y="8" width="1" height="1" />
      <rect id="extra_bound" inkscape:label="bound" x="0" y="0" width="10" height="1" />
      <rect id="wall" inkscape:label="obstacle" x="2" y="2" width="1" height="1" />
      <rect id="crowd" inkscape:label="ped_crowded_zone" x="3" y="3" width="1" height="1" />
      <rect id="decor" inkscape:label="decorative" x="4" y="4" width="1" height="1" />
    </svg>
    """
    svg_path = tmp_path / "rects.svg"
    svg_path.write_text(svg.strip(), encoding="utf-8")

    map_def = convert_map(str(svg_path))
    assert map_def is not None
    assert len(map_def.obstacles) == 1
    assert len(map_def.ped_crowded_zones) == 1
    assert len(map_def.robot_spawn_zones) == 1
    assert len(map_def.robot_goal_zones) == 1
    assert map_def.robot_routes  # fallback route generated when none defined


def test_single_pedestrians_and_pois_parsed_from_circles(tmp_path: Path) -> None:
    """Single pedestrian markers and POIs should be parsed from circles."""
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg"
         xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
         width="10" height="10">
      <rect id="robot_spawn_zone_0" x="1" y="1" width="1" height="1" />
      <rect id="robot_goal_zone_0" x="8" y="8" width="1" height="1" />
      <circle id="single_ped_demo_start" inkscape:label="single_ped_demo_start" cx="1" cy="1" r="0.2" />
      <circle id="single_ped_demo_goal" inkscape:label="single_ped_demo_goal" cx="2" cy="2" r="0.2" />
      <circle id="single_ped_static_start" inkscape:label="single_ped_static_start" cx="3" cy="3" r="0.2" />
      <circle id="poi_1" inkscape:label="poi_1" class="poi" cx="4" cy="4" r="0.2" />
    </svg>
    """
    svg_path = tmp_path / "circles.svg"
    svg_path.write_text(svg.strip(), encoding="utf-8")

    map_def = convert_map(str(svg_path))
    assert map_def is not None
    assert len(map_def.single_pedestrians) == 2
    demo = next(ped for ped in map_def.single_pedestrians if ped.id == "demo")
    assert demo.goal == (2.0, 2.0)
    assert map_def.poi_positions == [(4.0, 4.0)]
    assert map_def.poi_labels["poi_1"] == "poi_1"
