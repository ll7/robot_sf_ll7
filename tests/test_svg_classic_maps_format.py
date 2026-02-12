"""Structural validation tests for classical interaction SVG maps.

These tests ensure each new classical interaction SVG map:
  * Parses without exception using SvgMapConverter
  * Produces a MapDefinition object
  * Contains at least one robot spawn zone and goal zone (MapDefinition logs error otherwise)
  * Has width/height > 0
  * Contains obstacles (except explicitly documented minimal cases)
  * Ensures any route labels with indices reference existing spawn/goal zone indices
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from shapely.geometry import LineString, Polygon

from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.navigation import get_prepared_obstacles
from robot_sf.nav.svg_map_parser import SvgMapConverter
from robot_sf.ped_npc.ped_population import PedSpawnConfig, populate_ped_routes

ROOT = Path(__file__).resolve().parent.parent
SVG_DIR = ROOT / "maps" / "svg_maps"

CLASSIC_PREFIX = "classic_"


def _classic_svg_files():
    """TODO docstring. Document this function."""
    return sorted(p for p in SVG_DIR.glob(f"{CLASSIC_PREFIX}*.svg") if p.is_file())


@pytest.mark.parametrize("svg_path", _classic_svg_files())
def test_classic_svg_parse(svg_path: Path):
    """TODO docstring. Document this function.

    Args:
        svg_path: TODO docstring.
    """
    converter = SvgMapConverter(str(svg_path))
    md = converter.get_map_definition()
    assert isinstance(md, MapDefinition), "Converter did not produce MapDefinition"
    assert md.width > 0 and md.height > 0, "Invalid map dimensions"
    # robot spawn & goal zones (parser logs errors; we enforce here)
    assert md.robot_spawn_zones, "No robot spawn zones found"
    assert md.robot_goal_zones, "No robot goal zones found"
    # obstacles check (allow empty only for overtaking narrow lane case? keep required for now)
    assert md.obstacles, "Expected at least one obstacle to define geometry"


@pytest.mark.parametrize("svg_path", _classic_svg_files())
def test_classic_svg_route_indices(svg_path: Path):
    """If routes encode indices (e.g., ped_route_0_1) ensure indices within bounds."""
    converter = SvgMapConverter(str(svg_path))
    md = converter.get_map_definition()
    max_spawn_robot = len(md.robot_spawn_zones) - 1 if md.robot_spawn_zones else -1
    max_goal_robot = len(md.robot_goal_zones) - 1 if md.robot_goal_zones else -1
    max_spawn_ped = len(md.ped_spawn_zones) - 1 if md.ped_spawn_zones else -1
    max_goal_ped = len(md.ped_goal_zones) - 1 if md.ped_goal_zones else -1

    route_label_pattern = re.compile(r"(ped_route|robot_route)_(\d+)_(\d+)")
    # Access protected attributes path_info for labels (test-only introspection)
    for path in converter.path_info:
        if not path.label:
            continue
        m = route_label_pattern.fullmatch(path.label)
        if not m:
            continue
        kind, spawn_str, goal_str = m.groups()
        spawn_i, goal_i = int(spawn_str), int(goal_str)
        if kind == "robot_route":
            assert 0 <= spawn_i <= max_spawn_robot, (
                f"robot_route spawn index {spawn_i} out of range"
            )
            assert 0 <= goal_i <= max_goal_robot, f"robot_route goal index {goal_i} out of range"
        if kind == "ped_route":
            assert 0 <= spawn_i <= max_spawn_ped, f"ped_route spawn index {spawn_i} out of range"
            assert 0 <= goal_i <= max_goal_ped, f"ped_route goal index {goal_i} out of range"


def test_classic_crossing_ped_routes_avoid_obstacle_interior():
    """Crossing map pedestrian routes should not traverse obstacle interiors."""
    converter = SvgMapConverter(str(SVG_DIR / "classic_crossing.svg"))
    md = converter.get_map_definition()
    obstacle_polys = [Polygon(ob.vertices) for ob in md.obstacles]

    for route in md.ped_routes:
        if len(route.waypoints) < 2:
            continue
        route_line = LineString(route.waypoints)
        intersects_interior = any(
            route_line.crosses(obstacle_poly) or route_line.within(obstacle_poly)
            for obstacle_poly in obstacle_polys
        )
        assert not intersects_interior, (
            f"classic_crossing pedestrian route {route.spawn_id}->{route.goal_id} crosses obstacle "
            "interior"
        )


@pytest.mark.parametrize("svg_path", _classic_svg_files())
def test_classic_svg_ped_routes_spawn_nonzero_population(svg_path: Path) -> None:
    """Classic maps with pedestrian routes should produce at least one route pedestrian."""
    converter = SvgMapConverter(str(svg_path))
    md = converter.get_map_definition()
    if not md.ped_routes:
        pytest.skip("Map has no pedestrian routes")

    for route in md.ped_routes:
        assert len(route.waypoints) >= 2, "Pedestrian route must contain at least two waypoints"
        assert route.total_length > 0, "Pedestrian route must have non-zero length"

    spawn_cfg = PedSpawnConfig(
        peds_per_area_m2=0.06,
        max_group_members=3,
        route_spawn_distribution="spread",
        route_spawn_seed=123,
        route_spawn_jitter_frac=0.05,
    )
    ped_states, *_ = populate_ped_routes(
        spawn_cfg,
        md.ped_routes,
        obstacle_polygons=get_prepared_obstacles(md),
    )
    assert ped_states.shape[0] > 0, "Expected non-zero route pedestrian spawn count"
