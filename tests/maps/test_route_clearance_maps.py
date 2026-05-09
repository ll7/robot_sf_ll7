"""Regression checks for benchmark maps with repaired route-clearance geometry."""

from __future__ import annotations

from pathlib import Path

from shapely.geometry import LineString, Polygon

from robot_sf.nav.svg_map_parser import convert_map

REPO_ROOT = Path(__file__).resolve().parents[2]


def _obstacle_polygons(map_path: Path) -> list[Polygon]:
    """Return obstacle polygons parsed from an SVG map.

    Returns:
        Non-empty obstacle polygons.
    """
    map_def = convert_map(str(map_path))
    polygons: list[Polygon] = []
    for obstacle in map_def.obstacles:
        vertices = getattr(obstacle, "vertices", None)
        if vertices:
            polygon = Polygon(vertices)
            if polygon.is_valid and not polygon.is_empty:
                polygons.append(polygon)
    return polygons


def _route_lines(map_path: Path, *, kind: str) -> list[LineString]:
    """Return parsed route centerlines for one route kind.

    Returns:
        Non-empty route lines.
    """
    map_def = convert_map(str(map_path))
    routes = map_def.robot_routes if kind == "robot" else map_def.ped_routes
    return [LineString(route.waypoints) for route in routes if len(route.waypoints) >= 2]


def test_repaired_maps_do_not_route_through_obstacle_interiors() -> None:
    """Classic merging and station-platform routes should avoid obstacle interiors."""
    for rel_path in (
        Path("maps/svg_maps/classic_merging.svg"),
        Path("maps/svg_maps/classic_station_platform.svg"),
    ):
        map_path = REPO_ROOT / rel_path
        obstacles = _obstacle_polygons(map_path)
        assert obstacles, f"{rel_path} should parse obstacle geometry"

        for kind in ("robot", "ped"):
            for line in _route_lines(map_path, kind=kind):
                assert not any(line.crosses(obstacle) for obstacle in obstacles), (
                    f"{rel_path} {kind} route crosses obstacle interior"
                )


def test_repaired_robot_routes_keep_nonnegative_footprint_margin() -> None:
    """Affected robot routes should no longer report negative route-clearance margin."""
    expected_min_margin_m = {
        "classic_merging.svg": 0.5,
        "classic_station_platform.svg": 2.0,
    }

    for filename, min_expected_margin in expected_min_margin_m.items():
        map_path = REPO_ROOT / "maps" / "svg_maps" / filename
        obstacles = _obstacle_polygons(map_path)
        route_margins = []
        for line in _route_lines(map_path, kind="robot"):
            min_center_distance = min(line.distance(obstacle) for obstacle in obstacles)
            route_margins.append(min_center_distance - 1.0)

        assert route_margins, f"{filename} should have at least one robot route"
        assert min(route_margins) >= min_expected_margin
