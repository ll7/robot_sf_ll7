"""Regression checks for benchmark maps with repaired route-clearance geometry."""

from __future__ import annotations

from pathlib import Path

from shapely.geometry import LineString, Polygon

from robot_sf.nav.svg_map_parser import convert_map

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_RADIUS_M = 1.0


def _resolve_map_path(map_path: Path) -> Path:
    """Resolve map paths relative to the repository root used by this test module.

    Returns:
        Absolute repository-local map path.
    """

    assert map_path, "map_path must be provided"
    return map_path if map_path.is_absolute() else (REPO_ROOT / map_path).resolve()


def _obstacle_polygons(map_path: Path) -> list[Polygon]:
    """Return obstacle polygons parsed from an SVG map.

    Returns:
        Non-empty obstacle polygons.
    """
    resolved_map_path = _resolve_map_path(map_path)
    map_def = convert_map(str(resolved_map_path))
    assert map_def is not None, f"Failed to convert map: {resolved_map_path}"
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
    resolved_map_path = _resolve_map_path(map_path)
    map_def = convert_map(str(resolved_map_path))
    assert map_def is not None, f"Failed to convert map: {resolved_map_path}"
    routes = map_def.robot_routes if kind == "robot" else map_def.ped_routes
    return [LineString(route.waypoints) for route in routes if len(route.waypoints) >= 2]


def test_repaired_maps_do_not_route_through_obstacle_interiors() -> None:
    """Classic merging and station-platform routes should avoid obstacle interiors."""
    for rel_path in (
        Path("maps/svg_maps/classic_merging.svg"),
        Path("maps/svg_maps/classic_station_platform.svg"),
    ):
        map_path = rel_path
        obstacles = _obstacle_polygons(map_path)
        assert obstacles, f"{rel_path} should parse obstacle geometry"

        for kind in ("robot", "ped"):
            for line in _route_lines(map_path, kind=kind):
                assert not any(
                    line.crosses(obstacle) or line.within(obstacle) for obstacle in obstacles
                ), f"{rel_path} {kind} route intersects an obstacle interior"


def test_repaired_robot_routes_keep_nonnegative_footprint_margin() -> None:
    """Affected robot routes should no longer report negative route-clearance margin."""
    expected_min_margin_m = {
        "classic_merging.svg": 0.5,
        "classic_station_platform.svg": 2.0,
    }

    for filename, min_expected_margin in expected_min_margin_m.items():
        map_path = Path("maps/svg_maps") / filename
        obstacles = _obstacle_polygons(map_path)
        assert obstacles, f"{filename} should parse obstacle geometry"
        route_margins = []
        for line in _route_lines(map_path, kind="robot"):
            min_center_distance = min(line.distance(obstacle) for obstacle in obstacles)
            route_margins.append(min_center_distance - ROBOT_RADIUS_M)

        assert route_margins, f"{filename} should have at least one robot route"
        assert min(route_margins) >= min_expected_margin
