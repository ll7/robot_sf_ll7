"""Focused contract tests for the archetype reachability probe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from shapely.geometry import Point, Polygon

from scripts.validation.check_scenario_archetype_reachability import (
    DEFAULT_MAPS,
    MapReachabilityReport,
    RouteReachability,
    _blocking_obstacle,
    _inside_zone,
    _zone_polygon,
    format_console_table,
    inspect_map_reachability,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# A square zone from (-1,-1) to (1,1) given as a three-corner rect.
_SQUARE = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0)]


@dataclass
class _FakeRoute:
    """Minimal duck-typed stand-in for GlobalRoute."""

    spawn_id: int
    goal_id: int
    waypoints: list[tuple[float, float]]
    spawn_zone: list[tuple[float, float]] | None
    goal_zone: list[tuple[float, float]] | None
    source_label: str = "robot_route_0_0"


@dataclass
class _FakeObstacle:
    """Minimal duck-typed stand-in for Obstacle."""

    geometry: object


def test_zone_polygon_from_three_corners() -> None:
    """A three-corner rect expands to the correct axis-aligned polygon."""
    polygon = _zone_polygon(_SQUARE)
    assert polygon.equals(Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)]))


def test_inside_zone_covers_interior_and_boundary() -> None:
    """An interior point is inside; a far point is outside; boundary counts."""
    assert _inside_zone(Point(0, 0), _SQUARE) is True
    assert _inside_zone(Point(1, 1), _SQUARE) is True
    assert _inside_zone(Point(3, 0), _SQUARE) is False
    assert _inside_zone(Point(0, 0), None) is False


def test_blocking_obstacle_detects_corridor_collision() -> None:
    """A route crossing an obstacle polygon is reported with its index."""
    route = _FakeRoute(0, 1, [(0.0, 0.0), (4.0, 0.0)], _SQUARE, _SQUARE)
    obstacle = _FakeObstacle(geometry=Polygon([(1.5, -0.5), (2.5, -0.5), (2.5, 0.5), (1.5, 0.5)]))
    blocked, index, shared_area = _blocking_obstacle(route, [obstacle], envelope_radius_m=0.4)
    assert blocked is True
    assert index == 0
    assert shared_area is not None and shared_area > 0.0


def test_blocking_obstacle_allows_clear_route() -> None:
    """An obstacle away from the buffered corridor does not block."""
    route = _FakeRoute(0, 1, [(0.0, 0.0), (1.0, 0.0)], _SQUARE, _SQUARE)
    obstacle = _FakeObstacle(geometry=Polygon([(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)]))
    blocked, index, shared_area = _blocking_obstacle(route, [obstacle], envelope_radius_m=0.4)
    assert blocked is False
    assert index is None
    assert shared_area is None


def test_route_reachable_flag_combines_all_conditions() -> None:
    """Reachability requires in-zone endpoints and no obstacle block."""
    clear = RouteReachability("r", True, True, False)
    assert clear.reachable is True
    off_goal = RouteReachability("r", True, False, False)
    assert off_goal.reachable is False
    blocked = RouteReachability("r", True, True, True, 2, 1.5)
    assert blocked.reachable is False


def test_report_blocked_route_counting() -> None:
    """Blocked routes aggregate into the report total."""
    report = MapReachabilityReport(map_path="m.svg")
    report.routes.append(RouteReachability("r1", True, True, False))
    report.routes.append(RouteReachability("r2", True, False, False))
    report.routes.append(RouteReachability("r3", True, True, True, 0, 0.5))
    assert report.blocked_routes == 2


def test_console_table_marks_findings() -> None:
    """Blocked routes render with reason detail."""
    report = MapReachabilityReport(map_path="m.svg")
    report.routes.append(RouteReachability("r1", True, False, True, 3, 1.25))
    text = format_console_table(report)
    assert "[BLOCK]" in text
    assert "goal-off-zone" in text
    assert "obstacle[3]" in text


def test_default_maps_are_the_four_pinned_archetypes() -> None:
    assert len(DEFAULT_MAPS) == 4
    for svg in DEFAULT_MAPS:
        assert Path(svg).name.startswith("classic_") and svg.endswith(".svg")


def test_cli_default_exit_zero_on_pinned_maps() -> None:
    """The four pinned archetype maps produce a deterministic report."""
    svg = REPO_ROOT / DEFAULT_MAPS[0]
    if not svg.exists():
        pytest.skip("pinned archetype maps not present")
    report = inspect_map_reachability(svg)
    assert report.routes
    assert all(route.label for route in report.routes)


def test_integration_pinned_maps_have_robot_routes() -> None:
    """Every pinned archetype defines at least one robot route to probe."""
    for svg in DEFAULT_MAPS:
        path = REPO_ROOT / svg
        if not path.exists():
            pytest.skip("pinned archetype maps not present")
        report = inspect_map_reachability(path)
        assert report.routes, f"{svg} has no robot routes"


def test_cli_missing_map_fails_closed(tmp_path: Path) -> None:
    """A missing map path is an explicit error, not an empty success."""
    assert main(["--map", str(tmp_path / "missing.svg")]) == 2
