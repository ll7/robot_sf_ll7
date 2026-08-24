"""Focused contract tests for the archetype reachability probe.

The probe models the simulator's runtime route
(``[spawn_sample, *waypoints, goal_sample]``) so it must distinguish
interior-corridor blockers, spawn/goal connector blockers, and insufficient
proof from the accepted anchor-offset observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from scripts.validation.check_scenario_archetype_reachability import (
    DEFAULT_MAPS,
    MapReachabilityReport,
    RouteReachability,
    ZoneSample,
    _anchor_inside_zone,
    _corridor_blocked,
    _point_in_obstacles,
    _segment_connectors_blocked,
    _zone_range_points,
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


def _obstacle(*coords: list[tuple[float, float]]) -> list[Polygon]:
    """Build obstacle polygon(s) from coordinate rings."""
    return [Polygon(ring) for ring in coords]


def test_anchor_inside_zone_covers_interior_and_boundary() -> None:
    """An interior point is inside; a far point is outside; boundary counts."""
    assert _anchor_inside_zone(Point(0, 0), _SQUARE) is True
    assert _anchor_inside_zone(Point(1, 1), _SQUARE) is True
    assert _anchor_inside_zone(Point(3, 0), _SQUARE) is False
    assert _anchor_inside_zone(Point(0, 0), None) is False


def test_point_in_obstacles_rejects_inside_points() -> None:
    """Representative points inside an obstacle are not admissible spawns."""
    obstacles = _obstacle([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    assert _point_in_obstacles(Point(0, 0), obstacles) is True
    assert _point_in_obstacles(Point(2, 2), obstacles) is False


def test_interior_corridor_blocked_detects_collision() -> None:
    """A route crossing an obstacle polygon is reported with its index."""
    polyline = LineString([(0.0, 0.0), (4.0, 0.0)])
    obstacles = _obstacle([(1.5, -0.5), (2.5, -0.5), (2.5, 0.5), (1.5, 0.5)])
    blocked, index, shared_area = _corridor_blocked(polyline, 0.4, obstacles)
    assert blocked is True
    assert index == 0
    assert shared_area is not None and shared_area > 0.0


def test_interior_corridor_allows_clear_route() -> None:
    """An obstacle away from the buffered corridor does not block."""
    polyline = LineString([(0.0, 0.0), (1.0, 0.0)])
    obstacles = _obstacle([(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)])
    blocked, index, shared_area = _corridor_blocked(polyline, 0.4, obstacles)
    assert blocked is False
    assert index is None
    assert shared_area is None


def test_spawn_connector_blocked_detects_obstruction() -> None:
    """An obstacle across the spawn connector surfaces as blocked."""
    anchor = Point(4.0, 0.0)
    zone_points = [
        ZoneSample(x=0.0, y=0.0),
        ZoneSample(x=0.6, y=0.2),
        ZoneSample(x=-0.3, y=0.5),
    ]
    obstacles = _obstacle([(1.5, -0.5), (2.5, -0.5), (2.5, 0.5), (1.5, 0.5)])
    blocked, index, _area = _segment_connectors_blocked(anchor, zone_points, obstacles, 0.4)
    assert blocked is True
    assert index == 0


def test_goal_connector_blocked_detects_obstruction() -> None:
    """An obstacle across the goal connector surfaces as blocked."""
    anchor = Point(-4.0, 0.0)
    zone_points = [
        ZoneSample(x=0.0, y=0.0),
        ZoneSample(x=0.6, y=0.2),
        ZoneSample(x=-0.3, y=0.5),
    ]
    obstacles = _obstacle([(-2.5, -0.5), (-1.5, -0.5), (-1.5, 0.5), (-2.5, 0.5)])
    blocked, _index, _area = _segment_connectors_blocked(anchor, zone_points, obstacles, 0.4)
    assert blocked is True


def test_connector_free_when_no_obstacle() -> None:
    """No obstacle across the connector leaves every representative free."""
    anchor = Point(4.0, 0.0)
    zone_points = [ZoneSample(x=0.0, y=0.0), ZoneSample(x=0.6, y=0.2)]
    obstacles = _obstacle([(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)])
    blocked, _index, _area = _segment_connectors_blocked(anchor, zone_points, obstacles, 0.4)
    assert blocked is False


def test_connector_insufficient_proof_when_all_points_in_obstacle() -> None:
    """No admissible zone point yields ``None`` (insufficient proof)."""
    anchor = Point(4.0, 0.0)
    large_obstacle = _obstacle([(-3, -3), (3, -3), (3, 3), (-3, 3)])
    zone_points = [ZoneSample(x=0.0, y=0.0), ZoneSample(x=0.5, y=0.5)]
    blocked, _index, _area = _segment_connectors_blocked(anchor, zone_points, large_obstacle, 0.4)
    assert blocked is None


def test_route_reachable_flag_combines_all_conditions() -> None:
    """Reachability requires no blocker on any segment and no proof gap."""
    clear = RouteReachability(
        label="r",
        anchor_offset=True,  # accepted observation does not block
        interior_obstacle_blocks=False,
        spawn_connector_blocks=False,
        goal_connector_blocks=False,
    )
    assert clear.reachable is True
    interior = RouteReachability("r", False, True, False, False)
    assert interior.reachable is False
    spawn = RouteReachability(
        "r",
        False,
        False,
        True,
        False,
        blocking_obstacle_index=0,
        blocking_obstacle_shared_area_m2=1.5,
        blocked_class="spawn_connector_blocked",
    )
    assert spawn.reachable is False
    proof = RouteReachability(
        "r",
        False,
        False,
        False,
        False,
        insufficient_proof=True,
        blocked_class="insufficient_proof",
    )
    assert proof.reachable is False


def test_report_blocked_route_counting() -> None:
    """Blocked routes aggregate into the report total."""
    report = MapReachabilityReport(map_path="m.svg")
    report.routes.append(RouteReachability("r1", True, False, False, False))
    report.routes.append(RouteReachability("r2", False, False, True, False))
    report.routes.append(RouteReachability("r3", False, True, False, False))
    assert report.blocked_routes == 2


def test_console_table_marks_findings() -> None:
    """Blocked routes render with class detail and obstacle evidence."""
    report = MapReachabilityReport(map_path="m.svg")
    report.routes.append(
        RouteReachability(
            "r1",
            True,
            False,
            False,
            True,
            blocking_obstacle_index=3,
            blocking_obstacle_shared_area_m2=1.25,
            blocked_class="goal_connector_blocked",
        )
    )
    text = format_console_table(report)
    assert "[BLOCK]" in text
    assert "goal_connector_blocked" in text
    assert "obstacle[3]" in text
    assert "anchor-offset(observed)" in text


def test_zone_range_points_are_deterministic_and_bounded() -> None:
    """Zone representatives are fixed for a given seed and stay inside."""
    points = _zone_range_points(_SQUARE, 5, np.random.default_rng(20260824))
    points2 = _zone_range_points(_SQUARE, 5, np.random.default_rng(20260824))
    assert [(p.x, p.y) for p in points] == [(p.x, p.y) for p in points2]
    assert len(points) == 5
    for sample in points:
        assert _anchor_inside_zone(Point(sample.x, sample.y), _SQUARE) is True


def test_default_maps_are_the_four_pinned_archetypes() -> None:
    assert len(DEFAULT_MAPS) == 4
    for svg in DEFAULT_MAPS:
        assert Path(svg).name.startswith("classic_") and svg.endswith(".svg")


def test_pinned_maps_not_blocked_solely_by_anchor_offsets() -> None:
    """The accepted #7709 anchor offsets must not produce BLOCK verdicts.

    Regression for the runtime-faithful repair: the four pinned archetype robot
    routes are off-zone by design (documented offsets), but the runtime executes
    sampled zone positions plus connectors, so the corrected probe must report
    them reachable with only an anchor-offset observation.
    """
    for svg in DEFAULT_MAPS:
        path = REPO_ROOT / svg
        if not path.exists():
            pytest.skip("pinned archetype maps not present")
        report = inspect_map_reachability(path)
        assert report.routes, f"{svg} has no robot routes"
        for route in report.routes:
            assert route.reachable, f"{svg} {route.label} blocked: {route.blocked_class}"


def test_route_with_broken_spawn_connector_is_blocked() -> None:
    """A deliberately broken spawn connector yields a blocked verdict."""
    from scripts.validation.check_scenario_archetype_reachability import (
        DEFAULT_ZONE_SAMPLES,
        _segment_connectors_blocked,
    )

    anchor = Point(4.0, 0.0)
    zone_points = _zone_range_points(_SQUARE, DEFAULT_ZONE_SAMPLES, np.random.default_rng(1))
    # Wall across every connector from the zone to the anchor.
    obstacles = _obstacle([(1.5, -2.0), (2.5, -2.0), (2.5, 2.0), (1.5, 2.0)])
    blocked, index, _area = _segment_connectors_blocked(anchor, zone_points, obstacles, 0.4)
    assert blocked is True
    assert index == 0


def test_cli_default_exit_zero_on_pinned_maps() -> None:
    """The four pinned archetype maps produce a deterministic report."""
    svg = REPO_ROOT / DEFAULT_MAPS[0]
    if not svg.exists():
        pytest.skip("pinned archetype maps not present")
    report = inspect_map_reachability(svg)
    assert report.routes
    assert all(route.label for route in report.routes)


def test_cli_missing_map_fails_closed(tmp_path: Path) -> None:
    """A missing map path is an explicit error, not an empty success."""
    assert main(["--map", str(tmp_path / "missing.svg")]) == 2


def test_fake_route_duck_type_shape() -> None:
    """The fake route carries the attributes the probe reads."""
    route = _FakeRoute(0, 1, [(0.0, 0.0), (2.0, 0.0)], _SQUARE, _SQUARE)
    assert route.spawn_zone == _SQUARE
    assert route.goal_zone == _SQUARE
    assert _anchor_inside_zone(Point(*route.waypoints[0]), route.spawn_zone) is True


def test_fake_obstacle_duck_type_shape() -> None:
    """The fake obstacle carries the geometry the probe reads."""
    obstacle = _FakeObstacle(geometry=Polygon([(1.0, 1.0), (2.0, 1.0), (2.0, 2.0)]))
    assert obstacle.geometry is not None
    assert isinstance(obstacle.geometry, Polygon)
