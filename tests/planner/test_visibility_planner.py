"""Direct regression coverage for the visibility-graph global planner geometry contracts.

These tests lock five behaviours of ``robot_sf/planner/visibility_planner.py`` using
tiny in-memory :class:`MapDefinition` / Shapely geometry only:

* consecutive-point de-duplication (``_dedup_consecutive``) including the tolerance boundary,
* collinear-point pruning (``_prune_collinear``) including the tolerance boundary,
* an unobstructed start-to-goal route (direct visibility),
* an obstacle detour whose returned segments stay in allowed free space, and
* no-path / unreachable geometry raising the module's typed :class:`PlanningFailedError`.

No repository map, simulation, environment, benchmark, or map asset is loaded or started.
This is regression coverage only; it makes no optimality, performance, or safety claim.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING

import pytest
from shapely.geometry import LineString, Point, Polygon

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.planner.visibility_graph import VisibilityGraph
from robot_sf.planner.visibility_planner import (
    PlannerConfig,
    PlanningFailedError,
    VisibilityPlanner,
    _dedup_consecutive,
    _prune_collinear,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# A tiny reusable spawn/goal zone and route so MapDefinition can be built purely in memory.
_SPAWN_ZONE = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0))
_GOAL_ZONE = ((8.0, 8.0), (9.0, 8.0), (8.0, 9.0))


def _tiny_map(width: float, height: float, obstacles: Sequence[Obstacle]) -> MapDefinition:
    """Build a minimal valid in-memory MapDefinition without loading any map asset."""
    bounds = [
        (0.0, width, 0.0, 0.0),
        (0.0, width, height, height),
        (0.0, 0.0, 0.0, height),
        (width, width, 0.0, height),
    ]
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.5, 1.5), (max(1.5, width - 1.5), max(1.5, height - 1.5))],
        spawn_zone=_SPAWN_ZONE,
        goal_zone=_GOAL_ZONE,
    )
    return MapDefinition(
        width=width,
        height=height,
        obstacles=list(obstacles),
        robot_spawn_zones=[_SPAWN_ZONE],
        ped_spawn_zones=[_SPAWN_ZONE],
        robot_goal_zones=[_GOAL_ZONE],
        bounds=bounds,
        robot_routes=[route],
        ped_goal_zones=[_GOAL_ZONE],
        ped_crowded_zones=[],
        ped_routes=[route],
        single_pedestrians=[],
    )


def _collision_envelopes(map_def: MapDefinition, config: PlannerConfig) -> list[Polygon]:
    """Return obstacles inflated by the robot radius only (the collision envelope)."""
    return [Polygon(o.vertices).buffer(config.robot_radius) for o in map_def.obstacles]


def _path_stays_in_free_space(path: list[tuple[float, float]], envelopes: list[Polygon]) -> bool:
    """Return True when no waypoint or segment enters or touches a collision envelope."""
    for waypoint in path:
        pt = Point(waypoint)
        for poly in envelopes:
            if poly.contains(pt) or poly.touches(pt):
                return False
    for start_pt, end_pt in pairwise(path):
        segment = LineString([start_pt, end_pt])
        for poly in envelopes:
            if poly.crosses(segment) or poly.contains(segment) or poly.touches(segment):
                return False
    return True


@pytest.fixture(autouse=True)
def _isolate_visibility_graph_cache() -> None:
    """Clear the shared visibility-graph cache around every test for deterministic results."""
    VisibilityGraph.clear_cache()
    yield
    VisibilityGraph.clear_cache()


# --------------------------------------------------------------------------- #
# Geometry helper: consecutive-point de-duplication
# --------------------------------------------------------------------------- #


def test_dedup_consecutive_removes_exact_duplicates() -> None:
    """Exact consecutive duplicates collapse to a single point."""
    assert _dedup_consecutive([(0.0, 0.0), (0.0, 0.0), (1.0, 1.0), (1.0, 1.0)]) == [
        (0.0, 0.0),
        (1.0, 1.0),
    ]


def test_dedup_consecutive_preserves_non_consecutive_duplicates() -> None:
    """A repeated point that is not consecutive with its twin is preserved (closed loop)."""
    assert _dedup_consecutive([(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)]) == [
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
    ]


def test_dedup_consecutive_empty_and_single() -> None:
    """Empty input returns empty; a lone point is returned unchanged."""
    assert _dedup_consecutive([]) == []
    assert _dedup_consecutive([(5.0, 7.0)]) == [(5.0, 7.0)]


def test_dedup_consecutive_tolerance_boundary() -> None:
    """Points exactly ``tol`` apart are treated as duplicates; just over ``tol`` are kept.

    The helper keeps a point only when ``abs(delta) > tol`` is strictly true, so the boundary
    value itself collapses and any strictly larger separation survives.
    """
    tol = 1.0
    # Exactly at the tolerance -> treated as a duplicate and dropped.
    assert _dedup_consecutive([(0.0, 0.0), (tol, 0.0)], tol=tol) == [(0.0, 0.0)]
    # Strictly above the tolerance -> kept.
    assert _dedup_consecutive([(0.0, 0.0), (tol + 1e-4, 0.0)], tol=tol) == [
        (0.0, 0.0),
        (tol + 1e-4, 0.0),
    ]
    # Mixed run: the second point sits at the boundary (dropped), the third is well beyond (kept).
    assert _dedup_consecutive([(0.0, 0.0), (tol, 0.0), (2 * tol, 0.0)], tol=tol) == [
        (0.0, 0.0),
        (2 * tol, 0.0),
    ]


# --------------------------------------------------------------------------- #
# Geometry helper: collinear-point pruning
# --------------------------------------------------------------------------- #


def test_prune_collinear_removes_collinear_middle_point() -> None:
    """A perfectly collinear intermediate point is pruned."""
    assert _prune_collinear([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]) == [(0.0, 0.0), (2.0, 2.0)]


def test_prune_collinear_keeps_non_collinear_corner() -> None:
    """A real corner produces a non-zero cross product and is retained."""
    assert _prune_collinear([(0.0, 0.0), (1.0, 2.0), (2.0, 0.0)]) == [
        (0.0, 0.0),
        (1.0, 2.0),
        (2.0, 0.0),
    ]


def test_prune_collinear_leaves_short_paths_unchanged() -> None:
    """Paths with fewer than three points are returned untouched."""
    assert _prune_collinear([(0.0, 0.0)]) == [(0.0, 0.0)]
    assert _prune_collinear([(0.0, 0.0), (1.0, 1.0)]) == [(0.0, 0.0), (1.0, 1.0)]


def test_prune_collinear_tolerance_boundary() -> None:
    """A cross product exactly at ``tol`` is pruned; strictly above ``tol`` is kept.

    For points A=(0,0), B=(1,0), C=(2,h) the cross product equals ``h``, so the boundary is
    exercised directly by tuning ``h`` against ``tol``.
    """
    tol = 1.0
    # Cross product == tol -> not strictly greater -> pruned.
    assert _prune_collinear([(0.0, 0.0), (1.0, 0.0), (2.0, tol)], tol=tol) == [
        (0.0, 0.0),
        (2.0, tol),
    ]
    # Cross product > tol -> kept.
    assert _prune_collinear([(0.0, 0.0), (1.0, 0.0), (2.0, 2 * tol)], tol=tol) == [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 2 * tol),
    ]


# --------------------------------------------------------------------------- #
# Planner: direct visibility (unobstructed route)
# --------------------------------------------------------------------------- #


def test_empty_map_returns_direct_unobstructed_path() -> None:
    """With no obstacles the planner returns the straight start-to-goal route."""
    map_def = _tiny_map(width=30.0, height=12.0, obstacles=[])
    planner = VisibilityPlanner(map_def)

    start = (2.0, 6.0)
    goal = (28.0, 6.0)
    path = planner.plan(start, goal)

    assert path == [start, goal]


def test_direct_line_of_sight_with_offside_obstacle() -> None:
    """An obstacle off the start-goal line keeps the direct route and stays in free space."""
    obstacle = Obstacle([(30.0, 15.0), (34.0, 15.0), (34.0, 19.0), (30.0, 19.0)])
    map_def = _tiny_map(width=40.0, height=20.0, obstacles=[obstacle])
    config = PlannerConfig()
    planner = VisibilityPlanner(map_def, config=config)

    start = (2.0, 10.0)
    goal = (38.0, 10.0)
    path = planner.plan(start, goal)

    assert path[0] == start
    assert path[-1] == goal
    assert _path_stays_in_free_space(path, _collision_envelopes(map_def, config))


# --------------------------------------------------------------------------- #
# Planner: obstacle detour staying in allowed free space
# --------------------------------------------------------------------------- #


def test_obstacle_detour_stays_in_free_space() -> None:
    """A wall anchored to the boundary with one gap forces a detour through free space.

    The wall spans from y=6 to the top boundary, so the direct y=10 line is blocked and the
    planner must thread the bottom gap. The returned segments must stay in allowed free space
    (outside the robot-radius collision envelope) and must be longer than the blocked straight
    line, confirming an actual detour rather than an unsafe shortcut.
    """
    wall = Obstacle([(18.0, 6.0), (20.0, 6.0), (20.0, 20.0), (18.0, 20.0)])
    map_def = _tiny_map(width=40.0, height=20.0, obstacles=[wall])
    config = PlannerConfig()
    planner = VisibilityPlanner(map_def, config=config)

    start = (2.0, 10.0)
    goal = (38.0, 10.0)
    path = planner.plan(start, goal)

    # Endpoints preserved and the route is not the blocked two-point straight line.
    assert path[0] == start
    assert path[-1] == goal
    assert len(path) > 2

    # Every waypoint and segment stays outside the robot-radius collision envelope.
    assert _path_stays_in_free_space(path, _collision_envelopes(map_def, config))

    # The route is genuinely longer than the blocked straight line (a real detour).
    direct_distance = math.dist(start, goal)
    path_length = sum(math.dist(a, b) for a, b in pairwise(path))
    assert path_length > direct_distance + 1e-6

    # The detour must reach the far side of the wall.
    assert path[-1][0] > wall.vertices[2][0]


# --------------------------------------------------------------------------- #
# Planner: no-path failure raises the typed PlanningFailedError
# --------------------------------------------------------------------------- #


def test_no_path_raises_planning_failed_error() -> None:
    """A full-height sealed wall leaves no collision-free route, so the planner fails closed.

    With ``fallback_on_failure`` at its default (False) the planner must raise the module's typed
    :class:`PlanningFailedError` instead of silently returning an unsafe path that crosses the
    sealed wall or leaves the map.
    """
    sealed_wall = Obstacle([(9.0, 0.0), (11.0, 0.0), (11.0, 12.0), (9.0, 12.0)])
    map_def = _tiny_map(width=20.0, height=12.0, obstacles=[sealed_wall])
    config = PlannerConfig()
    assert config.fallback_on_failure is False
    planner = VisibilityPlanner(map_def, config=config)

    with pytest.raises(PlanningFailedError):
        planner.plan((2.0, 6.0), (18.0, 6.0))
