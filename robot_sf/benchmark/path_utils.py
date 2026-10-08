"""Helpers for map-aware shortest path calculations used in metrics."""

from __future__ import annotations

import heapq
from functools import lru_cache
from itertools import pairwise
from math import dist
from typing import TYPE_CHECKING

import numpy as np
from shapely import from_wkb
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from robot_sf.benchmark.footprint_metrics import require_radius
from robot_sf.planner.classic_global_planner import (
    ClassicGlobalPlanner,
    ClassicPlannerConfig,
    PlanningError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from robot_sf.nav.map_config import MapDefinition


def _path_length(points: Iterable[tuple[float, float]]) -> float:
    """Return total polyline length, or NaN when no segment exists."""
    pts = list(points)
    if len(pts) < 2:
        return float("nan")
    return float(sum(dist(a, b) for a, b in pairwise(pts)))


def compute_shortest_path_length(
    map_def: MapDefinition | None,
    start: np.ndarray,
    goal: np.ndarray,
) -> float:
    """Return shortest path length using the classic Theta* global planner.

    Returns NaN when map definition is missing or planning fails.
    """
    if map_def is None:
        return float("nan")
    if not np.isfinite(start).all() or not np.isfinite(goal).all():
        return float("nan")
    planner = ClassicGlobalPlanner(map_def, config=ClassicPlannerConfig())
    try:
        waypoints, _info = planner.plan(
            (float(start[0]), float(start[1])), (float(goal[0]), float(goal[1]))
        )
    except (PlanningError, ValueError):
        return float("nan")
    return _path_length(waypoints)


def remaining_route_length(positions: np.ndarray, route: np.ndarray) -> np.ndarray:
    """Project positions onto a frozen route polyline and return remaining arclength.

    Projection uses the closest segment, with the earliest segment breaking ties.
    It is not cumulative-max progress: retreat and a real stop must remain visible.

    Returns:
        Remaining metres for each projected position.
    """
    route = np.asarray(route, dtype=float)
    segments = np.diff(route, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    keep = lengths > 1e-12
    starts, segments, lengths = route[:-1][keep], segments[keep], lengths[keep]
    if not len(lengths):
        return np.zeros(len(positions), dtype=float)
    offsets = np.concatenate([[0.0], np.cumsum(lengths)])
    relative = np.asarray(positions, dtype=float)[:, None, :] - starts
    fractions = np.clip(np.sum(relative * segments, axis=2) / lengths**2, 0, 1)
    projections = starts + fractions[:, :, None] * segments
    nearest = np.argmin(np.sum((positions[:, None, :] - projections) ** 2, axis=2), axis=1)
    return offsets[-1] - (
        offsets[nearest] + fractions[np.arange(len(positions)), nearest] * lengths[nearest]
    )


def _polygon_parts(geometry):
    """Yield polygon components, preserving holes and compound obstacle geometry."""
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _polygon_parts(part)


@lru_cache(maxsize=512)
def _zone_reference_cached(
    scenario_id: str,
    seed: int | None,
    start: tuple[float, float],
    zone_wkb: bytes,
    obstacles_wkb: bytes,
    bounds: tuple[float, float, float, float],
    clip_obstacles: bool = True,
    physical_contract: tuple[bytes, float] | None = None,
) -> float:
    """Exact polygonal robot-centre geodesic to a goal set, cached by reset identity.

    A shortest polygonal path bends only at obstacle vertices. Its last segment
    terminates at a goal vertex or a perpendicular projection onto a goal edge.
    Dijkstra searches these visible candidates, including obstacle/goal boundary
    intersections. Boundaries are admissible; obstacle interiors are not.

    Returns:
        Shortest distance in metres, or NaN if the goal set is unreachable.
    """
    domain = box(*bounds)
    obstacles = from_wkb(obstacles_wkb)
    if clip_obstacles:
        obstacles = obstacles.intersection(domain)
    zone = from_wkb(zone_wkb).intersection(domain).difference(obstacles)
    start_point = Point(start)
    physical = from_wkb(physical_contract[0]) if physical_contract is not None else None
    robot_radius = physical_contract[1] if physical_contract is not None else 0.0
    blocked_start = (
        start_point.distance(physical) < robot_radius - 1e-10
        if physical is not None and not physical.is_empty
        else obstacles.contains(start_point)
    )
    if zone.is_empty or not domain.covers(start_point) or blocked_start:
        return float("nan")
    if zone.covers(start_point):
        return 0.0
    obstacle_parts = list(_polygon_parts(obstacles))
    goal_edges = []
    vertices = [start]
    if zone.geom_type == "Point":
        vertices.append(tuple(zone.coords[0]))
    for poly in [*obstacle_parts, *_polygon_parts(zone)]:
        for ring in [poly.exterior, *poly.interiors]:
            vertices.extend(tuple(p) for p in list(ring.coords)[:-1])
    for poly in _polygon_parts(zone):
        for ring in [poly.exterior, *poly.interiors]:
            goal_edges.extend(LineString([a, b]) for a, b in pairwise(ring.coords))
    vertices = list(dict.fromkeys(vertices))
    return _shortest_visible_distance(
        vertices,
        goal_edges,
        zone,
        obstacle_parts,
        domain,
        physical=physical,
        robot_radius=robot_radius,
    )


def _segment_is_clear(line, domain, obstacle_parts, physical, robot_radius) -> bool:
    """Test a line against exact disc clearance, or the historical point geometry.

    Returns:
        Whether the line stays in the centre domain and outside solid geometry.
    """
    if not domain.covers(line):
        return False
    if physical is not None:
        return physical.is_empty or line.distance(physical) >= robot_radius - 1e-10
    return not any(line.relate_pattern(poly, "T********") for poly in obstacle_parts)


def _shortest_visible_distance(
    vertices, goal_edges, zone, obstacle_parts, domain, *, physical=None, robot_radius=0.0
) -> float:
    """Search the continuous visibility graph, with the first vertex as the reset.

    Returns:
        The minimum visible distance to any admissible goal edge, or NaN.
    """

    def visible(a, b) -> bool:
        if a == b:
            return True
        line = LineString([a, b])
        return _segment_is_clear(line, domain, obstacle_parts, physical, robot_radius)

    distances = [float("inf")] * len(vertices)
    distances[0] = 0.0
    pending = [(0.0, 0)]
    best = float("inf")
    while pending:
        cost, i = heapq.heappop(pending)
        if cost != distances[i] or cost >= best:
            continue
        point = vertices[i]
        if zone.covers(Point(point)):
            best = min(best, cost)
        for edge in goal_edges:
            target = tuple(edge.interpolate(edge.project(Point(point))).coords[0])
            if visible(point, target):
                best = min(best, cost + dist(point, target))
        for j, target in enumerate(vertices):
            next_cost = cost + dist(point, target)
            if next_cost < min(distances[j], best) and visible(point, target):
                distances[j] = next_cost
                heapq.heappush(pending, (next_cost, j))
    return best if np.isfinite(best) else float("nan")


def compute_completion_reference_length(
    map_def: MapDefinition | None,
    start: np.ndarray,
    goal: np.ndarray,
    *,
    completion_policy: str = "waypoint_radius_v1",
    goal_zone: np.ndarray | None = None,
    scenario_id: str = "",
    seed: int | None = None,
    robot_radius: float = 0.0,
) -> float:
    """Return the completion-policy reference shared by efficiency and ideal time.

    Zone entry uses a continuous shortest path to the frozen polygon; waypoint
    radius retains the historical Theta* point reference. Geometry joins the cache
    key so a reused scenario/seed with changed map or reset cannot reuse old values.
    """
    robot_radius = require_radius(robot_radius)
    if completion_policy != "goal_zone_entry_v1" and robot_radius == 0:
        return compute_shortest_path_length(map_def, start, goal)
    if map_def is None or not np.isfinite(start).all():
        return float("nan")
    if completion_policy == "goal_zone_entry_v1" and goal_zone is None:
        return float("nan")
    corners = np.asarray(
        goal_zone if completion_policy == "goal_zone_entry_v1" else [goal], dtype=float
    )
    if not np.isfinite(corners).all():
        return float("nan")
    if len(corners) == 3:
        corners = np.vstack([corners, corners[0] + corners[2] - corners[1]])
    zone = Polygon(corners) if len(corners) > 1 else Point(corners[0])
    if not zone.is_valid or zone.is_empty:
        return float("nan")
    if robot_radius * 2 >= min(map_def.width, map_def.height):
        return float("nan")
    polygons = [poly for obstacle in map_def.obstacles for poly in obstacle.iter_polygons()]
    obstacles = unary_union(polygons)
    physical_wkb = obstacles.wkb if robot_radius else None
    if robot_radius:
        # Circumscribed 32-edge circles keep chord approximation outside the disc.
        obstacles = obstacles.buffer(robot_radius / np.cos(np.pi / 32), quad_segs=8)
    xs = [x for line in map_def.bounds for x in line[:2]]
    ys = [y for line in map_def.bounds for y in line[2:]]
    return _zone_reference_cached(
        str(scenario_id),
        seed,
        tuple(map(float, start)),
        zone.wkb,
        obstacles.wkb,
        (
            min(xs) + robot_radius,
            min(ys) + robot_radius,
            max(xs) - robot_radius,
            max(ys) - robot_radius,
        ),
        clip_obstacles=robot_radius == 0,
        physical_contract=(physical_wkb, robot_radius) if physical_wkb is not None else None,
    )


__all__ = [
    "compute_completion_reference_length",
    "compute_shortest_path_length",
    "remaining_route_length",
]
