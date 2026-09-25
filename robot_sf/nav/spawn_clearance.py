"""Clearance-aware spawn geometry for robots and pedestrians (issue #9725).

The simulator samples spawn *centres*. Collision checks, however, test a robot
circle against wall lines and a robot circle against pedestrian circles. A centre
that lies outside every obstacle polygon can therefore still start the episode
in collision. This module owns the geometry that closes that gap:

* :func:`robot_start_exclusions` pads walls and map bounds by the robot radius
  plus a margin, so robot starts are sampled only where the robot footprint fits.
* :func:`robot_obstacle_clearance` measures the robot-to-wall clearance that
  benchmark records keep for every episode (see ``robot_sf.sim.spawn_validation``).
* :func:`relocate_overlapping_pedestrians` moves pedestrians that overlap a robot
  footprint at reset to the nearest clear point, deterministically and without
  drawing random numbers, so seeds without an overlap keep their exact spawns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, dist, hypot, isfinite, pi, sin
from typing import TYPE_CHECKING, Any

from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import PreparedGeometry, prep

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from robot_sf.common.types import Vec2D
    from robot_sf.nav.map_config import MapDefinition

#: Extra surface clearance (metres) kept between spawned agents and walls or robots.
SPAWN_CLEARANCE_MARGIN_M = 0.1
#: Reason recorded on benchmark rows whose outcome is caused by a spawn overlap.
SPAWN_OVERLAP_INVALID_REASON = "spawn_overlap"

_RELOCATION_ANGLE_STEP_RAD = pi / 12.0
_RELOCATION_DISTANCE_STEPS_M = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)


def _obstacle_lines(map_def: MapDefinition) -> list[LineString]:
    """Return wall and map-bound segments exactly as the collision checker sees them."""
    lines: list[LineString] = []
    for x1, x2, y1, y2 in getattr(map_def, "obstacles_pysf", []) or []:
        if (x1, y1) == (x2, y2):
            continue
        lines.append(LineString([(x1, y1), (x2, y2)]))
    return lines


def _obstacle_polygons(map_def: MapDefinition) -> list[Polygon]:
    """Return obstacle polygons (interiors count as blocked space)."""
    polygons: list[Polygon] = []
    for obstacle in getattr(map_def, "obstacles", []) or []:
        for polygon in obstacle.iter_polygons():
            if not polygon.is_empty:
                polygons.append(polygon)
    return polygons


def _blocked_geometry(map_def: MapDefinition, clearance: float) -> Any:
    """Build the region whose points are closer than ``clearance`` to any wall or bound.

    Returns:
        A shapely geometry covering padded walls, obstacle interiors, and the
        area outside the ``clearance``-inset map bounds.
    """
    parts: list[Any] = [line.buffer(clearance) for line in _obstacle_lines(map_def)]
    parts.extend(_obstacle_polygons(map_def))
    x_min, x_max, y_min, y_max = map_def.get_map_bounds()
    outer = box(x_min - 1.0e3, y_min - 1.0e3, x_max + 1.0e3, y_max + 1.0e3)
    inner = box(x_min + clearance, y_min + clearance, x_max - clearance, y_max - clearance)
    parts.append(outer.difference(inner) if not inner.is_empty else outer)
    return unary_union(parts)


def robot_start_exclusions(
    map_def: MapDefinition,
    robot_radius: float,
    margin: float = SPAWN_CLEARANCE_MARGIN_M,
) -> list[PreparedGeometry]:
    """Return cached exclusion geometry for clearance-aware robot start sampling.

    A robot centre is admissible only when it is at least ``robot_radius + margin``
    away from every wall segment and map bound, and not inside an obstacle.

    Returns:
        One prepared geometry; a sample point intersecting it is rejected.
    """
    clearance = float(robot_radius) + float(margin)
    if not isfinite(clearance) or clearance < 0.0:
        raise ValueError(f"robot start clearance must be finite and >= 0 (got {clearance!r})")
    cache: dict[float, list[PreparedGeometry]] | None = getattr(
        map_def, "_robot_start_exclusion_cache", None
    )
    if cache is None:
        cache = {}
        map_def._robot_start_exclusion_cache = cache  # type: ignore[attr-defined]
    key = round(clearance, 9)
    if key not in cache:
        cache[key] = [prep(_blocked_geometry(map_def, clearance))]
    return cache[key]


def robot_obstacle_clearance(map_def: MapDefinition, xy: Vec2D, robot_radius: float) -> float:
    """Return the surface clearance between a robot footprint and the static map.

    Negative values mean the robot starts in collision with a wall, a map bound,
    or an obstacle interior (the latter reported as ``-robot_radius`` or less).

    Returns:
        Minimum distance from the robot surface to walls and bounds, in metres.
    """
    point = Point(float(xy[0]), float(xy[1]))
    x_min, x_max, y_min, y_max = map_def.get_map_bounds()
    if not (x_min <= point.x <= x_max and y_min <= point.y <= y_max):
        return -float(robot_radius)
    if any(polygon.contains(point) for polygon in _obstacle_polygons(map_def)):
        return -float(robot_radius)
    distances = [line.distance(point) for line in _obstacle_lines(map_def)]
    if not distances:
        return float("inf")
    return float(min(distances)) - float(robot_radius)


@dataclass
class PedestrianRelocationReport:
    """Rows moved off robot footprints at reset, and rows that could not be moved."""

    relocated: dict[int, tuple[Vec2D, Vec2D]] = field(default_factory=dict)
    unresolved: list[int] = field(default_factory=list)


def _pedestrian_blocked_geometry(map_def: MapDefinition, ped_radius: float) -> PreparedGeometry:
    """Return cached static geometry a pedestrian centre must not enter."""
    cache: dict[float, PreparedGeometry] | None = getattr(
        map_def, "_ped_relocation_block_cache", None
    )
    if cache is None:
        cache = {}
        map_def._ped_relocation_block_cache = cache  # type: ignore[attr-defined]
    key = round(float(ped_radius), 9)
    if key not in cache:
        cache[key] = prep(_blocked_geometry(map_def, float(ped_radius)))
    return cache[key]


def relocate_overlapping_pedestrians(
    ped_xy: Sequence[Vec2D],
    ped_radius: float,
    robots: Sequence[tuple[Vec2D, float]],
    map_def: MapDefinition,
    margin: float = SPAWN_CLEARANCE_MARGIN_M,
    *,
    rows: Sequence[int] | None = None,
) -> PedestrianRelocationReport:
    """Find clear positions for pedestrians that overlap a robot footprint.

    A pedestrian overlaps when its centre is closer than
    ``robot_radius + ped_radius + margin`` to a robot centre. Each such pedestrian is
    moved along the ray from the robot through its current position, to the
    nearest point on the exclusion circle; if that point is blocked (walls, other
    pedestrians, other robots), rotated rays and slightly larger radii are tried
    in a fixed order. No random numbers are drawn, so the global RNG stream and
    every non-overlapping spawn stay unchanged.

    Args:
        ped_xy: Current pedestrian positions.
        ped_radius: Pedestrian collision radius.
        robots: ``((x, y), radius)`` for each robot.
        map_def: Map providing walls and bounds.
        margin: Extra surface clearance to keep.
        rows: Optional subset of rows that may be moved (defaults to all rows).

    Returns:
        Report with ``row -> (old, new)`` moves and the rows left unresolved.
    """
    report = PedestrianRelocationReport()
    positions: list[Vec2D] = [(float(p[0]), float(p[1])) for p in ped_xy]
    movable = set(range(len(positions)) if rows is None else rows)
    blocked = _pedestrian_blocked_geometry(map_def, ped_radius)
    for row in range(len(positions)):
        hit = _overlapping_robot(positions[row], robots, ped_radius, margin)
        if hit is None:
            continue
        if row not in movable:
            report.unresolved.append(row)
            continue
        new_xy = next(
            (
                candidate
                for candidate in _relocation_candidates(positions[row], hit, ped_radius, margin)
                if _is_clear(candidate, row, positions, robots, ped_radius, margin, blocked)
            ),
            None,
        )
        if new_xy is None:
            report.unresolved.append(row)
            continue
        report.relocated[row] = (positions[row], new_xy)
        positions[row] = new_xy
    return report


def _overlapping_robot(
    point: Vec2D,
    robots: Sequence[tuple[Vec2D, float]],
    ped_radius: float,
    margin: float,
) -> tuple[Vec2D, float] | None:
    """Return the first robot whose exclusion circle contains ``point``."""
    for robot_xy, robot_radius in robots:
        if dist(point, robot_xy) < float(robot_radius) + float(ped_radius) + float(margin):
            return robot_xy, robot_radius
    return None


def _relocation_candidates(
    point: Vec2D,
    robot: tuple[Vec2D, float],
    ped_radius: float,
    margin: float,
) -> Iterator[Vec2D]:
    """Yield candidate positions on and just beyond a robot exclusion circle.

    The ray through the current position comes first, then rays rotated in 15 degree
    steps alternating sides, then the same sweep at slightly larger radii.
    """
    robot_xy, robot_radius = robot
    dx, dy = point[0] - robot_xy[0], point[1] - robot_xy[1]
    base_angle = atan2(dy, dx) if hypot(dx, dy) > 1e-9 else 0.0
    base_radius = float(robot_radius) + float(ped_radius) + float(margin) + 1e-6
    half_turn_steps = 12
    for extra in _RELOCATION_DISTANCE_STEPS_M:
        radius = base_radius + extra
        for k in range(half_turn_steps + 1):
            signs = (1.0, -1.0) if 0 < k < half_turn_steps else (1.0,)
            for sign in signs:
                angle = base_angle + sign * k * _RELOCATION_ANGLE_STEP_RAD
                yield (robot_xy[0] + radius * cos(angle), robot_xy[1] + radius * sin(angle))


def _is_clear(
    candidate: Vec2D,
    row: int,
    positions: Sequence[Vec2D],
    robots: Sequence[tuple[Vec2D, float]],
    ped_radius: float,
    margin: float,
    blocked: PreparedGeometry,
) -> bool:
    """Return whether a relocation candidate clears robots, walls, and other pedestrians."""
    if _overlapping_robot(candidate, robots, ped_radius, margin) is not None:
        return False
    if blocked.intersects(Point(candidate)):
        return False
    return all(
        other == row or dist(candidate, other_xy) >= 2.0 * float(ped_radius)
        for other, other_xy in enumerate(positions)
    )


__all__ = [
    "SPAWN_CLEARANCE_MARGIN_M",
    "SPAWN_OVERLAP_INVALID_REASON",
    "PedestrianRelocationReport",
    "relocate_overlapping_pedestrians",
    "robot_obstacle_clearance",
    "robot_start_exclusions",
]
