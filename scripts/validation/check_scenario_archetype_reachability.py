"""Read-only geometric reachability probe for pinned scenario archetype maps.

This probe answers, per archetype map and per declared robot route, whether the
robot can geometrically reach its goal zone from its spawn zone through the
declared obstacle layout, using the same route model the simulator executes:

``[initial_spawn, *route.waypoints, final_goal]``

where ``initial_spawn`` is sampled inside the bound spawn zone and ``final_goal``
inside the bound goal zone (``robot_sf.nav.navigation.sample_route``). The probe
therefore evaluates three geometry segments against the obstacle set:

1. the interior corridor: the SVG route anchors buffered by the robot envelope;
2. the spawn connector: each representative admissible spawn-zone point to the
   first SVG route anchor;
3. the goal connector: the last SVG route anchor to each representative
   admissible goal-zone point.

SVG route anchors are NOT required to lie inside their zones: the accepted
route-anchor offsets (documented in #7693/#7709) are recorded as an
``anchor_offset`` observation, never as a blockage. Zone ranges use deterministic
representative points (centroid plus fixed interior samples matching the runtime
folded-triangle sampling, minus obstacle intersections) so the verdict is
reproducible; a segment is ``*_connector_blocked`` when ANY representative
connector is obstructed, and ``insufficient_proof`` when no collision-free
connector can be established for the whole range (fail closed).

The probe never mutates map files and makes no simulation, agent-model,
planner-liveness, or benchmark claim. Its report is deterministic for a given
map set.

Exit-code policy: findings are informational by default so the first landing
records existing known deviations without reddening CI; ``--fail-on-violation``
promotes any finding to exit code 1 for later gate enforcement.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.svg_map_parser import SvgMapConverter

DEFAULT_ENVELOPE_RADIUS_M = 0.4  # robot radius used by the simulator.
DEFAULT_MAPS = (
    "maps/svg_maps/classic_doorway.svg",
    "maps/svg_maps/classic_head_on_corridor.svg",
    "maps/svg_maps/classic_group_crossing.svg",
    "maps/svg_maps/classic_crossing.svg",
)
DEFAULT_ZONE_SAMPLES = 5  # deterministic representatives per zone range.
ZONE_SAMPLE_SEED = 20260824  # fixed seed so reports are reproducible.


@dataclass(frozen=True, slots=True)
class ZoneSample:
    """A deterministic representative point inside a zone range."""

    x: float
    y: float


@dataclass
class RouteReachability:
    """Geometric reachability result for one robot route.

    A route is blocked only by an actual obstacle along one of its three
    runtime-faithful segments, or by insufficient proof of a zone connector.
    The accepted anchor-offset geometry is an observation, never a blockage.
    """

    label: str
    anchor_offset: bool
    interior_obstacle_blocks: bool
    spawn_connector_blocks: bool
    goal_connector_blocks: bool
    insufficient_proof: bool = False
    blocking_obstacle_index: int | None = None
    blocking_obstacle_shared_area_m2: float | None = None
    blocked_class: str | None = None

    @property
    def reachable(self) -> bool:
        """True only when every runtime-faithful segment is geometrically free."""
        return not (
            self.interior_obstacle_blocks
            or self.spawn_connector_blocks
            or self.goal_connector_blocks
            or self.insufficient_proof
        )


@dataclass
class MapReachabilityReport:
    """Aggregated geometric reachability findings for one SVG map."""

    map_path: str
    routes: list[RouteReachability] = field(default_factory=list)

    @property
    def blocked_routes(self) -> int:
        """Count routes that are not geometrically reachable end to end."""
        return sum(1 for route in self.routes if not route.reachable)


def _obstacle_polygon(obstacle) -> object:
    """Return the canonical obstacle geometry (polygon or multipolygon)."""
    geometry = getattr(obstacle, "geometry", None)
    if geometry is not None:
        return geometry
    return Polygon(obstacle.vertices)


def _obstacle_polygons(obstacles) -> list[Polygon]:
    """Return every obstacle polygon in declaration order."""
    polygons: list[Polygon] = []
    for obstacle in obstacles:
        geometry = _obstacle_polygon(obstacle)
        if geometry is None or geometry.is_empty:
            continue
        if geometry.geom_type == "MultiPolygon":
            polygons.extend(list(geometry.geoms))
        else:
            polygons.append(geometry)
    return polygons


def _zone_range_points(rect, k: int, rng: np.random.Generator) -> list[ZoneSample]:
    """Return deterministic representative points for a triangular zone.

    Mirrors ``sample_zone``'s folded-triangle sampling: points ``a, b, c`` define
    the triangle, and interior points use barycentric folding so no sample falls
    outside the triangle. The centroid is always included so a degenerate or
    obstacle-free small zone still yields a representative.
    """
    a, b, c = rect
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    vec_ba, vec_bc = a - b, c - b
    width = np.concatenate(([1 / 3], rng.uniform(0, 1, max(k - 1, 0))))
    height = np.concatenate(([1 / 3], rng.uniform(0, 1, max(k - 1, 0))))
    folded = width + height > 1.0
    width[folded] = 1.0 - width[folded]
    height[folded] = 1.0 - height[folded]
    points = b + width[:, None] * vec_ba + height[:, None] * vec_bc
    return [ZoneSample(x=float(p[0]), y=float(p[1])) for p in points]


def _point_in_obstacles(point: Point, obstacle_polygons: list[Polygon]) -> bool:
    """Return whether a representative point falls inside any obstacle."""
    return any(polygon.covers(point) for polygon in obstacle_polygons)


def _corridor_blocked(
    polyline: LineString,
    envelope_radius_m: float,
    obstacle_polygons: list[Polygon],
) -> tuple[bool, int | None, float | None]:
    """Return whether a buffered corridor intersects any obstacle.

    Returns ``(blocked, obstacle_index, shared_area_m2)`` in declaration order.
    """
    corridor = polyline.buffer(envelope_radius_m)
    for index, polygon in enumerate(obstacle_polygons):
        intersection = corridor.intersection(polygon)
        if intersection.is_empty:
            continue
        shared_area = float(intersection.area)
        if shared_area > 0.0:
            return True, index, round(shared_area, 6)
    return False, None, None


def _segment_connectors_blocked(
    anchor: Point,
    zone_points: list[ZoneSample],
    obstacle_polygons: list[Polygon],
    envelope_radius_m: float,
) -> tuple[bool | None, int | None, float | None]:
    """Return whether any representative connector to an anchor is obstructed.

    Representative points already lying inside an obstacle are skipped (they are
    not admissible spawns). Returns ``(blocked, index, area)``; when
    ``blocked is None`` no admissible connector could be established at all
    (the caller treats that as ``insufficient_proof``).
    """
    admissible = [
        point
        for point in zone_points
        if not _point_in_obstacles(Point(point.x, point.y), obstacle_polygons)
    ]
    if not admissible:
        return None, None, None
    for sample in admissible:
        segment = LineString([(sample.x, sample.y), (anchor.x, anchor.y)])
        blocked, index, area = _corridor_blocked(segment, envelope_radius_m, obstacle_polygons)
        if blocked:
            return True, index, area
    return False, None, None


def _anchor_inside_zone(point: Point, rect) -> bool:
    """Return whether an SVG route anchor lies inside its bound zone rect."""
    if rect is None:
        return False
    a, b, c = rect
    xs = sorted({a[0], b[0], c[0]})
    ys = sorted({a[1], b[1], c[1]})
    if len(xs) != 2 or len(ys) != 2:
        raise ValueError(f"Rect {rect} is not an axis-aligned three-corner rectangle")
    polygon = Polygon([(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])])
    return polygon.covers(point)


def inspect_map_reachability(
    svg_path: Path, envelope_radius_m: float = DEFAULT_ENVELOPE_RADIUS_M
) -> MapReachabilityReport:
    """Run all read-only geometric reachability checks for one SVG map.

    Only robot routes are probed: the archetype maps define the robot as the
    actor whose spawn/goal reachability the scenario is built to ensure.
    Pedestrian routes are intentionally excluded from this diagnostic
    classification.
    """
    converter = SvgMapConverter(str(svg_path))
    map_def: MapDefinition = converter.get_map_definition()
    obstacle_polygons = _obstacle_polygons(map_def.obstacles)
    rng = np.random.default_rng(ZONE_SAMPLE_SEED)
    report = MapReachabilityReport(map_path=str(svg_path))
    for route in map_def.robot_routes:
        label = route.source_label or f"robot_route_{route.spawn_id}_{route.goal_id}"
        if not route.waypoints or len(route.waypoints) < 2:
            report.routes.append(
                RouteReachability(
                    label=label,
                    anchor_offset=True,
                    interior_obstacle_blocks=False,
                    spawn_connector_blocks=False,
                    goal_connector_blocks=False,
                    insufficient_proof=True,
                    blocked_class="insufficient_proof",
                )
            )
            continue

        first_anchor = Point(float(route.waypoints[0][0]), float(route.waypoints[0][1]))
        last_anchor = Point(float(route.waypoints[-1][0]), float(route.waypoints[-1][1]))

        spawn_points = _zone_range_points(route.spawn_zone, DEFAULT_ZONE_SAMPLES, rng)
        goal_points = _zone_range_points(route.goal_zone, DEFAULT_ZONE_SAMPLES, rng)

        interior_line = LineString([(float(x), float(y)) for x, y in route.waypoints])
        interior_blocked, interior_index, interior_area = _corridor_blocked(
            interior_line, envelope_radius_m, obstacle_polygons
        )
        spawn_blocked, spawn_index, spawn_area = _segment_connectors_blocked(
            first_anchor, spawn_points, obstacle_polygons, envelope_radius_m
        )
        goal_blocked, goal_index, goal_area = _segment_connectors_blocked(
            last_anchor, goal_points, obstacle_polygons, envelope_radius_m
        )

        blocked_class: str | None = None
        blocking_index: int | None = None
        blocking_area: float | None = None
        insufficient = False
        if spawn_blocked is True:
            blocked_class = "spawn_connector_blocked"
            blocking_index = spawn_index
            blocking_area = spawn_area
        elif goal_blocked is True:
            blocked_class = "goal_connector_blocked"
            blocking_index = goal_index
            blocking_area = goal_area
        elif interior_blocked:
            blocked_class = "interior_route_blocked"
            blocking_index = interior_index
            blocking_area = interior_area
        elif spawn_blocked is None or goal_blocked is None:
            blocked_class = "insufficient_proof"
            insufficient = True

        anchor_offset = not (
            _anchor_inside_zone(first_anchor, route.spawn_zone)
            and _anchor_inside_zone(last_anchor, route.goal_zone)
        )

        report.routes.append(
            RouteReachability(
                label=label,
                anchor_offset=anchor_offset,
                interior_obstacle_blocks=interior_blocked,
                spawn_connector_blocks=spawn_blocked is True,
                goal_connector_blocks=goal_blocked is True,
                insufficient_proof=insufficient,
                blocking_obstacle_index=blocking_index,
                blocking_obstacle_shared_area_m2=blocking_area,
                blocked_class=blocked_class,
            )
        )
    return report


def format_console_table(report: MapReachabilityReport) -> str:
    """Render a compact human-readable table for one map report."""
    lines = [f"map: {report.map_path}"]
    for route in report.routes:
        flag = "ok  " if route.reachable else "BLOCK"
        notes = []
        if route.anchor_offset:
            notes.append("anchor-offset(observed)")
        if route.blocked_class:
            notes.append(route.blocked_class)
            if route.blocking_obstacle_index is not None:
                notes.append(
                    f"obstacle[{route.blocking_obstacle_index}]"
                    f"({route.blocking_obstacle_shared_area_m2} m2)"
                )
        lines.append(
            f"  [{flag}] {route.label:<28} "
            + (", ".join(notes) if notes else "runtime-faithful route reachable")
        )
    if report.blocked_routes == 0:
        lines.append("  all routes runtime-faithfully reachable")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point returning a process exit code."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        help="SVG map path; repeatable. Defaults to the four pinned archetype maps.",
    )
    parser.add_argument(
        "--envelope-radius-m",
        type=float,
        default=DEFAULT_ENVELOPE_RADIUS_M,
        help="Robot envelope radius used to buffer the route corridor (default 0.4 m).",
    )
    parser.add_argument("--json", action="store_true", help="Emit the JSON report only.")
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit 1 when any finding exists (for later CI enforcement).",
    )
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.map] or [Path(p) for p in DEFAULT_MAPS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"ERROR: map files not found: {', '.join(str(p) for p in missing)}", file=sys.stderr)
        return 2

    reports = [inspect_map_reachability(p, envelope_radius_m=args.envelope_radius_m) for p in paths]
    total = sum(r.blocked_routes for r in reports)

    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=1))
    else:
        for report in reports:
            print(format_console_table(report))
        print(f"\ntotal blocked routes: {total}")

    if args.fail_on_violation and total > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
