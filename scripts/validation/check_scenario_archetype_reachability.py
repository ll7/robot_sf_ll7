"""Read-only geometric reachability probe for pinned scenario archetype maps.

This probe answers, per archetype map and per declared robot route, whether the
robot can geometrically reach its goal zone from its spawn zone through the
declared obstacle layout:

1. the route start and end waypoints lie inside their bound spawn/goal zones;
2. the route corridor (waypoints buffered by the robot envelope radius) does not
   intersect any declared obstacle polygon;
3. the route has at least two waypoints so the corridor is a real path.

The probe never mutates map files and makes no simulation, agent-model, or
benchmark claim. Its report is deterministic for a given map set.

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

from shapely.geometry import LineString, Point, Polygon

from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.svg_map_parser import SvgMapConverter

DEFAULT_ENVELOPE_RADIUS_M = 0.4  # ped/robot radius used by the simulator.
DEFAULT_MAPS = (
    "maps/svg_maps/classic_doorway.svg",
    "maps/svg_maps/classic_head_on_corridor.svg",
    "maps/svg_maps/classic_group_crossing.svg",
    "maps/svg_maps/classic_crossing.svg",
)


@dataclass
class RouteReachability:
    """Geometric reachability result for one robot route."""

    label: str
    start_inside_zone: bool
    goal_inside_zone: bool
    obstacle_blocks: bool
    blocking_obstacle_index: int | None = None
    blocking_obstacle_shared_area_m2: float | None = None

    @property
    def reachable(self) -> bool:
        """True only when the route is geometrically free end to end."""
        return self.start_inside_zone and self.goal_inside_zone and not self.obstacle_blocks


@dataclass
class MapReachabilityReport:
    """Aggregated geometric reachability findings for one SVG map."""

    map_path: str
    routes: list[RouteReachability] = field(default_factory=list)

    @property
    def blocked_routes(self) -> int:
        """Count routes that are not geometrically reachable."""
        return sum(1 for route in self.routes if not route.reachable)


def _route_polyline(route) -> LineString:
    """Return the route waypoints as a Shapely line string."""
    return LineString([(float(x), float(y)) for x, y in route.waypoints])


def _obstacle_polygon(obstacle) -> object:
    """Return the canonical obstacle geometry (polygon or multipolygon)."""
    geometry = getattr(obstacle, "geometry", None)
    if geometry is not None:
        return geometry
    return Polygon(obstacle.vertices)


def _blocking_obstacle(
    route, obstacles, envelope_radius_m: float
) -> tuple[bool, int | None, float | None]:
    """Return whether the buffered route corridor intersects any obstacle.

    Returns ``(blocked, obstacle_index, shared_area_m2)``. Obstacles are checked
    in declaration order so the index is deterministic for a map.
    """
    corridor = _route_polyline(route).buffer(envelope_radius_m)
    for index, obstacle in enumerate(obstacles):
        polygon = _obstacle_polygon(obstacle)
        if polygon is None:
            continue
        intersection = corridor.intersection(polygon)
        if intersection.is_empty:
            continue
        shared_area = float(intersection.area)
        if shared_area > 0.0:
            return True, index, round(shared_area, 6)
    return False, None, None


def _zone_polygon(rect) -> object:
    """Return a spawn/goal zone rectangle as a Shapely polygon."""
    a, b, c = rect
    xs = sorted({a[0], b[0], c[0]})
    ys = sorted({a[1], b[1], c[1]})
    if len(xs) != 2 or len(ys) != 2:
        raise ValueError(f"Rect {rect} is not an axis-aligned three-corner rectangle")
    return Polygon([(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])])


def _inside_zone(point: Point, rect) -> bool:
    """Return whether a point lies inside (or on the boundary of) a zone rect."""
    if rect is None:
        return False
    polygon = _zone_polygon(rect)
    return polygon.covers(point)


def inspect_map_reachability(
    svg_path: Path, envelope_radius_m: float = DEFAULT_ENVELOPE_RADIUS_M
) -> MapReachabilityReport:
    """Run all read-only geometric reachability checks for one SVG map.

    Only robot routes are probed: the archetype maps define the robot as the
    actor whose spawn/goal reachability the scenario is built to ensure. Pedestrian
    routes are intentionally excluded from this diagnostic classification.
    """
    converter = SvgMapConverter(str(svg_path))
    map_def: MapDefinition = converter.get_map_definition()
    report = MapReachabilityReport(map_path=str(svg_path))
    for route in map_def.robot_routes:
        if not route.waypoints or len(route.waypoints) < 2:
            report.routes.append(
                RouteReachability(
                    label=route.source_label or f"robot_route_{route.spawn_id}_{route.goal_id}",
                    start_inside_zone=False,
                    goal_inside_zone=False,
                    obstacle_blocks=False,
                )
            )
            continue
        first = Point(float(route.waypoints[0][0]), float(route.waypoints[0][1]))
        last = Point(float(route.waypoints[-1][0]), float(route.waypoints[-1][1]))
        start_ok = _inside_zone(first, route.spawn_zone)
        goal_ok = _inside_zone(last, route.goal_zone)
        blocked, obstacle_index, shared_area = _blocking_obstacle(
            route, map_def.obstacles, envelope_radius_m
        )
        report.routes.append(
            RouteReachability(
                label=route.source_label or f"robot_route_{route.spawn_id}_{route.goal_id}",
                start_inside_zone=start_ok,
                goal_inside_zone=goal_ok,
                obstacle_blocks=blocked,
                blocking_obstacle_index=obstacle_index,
                blocking_obstacle_shared_area_m2=shared_area,
            )
        )
    return report


def format_console_table(report: MapReachabilityReport) -> str:
    """Render a compact human-readable table for one map report."""
    lines = [f"map: {report.map_path}"]
    for route in report.routes:
        flag = "ok  " if route.reachable else "BLOCK"
        reasons = []
        if not route.start_inside_zone:
            reasons.append("start-off-zone")
        if not route.goal_inside_zone:
            reasons.append("goal-off-zone")
        if route.obstacle_blocks:
            reasons.append(
                f"obstacle[{route.blocking_obstacle_index}]"
                f"({route.blocking_obstacle_shared_area_m2} m2)"
            )
        lines.append(
            f"  [{flag}] {route.label:<28} "
            + (", ".join(reasons) if reasons else "geometrically reachable")
        )
    if report.blocked_routes == 0:
        lines.append("  all routes geometrically reachable")
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
