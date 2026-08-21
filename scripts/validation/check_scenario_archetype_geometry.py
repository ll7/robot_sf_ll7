"""Read-only route-zone geometry checks for pinned scenario archetype maps.

This checker answers, per archetype map and per declared route, whether the
route geometry is consistent with the spawn/goal zones the map declares:

1. every route endpoint lies inside its bound zone (or within a stated
   tolerance of that zone's centre), with the offset distance reported;
2. no contiguous interior fragment of a route is disconnected from every
   declared zone (the cross-trap interior-fragment failure shape);
3. each route kind with routes but without any declared zones of that kind is
   flagged as missing zone coverage.

The checker never mutates map files and makes no simulation, reachability,
or runtime-parameter claim. Its report is deterministic for a given map set.

Exit-code policy: findings are informational by default so the first landing
records existing known deviations without reddening CI; ``--fail-on-violation``
promotes any finding to exit code 1 for later gate enforcement.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from math import dist
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon

from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.svg_map_parser import SvgMapConverter

DEFAULT_TOLERANCE_M = 0.5
DEFAULT_MAPS = (
    "maps/svg_maps/classic_doorway.svg",
    "maps/svg_maps/classic_head_on_corridor.svg",
    "maps/svg_maps/classic_group_crossing.svg",
    "maps/svg_maps/classic_crossing.svg",
)


@dataclass
class EndpointCheck:
    """Endpoint-to-bound-zone-centre offset for one route end."""

    route_kind: str
    label: str
    end: str
    zone_kind: str
    zone_index: int
    inside_zone: bool
    offset_to_centre_m: float


@dataclass
class FragmentCheck:
    """Interior-fragment disconnection result for one route."""

    route_kind: str
    label: str
    disconnected_fragment_count: int
    first_disconnected_segment: int | None = None


@dataclass
class MapGeometryReport:
    """Aggregated geometry-consistency findings for one SVG map."""

    map_path: str
    endpoints: list[EndpointCheck] = field(default_factory=list)
    fragments: list[FragmentCheck] = field(default_factory=list)
    missing_zone_kinds: list[str] = field(default_factory=list)

    @property
    def violations(self) -> int:
        """Count endpoint misses, disconnected fragments, and missing zone kinds."""

        missed = sum(1 for e in self.endpoints if not e.inside_zone)
        broken = sum(1 for f in self.fragments if f.disconnected_fragment_count > 0)
        return missed + broken + len(self.missing_zone_kinds)


def _rect_polygon(rect: tuple) -> Polygon:
    """Return the shapely polygon for a Rect (three-corner tuple).

    For an axis-aligned rectangle each coordinate value appears exactly twice
    across the four corners; with three corners given, the missing corner takes
    the x and y value that occur only once.
    """

    a, b, c = rect
    xs = sorted({a[0], b[0], c[0]})
    ys = sorted({a[1], b[1], c[1]})
    if len(xs) != 2 or len(ys) != 2:
        raise ValueError(f"Rect {rect} is not an axis-aligned three-corner rectangle")
    return Polygon([(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])])


def _rect_centre(rect: tuple) -> tuple[float, float]:
    """Return the axis-aligned centre of a Rect from three corners."""

    x = sorted(p[0] for p in rect)
    y = sorted(p[1] for p in rect)
    return ((x[0] + x[2]) / 2.0, (y[0] + y[2]) / 2.0)


def _zone_pairs(
    map_def: MapDefinition,
) -> dict[str, list[tuple[str, tuple]]]:
    """Group zone rects by kind with stable ordering."""

    return {
        "robot": [("robot_spawn_zone", r) for r in map_def.robot_spawn_zones]
        + [("robot_goal_zone", r) for r in map_def.robot_goal_zones],
        "ped": [("ped_spawn_zone", r) for r in map_def.ped_spawn_zones]
        + [("ped_goal_zone", r) for r in map_def.ped_goal_zones],
    }


def _endpoint_checks(
    map_def: MapDefinition,
    route_kind: str,
    routes,
    tolerance_m: float,
) -> list[EndpointCheck]:
    """Check each route endpoint against its bound spawn/goal zone."""

    results: list[EndpointCheck] = []
    for route in routes:
        if not route.waypoints or route.spawn_zone is None or route.goal_zone is None:
            continue
        first, last = route.waypoints[0], route.waypoints[-1]
        ends = (
            (
                "start",
                first,
                _rect_centre(route.spawn_zone),
                _rect_polygon(route.spawn_zone),
                f"{route_kind}_spawn_zone",
                route.spawn_id,
            ),
            (
                "end",
                last,
                _rect_centre(route.goal_zone),
                _rect_polygon(route.goal_zone),
                f"{route_kind}_goal_zone",
                route.goal_id,
            ),
        )
        for end_name, point, centre, poly, kind, zone_index in ends:
            inside = poly.contains(Point(*point)) or poly.touches(Point(*point))
            offset = dist(point, centre)
            results.append(
                EndpointCheck(
                    route_kind=route_kind,
                    label=route.source_label
                    or f"{route_kind}_route_{route.spawn_id}_{route.goal_id}",
                    end=end_name,
                    zone_kind=kind,
                    zone_index=zone_index,
                    inside_zone=bool(inside or offset <= tolerance_m),
                    offset_to_centre_m=round(offset, 3),
                )
            )
    return results


def _fragment_checks(map_def: MapDefinition, route_kind: str, routes) -> list[FragmentCheck]:
    """Detect contiguous route segments that touch no declared zone of any kind."""

    all_zones = [
        _rect_polygon(rect) for rects in _zone_pairs(map_def).values() for _, rect in rects
    ]
    results: list[FragmentCheck] = []
    for route in routes:
        waypoints = route.waypoints
        if len(waypoints) < 2 or not all_zones:
            continue
        disconnected: list[int] = []
        for index, (a, b) in enumerate(pairwise(waypoints)):
            segment = LineString([a, b])
            if not any(zone.intersects(segment) for zone in all_zones):
                disconnected.append(index)
        results.append(
            FragmentCheck(
                route_kind=route_kind,
                label=route.source_label or f"{route_kind}_route_{route.spawn_id}_{route.goal_id}",
                disconnected_fragment_count=len(disconnected),
                first_disconnected_segment=disconnected[0] if disconnected else None,
            )
        )
    return results


def _missing_zone_kinds(map_def: MapDefinition) -> list[str]:
    """Flag route kinds that have routes but no zones of that kind at all."""

    missing: list[str] = []
    if map_def.robot_routes and not (map_def.robot_spawn_zones or map_def.robot_goal_zones):
        missing.append("robot")
    if map_def.ped_routes and not (map_def.ped_spawn_zones or map_def.ped_goal_zones):
        missing.append("ped")
    return missing


def inspect_map_geometry(
    svg_path: Path, tolerance_m: float = DEFAULT_TOLERANCE_M
) -> MapGeometryReport:
    """Run all read-only geometry consistency checks for one SVG map."""

    converter = SvgMapConverter(str(svg_path))
    map_def = converter.get_map_definition()
    report = MapGeometryReport(map_path=str(svg_path))
    for kind, routes in (("robot", map_def.robot_routes), ("ped", map_def.ped_routes)):
        report.endpoints.extend(_endpoint_checks(map_def, kind, routes, tolerance_m))
        report.fragments.extend(_fragment_checks(map_def, kind, routes))
    report.missing_zone_kinds = _missing_zone_kinds(map_def)
    return report


def format_console_table(report: MapGeometryReport) -> str:
    """Render a compact human-readable table for one map report."""

    lines = [f"map: {report.map_path}"]
    for e in report.endpoints:
        flag = "ok " if e.inside_zone else "MISS"
        lines.append(
            f"  [{flag}] {e.route_kind:>5} {e.label:<28} {e.end:<5}"
            f" offset={e.offset_to_centre_m:>7.3f}m zone={e.zone_kind}[{e.zone_index}]"
        )
    for f in report.fragments:
        if f.disconnected_fragment_count:
            lines.append(
                f"  [FRAG] {f.route_kind:>5} {f.label:<28}"
                f" disconnected_segments={f.disconnected_fragment_count}"
                f" first={f.first_disconnected_segment}"
            )
    for kind in report.missing_zone_kinds:
        lines.append(f"  [ZONES] {kind} routes present without any {kind} zones")
    if report.violations == 0:
        lines.append("  all checks informational-clean")
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
    parser.add_argument("--tolerance-m", type=float, default=DEFAULT_TOLERANCE_M)
    parser.add_argument("--json", action="store_true", help="Emit the JSON report only.")
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit 1 when any finding exists (for later CI enforcement).",
    )
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.map] or [Path(p) for p in DEFAULT_MAPS]
    reports = [inspect_map_geometry(p, args.tolerance_m) for p in paths]
    total = sum(r.violations for r in reports)

    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=1))
    else:
        for report in reports:
            print(format_console_table(report))
        print(f"\ntotal findings: {total}")

    if args.fail_on_violation and total > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
