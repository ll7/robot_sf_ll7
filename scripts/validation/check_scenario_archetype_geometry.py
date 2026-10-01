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

Exit-code policy: findings are informational by default so existing known
deviations remain visible without reddening local diagnostics. In
``--fail-on-violation`` mode, an explicit exact-waiver file is required and
any missing, stale, duplicate, or changed-evidence row returns exit code 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from math import dist
from pathlib import Path

from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import unary_union

from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.svg_map_parser import SvgMapConverter
from scripts.validation.scenario_validation_waivers import (
    WaiverValidationError,
    canonical_repo_path,
    load_waiver_rows,
    validate_exact_waivers,
)

DEFAULT_TOLERANCE_M = 0.5
DEFAULT_WAIVER_FILE = Path("configs/scenarios/archetype_validation_waivers.yaml")
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
    route_counts: dict[str, int] = field(default_factory=dict)

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
    report.route_counts = {
        "robot": len(map_def.robot_routes),
        "ped": len(map_def.ped_routes),
    }
    return report


_GEOMETRY_IDENTITY_FIELDS = (
    "map",
    "finding_type",
    "route_kind",
    "label",
    "end",
    "zone_kind",
    "zone_index",
    "first_disconnected_segment",
)


def _geometry_findings(reports: list[MapGeometryReport]) -> list[dict[str, object]]:
    """Convert current geometry findings into exact-waiver identity rows."""

    findings: list[dict[str, object]] = []
    for report in reports:
        for endpoint in report.endpoints:
            if not endpoint.inside_zone:
                findings.append(
                    {
                        "map": canonical_repo_path(report.map_path),
                        "finding_type": "endpoint",
                        "route_kind": endpoint.route_kind,
                        "label": endpoint.label,
                        "end": endpoint.end,
                        "zone_kind": endpoint.zone_kind,
                        "zone_index": endpoint.zone_index,
                        "expected_offset_to_centre_m": endpoint.offset_to_centre_m,
                    }
                )
        for fragment in report.fragments:
            if fragment.disconnected_fragment_count:
                findings.append(
                    {
                        "map": canonical_repo_path(report.map_path),
                        "finding_type": "fragment",
                        "route_kind": fragment.route_kind,
                        "label": fragment.label,
                        "first_disconnected_segment": fragment.first_disconnected_segment,
                        "expected_disconnected_fragment_count": fragment.disconnected_fragment_count,
                    }
                )
        for route_kind in report.missing_zone_kinds:
            findings.append(
                {
                    "map": canonical_repo_path(report.map_path),
                    "finding_type": "missing_zone",
                    "route_kind": route_kind,
                    "expected_route_count": report.route_counts.get(route_kind, 0),
                }
            )
    return findings


def _validate_geometry_waiver_shape(row: dict[str, object], index: int) -> None:
    """Require the exact identity and evidence fields for one geometry waiver."""

    prefix = f"geometry[{index}]"
    for field_name in ("map", "finding_type", "route_kind"):
        if not isinstance(row.get(field_name), str) or not str(row[field_name]).strip():
            raise WaiverValidationError(f"{prefix} requires non-empty {field_name}")
    finding_type = row["finding_type"]
    if finding_type == "endpoint":
        required = ("label", "end", "zone_kind", "zone_index", "expected_offset_to_centre_m")
    elif finding_type == "fragment":
        required = (
            "label",
            "first_disconnected_segment",
            "expected_disconnected_fragment_count",
        )
    elif finding_type == "missing_zone":
        required = ("expected_route_count",)
    else:
        raise WaiverValidationError(f"{prefix} has unsupported finding_type {finding_type!r}")
    missing = [field_name for field_name in required if field_name not in row]
    if missing:
        raise WaiverValidationError(f"{prefix} is missing fields: {', '.join(missing)}")


def _geometry_evidence_matches(actual: dict[str, object], waiver: dict[str, object]) -> bool:
    """Compare the expected measurement/fingerprint fields for one geometry row."""

    finding_type = actual["finding_type"]
    if finding_type == "endpoint":
        return round(float(actual["expected_offset_to_centre_m"]), 3) == round(
            float(waiver["expected_offset_to_centre_m"]), 3
        )
    if finding_type == "fragment":
        return (
            actual["first_disconnected_segment"] == waiver["first_disconnected_segment"]
            and actual["expected_disconnected_fragment_count"]
            == waiver["expected_disconnected_fragment_count"]
        )
    return actual["expected_route_count"] == waiver["expected_route_count"]


def enforce_geometry_waivers(reports: list[MapGeometryReport], waiver_file: Path) -> None:
    """Fail closed unless the exact current geometry findings are waived."""

    waiver_rows = load_waiver_rows(waiver_file, "geometry")
    for index, row in enumerate(waiver_rows):
        _validate_geometry_waiver_shape(row, index)
    validate_exact_waivers(
        _geometry_findings(reports),
        waiver_rows,
        identity_fields=_GEOMETRY_IDENTITY_FIELDS,
        evidence_matches=_geometry_evidence_matches,
        label="geometry",
    )


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


RELEASE_MATRICES = (
    Path("configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"),
    Path("configs/scenarios/francis2023_narrow_doorway_three_width_release_0_0_8_v1.yaml"),
)


def _route_spawn_support(route, half_width: float):
    """Exact nominal support of per-anchor axis-clipped x/y route jitter.

    sample_route draws an anchor anywhere along the route and independently
    clips both coordinate offsets to +/- sidewalk_width/2. Each segment's
    support is its Minkowski sum with an axis-aligned square, not a round buffer.
    Obstacle rejection and live-pose guards are separate runtime constraints.
    """
    return unary_union(
        [
            MultiPoint(
                [
                    (x + dx, y + dy)
                    for x, y in (a, b)
                    for dx in (-half_width, half_width)
                    for dy in (-half_width, half_width)
                ]
            ).convex_hull
            for a, b in pairwise(route.waypoints)
        ]
    )


def _release_actors(
    definition: MapDefinition, density: float, population_size: int | None = None
) -> list[tuple]:
    """Resolve static single-pedestrian lanes and all declared crowd start zones."""
    actors = []
    for ped in definition.single_pedestrians:
        points = [ped.start] + (ped.trajectory or ([ped.goal] if ped.goal else []))
        lane = LineString(points) if len(points) > 1 else Point(points[0])
        actors.append(("single", ped.id, lane, {"points": points, "role": ped.role}))
    from robot_sf.ped_npc.ped_population import PedSpawnConfig

    sidewalk_width = PedSpawnConfig.__dataclass_fields__["sidewalk_width"].default
    for index, route in enumerate(definition.ped_routes):
        actors.append(
            (
                "crowd_route",
                str(index),
                _route_spawn_support(route, sidewalk_width / 2),
                {
                    "points": route.waypoints,
                    "sidewalk_width": sidewalk_width,
                    "density": density,
                    "population_size": population_size,
                },
            )
        )
    for kind, zones in (
        ("ped_spawn", definition.ped_spawn_zones),
        ("crowded", definition.ped_crowded_zones),
    ):
        for index, zone in enumerate(zones):
            actors.append(
                (
                    kind,
                    str(index),
                    _rect_polygon(zone),
                    {"density": density, "population_size": population_size},
                )
            )
    return actors


def inspect_release_zones(matrices=RELEASE_MATRICES) -> list[dict]:
    """Audit every full robot rectangle against resolved actors, without stepping.

    Use the scenario loader so YAML actor/route overrides and geometry contracts
    are applied. Distance <= pedestrian radius includes tangency and round endcaps.
    Report dormant crowd zones too: zero density is a disposition, not an omission.
    The nominal lane test is geometric; it makes no dynamic collision claim.
    """
    from robot_sf.training.scenario_loader import (
        build_robot_config_from_scenario,
        load_scenarios,
    )

    rows = []
    for matrix in matrices:
        matrix = Path(matrix).resolve()
        for scenario in load_scenarios(matrix):
            config = build_robot_config_from_scenario(scenario, scenario_path=matrix)
            if config.map_pool is None or not config.map_pool.map_defs:
                raise ValueError(f"Missing map for {scenario['name']}")
            radius = float(config.sim_config.ped_radius)
            for map_id, definition in sorted(config.map_pool.map_defs.items()):
                actors = _release_actors(
                    definition,
                    config.sim_config.peds_per_area_m2,
                    config.sim_config.population_size,
                )
                for kind, zones in (
                    ("spawn", definition.robot_spawn_zones),
                    ("goal", definition.robot_goal_zones),
                ):
                    for index, zone in enumerate(zones):
                        rectangle = _rect_polygon(zone)
                        hits = []
                        for actor_kind, actor, shape, detail in actors:
                            distance = rectangle.distance(shape)
                            if distance > radius:
                                continue
                            evidence = {
                                "zone_wkt": rectangle.wkt,
                                "actor_wkt": shape.wkt,
                                "ped_radius_m": radius,
                                **detail,
                            }
                            fingerprint = hashlib.sha256(
                                json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
                            ).hexdigest()
                            hits.append(
                                {
                                    "actor_kind": actor_kind,
                                    "actor": actor,
                                    "distance_m": distance,
                                    "geometry_sha256": fingerprint,
                                    "evidence": evidence,
                                }
                            )
                        rows.append(
                            {
                                "matrix": canonical_repo_path(str(matrix)),
                                "scenario": scenario["name"],
                                "map_id": map_id,
                                "zone": f"{kind}[{index}]",
                                "bounds": list(rectangle.bounds),
                                "ped_radius_m": radius,
                                "intersections": hits,
                            }
                        )
    return rows


def enforce_release_zone_waivers(rows: list[dict], waiver_file: Path) -> None:
    """Reject unreviewed, changed or stale endpoint intersections individually."""
    findings = [
        {"matrix": row["matrix"], "scenario": row["scenario"], "zone": row["zone"], **hit}
        for row in rows
        for hit in row["intersections"]
    ]
    waivers = load_waiver_rows(waiver_file, "release_zones")
    fields = ("matrix", "scenario", "zone", "actor_kind", "actor")
    for row in waivers:
        if any(
            not isinstance(row.get(key), str) or not row[key]
            for key in (*fields, "geometry_sha256")
        ):
            raise WaiverValidationError(
                "release zone waiver requires exact identity and geometry_sha256"
            )
    validate_exact_waivers(
        findings,
        waivers,
        identity_fields=fields,
        evidence_matches=lambda actual, waiver: (
            actual["geometry_sha256"] == waiver["geometry_sha256"]
        ),
        label="release zone overlap",
    )


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
    parser.add_argument(
        "--waiver-file",
        type=Path,
        help="Exact waiver YAML required with --fail-on-violation.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the JSON report only.")
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Require exact waivers for every finding (for CI enforcement).",
    )
    parser.add_argument(
        "--release-zones",
        action="store_true",
        help="Audit all 48 release scenarios and three doorway widths.",
    )
    args = parser.parse_args(argv)

    if args.release_zones:
        rows = inspect_release_zones()
        print(json.dumps(rows, indent=2))
        if args.waiver_file is None:
            print("ERROR: --release-zones requires --waiver-file", file=sys.stderr)
            return 2
        try:
            enforce_release_zone_waivers(rows, args.waiver_file)
        except WaiverValidationError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        return 0

    paths = [Path(p) for p in args.map] or [Path(p) for p in DEFAULT_MAPS]
    reports = [inspect_map_geometry(p, args.tolerance_m) for p in paths]
    total = sum(r.violations for r in reports)

    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=1))
    else:
        for report in reports:
            print(format_console_table(report))
        print(f"\ntotal findings: {total}")

    if args.fail_on_violation:
        if args.waiver_file is None:
            print("ERROR: --fail-on-violation requires --waiver-file", file=sys.stderr)
            return 2
        try:
            enforce_geometry_waivers(reports, args.waiver_file)
        except WaiverValidationError as exc:
            print(f"ERROR: geometry waiver validation failed: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
