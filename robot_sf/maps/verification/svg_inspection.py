"""Utilities for inspecting SVG map structure and parser-facing route geometry.

This module provides reusable helpers to inspect SVG maps for issues that are
easy to miss when only looking at rendered geometry:
  - route-only mode detection (routes without explicit spawn/goal rectangles)
  - route label/index mismatches
  - route intersection with obstacle interiors
  - risky SVG path commands for the regex-based path parser
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from math import dist
from pathlib import Path
from typing import TYPE_CHECKING

from shapely.geometry import LineString, Polygon

from robot_sf.nav.svg_map_parser import SvgMapConverter

if TYPE_CHECKING:
    from robot_sf.nav.map_config import MapDefinition

_SVG_NS = {
    "svg": "http://www.w3.org/2000/svg",
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
}
_PATH_COMMAND_PATTERN = re.compile(r"[MmLlHhVvCcQqSsTtAaZz]")
_ROUTE_LABEL_PATTERN = re.compile(r"^(ped_route|robot_route)(?:_(\d+)_(\d+))?$")
_PARSER_RISKY_COMMANDS = frozenset("HhVvCcQqSsTtAa")


@dataclass
class PathCommandInfo:
    """Raw SVG path command metadata for a single `<path>` element.

    Attributes:
        path_id: SVG element id, if present.
        label: Inkscape label, if present.
        commands: Unique path commands used in the `d` attribute.
    """

    path_id: str
    label: str
    commands: list[str]


@dataclass
class RouteSummary:
    """Computed summary for a parsed route.

    Attributes:
        kind: Route kind (`"ped"` or `"robot"`).
        path_id: Source SVG path id when available.
        label: Source SVG route label.
        spawn_id: Parsed spawn index from route label.
        goal_id: Parsed goal index from route label.
        waypoint_count: Number of parsed waypoints.
        route_length: Polyline length from parsed waypoints.
        crosses_obstacle_interior: True when line crosses/inside any obstacle polygon.
        commands: Raw SVG path commands used in the source `d` attribute.
    """

    kind: str
    path_id: str
    label: str
    spawn_id: int
    goal_id: int
    waypoint_count: int
    route_length: float
    crosses_obstacle_interior: bool
    commands: list[str] = field(default_factory=list)


@dataclass
class SvgInspectionFinding:
    """Single inspection finding.

    Attributes:
        severity: `info`, `warning`, or `error`.
        code: Stable machine-readable identifier.
        message: Human-readable finding message.
        path_id: Optional related SVG element id.
    """

    severity: str
    code: str
    message: str
    path_id: str | None = None


@dataclass
class SvgInspectionReport:
    """Inspection report for one SVG file.

    Attributes:
        svg_file: Input SVG path.
        map_width: Parsed map width in meters/units.
        map_height: Parsed map height in meters/units.
        robot_routes: Number of robot routes.
        ped_routes: Number of pedestrian routes.
        robot_spawn_zones: Number of robot spawn zones.
        robot_goal_zones: Number of robot goal zones.
        ped_spawn_zones: Number of ped spawn zones.
        ped_goal_zones: Number of ped goal zones.
        ped_route_only_mode: True when ped routes exist but ped spawn/goal zones are both absent.
        robot_route_only_mode: True when robot routes exist but robot spawn/goal zones are both absent.
        routes: Per-route summaries.
        findings: Collected findings for this SVG.
    """

    svg_file: str
    map_width: float
    map_height: float
    robot_routes: int
    ped_routes: int
    robot_spawn_zones: int
    robot_goal_zones: int
    ped_spawn_zones: int
    ped_goal_zones: int
    ped_route_only_mode: bool
    robot_route_only_mode: bool
    routes: list[RouteSummary] = field(default_factory=list)
    findings: list[SvgInspectionFinding] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize the report as a JSON-serializable dictionary.

        Returns:
            dict: Dictionary representation of the report.
        """

        return asdict(self)

    def max_severity_rank(self) -> int:
        """Return the highest severity rank in the report.

        Returns:
            int: `0` for info-only, `1` when warnings exist, `2` when errors exist.
        """

        rank = {"info": 0, "warning": 1, "error": 2}
        return max((rank.get(f.severity, 0) for f in self.findings), default=0)


def inspect_svg(svg_path: str | Path) -> SvgInspectionReport:
    """Inspect a single SVG file for route and parser-facing map issues.

    Args:
        svg_path: Path to the SVG file.

    Returns:
        SvgInspectionReport: Structured inspection report for the input SVG.
    """

    path = Path(svg_path)
    converter = SvgMapConverter(str(path))
    map_def = converter.get_map_definition()
    path_commands = _collect_path_commands(path)
    path_commands_by_id = {p.path_id: p for p in path_commands if p.path_id}

    report = SvgInspectionReport(
        svg_file=str(path),
        map_width=map_def.width,
        map_height=map_def.height,
        robot_routes=len(map_def.robot_routes),
        ped_routes=len(map_def.ped_routes),
        robot_spawn_zones=len(map_def.robot_spawn_zones),
        robot_goal_zones=len(map_def.robot_goal_zones),
        ped_spawn_zones=len(map_def.ped_spawn_zones),
        ped_goal_zones=len(map_def.ped_goal_zones),
        ped_route_only_mode=bool(map_def.ped_routes)
        and not map_def.ped_spawn_zones
        and not map_def.ped_goal_zones,
        robot_route_only_mode=bool(map_def.robot_routes)
        and not map_def.robot_spawn_zones
        and not map_def.robot_goal_zones,
    )

    _collect_route_summaries(
        report=report,
        map_def=map_def,
        converter=converter,
        path_commands_by_id=path_commands_by_id,
    )
    _add_zone_consistency_findings(report)
    _add_risky_command_findings(report, path_commands)
    _add_obstacle_crossing_findings(report)
    return report


def inspect_svg_targets(target: str | Path, pattern: str = "*.svg") -> list[SvgInspectionReport]:
    """Inspect one SVG file or all matching SVGs in a directory.

    Args:
        target: File path or directory path.
        pattern: Glob pattern used when target is a directory.

    Returns:
        list[SvgInspectionReport]: Inspection reports in deterministic path order.
    """

    path = Path(target)
    if path.is_file():
        return [inspect_svg(path)]
    if path.is_dir():
        return [inspect_svg(svg) for svg in sorted(path.glob(pattern)) if svg.is_file()]
    raise FileNotFoundError(f"SVG target does not exist: {path}")


def _collect_path_commands(svg_path: Path) -> list[PathCommandInfo]:
    """Collect raw path commands from an SVG file.

    Args:
        svg_path: Path to the SVG file.

    Returns:
        list[PathCommandInfo]: Raw command metadata for each SVG path element.
    """

    tree = ET.parse(svg_path)
    root = tree.getroot()
    info: list[PathCommandInfo] = []
    for path in root.findall(".//svg:path", _SVG_NS):
        d_value = path.attrib.get("d", "")
        commands = sorted(set(_PATH_COMMAND_PATTERN.findall(d_value)))
        info.append(
            PathCommandInfo(
                path_id=path.attrib.get("id", ""),
                label=path.attrib.get("{http://www.inkscape.org/namespaces/inkscape}label", ""),
                commands=commands,
            )
        )
    return info


def _collect_route_summaries(
    *,
    report: SvgInspectionReport,
    map_def: MapDefinition,
    converter: SvgMapConverter,
    path_commands_by_id: dict[str, PathCommandInfo],
) -> None:
    """Populate per-route summaries for robot and pedestrian routes.

    Args:
        report: Mutable report object to fill with route summaries.
        map_def: Parsed map definition.
        converter: SVG converter with `path_info` metadata.
        path_commands_by_id: Mapping of path-id to raw command metadata.
    """

    route_labels_by_id = {p.id: p.label for p in converter.path_info}
    route_pairs = [("robot", route) for route in map_def.robot_routes] + [
        ("ped", route) for route in map_def.ped_routes
    ]

    for kind, route in route_pairs:
        route_label = route.source_label or f"{kind}_route_{route.spawn_id}_{route.goal_id}"
        source_path_id = route.source_path_id
        match_id = source_path_id or _find_path_id_by_label(route_labels_by_id, route_label)
        commands = path_commands_by_id.get(match_id, PathCommandInfo("", route_label, [])).commands
        crosses = _route_crosses_obstacle_interior(route.waypoints, map_def)
        report.routes.append(
            RouteSummary(
                kind=kind,
                path_id=match_id or "",
                label=route_label,
                spawn_id=route.spawn_id,
                goal_id=route.goal_id,
                waypoint_count=len(route.waypoints),
                route_length=_polyline_length(route.waypoints),
                crosses_obstacle_interior=crosses,
                commands=commands,
            )
        )


def _find_path_id_by_label(path_labels: dict[str, str], label: str) -> str | None:
    """Find path id by exact label.

    Args:
        path_labels: Mapping of path id to path label.
        label: Route label to find.

    Returns:
        str | None: Matching path id, or None when no match is found.
    """

    for path_id, path_label in path_labels.items():
        if path_label == label:
            return path_id
    return None


def _polyline_length(waypoints: list[tuple[float, float]]) -> float:
    """Compute polyline length for waypoints.

    Args:
        waypoints: Ordered list of polyline points.

    Returns:
        float: Total Euclidean length across consecutive waypoint pairs.
    """

    if len(waypoints) < 2:
        return 0.0
    return float(sum(dist(p1, p2) for p1, p2 in pairwise(waypoints)))


def _route_crosses_obstacle_interior(
    waypoints: list[tuple[float, float]],
    map_def: MapDefinition,
) -> bool:
    """Return True when route crosses or is fully within any obstacle interior.

    Args:
        waypoints: Route polyline waypoints.
        map_def: Map definition containing obstacle polygons.

    Returns:
        bool: True when the route line crosses obstacle interiors.
    """

    if len(waypoints) < 2:
        return False
    line = LineString(waypoints)
    for obstacle in map_def.obstacles:
        poly = Polygon(obstacle.vertices)
        if line.crosses(poly) or line.within(poly):
            return True
    return False


def _add_zone_consistency_findings(report: SvgInspectionReport) -> None:
    """Add findings related to route-only mode and index consistency.

    Args:
        report: Mutable report receiving findings.
    """

    if report.ped_route_only_mode:
        report.findings.append(
            SvgInspectionFinding(
                severity="info",
                code="PED_ROUTE_ONLY_MODE",
                message=(
                    "Pedestrian routes are present without explicit ped_spawn_zone/ped_goal_zone; "
                    "endpoint-derived route-only mode is active."
                ),
            )
        )
    if report.robot_route_only_mode:
        report.findings.append(
            SvgInspectionFinding(
                severity="warning",
                code="ROBOT_ROUTE_ONLY_MODE",
                message=(
                    "Robot routes are present without explicit robot_spawn_zone/robot_goal_zone; "
                    "endpoint-derived zones are being used."
                ),
            )
        )

    if report.ped_routes and bool(report.ped_spawn_zones) != bool(report.ped_goal_zones):
        report.findings.append(
            SvgInspectionFinding(
                severity="warning",
                code="PARTIAL_PED_ZONE_DEFINITIONS",
                message="Pedestrian routes are defined but only one of spawn/goal zone sets is present.",
            )
        )
    if report.robot_routes and bool(report.robot_spawn_zones) != bool(report.robot_goal_zones):
        report.findings.append(
            SvgInspectionFinding(
                severity="warning",
                code="PARTIAL_ROBOT_ZONE_DEFINITIONS",
                message="Robot routes are defined but only one of spawn/goal zone sets is present.",
            )
        )

    for route in report.routes:
        if route.kind == "ped":
            _validate_route_indices_against_zones(
                report=report,
                route=route,
                max_spawn=report.ped_spawn_zones - 1,
                max_goal=report.ped_goal_zones - 1,
            )
        elif route.kind == "robot":
            _validate_route_indices_against_zones(
                report=report,
                route=route,
                max_spawn=report.robot_spawn_zones - 1,
                max_goal=report.robot_goal_zones - 1,
            )


def _validate_route_indices_against_zones(
    *,
    report: SvgInspectionReport,
    route: RouteSummary,
    max_spawn: int,
    max_goal: int,
) -> None:
    """Add index mismatch findings for a single route.

    Args:
        report: Mutable report receiving findings.
        route: Route summary to validate.
        max_spawn: Maximum valid spawn index; negative means no zones.
        max_goal: Maximum valid goal index; negative means no zones.
    """

    if max_spawn >= 0 and not (0 <= route.spawn_id <= max_spawn):
        report.findings.append(
            SvgInspectionFinding(
                severity="error",
                code="ROUTE_SPAWN_INDEX_OUT_OF_RANGE",
                message=(
                    f"Route {route.label} uses spawn index {route.spawn_id}, "
                    f"but max available spawn index is {max_spawn}."
                ),
                path_id=route.path_id or None,
            )
        )
    if max_goal >= 0 and not (0 <= route.goal_id <= max_goal):
        report.findings.append(
            SvgInspectionFinding(
                severity="error",
                code="ROUTE_GOAL_INDEX_OUT_OF_RANGE",
                message=(
                    f"Route {route.label} uses goal index {route.goal_id}, "
                    f"but max available goal index is {max_goal}."
                ),
                path_id=route.path_id or None,
            )
        )


def _add_risky_command_findings(
    report: SvgInspectionReport,
    path_commands: list[PathCommandInfo],
) -> None:
    """Add findings for SVG path commands that are risky for regex parsing.

    Args:
        report: Mutable report receiving findings.
        path_commands: Raw path command metadata from the source SVG.
    """

    for info in path_commands:
        if not info.label:
            continue
        match = _ROUTE_LABEL_PATTERN.match(info.label)
        if not match:
            continue
        risky = sorted(set(info.commands).intersection(_PARSER_RISKY_COMMANDS))
        if risky:
            report.findings.append(
                SvgInspectionFinding(
                    severity="warning",
                    code="ROUTE_USES_RISKY_PATH_COMMANDS",
                    message=(
                        f"Route path '{info.label}' uses commands {risky}; "
                        "regex waypoint extraction may not match intended geometry."
                    ),
                    path_id=info.path_id or None,
                )
            )
        if any(cmd.islower() for cmd in info.commands):
            report.findings.append(
                SvgInspectionFinding(
                    severity="warning",
                    code="ROUTE_USES_RELATIVE_COMMANDS",
                    message=(
                        f"Route path '{info.label}' uses relative commands {info.commands}; "
                        "prefer absolute commands for predictable parser behavior."
                    ),
                    path_id=info.path_id or None,
                )
            )


def _add_obstacle_crossing_findings(report: SvgInspectionReport) -> None:
    """Add findings when routes cross obstacle interiors.

    Args:
        report: Mutable report receiving findings.
    """

    for route in report.routes:
        if route.crosses_obstacle_interior:
            report.findings.append(
                SvgInspectionFinding(
                    severity="error",
                    code="ROUTE_CROSSES_OBSTACLE_INTERIOR",
                    message=f"Route {route.label} crosses obstacle interior.",
                    path_id=route.path_id or None,
                )
            )
