"""Audit the Francis narrow-doorway envelope binding for issue #6645.

The audit is deliberately diagnostic. It derives the doorway opening from the authored
SVG, checks the exact ``gap_width_m - 2 * envelope_radius_m`` boundary, and delegates
runtime binding checks to the existing issue #6641 Gate 1 canary. It does not alter maps,
defaults, frozen artifacts, or benchmark outputs.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shapely.geometry import LineString, Polygon

from robot_sf.benchmark.camera_ready._route_clearance import _scenario_robot_radius_m
from robot_sf.benchmark.radius_binding_canary import (
    CAMPAIGN_ENVELOPE_RADII_M,
    canary_verdict_to_dict,
    run_radius_binding_canary,
)
from robot_sf.common.robot_defaults import DEFAULT_ROBOT_RADIUS
from robot_sf.gym_env.base_env import attach_planner_to_map
from robot_sf.nav.occupancy_grid import GridConfig
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

AUDIT_SCHEMA = "narrow_doorway_radius_binding_audit.v1"
CLAIM_BOUNDARY = (
    "diagnostic geometry and within-simulator binding audit; not benchmark evidence, "
    "a physical-footprint validation, a realism result, or a safety guarantee"
)
DEFAULT_SCENARIO_PATH = Path("configs/scenarios/single/francis2023_narrow_doorway.yaml")
EXPECTED_SCENARIO_ID = "francis2023_narrow_doorway"
_INKSCAPE_LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"
_TOLERANCE_M = 1e-9


@dataclass(frozen=True, slots=True)
class DoorwayGeometry:
    """Geometry derived from the authored narrow-doorway SVG."""

    map_path: str
    obstacle_rects: tuple[dict[str, float], ...]
    gap_lower_edge_m: float
    gap_upper_edge_m: float
    gap_width_m: float
    route_waypoints: tuple[tuple[float, float], ...]
    route_min_center_distance_m: float


def envelope_clearance_margin_m(gap_width_m: float, envelope_radius_m: float) -> float:
    """Return free transverse clearance after subtracting the circular envelope diameter."""
    gap = float(gap_width_m)
    radius = float(envelope_radius_m)
    if not math.isfinite(gap) or gap <= 0.0:
        raise ValueError("gap_width_m must be finite and positive")
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("envelope_radius_m must be finite and non-negative")
    return gap - 2.0 * radius


def _float_attr(element: ET.Element, name: str) -> float:
    """Read one finite SVG numeric attribute.

    Returns:
        Parsed finite attribute value.
    """
    raw = element.attrib.get(name)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SVG obstacle is missing finite {name!r}: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"SVG obstacle {name!r} must be finite")
    return value


def _obstacle_rectangles(map_path: Path) -> list[dict[str, float]]:
    """Return authored obstacle rectangles from one SVG."""
    root = ET.parse(map_path).getroot()
    rectangles: list[dict[str, float]] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "rect":
            continue
        if element.attrib.get(_INKSCAPE_LABEL) != "obstacle":
            continue
        rectangles.append(
            {
                "x": _float_attr(element, "x"),
                "y": _float_attr(element, "y"),
                "width": _float_attr(element, "width"),
                "height": _float_attr(element, "height"),
            }
        )
    return rectangles


def _resolve_map_path(scenario_path: Path, scenario: dict[str, Any]) -> Path:
    """Resolve the scenario's authored map reference.

    Returns:
        Existing absolute map path.
    """
    raw_map = scenario.get("map_file")
    if not isinstance(raw_map, str) or not raw_map.strip():
        raise ValueError("narrow-doorway scenario must declare map_file")
    map_path = Path(raw_map)
    if not map_path.is_absolute():
        map_path = (scenario_path.parent / map_path).resolve()
    if not map_path.exists():
        raise FileNotFoundError(map_path)
    return map_path


def derive_doorway_geometry(scenario_path: Path, scenario: dict[str, Any]) -> DoorwayGeometry:
    """Derive the opening and route distance from the authored scenario and SVG.

    Returns:
        Geometry and route-distance measurements derived from the source files.
    """
    map_path = _resolve_map_path(scenario_path, scenario)
    rectangles = _obstacle_rectangles(map_path)
    candidates: list[tuple[dict[str, float], dict[str, float]]] = []
    for index, first in enumerate(rectangles):
        for second in rectangles[index + 1 :]:
            if not math.isclose(first["x"], second["x"], abs_tol=_TOLERANCE_M):
                continue
            if not math.isclose(first["width"], second["width"], abs_tol=_TOLERANCE_M):
                continue
            # The opening is bounded by the two vertical doorway segments. This excludes
            # the aligned horizontal room walls, which also form a positive y gap.
            if not (first["height"] > first["width"] and second["height"] > second["width"]):
                continue
            ordered = sorted((first, second), key=lambda item: item["y"])
            lower, upper = ordered
            lower_edge = lower["y"] + lower["height"]
            gap = upper["y"] - lower_edge
            if gap > _TOLERANCE_M:
                candidates.append((lower, upper))
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one positive opening between aligned obstacle rectangles; "
            f"found {len(candidates)}"
        )
    lower, upper = candidates[0]

    map_definition = convert_map(str(map_path))
    if map_definition is None or len(map_definition.robot_routes) != 1:
        raise ValueError("narrow-doorway map must convert to exactly one robot route")
    route = map_definition.robot_routes[0]
    route_line = LineString(route.waypoints)
    obstacle_polygons = [
        Polygon(obstacle.vertices)
        for obstacle in map_definition.obstacles
        if len(obstacle.vertices) >= 3
    ]
    if not obstacle_polygons:
        raise ValueError("narrow-doorway map must contain obstacle geometry")
    route_distance = min(float(route_line.distance(obstacle)) for obstacle in obstacle_polygons)
    return DoorwayGeometry(
        map_path=str(map_path),
        obstacle_rects=(lower, upper),
        gap_lower_edge_m=float(lower["y"] + lower["height"]),
        gap_upper_edge_m=float(upper["y"]),
        gap_width_m=float(upper["y"] - (lower["y"] + lower["height"])),
        route_waypoints=tuple((float(x), float(y)) for x, y in route.waypoints),
        route_min_center_distance_m=route_distance,
    )


def _planner_fallback_radius_m() -> float:
    """Read the documented fallback from the planner attachment source.

    Returns:
        The fallback radius used when the planner has no robot configuration.
    """
    source = inspect.getsource(attach_planner_to_map)
    match = re.search(r'getattr\(robot_cfg, "radius", ([0-9.]+)\)', source)
    if match is None:
        raise ValueError("could not locate the planner radius fallback in attach_planner_to_map")
    return float(match.group(1))


def build_radius_inventory(scenario_path: Path, scenario: dict[str, Any]) -> dict[str, Any]:
    """Capture relevant radius defaults and the effective scenario binding.

    Returns:
        JSON-safe inventory of collision and non-collision radius defaults.
    """
    config = build_robot_config_from_scenario(dict(scenario), scenario_path=scenario_path)
    scenario_radius = float(config.robot_config.radius)
    route_default = float(_scenario_robot_radius_m({}, default=DEFAULT_ROBOT_RADIUS))
    return {
        "authoritative_collision_envelope_default_m": float(DEFAULT_ROBOT_RADIUS),
        "scenario_effective_robot_config_radius_m": scenario_radius,
        "route_clearance_helper_default_m": route_default,
        "oracle_default_m": float(DEFAULT_ROBOT_RADIUS),
        "grid_rasterization_default_m": float(GridConfig().robot_radius),
        "planner_attachment_fallback_m": _planner_fallback_radius_m(),
        "non_collision_defaults_scope": {
            "grid_rasterization_default_m": "optional robot observation channel only",
            "planner_attachment_fallback_m": "legacy planner fallback when no robot config exists",
        },
        "scenario_robot_config_declared": scenario.get("robot_config", {}),
    }


def _sha256(path: Path) -> str:
    """Hash one source file for audit provenance.

    Returns:
        Lower-case SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_audit_report(
    scenario_path: Path,
    *,
    radii_m: tuple[float, ...] = CAMPAIGN_ENVELOPE_RADII_M,
) -> dict[str, Any]:
    """Run the geometry and runtime binding audit and return a JSON-safe report.

    Returns:
        Machine-readable audit report with geometry, radius, and canary checks.
    """
    if not radii_m:
        raise ValueError("radii_m must contain at least one positive radius")
    if any(not math.isfinite(float(radius)) or float(radius) <= 0.0 for radius in radii_m):
        raise ValueError("radii_m must contain only finite positive radii")
    scenario_path = scenario_path.resolve()
    scenarios = [dict(item) for item in load_scenarios(scenario_path)]
    scenario = next((item for item in scenarios if item.get("name") == EXPECTED_SCENARIO_ID), None)
    if scenario is None:
        raise ValueError(f"scenario {EXPECTED_SCENARIO_ID!r} not found in {scenario_path}")
    geometry = derive_doorway_geometry(scenario_path, scenario)
    inventory = build_radius_inventory(scenario_path, scenario)
    boundary_radii = tuple(sorted({0.0, *[float(radius) for radius in radii_m]}))
    boundary = [
        {
            "envelope_radius_m": radius,
            "envelope_diameter_m": 2.0 * radius,
            "clearance_margin_m": envelope_clearance_margin_m(geometry.gap_width_m, radius),
        }
        for radius in boundary_radii
    ]
    canary_reports: list[dict[str, Any]] = []
    for radius in radii_m:
        if float(radius) <= 0.0:
            raise ValueError("runtime canary radii must be positive")
        canary_reports.append(
            canary_verdict_to_dict(
                run_radius_binding_canary(
                    scenario,
                    float(radius),
                    scenario_path=scenario_path,
                )
            )
        )

    geometry_checks = {
        "scenario_id_matches": scenario.get("name") == EXPECTED_SCENARIO_ID,
        "derived_gap_width_m_is_2": math.isclose(geometry.gap_width_m, 2.0, abs_tol=_TOLERANCE_M),
        "route_centerline_distance_matches_radius": math.isclose(
            geometry.route_min_center_distance_m,
            float(inventory["authoritative_collision_envelope_default_m"]),
            abs_tol=_TOLERANCE_M,
        ),
        "zero_clearance_at_1m": math.isclose(
            envelope_clearance_margin_m(geometry.gap_width_m, 1.0),
            0.0,
            abs_tol=_TOLERANCE_M,
        ),
    }
    canary_checks = {
        "all_radii_go": all(report["go"] for report in canary_reports),
        "all_five_surfaces_present": all(len(report["surfaces"]) == 5 for report in canary_reports),
        "all_surface_observations_match": all(
            surface["bound"] for report in canary_reports for surface in report["surfaces"]
        ),
    }
    return {
        "schema": AUDIT_SCHEMA,
        "issue": 6645,
        "parent_issue": 6600,
        "scenario_id": EXPECTED_SCENARIO_ID,
        "scenario_path": str(scenario_path),
        "source_sha256": {
            "scenario": _sha256(scenario_path),
            "map": _sha256(Path(geometry.map_path)),
        },
        "geometry": {
            "map_path": geometry.map_path,
            "obstacle_rects": list(geometry.obstacle_rects),
            "gap_lower_edge_m": geometry.gap_lower_edge_m,
            "gap_upper_edge_m": geometry.gap_upper_edge_m,
            "gap_width_m": geometry.gap_width_m,
            "route_waypoints": [list(point) for point in geometry.route_waypoints],
            "route_min_center_distance_m": geometry.route_min_center_distance_m,
        },
        "radius_inventory": inventory,
        "boundary": boundary,
        "canary": {
            "reports": canary_reports,
            "surfaces": [
                "simulator_collision_geometry",
                "obstacle_pedestrian_contact_logic",
                "feasibility_oracle",
                "metric_metadata_and_output_rows",
                "planner_inputs",
            ],
        },
        "checks": {**geometry_checks, **canary_checks},
        "go": all((*geometry_checks.values(), *canary_checks.values())),
        "claim_boundary": CLAIM_BOUNDARY,
        "release_or_frozen_artifacts_changed": False,
        "interpretation": {
            "zero_margin_is_intentional_boundary": True,
            "positive_margin_does_not_override_grid_oracle_classification": True,
            "runtime_canary_is_not_campaign_evidence": True,
        },
    }


def render_report(report: dict[str, Any]) -> str:
    """Render a stable, sorted JSON report.

    Returns:
        JSON text terminated by one newline.
    """
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


__all__ = [
    "AUDIT_SCHEMA",
    "CLAIM_BOUNDARY",
    "DEFAULT_SCENARIO_PATH",
    "DoorwayGeometry",
    "build_audit_report",
    "build_radius_inventory",
    "derive_doorway_geometry",
    "envelope_clearance_margin_m",
    "render_report",
]
