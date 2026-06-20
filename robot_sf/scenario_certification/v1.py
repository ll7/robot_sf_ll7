"""Versioned scenario certification for benchmark-facing scenario manifests."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from shapely.geometry import GeometryCollection, LineString, Point, Polygon, box
from shapely.ops import unary_union

from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.planner.classic_global_planner import (
    ClassicGlobalPlanner,
    ClassicPlannerConfig,
    PlanningError,
)
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.robot.holonomic_drive import HolonomicDriveSettings
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

if TYPE_CHECKING:
    from pathlib import Path

    from robot_sf.common.types import Vec2D
    from robot_sf.nav.global_route import GlobalRoute
    from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition

CERT_SCHEMA_VERSION = "scenario_cert.v1"

VALID = "valid"
INVALID = "invalid"
GEOMETRICALLY_INFEASIBLE = "geometrically_infeasible"
KINODYNAMICALLY_INFEASIBLE = "kinodynamically_infeasible"
DYNAMICALLY_OVERCONSTRAINED = "dynamically_overconstrained"
KNIFE_EDGE = "knife_edge"
HARD_BUT_SOLVABLE = "hard_but_solvable"

EXCLUDED_STATUSES = {
    INVALID,
    GEOMETRICALLY_INFEASIBLE,
    KINODYNAMICALLY_INFEASIBLE,
    DYNAMICALLY_OVERCONSTRAINED,
}

_STATUS_SEVERITY = {
    INVALID: 60,
    GEOMETRICALLY_INFEASIBLE: 50,
    KINODYNAMICALLY_INFEASIBLE: 40,
    DYNAMICALLY_OVERCONSTRAINED: 30,
    KNIFE_EDGE: 20,
    HARD_BUT_SOLVABLE: 10,
    VALID: 0,
}


@dataclass(frozen=True)
class CertificationSettings:
    """Tunable thresholds for ``scenario_cert.v1`` classification."""

    planner_cells_per_meter: float = 2.0
    knife_edge_clearance_margin_m: float = 0.15
    hard_clearance_margin_m: float = 0.35
    hard_path_length_ratio: float = 1.5
    knife_edge_path_length_ratio: float = 2.5
    hard_turn_count: int = 3


@dataclass(frozen=True)
class RouteCertificate:
    """Certificate evidence for one applicable robot route."""

    route_id: str
    spawn_id: int
    goal_id: int
    classification: str
    benchmark_eligibility: str
    reasons: list[str]
    checks: dict[str, Any]
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioCertificate:
    """Machine-readable scenario certificate for benchmark inclusion decisions."""

    schema_version: str
    scenario_id: str
    source: str
    classification: str
    benchmark_eligibility: str
    reasons: list[str]
    checks: dict[str, Any]
    route_certificates: list[RouteCertificate]
    evidence: dict[str, Any] = field(default_factory=dict)


def certify_scenario_file(
    scenario_path: Path,
    *,
    scenario_id: str | None = None,
    settings: CertificationSettings | None = None,
) -> list[ScenarioCertificate]:
    """Certify all scenarios, or one selected scenario, from a scenario manifest.

    Returns:
        List of certificates in manifest order, or a single selected certificate.
    """

    scenarios = load_scenarios(scenario_path)
    selected = [
        scenario
        for scenario in scenarios
        if scenario_id is None or _scenario_id(scenario).lower() == scenario_id.lower()
    ]
    if scenario_id is not None and not selected:
        raise ValueError(f"Scenario id '{scenario_id}' not found in {scenario_path}")
    return [
        certify_scenario(scenario, scenario_path=scenario_path, settings=settings)
        for scenario in selected
    ]


def certify_scenario(
    scenario: Mapping[str, Any],
    *,
    scenario_path: Path,
    settings: CertificationSettings | None = None,
) -> ScenarioCertificate:
    """Build a ``scenario_cert.v1`` certificate from a scenario-loader entry.

    Returns:
        Scenario certificate with fail-closed classification and evidence.
    """

    cert_settings = settings or CertificationSettings()
    sid = _scenario_id(scenario)
    try:
        config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
    except Exception as exc:  # noqa: BLE001 - certificate must fail closed on loader errors.
        return _invalid_scenario_certificate(
            scenario,
            scenario_id=sid,
            source=scenario_path.as_posix(),
            reason=f"scenario_loader_error: {exc}",
        )

    map_defs = list(config.map_pool.map_defs.items())
    if not map_defs:
        return _invalid_scenario_certificate(
            scenario,
            scenario_id=sid,
            source=scenario_path.as_posix(),
            reason="map_pool_empty",
        )

    route_certs: list[RouteCertificate] = []
    for map_name, map_def in map_defs:
        route_certs.extend(
            _certify_routes_for_map(
                map_def,
                scenario=scenario,
                map_name=map_name,
                config=config,
                settings=cert_settings,
            )
        )

    if not route_certs:
        return _invalid_scenario_certificate(
            scenario,
            scenario_id=sid,
            source=scenario_path.as_posix(),
            reason="no_applicable_robot_routes",
        )
    return _aggregate_scenario_certificate(
        scenario,
        scenario_id=sid,
        source=scenario_path.as_posix(),
        route_certs=route_certs,
        settings=cert_settings,
    )


def certify_map_definition(
    map_def: MapDefinition,
    *,
    scenario: Mapping[str, Any] | None = None,
    source: str = "programmatic",
    robot_config: DifferentialDriveSettings
    | BicycleDriveSettings
    | HolonomicDriveSettings
    | None = None,
    sim_config: Any | None = None,
    settings: CertificationSettings | None = None,
) -> ScenarioCertificate:
    """Certify a programmatic map definition without loading a YAML scenario.

    Returns:
        Scenario certificate for the supplied map definition and optional scenario payload.
    """

    scenario_payload = scenario or {"name": "programmatic"}
    config = RobotSimulationConfig(map_pool=MapDefinitionPool(map_defs={"programmatic": map_def}))
    if robot_config is not None:
        config.robot_config = robot_config
    if sim_config is not None:
        config.sim_config = sim_config
    route_certs = _certify_routes_for_map(
        map_def,
        scenario=scenario_payload,
        map_name="programmatic",
        config=config,
        settings=settings or CertificationSettings(),
    )
    if not route_certs:
        return _invalid_scenario_certificate(
            scenario_payload,
            scenario_id=_scenario_id(scenario_payload),
            source=source,
            reason="no_applicable_robot_routes",
        )
    return _aggregate_scenario_certificate(
        scenario_payload,
        scenario_id=_scenario_id(scenario_payload),
        source=source,
        route_certs=route_certs,
        settings=settings or CertificationSettings(),
    )


def certificate_to_dict(certificate: ScenarioCertificate) -> dict[str, Any]:
    """Convert a certificate dataclass into JSON-safe primitives.

    Returns:
        JSON-serializable dictionary representation of the certificate.
    """

    return _sanitize_json_value(asdict(certificate))


def _certify_routes_for_map(
    map_def: MapDefinition,
    *,
    scenario: Mapping[str, Any],
    map_name: str,
    config: RobotSimulationConfig,
    settings: CertificationSettings,
) -> list[RouteCertificate]:
    """Certify the scenario-relevant robot routes for one parsed map.

    Returns:
        list[RouteCertificate]: Per-route certification records.
    """
    routes = _applicable_routes(map_def, scenario)
    return [
        _certify_route(
            route,
            map_def=map_def,
            map_name=map_name,
            config=config,
            settings=settings,
        )
        for route in routes
    ]


def _certify_route(
    route: GlobalRoute,
    *,
    map_def: MapDefinition,
    map_name: str,
    config: RobotSimulationConfig,
    settings: CertificationSettings,
) -> RouteCertificate:
    """Run geometric, kinodynamic, and dynamic checks for one route.

    Returns:
        RouteCertificate: Route classification, eligibility, checks, and evidence.
    """
    route_id = route.source_label or f"robot_route_{route.spawn_id}_{route.goal_id}"
    reasons: list[str] = []
    checks: dict[str, Any] = {"map_name": map_name}
    evidence: dict[str, Any] = {}
    robot_radius = float(getattr(config.robot_config, "radius", 1.0))
    ped_radius = float(getattr(config.sim_config, "ped_radius", 0.4))
    start, goal = _route_start_goal(route)
    route_line = _line_from_points(route.waypoints)
    obstacle_union = _obstacle_union(map_def)
    map_bounds = box(0.0, 0.0, float(map_def.width), float(map_def.height))

    checks.update(
        {
            "robot_radius_m": robot_radius,
            "ped_radius_m": ped_radius,
            "start": list(start) if start else None,
            "goal": list(goal) if goal else None,
            "waypoint_count": len(route.waypoints),
        }
    )

    invalid_reasons = _validate_route_shape(route, start, goal, map_bounds, obstacle_union)
    if invalid_reasons:
        reasons.extend(invalid_reasons)
        return _route_certificate(
            route,
            route_id=route_id,
            status=INVALID,
            reasons=reasons,
            checks=checks,
            evidence=evidence,
        )

    assert start is not None
    assert goal is not None
    assert route_line is not None
    direct_length = float(Point(start).distance(Point(goal)))
    checks["direct_start_goal_distance_m"] = direct_length
    checks["authored_route_length_m"] = _polyline_length(route.waypoints)
    checks["minimum_static_clearance_m"] = _minimum_static_clearance(
        route_line,
        obstacle_union=obstacle_union,
        robot_radius=robot_radius,
    )

    planned_path, planner_info, planner_error = _plan_inflated_shortest_path(
        map_def,
        start=start,
        goal=goal,
        robot_radius=robot_radius,
        settings=settings,
    )
    if planned_path is None:
        reasons.append(f"no_inflated_collision_free_path: {planner_error}")
        checks["inflated_collision_free_path"] = False
        return _route_certificate(
            route,
            route_id=route_id,
            status=GEOMETRICALLY_INFEASIBLE,
            reasons=reasons,
            checks=checks,
            evidence=evidence,
        )

    planned_line = _line_from_points(planned_path)
    shortest_length = _polyline_length(planned_path)
    path_length_ratio = _safe_ratio(shortest_length, direct_length)
    checks.update(
        {
            "inflated_collision_free_path": True,
            "shortest_path_length_m": shortest_length,
            "path_length_ratio": path_length_ratio,
            "planner": planner_info,
            "planned_waypoint_count": len(planned_path),
            "planned_turn_count": _turn_count(planned_path),
        }
    )

    kinodynamic_reasons, kinodynamic_checks = _kinodynamic_checks(
        route,
        robot_config=config.robot_config,
    )
    checks["kinodynamic"] = kinodynamic_checks
    if kinodynamic_reasons:
        reasons.extend(kinodynamic_reasons)
        return _route_certificate(
            route,
            route_id=route_id,
            status=KINODYNAMICALLY_INFEASIBLE,
            reasons=reasons,
            checks=checks,
            evidence=evidence,
        )

    infrastructure_reasons, infrastructure_checks = _infrastructure_checks(
        map_def,
        route_line=route_line,
    )
    checks["infrastructure"] = infrastructure_checks
    if infrastructure_reasons:
        reasons.extend(infrastructure_reasons)
        return _route_certificate(
            route,
            route_id=route_id,
            status=INVALID,
            reasons=reasons,
            checks=checks,
            evidence=evidence,
        )

    dynamic_reasons, dynamic_checks = _dynamic_checks(
        map_def,
        planned_line=planned_line,
        robot_radius=robot_radius,
        ped_radius=ped_radius,
    )
    checks["dynamic"] = dynamic_checks
    if dynamic_reasons:
        reasons.extend(dynamic_reasons)
        return _route_certificate(
            route,
            route_id=route_id,
            status=DYNAMICALLY_OVERCONSTRAINED,
            reasons=reasons,
            checks=checks,
            evidence=evidence,
        )

    route_status = _classify_solvable_route(checks, settings=settings)
    reasons.extend(_solvable_reasons(route_status, checks, settings=settings))
    evidence["difficulty_analysis"] = {
        "source": "docs/context/issue_692_scenario_difficulty_analysis.md",
        "role": "optional_diagnostic_evidence_only",
    }
    return _route_certificate(
        route,
        route_id=route_id,
        status=route_status,
        reasons=reasons,
        checks=checks,
        evidence=evidence,
    )


def _aggregate_scenario_certificate(
    scenario: Mapping[str, Any],
    *,
    scenario_id: str,
    source: str,
    route_certs: list[RouteCertificate],
    settings: CertificationSettings,
) -> ScenarioCertificate:
    """Aggregate per-route certificates into one scenario-level certificate.

    Returns:
        ScenarioCertificate: Scenario classification and route evidence summary.
    """
    status = max(route_certs, key=lambda cert: _STATUS_SEVERITY[cert.classification]).classification
    reasons = sorted({reason for cert in route_certs for reason in cert.reasons})
    checks = {
        "route_count": len(route_certs),
        "settings": asdict(settings),
        "all_routes_benchmark_eligible": all(
            cert.benchmark_eligibility == "eligible" for cert in route_certs
        ),
    }
    evidence = _scenario_evidence(scenario)
    return ScenarioCertificate(
        schema_version=CERT_SCHEMA_VERSION,
        scenario_id=scenario_id,
        source=source,
        classification=status,
        benchmark_eligibility=_benchmark_eligibility(status),
        reasons=reasons,
        checks=checks,
        route_certificates=route_certs,
        evidence=evidence,
    )


def _invalid_scenario_certificate(
    scenario: Mapping[str, Any],
    *,
    scenario_id: str,
    source: str,
    reason: str,
) -> ScenarioCertificate:
    """Build an excluded scenario certificate when route certification cannot run.

    Returns:
        ScenarioCertificate: Invalid certificate carrying the supplied exclusion reason.
    """
    return ScenarioCertificate(
        schema_version=CERT_SCHEMA_VERSION,
        scenario_id=scenario_id,
        source=source,
        classification=INVALID,
        benchmark_eligibility="excluded",
        reasons=[reason],
        checks={"route_count": 0},
        route_certificates=[],
        evidence=_scenario_evidence(scenario),
    )


def _route_certificate(
    route: GlobalRoute,
    *,
    route_id: str,
    status: str,
    reasons: list[str],
    checks: dict[str, Any],
    evidence: dict[str, Any],
) -> RouteCertificate:
    """Build a route certificate with benchmark eligibility derived from status.

    Returns:
        RouteCertificate: JSON-safe route certification payload.
    """
    return RouteCertificate(
        route_id=route_id,
        spawn_id=int(route.spawn_id),
        goal_id=int(route.goal_id),
        classification=status,
        benchmark_eligibility=_benchmark_eligibility(status),
        reasons=reasons,
        checks=checks,
        evidence=evidence,
    )


def _benchmark_eligibility(status: str) -> str:
    """Map a certification status to benchmark eligibility.

    Returns:
        str: ``excluded``, ``stress_only``, or ``eligible``.
    """
    if status in EXCLUDED_STATUSES:
        return "excluded"
    if status == KNIFE_EDGE:
        return "stress_only"
    return "eligible"


def _scenario_id(scenario: Mapping[str, Any]) -> str:
    """Resolve the stable scenario identifier from common manifest fields.

    Returns:
        str: Non-empty scenario id, or ``"unknown"``.
    """
    raw = scenario.get("name") or scenario.get("scenario_id") or scenario.get("id")
    return str(raw or "unknown").strip() or "unknown"


def _scenario_evidence(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Extract reusable provenance and plausibility evidence from a scenario.

    Returns:
        dict[str, Any]: JSON-safe evidence block for scenario certificates.
    """
    metadata = scenario.get("metadata")
    evidence: dict[str, Any] = {"scenario_fingerprint": _fingerprint_mapping(scenario)}
    if isinstance(metadata, Mapping):
        plausibility = metadata.get("plausibility")
        if isinstance(plausibility, Mapping):
            evidence["plausibility"] = _sanitize_json_value(dict(plausibility))
        for key in ("archetype", "primary_capability", "target_failure_mode", "determinism"):
            if key in metadata:
                evidence[key] = _sanitize_json_value(metadata[key])
    evidence["difficulty_analysis"] = {
        "source": "docs/context/issue_692_scenario_difficulty_analysis.md",
        "role": "optional_diagnostic_evidence_only",
    }
    return evidence


def _fingerprint_mapping(payload: Mapping[str, Any]) -> str:
    """Fingerprint a mapping after JSON-safe normalization.

    Returns:
        str: Short SHA-256 fingerprint for provenance comparisons.
    """
    encoded = json.dumps(_sanitize_json_value(dict(payload)), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _applicable_routes(map_def: MapDefinition, scenario: Mapping[str, Any]) -> list[GlobalRoute]:
    """Select routes that apply to the scenario override or full map.

    Returns:
        list[GlobalRoute]: Scenario-specific route when configured, otherwise all robot routes.
    """
    spawn_id = scenario.get("robot_spawn_id")
    goal_id = scenario.get("robot_goal_id")
    if isinstance(spawn_id, int) and isinstance(goal_id, int):
        route = map_def.find_route(spawn_id, goal_id)
        return [route] if route is not None else []
    return list(map_def.robot_routes)


def _route_start_goal(route: GlobalRoute) -> tuple[Vec2D | None, Vec2D | None]:
    """Return start and goal waypoints when a route has enough points.

    Returns:
        tuple[Vec2D | None, Vec2D | None]: Start and goal, or ``None`` pair.
    """
    if len(route.waypoints) < 2:
        return None, None
    return tuple(route.waypoints[0]), tuple(route.waypoints[-1])


def _validate_route_shape(
    route: GlobalRoute,
    start: Vec2D | None,
    goal: Vec2D | None,
    map_bounds: Polygon,
    obstacle_union: Any,
) -> list[str]:
    """Validate basic route geometry before planner-based certification.

    Returns:
        list[str]: Invalidity reasons; empty when the route shape is usable.
    """
    reasons: list[str] = []
    if start is None or goal is None:
        return ["route_requires_at_least_two_waypoints"]
    for label, point in (("start", start), ("goal", goal)):
        if not _finite_point(point):
            reasons.append(f"{label}_point_not_finite")
            continue
        if not map_bounds.covers(Point(point)):
            reasons.append(f"{label}_outside_map_bounds")
        if not obstacle_union.is_empty and obstacle_union.covers(Point(point)):
            reasons.append(f"{label}_inside_static_obstacle")
    for idx, waypoint in enumerate(route.waypoints):
        if not _finite_point(waypoint):
            reasons.append(f"waypoint_{idx}_not_finite")
    return reasons


def _finite_point(point: Vec2D) -> bool:
    """Return whether a point has two finite coordinates."""
    return len(point) == 2 and all(math.isfinite(float(coord)) for coord in point)


def _obstacle_union(map_def: MapDefinition) -> Any:
    """Union all obstacle polygons from a map definition.

    Returns:
        Geometry collection or unioned polygonal geometry.
    """
    polygons = []
    for obstacle in map_def.obstacles:
        polygons.extend(obstacle.iter_polygons())
    if not polygons:
        return GeometryCollection()
    return unary_union(polygons)


def _line_from_points(points: list[Vec2D]) -> LineString | None:
    """Build a valid Shapely line from route points.

    Returns:
        LineString | None: Valid route line, or ``None`` when unusable.
    """
    if len(points) < 2:
        return None
    line = LineString(points)
    if line.is_empty or not line.is_valid:
        return None
    return line


def _minimum_static_clearance(
    line: LineString,
    *,
    obstacle_union: Any,
    robot_radius: float,
) -> float | None:
    """Compute route clearance from static obstacles after robot-radius inflation.

    Returns:
        float | None: Clearance margin in meters, or ``None`` without obstacles.
    """
    if obstacle_union.is_empty:
        return None
    return float(line.distance(obstacle_union) - robot_radius)


def _plan_inflated_shortest_path(
    map_def: MapDefinition,
    *,
    start: Vec2D,
    goal: Vec2D,
    robot_radius: float,
    settings: CertificationSettings,
) -> tuple[list[Vec2D] | None, dict[str, Any], str | None]:
    """Plan an inflated A* path used as geometric feasibility proof.

    Returns:
        tuple[list[Vec2D] | None, dict[str, Any], str | None]: Planned path,
        planner metadata, and optional error text.
    """
    planner_config = ClassicPlannerConfig(
        cells_per_meter=settings.planner_cells_per_meter,
        inflate_radius_meters=robot_radius,
        algorithm="a_star",
    )
    planner = ClassicGlobalPlanner(map_def, planner_config)
    try:
        path, info = planner.plan(start, goal, algorithm="a_star", allow_inflation_fallback=False)
    except (PlanningError, ValueError, RuntimeError) as exc:
        return None, {"algorithm": "a_star", "inflation_fallback": False}, str(exc)
    if not path:
        return None, {"algorithm": "a_star", "inflation_fallback": False}, "empty_path"
    planner_info = {
        "algorithm": "a_star",
        "cells_per_meter": settings.planner_cells_per_meter,
        "inflation_radius_m": robot_radius,
        "inflation_fallback": False,
    }
    if isinstance(info, Mapping):
        planner_info["raw_info"] = _sanitize_json_value(dict(info))
    return [(float(x), float(y)) for x, y in path], planner_info, None


def _kinodynamic_checks(
    route: GlobalRoute,
    *,
    robot_config: DifferentialDriveSettings | BicycleDriveSettings | HolonomicDriveSettings,
) -> tuple[list[str], dict[str, Any]]:
    """Check route compatibility with the configured robot kinematics.

    Returns:
        tuple[list[str], dict[str, Any]]: Infeasibility reasons and check metadata.
    """
    checks: dict[str, Any] = {
        "robot_model": type(robot_config).__name__,
        "command_limits_valid": True,
    }
    reasons: list[str] = []
    if isinstance(robot_config, BicycleDriveSettings):
        max_steer = float(robot_config.max_steer)
        if max_steer <= 0:
            checks["command_limits_valid"] = False
            return ["bicycle_max_steer_non_positive"], checks
        min_turning_radius = float(robot_config.wheelbase / math.tan(max_steer))
        route_min_radius = _minimum_route_turn_radius(route.waypoints)
        checks.update(
            {
                "minimum_turning_radius_m": min_turning_radius,
                "route_minimum_turn_radius_m": route_min_radius,
            }
        )
        if route_min_radius is not None and route_min_radius < min_turning_radius:
            reasons.append(
                "route_turn_radius_below_bicycle_limit: "
                f"{route_min_radius:.3f} < {min_turning_radius:.3f}"
            )
    elif isinstance(robot_config, DifferentialDriveSettings):
        checks["minimum_turning_radius_m"] = 0.0
        checks["controller_semantics"] = "can_rotate_in_place"
    elif isinstance(robot_config, HolonomicDriveSettings):
        checks["minimum_turning_radius_m"] = 0.0
        checks["controller_semantics"] = robot_config.command_mode
    else:
        checks["command_limits_valid"] = False
        reasons.append(f"unsupported_robot_config: {type(robot_config).__name__}")
    return reasons, checks


def _minimum_route_turn_radius(points: list[Vec2D]) -> float | None:
    """Estimate the tightest turn radius across route triplets.

    Returns:
        float | None: Minimum finite circumradius, or ``None`` for straight/short routes.
    """
    radii: list[float] = []
    for p0, p1, p2 in zip(points[:-2], points[1:-1], points[2:], strict=False):
        radius = _circumradius(p0, p1, p2)
        if radius is not None:
            radii.append(radius)
    return min(radii) if radii else None


def _circumradius(p0: Vec2D, p1: Vec2D, p2: Vec2D) -> float | None:
    """Compute the circumradius through three route points.

    Returns:
        float | None: Circumradius, or ``None`` for degenerate triplets.
    """
    a = math.dist(p1, p2)
    b = math.dist(p0, p2)
    c = math.dist(p0, p1)
    area2 = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    if a <= 1e-9 or b <= 1e-9 or c <= 1e-9 or area2 <= 1e-9:
        return None
    return float((a * b * c) / (2.0 * area2))


def _infrastructure_checks(
    map_def: MapDefinition,
    *,
    route_line: LineString,
) -> tuple[list[str], dict[str, Any]]:
    """Check certification-only public-space restrictions for AMV routes.

    Returns:
        tuple[list[str], dict[str, Any]]: Invalidity reasons and diagnostics.
    """
    zones = list(getattr(map_def, "infrastructure_zones", []))
    checks: dict[str, Any] = {
        "zone_count": len(zones),
        "intersected_zone_ids": [],
        "illegal_amv_zone_ids": [],
    }
    if not zones:
        return [], checks

    illegal: list[dict[str, str]] = []
    intersected: list[str] = []
    for zone in zones:
        if not route_line.intersects(zone.polygon()):
            continue
        intersected.append(zone.id)
        allowed = set(zone.allowed_actor_types)
        if allowed.isdisjoint({"all", "amv", "robot", "vehicle"}):
            illegal.append({"id": zone.id, "zone_type": zone.zone_type})

    checks["intersected_zone_ids"] = intersected
    checks["illegal_amv_zone_ids"] = [entry["id"] for entry in illegal]
    checks["illegal_amv_zones"] = illegal
    if not illegal:
        return [], checks
    labels = ", ".join(f"{entry['id']}({entry['zone_type']})" for entry in illegal)
    return [f"illegal_amv_infrastructure_traversal: {labels}"], checks


def _dynamic_checks(
    map_def: MapDefinition,
    *,
    planned_line: LineString | None,
    robot_radius: float,
    ped_radius: float,
) -> tuple[list[str], dict[str, Any]]:
    """Check static pedestrian blockers and dynamic interaction presence.

    Returns:
        tuple[list[str], dict[str, Any]]: Dynamic overconstraint reasons and diagnostics.
    """
    pedestrians = list(getattr(map_def, "single_pedestrians", []))
    checks = {
        "single_pedestrian_count": len(pedestrians),
        "static_blocking_pedestrian_ids": [],
    }
    if planned_line is None or not pedestrians:
        return [], checks
    blocking_threshold = robot_radius + ped_radius
    blocking: list[str] = []
    dynamic_count = 0
    for ped in pedestrians:
        if _pedestrian_is_static(ped):
            if planned_line.distance(Point(ped.start)) <= blocking_threshold:
                blocking.append(str(ped.id))
        else:
            dynamic_count += 1
    checks["dynamic_interaction_count"] = dynamic_count
    checks["static_blocking_pedestrian_ids"] = blocking
    if blocking:
        return [f"static_pedestrians_block_route: {', '.join(blocking)}"], checks
    return [], checks


def _pedestrian_is_static(pedestrian: SinglePedestrianDefinition) -> bool:
    """Return whether a pedestrian definition has no dynamic target or role."""
    return (
        pedestrian.goal is None
        and pedestrian.trajectory is None
        and getattr(pedestrian, "role", None) is None
    )


def _classify_solvable_route(checks: Mapping[str, Any], *, settings: CertificationSettings) -> str:
    """Classify a feasible route by clearance, detour, turns, and dynamics.

    Returns:
        str: Certification status for a solvable route.
    """
    clearance = checks.get("minimum_static_clearance_m")
    ratio = checks.get("path_length_ratio")
    dynamic = checks.get("dynamic")
    dynamic_count = (
        int(dynamic.get("dynamic_interaction_count", 0)) if isinstance(dynamic, Mapping) else 0
    )
    turn_count = int(checks.get("planned_turn_count", 0))
    if isinstance(clearance, (int, float)) and clearance <= settings.knife_edge_clearance_margin_m:
        return KNIFE_EDGE
    if isinstance(ratio, (int, float)) and ratio >= settings.knife_edge_path_length_ratio:
        return KNIFE_EDGE
    if isinstance(clearance, (int, float)) and clearance <= settings.hard_clearance_margin_m:
        return HARD_BUT_SOLVABLE
    if isinstance(ratio, (int, float)) and ratio >= settings.hard_path_length_ratio:
        return HARD_BUT_SOLVABLE
    if turn_count >= settings.hard_turn_count:
        return HARD_BUT_SOLVABLE
    if dynamic_count > 0:
        return HARD_BUT_SOLVABLE
    return VALID


def _solvable_reasons(
    status: str,
    checks: Mapping[str, Any],
    *,
    settings: CertificationSettings,
) -> list[str]:
    """Explain why a solvable route received its hardness classification.

    Returns:
        list[str]: Human-readable certification reasons.
    """
    if status == VALID:
        return ["inflated_path_found_with_nominal_clearance"]
    if status == KNIFE_EDGE:
        return ["solvable_but_near_certification_threshold"]
    reasons: list[str] = []
    clearance = checks.get("minimum_static_clearance_m")
    ratio = checks.get("path_length_ratio")
    dynamic = checks.get("dynamic")
    if isinstance(clearance, (int, float)) and clearance <= settings.hard_clearance_margin_m:
        reasons.append("low_static_clearance_margin")
    if isinstance(ratio, (int, float)) and ratio >= settings.hard_path_length_ratio:
        reasons.append("long_detour_relative_to_direct_distance")
    if isinstance(dynamic, Mapping) and int(dynamic.get("dynamic_interaction_count", 0)) > 0:
        reasons.append("dynamic_interaction_present")
    if not reasons:
        reasons.append("solvable_with_hardness_caveat")
    return reasons


def _polyline_length(points: list[Vec2D]) -> float:
    """Compute total route polyline length.

    Returns:
        float: Sum of consecutive segment lengths.
    """
    if len(points) < 2:
        return 0.0
    return float(sum(math.dist(a, b) for a, b in pairwise(points)))


def _turn_count(points: list[Vec2D]) -> int:
    """Count non-collinear interior turns in a waypoint sequence.

    Returns:
        int: Number of detected turns.
    """
    turns = 0
    for p0, p1, p2 in zip(points[:-2], points[1:-1], points[2:], strict=False):
        dx1 = p1[0] - p0[0]
        dy1 = p1[1] - p0[1]
        dx2 = p2[0] - p1[0]
        dy2 = p2[1] - p1[1]
        if abs(dx1 * dy2 - dy1 * dx2) > 1e-9:
            turns += 1
    return turns


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    """Divide two positive-scale quantities with zero-denominator protection.

    Returns:
        float | None: Ratio, or ``None`` when the denominator is effectively zero.
    """
    if denominator <= 1e-9:
        return None
    return float(numerator / denominator)


def _sanitize_json_value(value: Any) -> Any:
    """Convert values into JSON-safe primitives for certificates.

    Returns:
        Any: JSON-safe representation with non-finite floats replaced by ``None``.
    """
    if isinstance(value, Mapping):
        return {str(key): _sanitize_json_value(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, bool | int | str):
        return value
    return str(value)
