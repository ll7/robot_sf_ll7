"""Route-clearance helpers for camera-ready benchmark campaigns."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from shapely.geometry import LineString, Polygon

from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.nav.svg_map_parser import convert_map

_ROUTE_CLEARANCE_WARN_THRESHOLD_M = 0.5
_ROUTE_CLEARANCE_CERTIFICATION_STATUSES = {
    "certified_stress_geometry",
    "excluded_from_planner_attribution",
    "repaired_geometry",
}


def _route_clearance_scenario_id(scenario: dict[str, Any]) -> str:
    """Return the stable scenario identifier for route-clearance log messages."""
    for key in ("name", "scenario_id", "id"):
        value = scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _resolve_map_path_for_scenario(scenario: dict[str, Any]) -> Path | None:
    """Resolve a scenario map path into an existing absolute path.

    Returns:
        Resolved map path when available, otherwise ``None``.
    """
    map_file = scenario.get("map_file")
    if not isinstance(map_file, str) or not map_file.strip():
        return None
    repo_root = get_repository_root().resolve()
    map_path = Path(map_file)
    candidate = map_path if map_path.is_absolute() else (repo_root / map_path)
    candidate = candidate.resolve()
    if candidate.exists():
        return candidate
    return None


def _scenario_robot_radius_m(scenario: dict[str, Any], *, default: float = 1.0) -> float:
    """Extract scenario robot radius for clearance checks.

    Returns:
        Positive finite radius in meters.
    """
    robot_config = scenario.get("robot_config")
    if isinstance(robot_config, dict):
        raw = robot_config.get("radius")
        try:
            radius = float(raw)
        except (TypeError, ValueError):
            radius = default
    else:
        radius = default
    if not math.isfinite(radius) or radius <= 0.0:
        return float(default)
    return float(radius)


def _valid_obstacle_polygons(map_def: Any) -> list[Polygon]:
    """Return valid obstacle polygons from a map definition."""
    obstacles = getattr(map_def, "obstacles", None)
    if not isinstance(obstacles, list):
        return []
    polygons: list[Polygon] = []
    for obstacle in obstacles:
        vertices = getattr(obstacle, "vertices", None)
        if not isinstance(vertices, list) or len(vertices) < 3:
            continue
        poly = Polygon(vertices)
        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)
    return polygons


def _valid_route_lines(map_def: Any) -> list[LineString]:
    """Return valid robot route centerlines from a map definition."""
    routes = getattr(map_def, "robot_routes", None)
    if not isinstance(routes, list):
        return []
    lines: list[LineString] = []
    for route in routes:
        waypoints = getattr(route, "waypoints", None)
        if not isinstance(waypoints, list) or len(waypoints) < 2:
            continue
        line = LineString(waypoints)
        if line.is_valid and not line.is_empty:
            lines.append(line)
    return lines


def _scenario_route_lines(map_def: Any, scenario: dict[str, Any]) -> tuple[list[LineString], str]:
    """Return route lines applicable to one scenario and the warning scope."""
    routes = getattr(map_def, "robot_routes", None)
    if not isinstance(routes, list):
        return [], "scenario"
    spawn_id = scenario.get("robot_spawn_id")
    goal_id = scenario.get("robot_goal_id")
    if isinstance(spawn_id, int) and isinstance(goal_id, int):
        lines: list[LineString] = []
        for route in routes:
            if (
                getattr(route, "spawn_id", None) != spawn_id
                or getattr(route, "goal_id", None) != goal_id
            ):
                continue
            waypoints = getattr(route, "waypoints", None)
            if isinstance(waypoints, list) and len(waypoints) >= 2:
                line = LineString(waypoints)
                if line.is_valid and not line.is_empty:
                    lines.append(line)
        return lines, "scenario"
    route_lines = _valid_route_lines(map_def)
    scope = "scenario" if len(route_lines) <= 1 else "map"
    return route_lines, scope


def _map_route_clearance_center_min_m(route_lines: list[LineString], map_def: Any) -> float | None:
    """Compute minimum centerline distance from route lines to obstacle polygons.

    Returns:
        Minimum route-to-obstacle center distance in meters, or ``None`` when unavailable.
    """
    obstacle_polygons = _valid_obstacle_polygons(map_def)
    if not obstacle_polygons or not route_lines:
        return None

    min_distance: float | None = None
    for line in route_lines:
        for poly in obstacle_polygons:
            distance = float(line.distance(poly))
            if min_distance is None or distance < min_distance:
                min_distance = distance
    return min_distance


def _load_route_clearance_certifications(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load route-clearance certification records keyed by scenario id.

    Returns:
        Certification metadata keyed by scenario name, or an empty mapping when disabled.
    """
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Route-clearance certification file must be a mapping: {path}")
    records_raw = payload.get("certifications")
    if not isinstance(records_raw, dict):
        raise ValueError(f"Route-clearance certification file requires 'certifications': {path}")

    records: dict[str, dict[str, Any]] = {}
    seen_scenarios: set[str] = set()
    for scenario_name, record_raw in records_raw.items():
        scenario = str(scenario_name or "").strip()
        if not scenario:
            raise ValueError("Route-clearance certification scenario keys must be non-empty")
        scenario_lower = scenario.lower()
        if scenario_lower in seen_scenarios:
            raise ValueError(f"Duplicate scenario name detected (case-insensitive): '{scenario}'")
        seen_scenarios.add(scenario_lower)
        if not isinstance(record_raw, dict):
            raise ValueError(f"Certification for '{scenario}' must be a mapping")
        status = str(record_raw.get("status") or "").strip()
        if status not in _ROUTE_CLEARANCE_CERTIFICATION_STATUSES:
            expected = ", ".join(sorted(_ROUTE_CLEARANCE_CERTIFICATION_STATUSES))
            raise ValueError(
                f"Unsupported route-clearance certification status '{status}' for "
                f"'{scenario}'. Expected one of: {expected}"
            )
        claim_scope = str(record_raw.get("claim_scope") or "").strip()
        rationale = str(record_raw.get("rationale") or "").strip()
        if not claim_scope or not rationale:
            raise ValueError(
                f"Certification for '{scenario}' requires non-empty claim_scope and rationale"
            )
        records[scenario] = {
            "status": status,
            "claim_scope": claim_scope,
            "rationale": rationale,
            "reviewed_on": str(record_raw.get("reviewed_on") or "").strip() or None,
            "reviewed_by": str(record_raw.get("reviewed_by") or "").strip() or None,
            "issue": str(record_raw.get("issue") or "").strip() or None,
        }
    return records


def _route_clearance_warning_summary(
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize route-clearance warning certification state for preflight payloads.

    Returns:
        Counts and unresolved scenario ids for the emitted warning rows.
    """
    status_counts: dict[str, int] = {}
    unresolved: list[str] = []
    for warning in warnings:
        scenario = str(warning.get("scenario", "unknown"))
        status = warning.get("certification_status")
        if isinstance(status, str) and status:
            status_counts[status] = status_counts.get(status, 0) + 1
        else:
            unresolved.append(scenario)
    return {
        "warning_count": len(warnings),
        "certified_warning_count": len(warnings) - len(unresolved),
        "unresolved_warning_count": len(unresolved),
        "status_counts": dict(sorted(status_counts.items())),
        "unresolved_scenarios": sorted(unresolved),
    }


def _build_route_clearance_warnings(
    scenarios: list[dict[str, Any]],
    *,
    certifications: dict[str, dict[str, Any]] | None = None,
    margin_warn_threshold_m: float = _ROUTE_CLEARANCE_WARN_THRESHOLD_M,
) -> list[dict[str, Any]]:
    """Build informational preflight warnings for low route-obstacle clearance margins.

    Returns:
        List of warning dictionaries for scenarios with margin below ``margin_warn_threshold_m``.
    """
    warnings: list[dict[str, Any]] = []
    certifications = certifications or {}
    for scenario in scenarios:
        map_path = _resolve_map_path_for_scenario(scenario)
        if map_path is None:
            continue
        try:
            map_def = convert_map(str(map_path))
        except (OSError, ValueError, RuntimeError) as exc:
            scenario_name = _route_clearance_scenario_id(scenario)
            logger.opt(exception=True).warning(
                "Skipping route-clearance preflight for scenario '{}' on map '{}' due to map "
                "parse failure: {}",
                scenario_name,
                map_path,
                exc,
            )
            continue
        route_lines, warning_scope = _scenario_route_lines(map_def, scenario)
        min_center_distance = _map_route_clearance_center_min_m(route_lines, map_def)
        if min_center_distance is None or not math.isfinite(min_center_distance):
            continue
        robot_radius_m = _scenario_robot_radius_m(scenario)
        min_margin_m = float(min_center_distance - robot_radius_m)
        if min_margin_m >= float(margin_warn_threshold_m):
            continue
        scenario_name = _route_clearance_scenario_id(scenario)
        warning = {
            "scenario": scenario_name,
            "map_file": str(scenario.get("map_file", "")),
            "robot_radius_m": round(robot_radius_m, 6),
            "min_center_distance_m": round(float(min_center_distance), 6),
            "min_clearance_margin_m": round(min_margin_m, 6),
            "warning_threshold_m": float(margin_warn_threshold_m),
            "warning_scope": warning_scope,
        }
        certification = certifications.get(scenario_name)
        if certification is not None:
            warning.update(
                {
                    "certification_status": certification["status"],
                    "certification_claim_scope": certification["claim_scope"],
                    "certification_rationale": certification["rationale"],
                    "certification_reviewed_on": certification["reviewed_on"],
                    "certification_reviewed_by": certification["reviewed_by"],
                    "certification_issue": certification["issue"],
                }
            )
        warnings.append(warning)
    warnings.sort(key=lambda item: (item.get("scenario", ""), item.get("map_file", "")))
    return warnings
