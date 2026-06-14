"""Validation script for issue 2728: 2D semantic boundary metadata.

Loads the fixture map and scenarios, then emits compact JSON proving:
- Semantic boundary flags are parsed correctly.
- Unsupported tokens fail closed with ValueError.
- The robot authored route avoids the vehicle_blocking separator.
- The pedestrian emergence route starts near the spawn-edge boundary.
- Evidence classification is diagnostic_only_not_benchmark.
"""

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAP_PATH = REPO_ROOT / "maps" / "svg_maps" / "issue_2728_semantic_boundaries.svg"
SCENARIO_PATH = (
    REPO_ROOT / "configs" / "scenarios" / "single" / "issue_2728_semantic_boundaries.yaml"
)


def _line_segment_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> bool:
    """Check if segment p1-p2 intersects segment p3-p4 using orientation tests."""

    def _orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> int:
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(val) < 1e-9:
            return 0
        return 1 if val > 0 else 2

    o1 = _orient(p1, p2, p3)
    o2 = _orient(p1, p2, p4)
    o3 = _orient(p3, p4, p1)
    o4 = _orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    return False


def _segment_distance_to_point(
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
    point: tuple[float, float],
) -> float:
    """Approximate distance from a point to a polyline segment."""
    dx = seg_end[0] - seg_start[0]
    dy = seg_end[1] - seg_start[1]
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.hypot(point[0] - seg_start[0], point[1] - seg_start[1])
    t = max(
        0.0, min(1.0, ((point[0] - seg_start[0]) * dx + (point[1] - seg_start[1]) * dy) / length_sq)
    )
    proj_x = seg_start[0] + t * dx
    proj_y = seg_start[1] + t * dy
    return math.hypot(point[0] - proj_x, point[1] - proj_y)


def _route_crosses_boundary(
    route_waypoints: list[tuple[float, float]],
    boundary_coords: tuple[tuple[float, float], ...],
) -> bool:
    """Check if any segment of the route crosses any segment of the boundary."""
    for i in range(len(route_waypoints) - 1):
        r_start = route_waypoints[i]
        r_end = route_waypoints[i + 1]
        for j in range(len(boundary_coords) - 1):
            b_start = boundary_coords[j]
            b_end = boundary_coords[j + 1]
            if _line_segment_intersection(r_start, r_end, b_start, b_end):
                return True
    return False


def _min_distance_to_boundary(
    point: tuple[float, float],
    boundary_coords: tuple[tuple[float, float], ...],
) -> float:
    """Minimum distance from a point to any segment of the boundary."""
    min_dist = float("inf")
    for j in range(len(boundary_coords) - 1):
        d = _segment_distance_to_point(boundary_coords[j], boundary_coords[j + 1], point)
        min_dist = min(min_dist, d)
    return min_dist


def run_validation() -> dict:
    """Run all validation checks and return a summary dict."""
    import yaml

    from robot_sf.nav.svg_map_parser import convert_map

    results: dict = {
        "ok": True,
        "evidence_classification": "diagnostic_only_not_benchmark",
        "checks": [],
    }

    def _check(name: str, passed: bool, detail: str = "") -> None:
        results["checks"].append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            results["ok"] = False

    # --- 1. Load map and verify semantic boundary flags parsed ---
    try:
        map_def = convert_map(str(MAP_PATH))
        assert map_def is not None
    except Exception as exc:
        results["ok"] = False
        results["checks"].append({"name": "map_load", "passed": False, "detail": str(exc)})
        return results

    _check(
        "map_load", True, f"Loaded map with {len(map_def.semantic_boundaries)} semantic boundaries"
    )

    boundaries_by_id = {b.id_: b for b in map_def.semantic_boundaries}
    _check(
        "boundary_count",
        len(map_def.semantic_boundaries) == 2,
        f"Expected 2, got {len(map_def.semantic_boundaries)}",
    )

    sep = boundaries_by_id.get("separator")
    ped_edge = boundaries_by_id.get("ped_spawn_edge")

    if sep:
        _check(
            "separator_vehicle_blocking",
            sep.vehicle_blocking,
            f"vehicle_blocking={sep.vehicle_blocking}",
        )
        _check("separator_occluding", sep.occluding, f"occluding={sep.occluding}")
        _check("separator_not_pedestrian_passable", not sep.pedestrian_passable)
        _check("separator_not_spawn_edge", not sep.spawn_edge)
    else:
        _check("separator_found", False, "separator boundary not found")

    if ped_edge:
        _check("ped_edge_pedestrian_passable", ped_edge.pedestrian_passable)
        _check("ped_edge_spawn_edge", ped_edge.spawn_edge)
        _check("ped_edge_not_vehicle_blocking", not ped_edge.vehicle_blocking)
        _check("ped_edge_not_occluding", not ped_edge.occluding)
    else:
        _check("ped_edge_found", False, "ped_spawn_edge boundary not found")

    # --- 2. Unsupported token fails closed ---
    from robot_sf.nav.svg_map_parser import SvgMapConverter

    try:
        SvgMapConverter._parse_semantic_boundary_label(
            "semantic_boundary_test__vehicle_blocking__bogus_token"
        )
        _check("unsupported_token_fail_closed", False, "No ValueError raised for bogus token")
    except ValueError as exc:
        _check("unsupported_token_fail_closed", True, str(exc))

    # --- 3. Robot route does not cross vehicle_blocking separator ---
    if sep and map_def.robot_routes:
        route = map_def.robot_routes[0]
        crosses = _route_crosses_boundary(list(route.waypoints), sep.coordinates)
        direct_crosses = _line_segment_intersection(
            route.waypoints[0],
            route.waypoints[-1],
            sep.coordinates[0],
            sep.coordinates[-1],
        )
        _check(
            "robot_route_avoids_separator",
            not crosses,
            f"route crosses={crosses}, direct line crosses={direct_crosses}",
        )
    else:
        _check("robot_route_avoids_separator", False, "No robot routes or separator found")

    # --- 4. Pedestrian emergence near spawn-edge boundary ---
    scenario_path = SCENARIO_PATH
    with open(scenario_path, encoding="utf-8") as f:
        scenario_data = yaml.safe_load(f)

    ped_scenario = scenario_data["scenarios"][1]
    ped_start = tuple(ped_scenario["single_pedestrians"][0]["start"])

    if ped_edge:
        dist = _min_distance_to_boundary(ped_start, ped_edge.coordinates)
        max_dist = float(ped_scenario["metadata"]["emergence_assertions"][0]["max_distance"])
        _check(
            "ped_start_near_spawn_edge",
            dist <= max_dist,
            f"distance={dist:.2f}, max={max_dist}",
        )
    else:
        _check("ped_start_near_spawn_edge", False, "ped_spawn_edge boundary not found")

    return results


def main() -> None:
    """Entry point for the validation script."""
    parser = argparse.ArgumentParser(description="Validate issue 2728 semantic boundaries")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON summary",
    )
    args = parser.parse_args()

    results = run_validation()
    json_str = json.dumps(results, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str, encoding="utf-8")
        print(f"Written to {out_path}")
    else:
        print(json_str)

    sys.exit(0 if results["ok"] else 1)


if __name__ == "__main__":
    main()
