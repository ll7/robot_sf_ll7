"""Reset-only spawn-clearance preflight over a scenario matrix (issue #9725).

For every scenario x seed, the preflight builds the environment exactly as the map
runner does, resets it with the episode seed, and measures robot-pedestrian and
robot-obstacle clearance. Any reset in contact fails the preflight. No planner runs.
The release-manifest path also checks a bounded respawn window with zero action
and a stationary robot; the direct matrix path optionally takes one zero-action
step when ``--step-zero`` is requested.

It also reports, per map, static geometry that places pedestrian route waypoints
inside a robot spawn zone or on a robot route waypoint (padded by both radii).
Those are warnings, not failures: route waypoints are fixed scenario design.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from math import dist
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.benchmark.map_runner.map_runner_identity import (
    _scenario_with_episode_seed_defaults,
)
from robot_sf.benchmark.release_candidate import (
    PrepublicationCandidate,
    load_preflight_input,
    verify_prepublication_candidate_after_preflight,
)
from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.nav.occupancy_grid import (
    OCCUPANCY_FREE_THRESHOLD,
    GridChannel,
    GridConfig,
    OccupancyGrid,
)
from robot_sf.sim.spawn_validation import reset_spawn_clearance
from robot_sf.training.scenario_loader import load_scenarios

DEFAULT_MATRIX = Path("configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v2.yaml")
DEFAULT_CLEARANCE_MARGIN_M = 0.1
DEFAULT_RESPAWN_WINDOW_STEPS = 20
DEFAULT_GRID_RESOLUTION_M = 0.1
CONTINUOUS_ORACLE_QUAD_SEGS = 32
_EXPECTED_OUTCOMES = frozenset({"infeasible_safe_hold"})
INFEASIBILITY_PROBE_RELEASE_KIND = "benchmark-infeasibility-probe"


def _parse_seeds(text: str) -> list[int]:
    """Parse ``1001-1030`` or ``1001,1005,1008`` development seed lists.

    Returns:
        Sorted unique seeds.
    """
    seeds: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            seeds.update(range(int(low), int(high) + 1))
        else:
            seeds.add(int(part))
    return sorted(seeds)


def _load_matrix(matrix: Path) -> list[dict[str, Any]]:
    """Load scenario rows from a matrix file.

    Returns:
        Scenario mappings as the map runner receives them.
    """
    return [dict(row) for row in load_scenarios(matrix)]


def _static_map_warnings(simulator: Any) -> list[dict[str, Any]]:
    """Report pedestrian route waypoints inside padded robot spawn zones or route waypoints.

    Returns:
        One warning mapping per offending pedestrian waypoint.
    """
    map_def = simulator.map_def
    robot_radius = max(float(robot.config.radius) for robot in simulator.robots)
    padding = robot_radius + float(simulator.config.ped_radius)
    spawn_zones = []
    for a, b, c in map_def.robot_spawn_zones:
        d = (a[0] + c[0] - b[0], a[1] + c[1] - b[1])
        spawn_zones.append(Polygon((a, b, c, d)).buffer(padding))
    robot_waypoints = [wp for route in map_def.robot_routes for wp in route.waypoints]
    warnings: list[dict[str, Any]] = []
    for route_index, route in enumerate(map_def.ped_routes):
        for wp_index, waypoint in enumerate(route.waypoints):
            point = Point(waypoint)
            if any(zone.intersects(point) for zone in spawn_zones):
                warnings.append(
                    {
                        "kind": "ped_waypoint_in_robot_spawn_zone",
                        "ped_route": route_index,
                        "waypoint_index": wp_index,
                        "waypoint": [float(waypoint[0]), float(waypoint[1])],
                    }
                )
            near = [wp for wp in robot_waypoints if dist(wp, waypoint) < padding]
            if near:
                warnings.append(
                    {
                        "kind": "ped_waypoint_on_robot_route_waypoint",
                        "ped_route": route_index,
                        "waypoint_index": wp_index,
                        "waypoint": [float(waypoint[0]), float(waypoint[1])],
                        "robot_waypoint": [float(near[0][0]), float(near[0][1])],
                    }
                )
    return warnings


def _validate_expected_outcome(scenario: dict[str, Any]) -> tuple[str | None, str | None]:
    """Validate the only currently supported geometry exemption.

    Returns:
        A normalized declaration and an error code, if the declaration is malformed.
    """
    if "expected_outcome" not in scenario:
        return None, None
    value = scenario["expected_outcome"]
    if not isinstance(value, str) or value not in _EXPECTED_OUTCOMES:
        return None, "unsupported_expected_outcome"
    return value, None


def _check_reset_clearance(
    clearance: dict[str, Any],
    *,
    margin_m: float,
    pedestrian_count: int,
) -> dict[str, Any]:
    """Apply the configured margin to the simulator's reset-clearance measurements.

    Returns:
        Check status, reason codes, and measured surface clearances.
    """
    reasons: list[str] = []
    wall_clearance = clearance.get("robot_obstacle_min_surface_clearance_m")
    ped_clearance = clearance.get("robot_pedestrian_min_surface_clearance_m")
    if not isinstance(wall_clearance, (int, float)) or not math.isfinite(float(wall_clearance)):
        reasons.append("missing_or_invalid_wall_clearance")
    elif float(wall_clearance) < margin_m:
        reasons.append("robot_wall_below_clearance_margin")

    if pedestrian_count:
        if not isinstance(ped_clearance, (int, float)) or not math.isfinite(float(ped_clearance)):
            reasons.append("missing_or_invalid_pedestrian_clearance")
        elif float(ped_clearance) < margin_m:
            reasons.append("robot_pedestrian_below_clearance_margin")
    elif ped_clearance is not None:
        if not isinstance(ped_clearance, (int, float)) or not math.isfinite(float(ped_clearance)):
            reasons.append("invalid_pedestrian_clearance")
        elif float(ped_clearance) < margin_m:
            reasons.append("robot_pedestrian_below_clearance_margin")

    if clearance.get("overlap"):
        reasons.append("reset_footprint_overlap")
    status = "fail" if reasons else "pass"
    return {
        "status": status,
        "reason": ";".join(dict.fromkeys(reasons)) if reasons else "clearance_meets_margin",
        "robot_wall_clearance_m": wall_clearance,
        "robot_pedestrian_clearance_m": ped_clearance,
    }


def _build_occupancy_analysis(
    env: Any,
    *,
    robot_radius_m: float,
    margin_m: float,
    resolution_m: float,
) -> dict[str, Any]:
    """Build a full-map obstacle grid and its robot-footprint-inflated variant.

    Returns:
        Occupancy, inflated occupancy, world origin, resolution, and wall geometry.
    """
    simulator = env.simulator
    map_def = simulator.map_def
    x_min, x_max, y_min, y_max = map_def.get_map_bounds()
    required_radius = robot_radius_m + margin_m
    padding = required_radius + 2.0 * resolution_m
    center_x = (float(x_min) + float(x_max)) / 2.0
    center_y = (float(y_min) + float(y_max)) / 2.0
    width = float(x_max) - float(x_min) + 2.0 * padding
    height = float(y_max) - float(y_min) + 2.0 * padding
    config = GridConfig(
        resolution=resolution_m,
        width=width,
        height=height,
        channels=[GridChannel.OBSTACLES],
        center_on_robot=True,
    )
    static_geometry = getattr(env, "_get_static_grid_obstacles", None)
    if not callable(static_geometry):
        raise ValueError("canonical occupancy-grid obstacle geometry is unavailable")
    obstacle_lines, obstacle_polygons = static_geometry()
    if not obstacle_lines and not obstacle_polygons:
        raise ValueError("scenario map contains no static occupancy-grid geometry")

    grid = OccupancyGrid(config=config)
    occupancy = (
        grid.generate(
            obstacles=obstacle_lines,
            pedestrians=(np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)),
            robot_pose=((center_x, center_y), 0.0),
            ego_frame=False,
            obstacle_polygons=obstacle_polygons,
        )[0]
        >= OCCUPANCY_FREE_THRESHOLD
    )
    distance_to_obstacle = distance_transform_edt(~occupancy, sampling=resolution_m)
    cell_half_diagonal = resolution_m * math.sqrt(2.0) / 2.0
    inflated = occupancy | (distance_to_obstacle <= required_radius + cell_half_diagonal)

    wall_geometries: list[Any] = [LineString(line) for line in obstacle_lines]
    wall_geometries.extend(obstacle_polygons)
    wall_geometry = unary_union(wall_geometries)
    return {
        "occupancy": occupancy,
        "inflated": inflated,
        "map_bounds": (float(x_min), float(x_max), float(y_min), float(y_max)),
        "origin": (center_x - width / 2.0, center_y - height / 2.0),
        "resolution": resolution_m,
        "wall_geometry": wall_geometry,
    }


def _grid_path(  # noqa: C901
    blocked: np.ndarray,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    *,
    origin: tuple[float, float],
    resolution: float,
) -> list[tuple[int, int]] | None:
    """Find an 8-connected grid path without diagonal corner cutting.

    Returns:
        Grid cells from start to goal, or ``None`` when no path exists.
    """
    rows, cols = blocked.shape

    def _cell(point: tuple[float, float]) -> tuple[int, int] | None:
        col = math.floor((point[0] - origin[0]) / resolution)
        row = math.floor((point[1] - origin[1]) / resolution)
        if not (0 <= row < rows and 0 <= col < cols):
            return None
        return row, col

    start = _cell(start_xy)
    goal = _cell(goal_xy)
    if start is None or goal is None or blocked[start] or blocked[goal]:
        return None
    if start == goal:
        return [start]

    neighbors = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )

    def _heuristic(point: tuple[int, int]) -> float:
        dx, dy = abs(point[1] - goal[1]), abs(point[0] - goal[0])
        return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)

    queue: list[tuple[float, float, tuple[int, int]]] = [(_heuristic(start), 0.0, start)]
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    costs = {start: 0.0}
    closed: set[tuple[int, int]] = set()
    while queue:
        _priority, cost, current = heapq.heappop(queue)
        if current in closed or cost != costs.get(current):
            continue
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return list(reversed(path))
        closed.add(current)
        row, col = current
        for dr, dc, step_cost in neighbors:
            nxt = (row + dr, col + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or blocked[nxt]:
                continue
            if dr and dc and (blocked[row + dr, col] or blocked[row, col + dc]):
                continue
            next_cost = cost + step_cost
            if next_cost >= costs.get(nxt, math.inf):
                continue
            costs[nxt] = next_cost
            parent[nxt] = current
            heapq.heappush(queue, (next_cost + _heuristic(nxt), next_cost, nxt))
    return None


def _path_world_points(
    path: list[tuple[int, int]], *, origin: tuple[float, float], resolution: float
) -> list[tuple[float, float]]:
    """Convert grid path cells to their world-coordinate centers.

    Returns:
        World-coordinate center for each grid cell.
    """
    return [
        (origin[0] + (col + 0.5) * resolution, origin[1] + (row + 0.5) * resolution)
        for row, col in path
    ]


def _ordered_route_grid_path(
    blocked: np.ndarray,
    route_points: list[tuple[float, float]],
    *,
    origin: tuple[float, float],
    resolution: float,
) -> tuple[list[tuple[int, int]] | None, int | None]:
    """Join grid paths through every required navigator waypoint in order.

    Returns:
        The complete cell path and no failure index, or the first blocked segment index.
    """
    route_path: list[tuple[int, int]] = []
    for segment_index, (start, goal) in enumerate(pairwise(route_points)):
        segment = _grid_path(blocked, start, goal, origin=origin, resolution=resolution)
        if segment is None:
            return None, segment_index
        route_path.extend(segment[1:] if route_path else segment)
    return route_path, None


def _continuous_margin_route(
    analysis: dict[str, Any],
    route_points: list[tuple[float, float]],
    *,
    required_radius_m: float,
) -> dict[str, Any]:
    """Certify each ordered leg within continuous footprint-safe free space.

    The polygonal GEOS buffer approximates circular arcs with chords. Increase its
    radius by the maximum chord deficit so the approximation cannot admit a
    point closer than the required robot radius plus margin to a wall.

    Returns:
        A pass only when every ordered route leg shares a margin-safe component.
    """
    bounds = analysis.get("map_bounds")
    wall = analysis.get("wall_geometry")
    if (
        not isinstance(bounds, (tuple, list))
        or len(bounds) != 4
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in bounds)
        or wall is None
        or wall.is_empty
        or not wall.is_valid
        or not math.isfinite(required_radius_m)
        or required_radius_m <= 0
    ):
        return {"status": "invalid", "reason": "continuous_geometry_unavailable"}
    x_min, x_max, y_min, y_max = (float(value) for value in bounds)
    if x_min >= x_max or y_min >= y_max:
        return {"status": "invalid", "reason": "invalid_map_bounds"}

    # A straight segment's distance to the full wall geometry is exact. This
    # certifies boundary-width corridors (including the preregistered 2.2 m
    # doorway) where a positive-area buffered component cannot represent the
    # single safe centerline. The check keeps the full 0.10 m margin.
    direct_segments = [LineString([start, goal]) for start, goal in pairwise(route_points)]
    if direct_segments and all(
        segment.distance(wall) + 1.0e-12 >= required_radius_m
        and all(
            x_min + required_radius_m <= point[0] <= x_max - required_radius_m
            and y_min + required_radius_m <= point[1] <= y_max - required_radius_m
            for point in segment.coords
        )
        for segment in direct_segments
    ):
        return {
            "status": "pass",
            "reason": "exact_straight_route_meets_continuous_margin",
            "certified_center_clearance_lower_bound_m": round(required_radius_m, 6),
        }

    # A chord of a quad_segs-buffered quarter circle lies at least this far
    # from the exact circular boundary. The extra micron handles roundoff.
    oracle_radius = (
        required_radius_m / math.cos(math.pi / (4 * CONTINUOUS_ORACLE_QUAD_SEGS)) + 1.0e-6
    )
    point_geometries = [Point(point) for point in route_points]
    for index, point in enumerate(point_geometries):
        boundary_clearance = min(point.x - x_min, x_max - point.x, point.y - y_min, y_max - point.y)
        if min(boundary_clearance, wall.distance(point)) < oracle_radius:
            return {
                "status": "fail",
                "reason": "required_route_point_below_continuous_margin",
                "first_unsafe_route_point_index": index,
            }

    safe_bounds = box(
        x_min + oracle_radius,
        y_min + oracle_radius,
        x_max - oracle_radius,
        y_max - oracle_radius,
    )
    if safe_bounds.is_empty or safe_bounds.area <= 0:
        return {"status": "fail", "reason": "map_has_no_footprint_safe_area"}
    free_space = safe_bounds.difference(
        wall.buffer(oracle_radius, quad_segs=CONTINUOUS_ORACLE_QUAD_SEGS)
    )
    components = [
        geometry
        for geometry in (list(free_space.geoms) if hasattr(free_space, "geoms") else [free_space])
        if isinstance(geometry, Polygon) and geometry.area > 0
    ]
    point_components = [
        {index for index, component in enumerate(components) if component.covers(point)}
        for point in point_geometries
    ]
    for index, (start_components, goal_components) in enumerate(pairwise(point_components)):
        if not start_components.intersection(goal_components):
            return {
                "status": "fail",
                "reason": "ordered_route_segment_disconnected_in_continuous_free_space",
                "first_blocked_segment_index": index,
            }
    return {
        "status": "pass",
        "reason": "ordered_route_connected_in_continuous_margin_safe_free_space",
        "certified_center_clearance_lower_bound_m": round(required_radius_m, 6),
        "buffer_approximation_guard_m": round(oracle_radius - required_radius_m, 6),
    }


def _verified_main_grid_probe(scenario: dict[str, Any], matrix_path: Path) -> bool:
    """Admit only the pinned historical doorway as a main-grid probe.

    Returns:
        True only for the exact declaration and immutable 2.0 m map bytes.
    """
    declaration = scenario.get("infeasibility_probe")
    if not isinstance(declaration, dict):
        return False
    expected = {
        "schema_version": "benchmark-main-grid-infeasibility-probe.v1",
        "expected_outcome": "timeout",
        "classification": "infeasible_by_construction",
        "map_sha256": "7538ed173d462a5107afc1a1e43b5b2e6d2bc5c9604035cdec9a551e20a8b15e",
        "opening_width_m": 2.0,
        "robot_diameter_m": 2.0,
        "required_width_with_margin_m": 2.2,
        "safe_failure_metric": {
            "numerator": "timeout_without_robot_attributable_contact",
            "denominator": "all_420_probe_episodes",
            "unresolved_rows": "fail_admission",
        },
    }
    if (
        scenario.get("name") != "francis2023_narrow_doorway"
        or scenario.get("expected_outcome") != "infeasible_safe_hold"
        or declaration != expected
        or scenario.get("map_id") is not None
    ):
        return False
    map_file = scenario.get("map_file")
    if not isinstance(map_file, str):
        return False
    resolved_map = (matrix_path.parent / map_file).resolve()
    return (
        resolved_map
        == get_repository_root().resolve()
        / "maps/svg_maps/francis2023/francis2023_narrow_doorway.svg"
        and sha256_file(resolved_map) == expected["map_sha256"]
    )


def _apply_infeasibility_declaration(
    reachability: dict[str, Any],
    passage: dict[str, Any],
    *,
    probe_manifest: bool,
) -> None:
    """Require a pinned declaration and an observed infeasible continuous route."""
    if not probe_manifest:
        for check in (reachability, passage):
            check["observed_status"] = check["status"]
            check["observed_reason"] = check["reason"]
            check["status"] = "invalid"
            check["reason"] = "infeasibility_probe_requires_separate_manifest"
    elif reachability.get("continuous_oracle", {}).get("status") != "fail" or reachability.get(
        "continuous_oracle", {}
    ).get("reason") not in {
        "ordered_route_segment_disconnected_in_continuous_free_space",
        "required_route_point_below_continuous_margin",
    }:
        for check in (reachability, passage):
            check["status"] = "invalid"
            check["reason"] = "declared_infeasibility_not_confirmed_by_continuous_oracle"
    else:
        for check in (reachability, passage):
            if check["status"] == "fail":
                check["status"] = "exempt_expected_outcome"
                check["reason"] = "infeasibility_declared_expected_safe_hold"


def _check_footprint_path(
    env: Any,
    analysis: dict[str, Any],
    *,
    scenario: dict[str, Any],
    margin_m: float,
    probe_manifest: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check route reachability in inflated grid cells and estimate path opening widths.

    Returns:
        Footprint reachability and passage-width check results.
    """
    simulator = env.simulator
    if len(simulator.robots) != 1 or len(simulator.robot_navs) != len(simulator.robots):
        invalid = {"status": "invalid", "reason": "robot_route_state_unavailable"}
        return invalid, invalid.copy()
    robot = simulator.robots[0]
    robot_xy = tuple(float(value) for value in robot.pose[0])
    waypoints = list(getattr(simulator.robot_navs[0], "waypoints", []) or [])
    if not waypoints:
        invalid = {"status": "invalid", "reason": "scenario_goal_route_unavailable"}
        return invalid, invalid.copy()
    route_points = [
        robot_xy,
        *(tuple(float(value) for value in waypoint) for waypoint in waypoints),
    ]
    origin = analysis["origin"]
    resolution = float(analysis["resolution"])

    raw_path, raw_blocked_segment = _ordered_route_grid_path(
        analysis["occupancy"], route_points, origin=origin, resolution=resolution
    )
    inflated_path, inflated_blocked_segment = _ordered_route_grid_path(
        analysis["inflated"], route_points, origin=origin, resolution=resolution
    )
    expected_outcome, declaration_error = _validate_expected_outcome(scenario)
    if declaration_error:
        reachability = {"status": "invalid", "reason": declaration_error}
        passage = {"status": "invalid", "reason": declaration_error}
    else:
        reachability = {
            "status": "pass" if inflated_path is not None else "fail",
            "reason": (
                "collision_free_footprint_path_found"
                if inflated_path is not None
                else "no_collision_free_footprint_path"
            ),
            "route_waypoint_count": len(waypoints),
            "first_blocked_segment_index": inflated_blocked_segment,
            "path_length_m": (
                round(
                    sum(
                        math.dist(a, b)
                        for a, b in zip(
                            _path_world_points(inflated_path, origin=origin, resolution=resolution),
                            _path_world_points(inflated_path, origin=origin, resolution=resolution)[
                                1:
                            ],
                            strict=False,
                        )
                    ),
                    3,
                )
                if inflated_path is not None
                else None
            ),
        }
        measurement_path = inflated_path or raw_path
        if measurement_path is None:
            passage = {
                "status": "fail",
                "reason": "no_grid_path_available_for_width_check",
                "first_blocked_segment_index": raw_blocked_segment,
            }
        else:
            points = _path_world_points(measurement_path, origin=origin, resolution=resolution)
            sampled_widths = [
                2.0 * float(analysis["wall_geometry"].distance(Point(point))) for point in points
            ]
            measured_width = min(sampled_widths)
            required_width = 2.0 * float(robot.config.radius) + 2.0 * margin_m
            narrow_samples = [
                (point, width)
                for point, width in zip(points, sampled_widths, strict=True)
                if width + 1.0e-9 < required_width
            ]
            passage = {
                "status": "fail" if narrow_samples else "pass",
                "reason": (
                    "path_openings_meet_required_width"
                    if not narrow_samples
                    else "path_opening_below_required_width"
                ),
                "minimum_opening_width_estimate_m": round(measured_width, 3),
                "required_opening_width_m": round(required_width, 3),
                "narrow_path_cell_count": len(narrow_samples),
                "first_narrow_path_cell_xy": (
                    [round(float(value), 3) for value in narrow_samples[0][0]]
                    if narrow_samples
                    else None
                ),
                "measurement_path": "inflated" if inflated_path is not None else "raw",
            }

        if reachability["status"] == "fail" or passage["status"] == "fail":
            required_radius = float(robot.config.radius) + margin_m
            oracle = _continuous_margin_route(
                analysis, route_points, required_radius_m=required_radius
            )
            reachability["continuous_oracle"] = oracle
            passage["continuous_oracle"] = oracle
            if oracle["status"] == "pass":
                reachability.update(
                    grid_status=reachability["status"],
                    grid_reason=reachability["reason"],
                    grid_path_length_m=reachability.get("path_length_m"),
                    status="pass",
                    reason="continuous_margin_safe_route_found",
                    first_blocked_segment_index=None,
                    path_length_m=None,
                )
                passage.update(
                    grid_status=passage["status"],
                    grid_reason=passage["reason"],
                    grid_minimum_opening_width_estimate_m=passage.get(
                        "minimum_opening_width_estimate_m"
                    ),
                    grid_measurement_path=passage.get("measurement_path"),
                    status="pass",
                    reason="continuous_route_opening_meets_required_width",
                    first_blocked_segment_index=None,
                    minimum_opening_width_estimate_m=None,
                    minimum_opening_width_certified_lower_bound_m=2.0 * required_radius,
                    narrow_path_cell_count=0,
                    first_narrow_path_cell_xy=None,
                    measurement_path="continuous_clearance_component",
                    required_opening_width_m=2.0 * required_radius,
                )

        if expected_outcome == "infeasible_safe_hold":
            _apply_infeasibility_declaration(reachability, passage, probe_manifest=probe_manifest)

    if declaration_error:
        reachability["expected_outcome"] = scenario.get("expected_outcome")
        passage["expected_outcome"] = scenario.get("expected_outcome")
    return reachability, passage


def _check_respawn_window(  # noqa: C901
    env: Any, *, window_steps: int
) -> dict[str, Any]:
    """Keep a robot at zero action and detect fallback respawns inside its footprint.

    Returns:
        Respawn safety status, reason, and observed overlap details.
    """
    simulator = env.simulator
    pedestrian_count = len(getattr(simulator, "ped_pos", []))
    if pedestrian_count == 0:
        return {"status": "pass", "reason": "no_pedestrians_to_respawn", "steps_checked": 0}
    behaviors = list(getattr(simulator, "peds_behaviors", []) or [])
    ledger_behaviors = [
        behavior for behavior in behaviors if hasattr(behavior, "respawn_overlap_events")
    ]
    untracked_route_behaviors = [
        behavior
        for behavior in behaviors
        if getattr(behavior, "navigators", None) and not hasattr(behavior, "respawn_overlap_events")
    ]
    if untracked_route_behaviors:
        return {
            "status": "invalid",
            "reason": "respawn_event_ledger_unavailable",
            "steps_checked": 0,
        }
    ledgers = [behavior.respawn_overlap_events for behavior in ledger_behaviors]
    if any(not isinstance(ledger, list) for ledger in ledgers):
        return {
            "status": "invalid",
            "reason": "respawn_event_ledger_unavailable",
            "steps_checked": 0,
        }
    if not ledgers:
        return {
            "status": "pass",
            "reason": "no_route_end_respawn_groups",
            "steps_checked": 0,
        }

    initial_poses = [
        (np.asarray(robot.pose[0], dtype=float), float(robot.pose[1])) for robot in simulator.robots
    ]
    initial_event_counts = [len(ledger) for ledger in ledgers]
    zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    first_event: dict[str, Any] | None = None
    max_translation = 0.0
    for step in range(1, window_steps + 1):
        _observation, _reward, terminated, truncated, _info = env.step(zero_action)
        for robot, (start_xy, start_heading) in zip(simulator.robots, initial_poses, strict=True):
            max_translation = max(
                max_translation,
                float(np.linalg.norm(np.asarray(robot.pose[0], dtype=float) - start_xy)),
            )
            heading_delta = abs(
                (float(robot.pose[1]) - start_heading + math.pi) % (2 * math.pi) - math.pi
            )
            if max_translation > 1.0e-4 or heading_delta > 1.0e-4:
                return {
                    "status": "invalid",
                    "reason": "robot_did_not_remain_stationary",
                    "steps_checked": step,
                    "max_robot_translation_m": round(max_translation, 6),
                }
        for index, ledger in enumerate(ledgers):
            new_events = ledger[initial_event_counts[index] :]
            if new_events and first_event is None:
                first_event = dict(new_events[0])
        if terminated or truncated:
            return {
                "status": "invalid",
                "reason": "episode_ended_before_respawn_window",
                "steps_checked": step,
                "window_steps": window_steps,
            }

    if first_event is not None:
        return {
            "status": "fail",
            "reason": "pedestrian_respawn_inside_robot_exclusion_radius",
            "steps_checked": window_steps,
            "first_overlap_event": first_event,
            "max_robot_translation_m": round(max_translation, 6),
        }
    return {
        "status": "pass",
        "reason": "no_respawn_inside_robot_exclusion_radius",
        "steps_checked": window_steps,
        "max_robot_translation_m": round(max_translation, 6),
    }


def _check_scenario(job: tuple[dict[str, Any], str, list[int], bool, bool]) -> dict[str, Any]:
    """Reset one scenario for every seed and measure spawn clearance.

    Returns:
        Per-scenario result with one row per seed and static map warnings.
    """
    scenario, matrix_path, seeds, step_zero, dump_spawns = job
    name = str(scenario.get("name") or scenario.get("scenario_id") or "unknown")
    rows: list[dict[str, Any]] = []
    map_warnings: list[dict[str, Any]] | None = None
    for seed in seeds:
        episode_scenario = _scenario_with_episode_seed_defaults(scenario, seed=seed)
        config = build_env_config(episode_scenario, scenario_path=Path(matrix_path))
        env = make_robot_env(config=config, seed=int(seed), debug=False)
        try:
            env.reset(seed=int(seed))
            simulator = env.simulator
            clearance = reset_spawn_clearance(simulator)
            row: dict[str, Any] = {"scenario": name, "seed": int(seed), **clearance}
            relocation = getattr(simulator, "last_spawn_relocation", None)
            row["relocated_pedestrian_rows"] = (
                sorted(relocation.relocated) if relocation is not None else []
            )
            if dump_spawns:
                row["robot_start"] = [list(map(float, pose[0])) for pose in simulator.robot_poses]
                row["ped_positions"] = np.asarray(simulator.ped_pos, dtype=float).tolist()
            if map_warnings is None:
                map_warnings = _static_map_warnings(simulator)
            if step_zero:
                _obs, _reward, _terminated, _truncated, info = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                meta = info.get("meta", info) if isinstance(info, dict) else {}
                row["step1_collision"] = bool(
                    meta.get("is_pedestrian_collision")
                    or meta.get("is_obstacle_collision")
                    or meta.get("is_robot_collision")
                )
            rows.append(row)
        finally:
            env.close()
    return {"scenario": name, "rows": rows, "map_warnings": map_warnings or []}


def _check_release_scenario(  # noqa: C901
    job: tuple[dict[str, Any], str, tuple[int, ...], float, int, float, bool],
) -> dict[str, Any]:
    """Run every release seed for one scenario and return stable check rows.

    Returns:
        Scenario identity, one check row per seed, and static map warnings.
    """
    (
        scenario,
        matrix_path,
        seeds,
        margin_m,
        respawn_window_steps,
        grid_resolution_m,
        probe_manifest,
    ) = job
    name = str(scenario.get("name") or scenario.get("scenario_id") or "unknown")
    rows: list[dict[str, Any]] = []
    map_warnings: list[dict[str, Any]] | None = None
    map_analysis: dict[str, Any] | None = None
    map_analysis_error: str | None = None
    for seed in seeds:
        row: dict[str, Any] = {
            "scenario": name,
            "seed": int(seed),
            "clearance_margin_m": margin_m,
            "respawn_window_steps": respawn_window_steps,
            "expected_outcome": scenario.get("expected_outcome"),
        }
        env: Any | None = None
        try:
            episode_scenario = _scenario_with_episode_seed_defaults(scenario, seed=int(seed))
            config = build_env_config(episode_scenario, scenario_path=Path(matrix_path))
            env = make_robot_env(config=config, seed=int(seed), debug=False)
            env.reset(seed=int(seed))
            simulator = env.simulator
            clearance = reset_spawn_clearance(simulator)
            row["reset_clearance"] = _check_reset_clearance(
                clearance,
                margin_m=margin_m,
                pedestrian_count=len(getattr(simulator, "ped_pos", [])),
            )
            row["robot_radius_m"] = max(float(robot.config.radius) for robot in simulator.robots)
            row["pedestrian_radius_m"] = float(simulator.config.ped_radius)
            row["relocated_pedestrian_rows"] = sorted(
                getattr(getattr(simulator, "last_spawn_relocation", None), "relocated", {})
            )
            unresolved = sorted(
                getattr(getattr(simulator, "last_spawn_relocation", None), "unresolved", [])
            )
            if unresolved:
                row["reset_clearance"]["status"] = "fail"
                row["reset_clearance"]["reason"] += ";unresolved_reset_relocation"
                row["reset_clearance"]["unresolved_pedestrian_rows"] = unresolved
            if map_warnings is None:
                map_warnings = _static_map_warnings(simulator)
            if map_analysis is None and map_analysis_error is None:
                try:
                    robot_radius = max(float(robot.config.radius) for robot in simulator.robots)
                    map_analysis = _build_occupancy_analysis(
                        env,
                        robot_radius_m=robot_radius,
                        margin_m=margin_m,
                        resolution_m=grid_resolution_m,
                    )
                # Preserve the scenario-by-seed row and mark geometry unavailable.
                except Exception as exc:  # noqa: BLE001
                    map_analysis_error = f"{type(exc).__name__}: {exc}"
            if map_analysis is None:
                reason = "occupancy_grid_unavailable: " + str(map_analysis_error)
                row["footprint_reachability"] = {"status": "invalid", "reason": reason}
                row["passage_width"] = {"status": "invalid", "reason": reason}
            else:
                reachability, passage = _check_footprint_path(
                    env,
                    map_analysis,
                    scenario=scenario,
                    margin_m=margin_m,
                    probe_manifest=probe_manifest,
                )
                row["footprint_reachability"] = reachability
                row["passage_width"] = passage
            if clearance.get("overlap") or unresolved:
                row["respawn_safety"] = {
                    "status": "not_assessed",
                    "reason": "reset_state_invalid_before_respawn_window",
                    "steps_checked": 0,
                }
            else:
                row["respawn_safety"] = _check_respawn_window(
                    env,
                    window_steps=respawn_window_steps,
                )
            row["step1_collision"] = bool(
                row["respawn_safety"].get("first_overlap_event")
                or row["respawn_safety"].get("reason") == "episode_ended_before_respawn_window"
            )
        # A failing cell must remain visible as invalid rather than disappear.
        except Exception as exc:  # noqa: BLE001
            row["cell_error"] = f"{type(exc).__name__}: {exc}"
            for key in (
                "reset_clearance",
                "footprint_reachability",
                "passage_width",
                "respawn_safety",
            ):
                row.setdefault(
                    key,
                    {"status": "invalid", "reason": "cell_initialization_or_check_failed"},
                )
            row.setdefault("step1_collision", False)
        finally:
            if env is not None:
                env.close()
        check_statuses = [
            row[key]["status"]
            for key in (
                "reset_clearance",
                "footprint_reachability",
                "passage_width",
                "respawn_safety",
            )
        ]
        row["overall_status"] = (
            "blocked"
            if any(status in {"fail", "invalid", "not_assessed"} for status in check_statuses)
            else ("infeasibility_probe" if "exempt_expected_outcome" in check_statuses else "valid")
        )
        rows.append(row)
    return {"scenario": name, "rows": rows, "map_warnings": map_warnings or []}


def _release_manifest_inputs(  # noqa: C901, PLR0912, PLR0915
    manifest: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], tuple[int, ...]]:
    """Resolve and checksum the exact scenario matrix and seed set from a release manifest.

    Returns:
        Input identities, resolved scenarios, and the exact resolved seed tuple.
    """
    manifest_path = Path(manifest.path).resolve()
    candidate = isinstance(manifest, PrepublicationCandidate)
    repository_root = manifest.repository_root if candidate else get_repository_root().resolve()
    matrix_path = Path(manifest.scenario_matrix_path).resolve()
    manifest_sha256 = sha256_file(manifest_path)
    matrix_sha256 = sha256_file(matrix_path)
    expected_matrix_sha256 = str(manifest.scenario_matrix_sha256).lower()
    if matrix_sha256 != expected_matrix_sha256:
        raise ValueError("scenario matrix SHA-256 does not match the release manifest")

    raw_seeds = getattr(manifest, "resolved_seeds", ())
    if not raw_seeds:
        raw_seeds = manifest.seed_policy.get("resolved_seeds", ())
    if not raw_seeds and manifest.seed_policy.get("mode") == "fixed-list":
        raw_seeds = manifest.seed_policy.get("seeds", ())
    if not raw_seeds or any(type(seed) is not int or seed < 0 for seed in raw_seeds):
        raise ValueError("release manifest has no valid resolved seed set")
    seeds = tuple(raw_seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("release manifest resolved seed set contains duplicates")

    seed_policy = dict(manifest.seed_policy)
    seed_mode = seed_policy.get("mode")
    if seed_mode not in {"seed-set", "fixed-list"}:
        raise ValueError(
            f"unsupported seed_policy mode {seed_mode!r}; expected seed-set or fixed-list"
        )
    seed_set_name = seed_policy.get("seed_set")
    seed_sets_path: Path | None = None
    declared_seed_sha256 = getattr(manifest, "seed_sets_sha256", None) or seed_policy.get(
        "seed_sets_sha256"
    )
    seed_sets_sha256: str | None = None
    seed_sets_path_raw = seed_policy.get("seed_sets_path")
    if seed_sets_path_raw and seed_policy.get("mode") == "seed-set":
        seed_sets_path = (
            (repository_root if candidate else manifest_path.parent) / str(seed_sets_path_raw)
        ).resolve()
        seed_sets_sha256 = sha256_file(seed_sets_path)
        if candidate:
            pinned = dict(manifest.pinned_files)
            declared_seed_sha256 = pinned.get(seed_sets_path)
        if not declared_seed_sha256:
            raise ValueError("release seed-set path is missing its declared SHA-256")
        if seed_sets_sha256 != str(declared_seed_sha256).lower():
            raise ValueError("seed-set file SHA-256 does not match the release manifest")
        if not isinstance(seed_set_name, str) or not seed_set_name:
            raise ValueError("release seed-set path is present without a seed_set identity")
        seed_set_payload = yaml.safe_load(seed_sets_path.read_text(encoding="utf-8"))
        if not isinstance(seed_set_payload, dict) or not isinstance(
            seed_set_payload.get(seed_set_name), list
        ):
            raise ValueError("named release seed set is missing or malformed")
        declared_seed_values = seed_set_payload[seed_set_name]
        if any(type(seed) is not int or seed < 0 for seed in declared_seed_values):
            raise ValueError("named release seed set contains an invalid seed")
        declared_seeds = tuple(declared_seed_values)
        if declared_seeds != seeds:
            raise ValueError("resolved release seeds do not match the named seed set")
    elif seed_policy.get("mode") == "seed-set":
        raise ValueError("seed-set mode requires a checksummed seed_sets_path")
    else:
        fixed_seeds = seed_policy.get("seeds")
        if not isinstance(fixed_seeds, list):
            raise ValueError("fixed-list mode requires seed_policy.seeds as a list")
        if tuple(int(seed) for seed in fixed_seeds) != seeds:
            raise ValueError("resolved release seeds do not match seed_policy.seeds")

    scenarios = _load_matrix(matrix_path)
    if not scenarios:
        raise ValueError("release scenario matrix is empty")
    names = [str(row.get("name") or row.get("scenario_id") or "") for row in scenarios]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("release scenario matrix has missing or duplicate scenario identities")
    for scenario in scenarios:
        if "infeasibility_probe" in scenario and not _verified_main_grid_probe(
            scenario, matrix_path
        ):
            raise ValueError("main-grid infeasibility probe declaration or map pin is invalid")

    expected_cells = getattr(manifest, "expected_episode_cells", None)
    planner_count = len(getattr(manifest, "planner_keys", ()) or ())
    if expected_cells is not None and planner_count:
        if int(expected_cells) != len(scenarios) * len(seeds) * planner_count:
            raise ValueError(
                "release manifest cell count disagrees with matrix, seeds, and planners"
            )

    def _portable_path(path: Path) -> str:
        """Keep repository inputs relative and avoid publishing machine-local paths.

        Returns:
            Repository-relative path, or the basename for external inputs.
        """
        try:
            return path.resolve().relative_to(repository_root).as_posix()
        except ValueError:
            return path.name

    identity = {
        "manifest_path": _portable_path(manifest_path),
        "manifest_sha256": manifest_sha256,
        "release_id": str(getattr(manifest, "release_id", "")),
        "release_kind": str(getattr(manifest, "release_kind", "") or ""),
        "manifest_schema_version": str(getattr(manifest, "schema_version", "")),
        "scenario_matrix_path": _portable_path(matrix_path),
        "scenario_matrix_sha256": matrix_sha256,
        "scenario_count": len(scenarios),
        "seed_policy": seed_policy,
        "seed_set": seed_set_name,
        "seed_sets_path": _portable_path(seed_sets_path) if seed_sets_path is not None else None,
        "seed_sets_sha256": seed_sets_sha256,
        "resolved_seeds": list(seeds),
        "expected_episode_cells": expected_cells,
        "planner_count": planner_count or None,
    }
    return identity, scenarios, seeds


def _preflight_markdown(report: dict[str, Any]) -> str:
    """Render a complete Markdown matrix report from the JSON report payload.

    Returns:
        Markdown report text.
    """
    identity = report.get("release_inputs", {})
    lines = [
        "# Spawn matrix preflight",
        "",
        "This is preflight diagnostic output. It is not planner-performance or benchmark-success evidence.",
        "",
        f"- Status: `{report.get('status', 'invalid')}`",
        f"- Release: `{identity.get('release_id', 'unknown')}`",
        f"- Release kind: `{identity.get('release_kind', 'unknown')}`",
        f"- Manifest SHA-256: `{identity.get('manifest_sha256', 'unavailable')}`",
        f"- Matrix: `{identity.get('scenario_matrix_path', 'unavailable')}`",
        f"- Matrix SHA-256: `{identity.get('scenario_matrix_sha256', 'unavailable')}`",
        f"- Seed set: `{identity.get('seed_set', 'fixed')}`",
        f"- Seed-set SHA-256: `{identity.get('seed_sets_sha256', 'unavailable')}`",
        f"- Resolved seeds: `{identity.get('resolved_seeds', [])}`",
        f"- Clearance margin: `{report.get('clearance_margin_m', 'unavailable')} m`",
        f"- Respawn window: `{report.get('respawn_window_steps', 'unavailable')} steps`",
        f"- Grid resolution: `{report.get('grid_resolution_m', 'unavailable')} m`",
        f"- Scenario workers: `{report.get('workers', 'unavailable')}`",
        f"- Rows: `{report.get('cell_count', 0)}`; blocked: `{report.get('blocked_cell_count', 0)}`",
        "",
        "| Scenario | Seed | Overall | Reset clearance | Footprint path | Passage width | Respawn | Reasons |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    check_keys = (
        ("reset_clearance", "reset"),
        ("footprint_reachability", "path"),
        ("passage_width", "width"),
        ("respawn_safety", "respawn"),
    )
    for row in report.get("rows", []):
        reasons = [
            f"{label}: {row.get(key, {}).get('reason', 'missing')}"
            for key, label in check_keys
            if row.get(key, {}).get("status") not in {"pass", "exempt_expected_outcome"}
        ]
        reason = "; ".join(reasons) or "—"
        fields = [
            str(row.get("scenario", "unknown")),
            str(row.get("seed", "unknown")),
            str(row.get("overall_status", "invalid")),
            str(row.get("reset_clearance", {}).get("status", "invalid")),
            str(row.get("footprint_reachability", {}).get("status", "invalid")),
            str(row.get("passage_width", {}).get("status", "invalid")),
            str(row.get("respawn_safety", {}).get("status", "invalid")),
            reason,
        ]
        lines.append(
            "| "
            + " | ".join(field.replace("|", "\\|").replace("\n", " ") for field in fields)
            + " |"
        )
    if report.get("input_error"):
        lines.extend(("", f"Input error: `{report['input_error']}`"))
    lines.append("")
    return "\n".join(lines)


def write_preflight_reports(
    report: dict[str, Any], *, json_path: Path, markdown_path: Path
) -> tuple[str, str]:
    """Write both representations and return their SHA-256 digests.

    Returns:
        JSON and Markdown file SHA-256 digests, in that order.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_preflight_markdown(report), encoding="utf-8")
    return sha256_file(json_path), sha256_file(markdown_path)


def _verify_candidate_report_inputs(
    manifest: Any, identity: dict[str, Any], report: dict[str, Any]
) -> None:
    """Invalidate a candidate report when its pinned source or inputs drifted."""
    if not isinstance(manifest, PrepublicationCandidate):
        return
    try:
        verify_prepublication_candidate_after_preflight(
            manifest,
            manifest_sha256=identity["manifest_sha256"],
            repository_root=manifest.repository_root,
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        report["status"] = "invalid"
        drift_error = f"candidate_input_drift: {type(exc).__name__}: {exc}"
        prior_error = report.get("input_error")
        report["input_error"] = f"{prior_error}; {drift_error}" if prior_error else drift_error


def guard_manifest_execution(
    manifest: Any, *, source_commit: str | None = None, repository_root: Path | None = None
) -> None:
    """Refuse held-out execution before any environment or campaign is created.

    Historical pins permit static validation only. Sealed execution requires a
    named release identity bound to the checked-out freeze commit.
    """
    from robot_sf.benchmark.release_protocol import (  # noqa: PLC0415
        _resolved_seed_inventory,
        load_release_campaign_config,
        sealed_seed_execution_problem,
    )
    from robot_sf.benchmark.seed_bands import RETIRED_EVAL_SEEDS_0_0_7  # noqa: PLC0415

    root = (
        repository_root or getattr(manifest, "repository_root", None) or get_repository_root()
    ).resolve()
    _identity, _scenarios, seeds = _release_manifest_inputs(manifest)
    effective = set(seeds)
    if getattr(manifest, "canonical_campaign_config_path", None) is not None and not isinstance(
        manifest, PrepublicationCandidate
    ):
        from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios  # noqa: PLC0415

        cfg = load_release_campaign_config(manifest, repository_root=root)
        effective.update(
            _resolved_seed_inventory(_load_campaign_scenarios(cfg, repository_root=root))
        )
    if effective.intersection(RETIRED_EVAL_SEEDS_0_0_7):
        raise ValueError("retired evaluation seeds are forbidden for execution (D-049)")
    from robot_sf.benchmark.release_protocol import (  # noqa: PLC0415
        _require_clean_exact_checkout,
        _require_sealed_runtime_sources,
        _require_sealed_source_inputs,
        _validated_development_seeds,
        is_development_rehearsal,
    )

    if is_development_rehearsal(manifest):
        _validated_development_seeds(tuple(sorted(effective)))
        if manifest.resolved_identity_path is None:
            raise ValueError("development rehearsal requires a verified materialized identity")
        _require_clean_exact_checkout(
            root,
            source_commit=source_commit or manifest.source_sha,
            template_path=manifest.identity_template_path,
        )
        _require_sealed_source_inputs(manifest, root, manifest.source_sha)
        _require_sealed_runtime_sources(root, manifest.source_sha)
    problem = sealed_seed_execution_problem(
        manifest,
        tuple(sorted(effective)),
        source_commit=source_commit,
        repository_root=root,
    )
    if problem is not None:
        raise ValueError(problem)


def run_manifest_preflight(  # noqa: C901
    manifest: Any,
    *,
    workers: int = 1,
    clearance_margin_m: float = DEFAULT_CLEARANCE_MARGIN_M,
    respawn_window_steps: int = DEFAULT_RESPAWN_WINDOW_STEPS,
    grid_resolution_m: float = DEFAULT_GRID_RESOLUTION_M,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Run a fail-closed spawn/reachability matrix check from exact release inputs.

    Returns:
        Diagnostic report bound to the release manifest and all scenario-seed checks.
    """
    started = time.perf_counter()
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    if not math.isfinite(clearance_margin_m) or clearance_margin_m < 0.0:
        raise ValueError("clearance_margin_m must be finite and >= 0")
    if respawn_window_steps < 1:
        raise ValueError("respawn_window_steps must be >= 1")
    if not math.isfinite(grid_resolution_m) or grid_resolution_m <= 0.0:
        raise ValueError("grid_resolution_m must be finite and > 0")

    try:
        guard_manifest_execution(manifest, source_commit=source_commit)
        identity, scenarios, seeds = _release_manifest_inputs(manifest)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        report = {
            "schema_version": "spawn_matrix_preflight.v1",
            "status": "invalid",
            "evidence_class": "preflight_diagnostic_only",
            "benchmark_success": None,
            "release_inputs": {},
            "clearance_margin_m": clearance_margin_m,
            "respawn_window_steps": respawn_window_steps,
            "grid_resolution_m": grid_resolution_m,
            "scenario_count": 0,
            "seed_count": 0,
            "expected_cell_count": 0,
            "cell_count": 0,
            "blocked_cell_count": 0,
            "rows": [],
            "input_error": f"{type(exc).__name__}: {exc}",
            "runtime_s": round(time.perf_counter() - started, 1),
        }
        return report

    probe_manifest = identity.get("release_kind") == INFEASIBILITY_PROBE_RELEASE_KIND
    jobs = [
        (
            scenario,
            str(manifest.scenario_matrix_path),
            seeds,
            clearance_margin_m,
            respawn_window_steps,
            grid_resolution_m,
            probe_manifest
            or _verified_main_grid_probe(scenario, Path(manifest.scenario_matrix_path)),
        )
        for scenario in scenarios
    ]
    execution_error: str | None = None
    try:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_check_release_scenario, jobs))
        else:
            results = [_check_release_scenario(job) for job in jobs]
    # Worker failures invalidate all unfinished matrix cells; retain a complete failed report.
    except Exception as exc:  # noqa: BLE001
        execution_error = f"matrix_worker_failed: {type(exc).__name__}: {exc}"
        results = [
            {
                "scenario": str(scenario.get("name") or scenario.get("scenario_id") or "unknown"),
                "map_warnings": [],
                "rows": [
                    {
                        "scenario": str(
                            scenario.get("name") or scenario.get("scenario_id") or "unknown"
                        ),
                        "seed": int(seed),
                        "clearance_margin_m": clearance_margin_m,
                        "respawn_window_steps": respawn_window_steps,
                        "overall_status": "blocked",
                        "cell_error": execution_error,
                        **{
                            key: {"status": "invalid", "reason": "matrix_worker_failed"}
                            for key in (
                                "reset_clearance",
                                "footprint_reachability",
                                "passage_width",
                                "respawn_safety",
                            )
                        },
                    }
                    for seed in seeds
                ],
            }
            for scenario in scenarios
        ]
    rows = [row for result in results for row in result["rows"]]
    for row in rows:
        row["manifest_path"] = identity["manifest_path"]
        row["manifest_sha256"] = identity["manifest_sha256"]
        row["scenario_matrix_path"] = identity["scenario_matrix_path"]
        row["scenario_matrix_sha256"] = identity["scenario_matrix_sha256"]
        row["seed_set"] = identity["seed_set"]
        row["seed_sets_path"] = identity["seed_sets_path"]
        row["seed_sets_sha256"] = identity["seed_sets_sha256"]
    expected_rows = len(scenarios) * len(seeds)
    observed_cells = [(row["scenario"], row["seed"]) for row in rows]
    input_error = execution_error
    if len(rows) != expected_rows or len(set(observed_cells)) != expected_rows:
        input_error = "matrix execution did not emit exactly one row per scenario and seed"
    try:
        if sha256_file(Path(manifest.path)) != identity["manifest_sha256"]:
            input_error = "release manifest changed while preflight was running"
        if sha256_file(Path(manifest.scenario_matrix_path)) != identity["scenario_matrix_sha256"]:
            input_error = "scenario matrix changed while preflight was running"
        seed_sets_path_raw = manifest.seed_policy.get("seed_sets_path")
        if seed_sets_path_raw and manifest.seed_policy.get("mode") == "seed-set":
            seed_sets_base = (
                manifest.repository_root
                if isinstance(manifest, PrepublicationCandidate)
                else Path(manifest.path).resolve().parent
            )
            seed_sets_path = (seed_sets_base / str(seed_sets_path_raw)).resolve()
            if sha256_file(seed_sets_path) != identity["seed_sets_sha256"]:
                input_error = "seed-set file changed while preflight was running"
    except OSError as exc:
        input_error = f"input disappeared while preflight was running: {exc}"
    blocked = sum(
        row["overall_status"]
        not in ({"valid"} if probe_manifest else {"valid", "infeasibility_probe"})
        for row in rows
    )
    report = {
        "schema_version": "spawn_matrix_preflight.v1",
        "status": (
            "valid" if blocked == 0 and input_error is None and not probe_manifest else "blocked"
        ),
        "evidence_class": "preflight_diagnostic_only",
        "benchmark_success": None,
        "source_commit": source_commit,
        "release_inputs": identity,
        "clearance_margin_m": clearance_margin_m,
        "respawn_window_steps": respawn_window_steps,
        "grid_resolution_m": grid_resolution_m,
        "workers": workers,
        "scenario_count": len(scenarios),
        "seed_count": len(seeds),
        "expected_cell_count": expected_rows,
        "cell_count": len(rows),
        "blocked_cell_count": blocked,
        "overlap_count": sum(
            bool(row.get("reset_clearance", {}).get("status") == "fail") for row in rows
        ),
        "respawn_failure_count": sum(
            row.get("respawn_safety", {}).get("status") == "fail" for row in rows
        ),
        "map_warnings": {
            result["scenario"]: result["map_warnings"]
            for result in results
            if result["map_warnings"]
        },
        "rows": rows,
        "input_error": input_error,
        "runtime_s": round(time.perf_counter() - started, 1),
    }
    _verify_candidate_report_inputs(manifest, identity, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    """Return the exact release-manifest or diagnostic-candidate parser."""
    parser = argparse.ArgumentParser(
        description="Run the fail-closed spawn and footprint preflight for pinned release inputs."
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Release manifest or DOI-free candidate."
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel scenario workers (1-8).")
    parser.add_argument(
        "--clearance-margin-m",
        type=float,
        default=DEFAULT_CLEARANCE_MARGIN_M,
        help="Required surface clearance (default: 0.10 m).",
    )
    parser.add_argument(
        "--respawn-window-steps",
        type=int,
        default=DEFAULT_RESPAWN_WINDOW_STEPS,
        help="Stationary-robot respawn check window (default: 20 steps).",
    )
    parser.add_argument(
        "--grid-resolution-m",
        type=float,
        default=DEFAULT_GRID_RESOLUTION_M,
        help="Static occupancy-grid resolution (default: 0.10 m).",
    )
    parser.add_argument("--json-output", type=Path, required=True, help="JSON report path.")
    parser.add_argument("--markdown-output", type=Path, required=True, help="Markdown report path.")
    return parser


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Run the preflight and return the report.

    Returns:
        JSON-serializable report with per-cell rows, overlaps, and map warnings.
    """
    started = time.perf_counter()
    scenarios = _load_matrix(args.matrix)
    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [row for row in scenarios if row.get("name") in wanted]
        missing = wanted - {row.get("name") for row in scenarios}
        if missing:
            raise SystemExit(f"unknown scenario(s): {sorted(missing)}")
    seeds = _parse_seeds(args.seeds)
    jobs = [(row, str(args.matrix), seeds, args.step_zero, args.dump_spawns) for row in scenarios]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_check_scenario, jobs))
    else:
        results = [_check_scenario(job) for job in jobs]
    rows = [row for result in results for row in result["rows"]]
    overlaps = [row for row in rows if row["overlap"]]
    step1 = [row for row in rows if row.get("step1_collision")]
    return {
        "schema_version": "spawn_clearance_preflight.v1",
        "matrix": str(args.matrix),
        "seeds": seeds,
        "scenario_count": len(scenarios),
        "cell_count": len(rows),
        "overlap_count": len(overlaps),
        "overlaps": [
            {key: row[key] for key in ("scenario", "seed") if key in row}
            | {
                "robot_pedestrian_min_surface_clearance_m": row[
                    "robot_pedestrian_min_surface_clearance_m"
                ],
                "robot_obstacle_min_surface_clearance_m": row[
                    "robot_obstacle_min_surface_clearance_m"
                ],
            }
            for row in overlaps
        ],
        "step1_collision_count": len(step1) if args.step_zero else None,
        "relocated_cell_count": sum(1 for row in rows if row["relocated_pedestrian_rows"]),
        "map_warnings": {
            result["scenario"]: result["map_warnings"]
            for result in results
            if result["map_warnings"]
        },
        "runtime_s": round(time.perf_counter() - started, 1),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the exact pinned matrix and write JSON and Markdown diagnostic reports.

    Returns:
        Zero for a valid matrix and nonzero when inputs or checks fail.
    """
    args = build_parser().parse_args(argv)
    try:
        manifest = load_preflight_input(args.manifest)
        guard_manifest_execution(manifest)
        report = run_manifest_preflight(
            manifest,
            workers=args.workers,
            clearance_margin_m=args.clearance_margin_m,
            respawn_window_steps=args.respawn_window_steps,
            grid_resolution_m=args.grid_resolution_m,
            source_commit=manifest.source_sha
            if isinstance(manifest, PrepublicationCandidate)
            else None,
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        report = {
            "schema_version": "spawn_matrix_preflight.v1",
            "status": "invalid",
            "evidence_class": "preflight_diagnostic_only",
            "benchmark_success": None,
            "release_inputs": {},
            "clearance_margin_m": args.clearance_margin_m,
            "respawn_window_steps": args.respawn_window_steps,
            "grid_resolution_m": args.grid_resolution_m,
            "scenario_count": 0,
            "seed_count": 0,
            "expected_cell_count": 0,
            "cell_count": 0,
            "blocked_cell_count": 0,
            "rows": [],
            "input_error": f"{type(exc).__name__}: {exc}",
        }
    json_sha256, markdown_sha256 = write_preflight_reports(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = {key: value for key, value in report.items() if key != "rows"}
    summary["json_sha256"] = json_sha256
    summary["markdown_sha256"] = markdown_sha256
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0 if report.get("status") == "valid" else 2


if __name__ == "__main__":  # pragma: no cover - CLI shim
    sys.exit(main())
