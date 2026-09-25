"""Reset-only spawn-clearance preflight over a scenario matrix (issue #9725).

For every scenario x seed, the preflight builds the environment exactly as the map
runner does, resets it with the episode seed, and measures robot-pedestrian and
robot-obstacle clearance. Any reset in contact fails the preflight. No planner runs
and no step is taken unless ``--step-zero`` asks for one zero-action step, which
also reports the simulator's own step-1 collision flags.

It also reports, per map, static geometry that places pedestrian route waypoints
inside a robot spawn zone or on a robot route waypoint (padded by both radii).
Those are warnings, not failures: route waypoints are fixed scenario design.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from math import dist
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon

from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.benchmark.map_runner.map_runner_identity import (
    _scenario_with_episode_seed_defaults,
)
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.sim.spawn_validation import reset_spawn_clearance
from robot_sf.training.scenario_loader import load_scenarios

DEFAULT_MATRIX = Path("configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v2.yaml")


def _parse_seeds(text: str) -> list[int]:
    """Parse ``111-140`` or ``111,115,118`` seed lists.

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


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--seeds", default="111-140", help="e.g. 111-140 or 111,115")
    parser.add_argument("--scenario", action="append", default=[], help="Limit to scenario name")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--step-zero", action="store_true", help="Also take one zero step")
    parser.add_argument("--dump-spawns", action="store_true", help="Record robot/ped positions")
    parser.add_argument("--output", type=Path, default=None, help="Write the JSON report here")
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
    """CLI entry point.

    Returns:
        0 when no reset overlaps (and, with ``--step-zero``, no step-1 collision); else 1.
    """
    args = build_parser().parse_args(argv)
    report = run_preflight(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "rows"}
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    failed = report["overlap_count"] > 0 or bool(report["step1_collision_count"])
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover - CLI shim
    sys.exit(main())
