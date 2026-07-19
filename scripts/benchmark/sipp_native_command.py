#!/usr/bin/env python3
"""Run the tracked geometry-aware SIPP planner through native-command protocol.

The process reads one JSON request per line and emits one JSON command per line.
It reconstructs canonical occupancy channels from map segments supplied by the
runner. It never reads a scenario/map file itself, so missing geometry fails
closed rather than silently becoming a goal-only planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.nav.occupancy_grid import GridChannel, GridConfig, OccupancyGrid
from robot_sf.planner.sipp_lattice import build_sipp_lattice_search_adapter


class RequestError(ValueError):
    """Raised when a native-command request cannot prove its required inputs."""


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RequestError(f"{label} must be an object")
    return value


def _vector(value: object, label: str, size: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < size:
        raise RequestError(f"{label} must contain {size} finite values")
    try:
        result = [float(item) for item in value[:size]]
    except (TypeError, ValueError) as exc:
        raise RequestError(f"{label} must contain numeric values") from exc
    if not all(math.isfinite(item) for item in result):
        raise RequestError(f"{label} must contain finite values")
    return result


def _segments(value: object, label: str) -> list[tuple[np.ndarray, np.ndarray]]:
    if not isinstance(value, list):
        raise RequestError(f"static_geometry.{label} must be a list")
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, list) or len(raw) != 2:
            raise RequestError(f"static_geometry.{label}[{index}] schema mismatch")
        start = np.asarray(_vector(raw[0], f"{label}[{index}][0]", 2), dtype=float)
        end = np.asarray(_vector(raw[1], f"{label}[{index}][1]", 2), dtype=float)
        result.append((start, end))
    return result


def _geometry(
    request: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[np.ndarray, np.ndarray]]]:
    payload = _mapping(request.get("static_geometry"), "static_geometry")
    if payload.get("schema_version") != "native-command-static-geometry.v1":
        raise RequestError("static_geometry schema_version mismatch")
    scenario_id = payload.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise RequestError("static_geometry.scenario_id is required")
    obstacles = _segments(payload.get("obstacle_segments"), "obstacle_segments")
    boundaries = _segments(payload.get("boundary_segments"), "boundary_segments")
    if not obstacles and not boundaries:
        raise RequestError("static_geometry must contain obstacle or boundary segments")
    declared_hash = payload.get("sha256")
    hash_payload = {key: value for key, value in payload.items() if key != "sha256"}
    expected_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if declared_hash != expected_hash:
        raise RequestError("static_geometry sha256 mismatch")
    return payload, obstacles + boundaries


def _occupancy_observation(
    request: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct canonical planner observation from one native request."""
    geometry, segments = _geometry(request)
    robot = _mapping(request.get("robot"), "robot")
    goal = _mapping(request.get("goal"), "goal")
    pedestrians = _mapping(request.get("pedestrians", {}), "pedestrians")

    robot_position = _vector(robot.get("position"), "robot.position", 2)
    heading = _vector(robot.get("heading"), "robot.heading", 1)
    speed = _vector(robot.get("speed"), "robot.speed", 1)
    angular_velocity = _vector(
        robot.get("angular_velocity", robot.get("omega", [0.0])),
        "robot.angular_velocity",
        1,
    )
    goal_current = _vector(goal.get("current"), "goal.current", 2)
    goal_next = _vector(goal.get("next", goal_current), "goal.next", 2)

    positions = pedestrians.get("positions", [])
    velocities = pedestrians.get("velocities", [])
    if not isinstance(positions, list) or not isinstance(velocities, list):
        raise RequestError("pedestrians.positions and pedestrians.velocities must be lists")
    if len(positions) != len(velocities):
        raise RequestError("pedestrian position/velocity counts differ")
    ped_positions = [_vector(value, "pedestrians.positions", 2) for value in positions]
    ped_velocities = [_vector(value, "pedestrians.velocities", 2) for value in velocities]
    count_values = _vector(
        pedestrians.get("count", [len(ped_positions)]),
        "pedestrians.count",
        1,
    )
    count = int(count_values[0])
    if count < 0 or float(count) != count_values[0] or count > len(ped_positions):
        raise RequestError("pedestrians.count must select available active rows")
    ped_positions = ped_positions[:count]
    ped_velocities = ped_velocities[:count]

    resolution = float(config.get("native_occupancy_resolution", 0.1))
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise RequestError("native_occupancy_resolution must be positive")
    pedestrian_radius = float(
        config.get("pedestrian_radius", config.get("pedestrian_radius_default", 0.3))
    )
    if not math.isfinite(pedestrian_radius) or pedestrian_radius <= 0.0:
        raise RequestError("pedestrian radius must be positive")

    points = [point for segment in segments for point in segment]
    points.extend(np.asarray(point, dtype=float) for point in ped_positions)
    robot_point = np.asarray(robot_position, dtype=float)
    points.append(robot_point)
    min_corner = np.min(np.asarray(points, dtype=float), axis=0)
    max_corner = np.max(np.asarray(points, dtype=float), axis=0)
    padding = max(2.0, resolution * 2.0)
    extent = np.maximum(np.abs(min_corner - robot_point), np.abs(max_corner - robot_point))
    size = np.maximum(2.0 * (extent + padding), resolution * 4.0)
    width_cells, height_cells = (math.ceil(value / resolution) for value in size)
    if width_cells > 1024 or height_cells > 1024:
        raise RequestError("static geometry occupancy grid exceeds 1024 cells per axis")

    occupancy = OccupancyGrid(
        GridConfig(
            resolution=resolution,
            width=width_cells * resolution,
            height=height_cells * resolution,
            channels=[
                GridChannel.OBSTACLES,
                GridChannel.PEDESTRIANS,
                GridChannel.COMBINED,
            ],
            center_on_robot=True,
            use_ego_frame=False,
        )
    )
    grid = occupancy.generate(
        obstacles=[
            (
                (float(start[0]), float(start[1])),
                (float(end[0]), float(end[1])),
            )
            for start, end in segments
        ],
        pedestrians=[
            ((float(position[0]), float(position[1])), pedestrian_radius)
            for position in ped_positions
        ],
        robot_pose=((robot_position[0], robot_position[1]), heading[0]),
        ego_frame=False,
    )
    grid_meta = occupancy.metadata_observation()
    grid_meta["scenario_id"] = geometry["scenario_id"]
    grid_meta["geometry_sha256"] = geometry["sha256"]

    return {
        "robot": {
            "position": robot_position,
            "heading": heading,
            "speed": speed,
            "angular_velocity": angular_velocity,
        },
        "goal": {"current": goal_current, "next": goal_next},
        "pedestrians": {
            "positions": ped_positions,
            "velocities": ped_velocities,
            "count": [len(ped_positions)],
        },
        "occupancy_grid": grid,
        "occupancy_grid_meta": grid_meta,
    }


def _geometry_consumption(observation: dict[str, Any]) -> dict[str, Any]:
    """Return proof that planner-facing canonical occupancy channels carry geometry."""
    grid = np.asarray(observation["occupancy_grid"])
    meta = _mapping(observation["occupancy_grid_meta"], "occupancy_grid_meta")
    indices = list(np.asarray(meta["channel_indices"], dtype=int).reshape(-1))
    obstacle_index, pedestrian_index, _, combined_index = indices
    if min(obstacle_index, pedestrian_index, combined_index) < 0:
        raise RequestError("canonical occupancy channels are missing")
    obstacles = grid[obstacle_index]
    pedestrians = grid[pedestrian_index]
    combined = grid[combined_index]
    union = np.maximum(obstacles, pedestrians)
    return {
        "schema_version": "native-command-geometry-consumption.v1",
        "geometry_sha256": str(meta["geometry_sha256"]),
        "obstacle_occupied_cells": int(np.count_nonzero(obstacles)),
        "pedestrian_occupied_cells": int(np.count_nonzero(pedestrians)),
        "combined_occupied_cells": int(np.count_nonzero(combined)),
        "combined_matches_union": bool(np.array_equal(combined, union)),
    }


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RequestError("SIPP native config must be a mapping")
    return payload


def run(config_path: Path) -> int:
    """Serve native-command requests using existing bounded SIPP search adapter."""
    config = _load_config(config_path)
    planner = build_sipp_lattice_search_adapter(config)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise RequestError("request must be a JSON object")
            observation = _occupancy_observation(request, config)
            command = planner.plan(observation)
            diagnostics = planner.diagnostics().get("last_decision", {})
            print(
                json.dumps(
                    {
                        "linear_velocity": float(command[0]),
                        "angular_velocity": float(command[1]),
                        "sipp_diagnostics": diagnostics,
                        "geometry_consumption": _geometry_consumption(observation),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except (
            ArithmeticError,
            AssertionError,
            AttributeError,
            IndexError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            print(
                json.dumps({"error": str(exc), "status": "invalid_request"}),
                file=sys.stderr,
            )
            return 2
    return 0


def main() -> int:
    """Parse command-line persistent native SIPP server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    return run(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
