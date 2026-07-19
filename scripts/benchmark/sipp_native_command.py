#!/usr/bin/env python3
"""Run the tracked geometry-aware SIPP planner through the native-command protocol.

The process reads one JSON request per line and emits one JSON command per line.
It deliberately reconstructs an occupancy grid from the map segments supplied by
the runner; it never reads a scenario/map file itself, so missing geometry fails
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

from robot_sf.nav.occupancy_grid import OBSERVATION_CHANNEL_ORDER
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
    result = []
    for index, raw in enumerate(value):
        if not isinstance(raw, list) or len(raw) != 2:
            raise RequestError(f"static_geometry.{label}[{index}] must be a point pair")
        result.append(
            (
                np.asarray(_vector(raw[0], f"static_geometry.{label}[{index}][0]", 2)),
                np.asarray(_vector(raw[1], f"static_geometry.{label}[{index}][1]", 2)),
            )
        )
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


def _rasterize_segments(
    *,
    segments: list[tuple[np.ndarray, np.ndarray]],
    origin: np.ndarray,
    resolution: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize segment neighborhoods without scanning the full grid per cell."""
    occupied = np.zeros((height, width), dtype=bool)
    radius = resolution * 0.75
    radius_sq = radius * radius
    for start, end in segments:
        lower = np.minimum(start, end) - radius
        upper = np.maximum(start, end) + radius
        col_start = max(0, math.floor((lower[0] - origin[0]) / resolution - 0.5))
        col_stop = min(width - 1, math.ceil((upper[0] - origin[0]) / resolution - 0.5))
        row_start = max(0, math.floor((lower[1] - origin[1]) / resolution - 0.5))
        row_stop = min(height - 1, math.ceil((upper[1] - origin[1]) / resolution - 0.5))
        if col_start > col_stop or row_start > row_stop:
            continue

        columns = np.arange(col_start, col_stop + 1, dtype=float)
        rows = np.arange(row_start, row_stop + 1, dtype=float)
        x_values = origin[0] + (columns + 0.5) * resolution
        y_values = origin[1] + (rows + 0.5) * resolution
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        points = np.stack((x_grid, y_grid), axis=-1)

        direction = end - start
        length_sq = float(np.dot(direction, direction))
        if length_sq <= 1e-12:
            distance_sq = np.sum((points - start) ** 2, axis=-1)
        else:
            fractions = np.clip(np.sum((points - start) * direction, axis=-1) / length_sq, 0, 1)
            projections = start + fractions[..., np.newaxis] * direction
            distance_sq = np.sum((points - projections) ** 2, axis=-1)
        occupied[row_start : row_stop + 1, col_start : col_stop + 1] |= distance_sq <= radius_sq
    return occupied


def _occupancy_observation(request: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
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
    count_values = _vector(pedestrians.get("count", [len(ped_positions)]), "pedestrians.count", 1)
    count = int(count_values[0])
    if count < 0 or float(count) != count_values[0] or count > len(ped_positions):
        raise RequestError("pedestrians.count must select available active rows")
    ped_positions = ped_positions[:count]
    ped_velocities = ped_velocities[:count]

    all_points = [point for segment in segments for point in segment]
    min_corner = np.min(np.asarray(all_points, dtype=float), axis=0)
    max_corner = np.max(np.asarray(all_points, dtype=float), axis=0)
    resolution = float(config.get("native_occupancy_resolution", 0.1))
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise RequestError("native_occupancy_resolution must be positive")
    padding = max(2.0, resolution * 2.0)
    origin = min_corner - padding
    size = np.maximum(max_corner - min_corner + 2.0 * padding, resolution * 4.0)
    width, height = (math.ceil(value / resolution) + 1 for value in size)
    if width > 1024 or height > 1024:
        raise RequestError("static geometry occupancy grid exceeds 1024 cells per axis")
    obstacle_channel = tuple(channel.value for channel in OBSERVATION_CHANNEL_ORDER).index(
        "obstacles"
    )
    grid = np.zeros((len(OBSERVATION_CHANNEL_ORDER), height, width), dtype=float)
    grid[obstacle_channel] = _rasterize_segments(
        segments=segments,
        origin=origin,
        resolution=resolution,
        width=width,
        height=height,
    )
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
        "occupancy_grid_meta": {
            "origin": origin.tolist(),
            "resolution": [resolution],
            "size": size.tolist(),
            "use_ego_frame": [0.0],
            "center_on_robot": [0.0],
            "channel_indices": list(range(len(OBSERVATION_CHANNEL_ORDER))),
            "robot_pose": [robot_position[0], robot_position[1], heading[0]],
            "scenario_id": geometry["scenario_id"],
            "geometry_sha256": geometry["sha256"],
        },
    }


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RequestError("SIPP native config must be a mapping")
    return payload


def run(config_path: Path) -> int:
    """Serve native-command requests using the existing bounded SIPP search adapter."""
    config = _load_config(config_path)
    planner = build_sipp_lattice_search_adapter(config)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise RequestError("request must be a JSON object")
            command = planner.plan(_occupancy_observation(request, config))
            diagnostics = planner.diagnostics().get("last_decision", {})
            print(
                json.dumps(
                    {
                        "linear_velocity": float(command[0]),
                        "angular_velocity": float(command[1]),
                        "sipp_diagnostics": diagnostics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        # The persistent protocol must leave callers with a structured, non-zero
        # result for every ordinary request-time failure, including planner bugs.
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"error": str(exc), "status": "invalid_request"}), file=sys.stderr)
            return 2
    return 0


def main() -> int:
    """Parse command-line arguments and run the persistent native SIPP server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    return run(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
