"""LiDAR-derived ego occupancy adapter for grid-route planner tests.

The adapter in this module deliberately consumes only the sensor-fusion
``drive_state`` and ``rays`` payload. It reconstructs a small ego-frame
occupancy grid from ray endpoints and feeds that grid to the existing
``GridRoutePlannerAdapter``. It is testing-only compatibility glue, not a
privileged map reconstruction path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_sf.planner.grid_route import (
    GridRoutePlannerAdapter,
    GridRoutePlannerConfig,
    build_grid_route_config,
)
from robot_sf.sensor.goal_sensor import TARGET_DISTANCE_CAP_M
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS


@dataclass(frozen=True)
class LidarOccupancyGridConfig:
    """Configuration for LiDAR ray endpoint rasterization."""

    resolution: float = 0.2
    width: float = 4.2
    height: float = 4.2
    max_scan_dist: float = 10.0
    visual_angle_portion: float = 1.0
    obstacle_thickness_cells: int = 1
    normalized_observation: bool = True
    target_distance_scale: float = TARGET_DISTANCE_CAP_M
    linear_speed_scale: float = 2.0
    angular_speed_scale: float = 1.0
    robot_radius: float = 0.3

    def __post_init__(self) -> None:
        """Validate rasterization parameters."""
        if self.resolution <= 0.0:
            raise ValueError("resolution must be > 0")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("width and height must be > 0")
        if self.max_scan_dist <= 0.0:
            raise ValueError("max_scan_dist must be > 0")
        if not 0.0 < self.visual_angle_portion <= 1.0:
            raise ValueError("visual_angle_portion must be within (0, 1]")
        if self.obstacle_thickness_cells < 0:
            raise ValueError("obstacle_thickness_cells must be >= 0")
        if self.target_distance_scale <= 0.0:
            raise ValueError("target_distance_scale must be > 0")
        if self.linear_speed_scale <= 0.0 or self.angular_speed_scale <= 0.0:
            raise ValueError("speed scales must be > 0")
        if self.robot_radius <= 0.0:
            raise ValueError("robot_radius must be > 0")

    @property
    def grid_width(self) -> int:
        """Number of grid columns."""
        return int(np.ceil(self.width / self.resolution))

    @property
    def grid_height(self) -> int:
        """Number of grid rows."""
        return int(np.ceil(self.height / self.resolution))


@dataclass(frozen=True)
class LidarGridRouteBuildConfig:
    """Parsed build config for :class:`LidarOccupancyGridRouteAdapter`."""

    lidar_occupancy: LidarOccupancyGridConfig
    grid_route: GridRoutePlannerConfig


def _latest_row(values: Any, *, name: str) -> np.ndarray:
    """Return the newest row from a scalar, vector, or stacked sensor array."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim >= 2:
        arr = arr.reshape((-1, arr.shape[-1]))[-1]
    else:
        arr = arr.reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr.astype(float, copy=False)


def lidar_ray_angles(num_rays: int, *, visual_angle_portion: float = 1.0) -> np.ndarray:
    """Return ray angles matching :mod:`robot_sf.sensor.range_sensor` convention."""
    if num_rays <= 0:
        raise ValueError("num_rays must be > 0")
    if not 0.0 < visual_angle_portion <= 1.0:
        raise ValueError("visual_angle_portion must be within (0, 1]")
    lower = -np.pi * visual_angle_portion
    upper = np.pi * visual_angle_portion
    return np.linspace(lower, upper, int(num_rays) + 1, dtype=float)[:-1]


def lidar_rays_to_ego_occupancy_grid(
    rays: Any,
    config: LidarOccupancyGridConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rasterize LiDAR endpoint returns into an ego-frame occupancy grid.

    Returns:
        tuple[np.ndarray, dict[str, Any]]: A three-channel grid
        ``[obstacles, pedestrians, combined]`` plus metadata compatible with
        ``OccupancyAwarePlannerMixin``.
    """
    cfg = config or LidarOccupancyGridConfig()
    ray_values = _latest_row(rays, name=OBS_RAYS)
    if bool(cfg.normalized_observation):
        ray_values = ray_values * float(cfg.max_scan_dist)
    ray_values = np.clip(ray_values, 0.0, float(cfg.max_scan_dist))

    grid = np.zeros((3, cfg.grid_height, cfg.grid_width), dtype=np.float32)
    origin = np.array([-float(cfg.width) / 2.0, -float(cfg.height) / 2.0], dtype=float)
    angles = lidar_ray_angles(
        int(ray_values.size),
        visual_angle_portion=float(cfg.visual_angle_portion),
    )

    for distance, angle in zip(ray_values, angles, strict=True):
        # Max-range rays are treated as no hit: the sensor observed free range
        # up to its cap, but did not reveal an obstacle endpoint.
        if distance <= 0.0 or distance >= float(cfg.max_scan_dist):
            continue
        x = float(distance * np.cos(angle))
        y = float(distance * np.sin(angle))
        col = int((x - origin[0]) / float(cfg.resolution))
        row = int((y - origin[1]) / float(cfg.resolution))
        if row < 0 or row >= cfg.grid_height or col < 0 or col >= cfg.grid_width:
            continue
        radius = int(cfg.obstacle_thickness_cells)
        r0 = max(0, row - radius)
        r1 = min(cfg.grid_height, row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(cfg.grid_width, col + radius + 1)
        grid[0, r0:r1, c0:c1] = 1.0
        grid[2, r0:r1, c0:c1] = 1.0

    meta: dict[str, Any] = {
        "origin": origin.tolist(),
        "resolution": [float(cfg.resolution)],
        "size": [float(cfg.width), float(cfg.height)],
        "use_ego_frame": [1.0],
        "center_on_robot": [1.0],
        "channel_indices": [0, 1, -1, 2],
        "robot_pose": [0.0, 0.0, 0.0],
    }
    return grid, meta


def sensor_fusion_to_grid_route_observation(
    observation: dict[str, Any],
    config: LidarOccupancyGridConfig | None = None,
) -> dict[str, Any]:
    """Build a grid-route observation from normalized sensor-fusion inputs.

    Returns:
        dict[str, Any]: Structured observation compatible with
        :class:`GridRoutePlannerAdapter`.
    """
    cfg = config or LidarOccupancyGridConfig()
    if not isinstance(observation, dict):
        raise ValueError("observation must be a mapping")
    if OBS_DRIVE_STATE not in observation or OBS_RAYS not in observation:
        raise ValueError("observation must contain drive_state and rays")

    drive = _latest_row(observation[OBS_DRIVE_STATE], name=OBS_DRIVE_STATE)
    if drive.size < 5:
        raise ValueError("drive_state must contain speed, turn rate, distance, and angles")
    if bool(cfg.normalized_observation):
        drive = drive.copy()
        drive[0] *= float(cfg.linear_speed_scale)
        drive[1] *= float(cfg.angular_speed_scale)
        drive[2] *= float(cfg.target_distance_scale)
        drive[3] *= np.pi
        drive[4] *= np.pi

    target_distance = float(np.clip(drive[2], 0.0, float(cfg.target_distance_scale)))
    target_angle = float(np.clip(drive[3], -np.pi, np.pi))
    goal = [
        float(target_distance * np.cos(target_angle)),
        float(target_distance * np.sin(target_angle)),
    ]
    grid, meta = lidar_rays_to_ego_occupancy_grid(observation[OBS_RAYS], cfg)
    return {
        "robot": {
            "position": [0.0, 0.0],
            "heading": [0.0],
            "speed": [float(drive[0])],
            "radius": [float(cfg.robot_radius)],
        },
        "goal": {"current": goal, "next": goal},
        "pedestrians": {"positions": [], "velocities": [], "count": [0], "radius": [0.3]},
        "occupancy_grid": grid,
        "occupancy_grid_meta": meta,
    }


class LidarOccupancyGridRouteAdapter:
    """Testing-only LiDAR-to-grid-route adapter."""

    def __init__(
        self,
        *,
        config: LidarOccupancyGridConfig | None = None,
        grid_route: GridRoutePlannerAdapter | None = None,
    ) -> None:
        """Initialize the adapter around a grid-route planner instance."""
        self.config = config or LidarOccupancyGridConfig()
        self.grid_route = grid_route or GridRoutePlannerAdapter()
        self._last_status = "not_run"
        self._last_error: str | None = None
        self._last_occupied_cells = 0

    def reset(self, seed: int | None = None) -> None:
        """Reset wrapped planner state when supported."""
        reset = getattr(self.grid_route, "reset", None)
        if callable(reset):
            if seed is None:
                reset()
            else:
                try:
                    reset(seed=seed)
                except TypeError:
                    reset()
        self._last_status = "not_run"
        self._last_error = None
        self._last_occupied_cells = 0

    def adapt_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Convert sensor-fusion observation into grid-route observation.

        Returns:
            dict[str, Any]: Structured grid-route observation.
        """
        return sensor_fusion_to_grid_route_observation(observation, self.config)

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Plan from LiDAR-only sensor-fusion inputs, failing closed to stop.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        try:
            adapted = self.adapt_observation(observation)
            grid = np.asarray(adapted["occupancy_grid"], dtype=float)
            self._last_occupied_cells = int(np.count_nonzero(grid[2] >= 0.5))
            linear, angular = self.grid_route.plan(adapted)
        except Exception as exc:  # noqa: BLE001 - benchmark adapter must fail closed.
            self._last_status = "not_available"
            self._last_error = str(exc)
            self._last_occupied_cells = 0
            return 0.0, 0.0
        self._last_status = "ok"
        self._last_error = None
        return float(linear), float(angular)

    def diagnostics(self) -> dict[str, Any]:
        """Expose adapter contract and last-run status for benchmark metadata.

        Returns:
            dict[str, Any]: JSON-compatible adapter diagnostics.
        """
        return {
            "adapter": "LidarOccupancyGridRouteAdapter",
            "status": self._last_status,
            "error": self._last_error,
            "observation_level": "lidar_2d",
            "execution_mode": "adapter",
            "runtime_inputs": [OBS_DRIVE_STATE, OBS_RAYS],
            "derived_payload": "ego_occupancy_grid",
            "forbidden_runtime_inputs_not_read": [
                "robot",
                "pedestrians",
                "occupancy_grid",
                "map",
                "static_obstacles",
            ],
            "occupied_cells": self._last_occupied_cells,
            "grid_resolution": float(self.config.resolution),
            "grid_size": [float(self.config.width), float(self.config.height)],
            "normalized_observation": bool(self.config.normalized_observation),
        }


def build_lidar_grid_route_config(cfg: dict[str, Any] | None) -> LidarGridRouteBuildConfig:
    """Build parsed LiDAR occupancy and grid-route planner configs.

    Returns:
        LidarGridRouteBuildConfig: Parsed adapter and wrapped planner config.
    """
    payload = cfg if isinstance(cfg, dict) else {}
    lidar_payload = payload.get("lidar_occupancy")
    if not isinstance(lidar_payload, dict):
        lidar_payload = payload
    grid_payload = payload.get("grid_route")
    if not isinstance(grid_payload, dict):
        grid_payload = payload

    def _get(mapping: dict[str, Any], key: str, default: Any) -> Any:
        """Return a config value, treating explicit YAML null as absent."""
        value = mapping.get(key, default)
        return default if value is None else value

    return LidarGridRouteBuildConfig(
        lidar_occupancy=LidarOccupancyGridConfig(
            resolution=float(_get(lidar_payload, "resolution", 0.2)),
            width=float(_get(lidar_payload, "width", 4.2)),
            height=float(_get(lidar_payload, "height", 4.2)),
            max_scan_dist=float(_get(lidar_payload, "max_scan_dist", 10.0)),
            visual_angle_portion=float(_get(lidar_payload, "visual_angle_portion", 1.0)),
            obstacle_thickness_cells=int(_get(lidar_payload, "obstacle_thickness_cells", 1)),
            normalized_observation=bool(_get(lidar_payload, "normalized_observation", True)),
            target_distance_scale=float(
                _get(lidar_payload, "target_distance_scale", TARGET_DISTANCE_CAP_M)
            ),
            linear_speed_scale=float(_get(lidar_payload, "linear_speed_scale", 2.0)),
            angular_speed_scale=float(_get(lidar_payload, "angular_speed_scale", 1.0)),
            robot_radius=float(_get(lidar_payload, "robot_radius", 0.3)),
        ),
        grid_route=build_grid_route_config(grid_payload),
    )


def build_lidar_grid_route_adapter(
    cfg: dict[str, Any] | None,
) -> LidarOccupancyGridRouteAdapter:
    """Build the testing-only LiDAR occupancy grid-route adapter.

    Returns:
        LidarOccupancyGridRouteAdapter: Configured adapter instance.
    """
    parsed = build_lidar_grid_route_config(cfg)
    return LidarOccupancyGridRouteAdapter(
        config=parsed.lidar_occupancy,
        grid_route=GridRoutePlannerAdapter(config=parsed.grid_route),
    )


__all__ = [
    "LidarGridRouteBuildConfig",
    "LidarOccupancyGridConfig",
    "LidarOccupancyGridRouteAdapter",
    "build_lidar_grid_route_adapter",
    "build_lidar_grid_route_config",
    "lidar_ray_angles",
    "lidar_rays_to_ego_occupancy_grid",
    "sensor_fusion_to_grid_route_observation",
]
