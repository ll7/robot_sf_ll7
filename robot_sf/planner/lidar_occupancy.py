"""LiDAR-derived ego occupancy adapter for testing-only local planners."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from robot_sf.errors import RobotSfError


@dataclass(frozen=True)
class LidarOccupancyGridConfig:
    """Configuration for converting range rays into an ego-frame occupancy grid."""

    resolution: float = 0.2
    width: float = 10.0
    height: float = 10.0
    max_range: float = 10.0
    angle_min: float = -np.pi
    angle_max: float = np.pi
    obstacle_inflation_cells: int = 1
    normalized_rays: bool = True
    normalized_drive_state: bool = True
    target_distance_scale: float = 10.0
    target_angle_scale: float = np.pi
    robot_radius: float = 0.3


class LidarOccupancyAdapterError(RobotSfError, ValueError):
    """Raised when a LiDAR observation cannot be converted safely."""


def build_lidar_occupancy_config(cfg: dict[str, Any] | None) -> LidarOccupancyGridConfig:
    """Build a LiDAR occupancy adapter config from an algorithm config mapping.

    Returns:
        LidarOccupancyGridConfig: Parsed adapter configuration.
    """
    if not isinstance(cfg, dict):
        return LidarOccupancyGridConfig()
    source = cfg.get("lidar_occupancy_adapter")
    payload = source if isinstance(source, dict) else cfg
    defaults = LidarOccupancyGridConfig()
    return LidarOccupancyGridConfig(
        resolution=float(payload.get("lidar_grid_resolution", defaults.resolution)),
        width=float(payload.get("lidar_grid_width", defaults.width)),
        height=float(payload.get("lidar_grid_height", defaults.height)),
        max_range=float(payload.get("lidar_max_range", defaults.max_range)),
        angle_min=float(payload.get("lidar_angle_min", defaults.angle_min)),
        angle_max=float(payload.get("lidar_angle_max", defaults.angle_max)),
        obstacle_inflation_cells=int(
            payload.get("lidar_obstacle_inflation_cells", defaults.obstacle_inflation_cells)
        ),
        normalized_rays=bool(payload.get("lidar_rays_normalized", defaults.normalized_rays)),
        normalized_drive_state=bool(
            payload.get("drive_state_normalized", defaults.normalized_drive_state)
        ),
        target_distance_scale=float(
            payload.get("target_distance_scale", defaults.target_distance_scale)
        ),
        target_angle_scale=float(payload.get("target_angle_scale", defaults.target_angle_scale)),
        robot_radius=float(payload.get("robot_radius", defaults.robot_radius)),
    )


def _latest_row(values: Any, *, key: str) -> np.ndarray:
    """Return the latest row from a possibly stacked observation value."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise LidarOccupancyAdapterError(f"{key} must be numeric") from exc
    if arr.size == 0:
        raise LidarOccupancyAdapterError(f"{key} is empty")
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(-1, arr.shape[-1])[-1]
    return arr.reshape(-1)


def _extract_lidar_rays(
    observation: dict[str, Any], config: LidarOccupancyGridConfig
) -> np.ndarray:
    """Extract latest LiDAR ray distances from a planner observation.

    Returns:
        np.ndarray: Finite ray distances clipped to the configured maximum range.
    """
    for key in ("rays", "lidar_rays", "lidar", "laser_scan", "ranges"):
        if key in observation:
            rays = _latest_row(observation[key], key=key)
            break
    else:
        raise LidarOccupancyAdapterError("LiDAR occupancy adapter requires rays")
    if config.normalized_rays:
        rays = rays * float(config.max_range)
    rays = np.where(np.isfinite(rays), rays, float(config.max_range))
    return np.clip(rays, 0.0, float(config.max_range))


def _extract_goal_from_drive_state(
    observation: dict[str, Any], config: LidarOccupancyGridConfig
) -> np.ndarray:
    """Return an ego-frame goal point from sensor-fusion drive state."""
    if "drive_state" not in observation:
        return np.array([float(config.max_range), 0.0], dtype=float)
    drive_state = _latest_row(observation["drive_state"], key="drive_state")
    if drive_state.size < 4:
        raise LidarOccupancyAdapterError("drive_state must include target distance and angle")
    distance = float(drive_state[2])
    angle = float(drive_state[3])
    if config.normalized_drive_state:
        distance *= float(config.target_distance_scale)
        angle *= float(config.target_angle_scale)
    distance = max(0.0, distance)
    return np.array([distance * np.cos(angle), distance * np.sin(angle)], dtype=float)


def _ray_angles(count: int, config: LidarOccupancyGridConfig) -> np.ndarray:
    """Return ego-frame ray angles matching the range-sensor linspace convention."""
    if count <= 0:
        raise LidarOccupancyAdapterError("at least one LiDAR ray is required")
    angle_min = float(config.angle_min)
    angle_max = float(config.angle_max)
    return _cached_ray_angles(int(count), angle_min, angle_max)


@lru_cache(maxsize=32)
def _cached_ray_angles(count: int, angle_min: float, angle_max: float) -> np.ndarray:
    """Return immutable cached ray angles for stable legacy LiDAR adapter configs."""
    if count == 1:
        angles = np.asarray([angle_min], dtype=float)
        angles.setflags(write=False)
        return angles
    span = angle_max - angle_min
    endpoint = not np.isclose(abs(span), 2.0 * np.pi)
    angles = np.linspace(angle_min, angle_max, count, endpoint=endpoint)
    angles.setflags(write=False)
    return angles


def _inflate_cell(channel: np.ndarray, row: int, col: int, radius: int) -> None:
    """Mark one occupied cell and its local inflation radius."""
    rows, cols = channel.shape
    r0 = max(0, int(row) - radius)
    r1 = min(rows, int(row) + radius + 1)
    c0 = max(0, int(col) - radius)
    c1 = min(cols, int(col) + radius + 1)
    channel[r0:r1, c0:c1] = 1.0


def lidar_rays_to_occupancy_observation(
    observation: dict[str, Any],
    config: LidarOccupancyGridConfig | None = None,
) -> dict[str, Any]:
    """Convert LiDAR rays into a safety-barrier-compatible planner observation.

    Returns:
        dict[str, Any]: Synthetic ego-frame planner observation with only robot/goal
        state derived from allowed ego inputs and occupancy derived from rays.
    """
    cfg = config or LidarOccupancyGridConfig()
    if cfg.resolution <= 0.0 or cfg.width <= 0.0 or cfg.height <= 0.0 or cfg.max_range <= 0.0:
        raise LidarOccupancyAdapterError("LiDAR occupancy grid dimensions must be positive")

    rays = _extract_lidar_rays(observation, cfg)
    goal = _extract_goal_from_drive_state(observation, cfg)
    cols = max(int(np.ceil(float(cfg.width) / float(cfg.resolution))), 1)
    rows = max(int(np.ceil(float(cfg.height) / float(cfg.resolution))), 1)
    origin = np.array([-float(cfg.width) / 2.0, -float(cfg.height) / 2.0], dtype=np.float32)
    grid = np.zeros((3, rows, cols), dtype=np.float32)
    obstacle_channel = grid[0]
    combined_channel = grid[2]
    angles = _ray_angles(rays.size, cfg)
    inflation = max(int(cfg.obstacle_inflation_cells), 0)

    hit_mask = np.isfinite(rays) & (rays > 0.0) & (rays < float(cfg.max_range))
    hit_distances = rays[hit_mask]
    hit_angles = angles[hit_mask]
    endpoints = np.stack(
        (hit_distances * np.cos(hit_angles), hit_distances * np.sin(hit_angles)),
        axis=1,
    )
    cols_arr = np.floor((endpoints[:, 0] - float(origin[0])) / float(cfg.resolution)).astype(int)
    rows_arr = np.floor((endpoints[:, 1] - float(origin[1])) / float(cfg.resolution)).astype(int)

    for row, col in zip(rows_arr, cols_arr, strict=True):
        if row < 0 or row >= rows or col < 0 or col >= cols:
            continue
        _inflate_cell(obstacle_channel, row, col, inflation)
        _inflate_cell(combined_channel, row, col, inflation)

    metadata = {
        "origin": origin,
        "resolution": np.asarray([float(cfg.resolution)], dtype=np.float32),
        "size": np.asarray([float(cfg.width), float(cfg.height)], dtype=np.float32),
        "use_ego_frame": np.asarray([1.0], dtype=np.float32),
        "center_on_robot": np.asarray([1.0], dtype=np.float32),
        "channel_indices": np.asarray([0, 1, -1, 2], dtype=np.int32),
        "robot_pose": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    }
    planner_obs: dict[str, Any] = {
        "robot": {
            "position": np.asarray([0.0, 0.0], dtype=np.float32),
            "heading": np.asarray([0.0], dtype=np.float32),
            "speed": np.asarray([0.0], dtype=np.float32),
            "radius": np.asarray([float(cfg.robot_radius)], dtype=np.float32),
        },
        "goal": {"current": goal.astype(np.float32), "next": np.zeros(2, dtype=np.float32)},
        "pedestrians": {
            "positions": np.zeros((0, 2), dtype=np.float32),
            "velocities": np.zeros((0, 2), dtype=np.float32),
            "count": np.asarray([0], dtype=np.float32),
            "radius": np.asarray([0.0], dtype=np.float32),
        },
        "occupancy_grid": grid,
    }
    planner_obs.update({f"occupancy_grid_meta_{key}": value for key, value in metadata.items()})
    return planner_obs


class LidarOccupancyPlannerAdapter:
    """Wrap an occupancy-aware planner so it consumes LiDAR-derived occupancy only."""

    def __init__(self, planner: Any, config: LidarOccupancyGridConfig | None = None) -> None:
        """Initialize the wrapper with a concrete planner and adapter config."""
        self.planner = planner
        self.config = config or LidarOccupancyGridConfig()
        self._converted_count = 0
        self._unavailable_count = 0
        self._last_error: str | None = None

    def reset(self, seed: int | None = None) -> None:
        """Reset wrapper counters and the wrapped planner when supported."""
        self._converted_count = 0
        self._unavailable_count = 0
        self._last_error = None
        reset = getattr(self.planner, "reset", None)
        if callable(reset):
            if seed is None:
                reset()
                return
            try:
                reset(seed=seed)
            except TypeError:
                reset()

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Convert rays to occupancy and delegate to the wrapped planner.

        Returns:
            tuple[float, float]: Linear and angular command, or a stop command
            when the LiDAR adapter cannot produce a valid planner observation.
        """
        try:
            planner_obs = lidar_rays_to_occupancy_observation(observation, self.config)
        except LidarOccupancyAdapterError as exc:
            self._unavailable_count += 1
            self._last_error = str(exc)
            return 0.0, 0.0
        self._converted_count += 1
        self._last_error = None
        return self.planner.plan(planner_obs)

    def diagnostics(self) -> dict[str, Any]:
        """Return adapter execution metadata for benchmark episode records."""
        return {
            "lidar_occupancy_adapter": {
                "execution_mode": "adapter",
                "source": "lidar_rays",
                "output": "ego_occupancy_grid",
                "converted_observations": self._converted_count,
                "unavailable_observations": self._unavailable_count,
                "last_error": self._last_error,
            }
        }


__all__ = [
    "LidarOccupancyAdapterError",
    "LidarOccupancyGridConfig",
    "LidarOccupancyPlannerAdapter",
    "build_lidar_occupancy_config",
    "lidar_rays_to_occupancy_observation",
]
