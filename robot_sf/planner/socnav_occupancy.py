"""Occupancy-grid helpers shared by SocNav-family planner adapters."""

from typing import Any

import numpy as np

from robot_sf.nav.occupancy_grid import OBSERVATION_CHANNEL_ORDER
from robot_sf.nav.occupancy_grid_utils import world_to_ego


class OccupancyAwarePlannerMixin:
    """Shared helpers for planners that can leverage occupancy grid observations."""

    _CHANNEL_KEYS = tuple(channel.value for channel in OBSERVATION_CHANNEL_ORDER)

    @staticmethod
    def _as_1d_float(values: Any, *, pad: int | None = None, default: float = 0.0) -> np.ndarray:
        """Normalize metadata values to at least 1D float array with optional padding.

        Returns:
            np.ndarray: At-least-1D float array, padded to length ``pad`` when provided.
        """
        arr = np.atleast_1d(np.asarray(values, dtype=float))
        if pad is not None and arr.size < pad:
            arr = np.pad(arr, (0, pad - arr.size), constant_values=default)
        return arr

    def _extract_grid_payload(self, observation: dict) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Extract occupancy grid tensor and metadata from observation.

        Returns:
            tuple[np.ndarray, dict[str, Any]] | None: Grid array and metadata dict
            when present, otherwise ``None``.
        """
        grid = observation.get("occupancy_grid")
        if grid is None:
            return None

        # Reconstruct metadata from flattened fields (SB3 compatibility format)
        meta = {}
        for key in (
            "origin",
            "resolution",
            "size",
            "use_ego_frame",
            "center_on_robot",
            "channel_indices",
            "robot_pose",
        ):
            flat_key = f"occupancy_grid_meta_{key}"
            if flat_key in observation:
                meta[key] = observation[flat_key]

        # If no metadata fields found, try old format (backward compatibility)
        if not meta:
            meta = observation.get("occupancy_grid_meta")

        if meta is None or not meta:
            return None

        try:
            grid_arr = np.asarray(grid)
        except (TypeError, ValueError):
            return None
        if grid_arr.ndim < 3:
            return None
        return grid_arr, meta

    def _cache_grid_payload(self, observation: dict) -> tuple[np.ndarray, dict[str, Any]]:
        """Return one per-step grid payload, including a no-grid sentinel.

        The empty payload distinguishes a cached "no grid" result from an omitted cache
        argument, so downstream rollout helpers do not repeat extraction on every candidate.

        Returns:
            tuple[np.ndarray, dict[str, Any]]: Grid payload or an empty no-grid sentinel.
        """
        payload = self._extract_grid_payload(observation)
        if payload is not None:
            return payload
        return np.empty((0, 0, 0), dtype=float), {}

    def _socnav_fields(self, observation: dict) -> tuple[dict, dict, dict]:
        """Normalize SocNav observation (nested or flattened) into standard dicts.

        Returns:
            tuple[dict, dict, dict]: (robot_state, goal_state, ped_state) dictionaries.
        """
        if "robot" in observation:
            robot_state = observation["robot"]
            goal_state = observation.get("goal", {})
            ped_state = observation.get("pedestrians", {})
        else:
            pos_arr = self._as_1d_float(observation.get("robot_position", [0.0, 0.0]), pad=2)
            robot_state = {
                "position": pos_arr,
                "heading": self._as_1d_float(observation.get("robot_heading", [0.0]), pad=1),
                "speed": self._as_1d_float(observation.get("robot_speed", [0.0]), pad=1),
                "radius": self._as_1d_float(observation.get("robot_radius", [0.0]), pad=1),
            }
            goal_state = {
                "current": self._as_1d_float(observation.get("goal_current", [0.0, 0.0]), pad=2),
                "next": self._as_1d_float(observation.get("goal_next", [0.0, 0.0]), pad=2),
            }
            ped_state = {
                "positions": observation.get("pedestrians_positions"),
                "velocities": observation.get("pedestrians_velocities"),
                "count": self._as_1d_float(observation.get("pedestrians_count", [0]), pad=1),
                "radius": self._as_1d_float(observation.get("pedestrians_radius", [0.0]), pad=1)[0],
            }
        return robot_state, goal_state, ped_state

    def _grid_channel_index(self, meta: dict[str, Any], key: str) -> int:
        """Return channel index for a semantic key, or -1 when unavailable.

        Returns:
            int: Channel index or -1 when the channel is missing.
        """
        indices = meta.get("channel_indices")
        if indices is None:
            return -1
        try:
            pos = self._CHANNEL_KEYS.index(key)
            idx_arr = self._as_1d_float(indices)
            if pos >= idx_arr.size:
                return -1
            return int(idx_arr[pos])
        except (ValueError, TypeError, IndexError):
            return -1

    def _preferred_channel(self, meta: dict[str, Any]) -> int:
        """Prefer combined channel, else obstacles, else pedestrians.

        Returns:
            int: Channel index or -1 when no occupancy channel is present.
        """
        combined_idx = self._grid_channel_index(meta, "combined")
        if combined_idx >= 0:
            return combined_idx
        obstacle_idx = self._grid_channel_index(meta, "obstacles")
        if obstacle_idx >= 0:
            return obstacle_idx
        return self._grid_channel_index(meta, "pedestrians")

    def _world_to_grid(
        self,
        point: np.ndarray,
        meta: dict[str, Any],
        grid_shape: tuple[int, int],
    ) -> tuple[int, int] | None:
        """Convert world coordinates to grid row/col using metadata.

        Returns:
            tuple[int, int] | None: Grid indices or None when point is out of bounds/invalid.
        """
        origin = np.asarray(meta.get("origin", [0.0, 0.0]), dtype=float)
        size = np.asarray(meta.get("size", [0.0, 0.0]), dtype=float)
        resolution_arr = self._as_1d_float(meta.get("resolution", [0.0]))
        if resolution_arr.size == 0 or resolution_arr[0] <= 0:
            return None
        resolution = float(resolution_arr[0])

        origin = self._as_1d_float(origin, pad=2)
        size = self._as_1d_float(size, pad=2)

        use_ego_arr = self._as_1d_float(meta.get("use_ego_frame", [0.0]), pad=1)
        use_ego = bool(use_ego_arr[0] > 0.5)
        pose_arr = self._as_1d_float(meta.get("robot_pose", [0.0, 0.0, 0.0]), pad=3)
        if use_ego:
            pose_tuple = ((float(pose_arr[0]), float(pose_arr[1])), float(pose_arr[2]))
            point = np.asarray(
                world_to_ego(float(point[0]), float(point[1]), pose_tuple), dtype=float
            )

        local = point - origin
        if np.any(local < 0.0) or local[0] > size[0] or local[1] > size[1]:
            return None

        col = int(local[0] / resolution)
        row = int(local[1] / resolution)
        row = min(max(row, 0), grid_shape[0] - 1)
        col = min(max(col, 0), grid_shape[1] - 1)
        return row, col

    def _grid_value(
        self,
        point: np.ndarray,
        grid: np.ndarray,
        meta: dict[str, Any],
        channel_idx: int,
    ) -> float:
        """Return occupancy value at a world point (treat OOB as occupied).

        Returns:
            float: Occupancy value in [0, 1], 1.0 when out of bounds.
        """
        if channel_idx < 0:
            return 0.0
        if grid.ndim < 3:
            return 1.0
        channels, height, width = grid.shape[0], grid.shape[1], grid.shape[2]
        grid_shape = (height, width)
        indices = self._world_to_grid(point, meta, grid_shape)
        if indices is None:
            return 1.0
        row, col = indices
        if channel_idx >= channels:
            return 1.0
        return float(grid[channel_idx, row, col])

    def _obstacle_grid_payload(
        self, observation: dict
    ) -> tuple[np.ndarray, dict[str, Any], int, float] | None:
        """Return validated obstacle-grid payload shared by obstacle-aware planners.

        Returns:
            tuple[np.ndarray, dict[str, Any], int, float] | None: Grid, metadata, obstacle
            channel, and resolution when available.
        """
        payload = self._extract_grid_payload(observation)
        if payload is None:
            return None
        grid, meta = payload
        if grid.ndim < 3:
            return None
        channel_idx = self._grid_channel_index(meta, "obstacles")
        if channel_idx < 0:
            channel_idx = self._grid_channel_index(meta, "combined")
        if channel_idx < 0 or channel_idx >= grid.shape[0]:
            return None
        resolution_arr = self._as_1d_float(meta.get("resolution", [0.0]), pad=1)
        resolution = float(resolution_arr[0])
        if resolution <= 0.0:
            return None
        return grid, meta, channel_idx, resolution

    def _path_penalty(
        self,
        robot_pos: np.ndarray,
        direction: np.ndarray,
        observation: dict,
        base_distance: float,
        num_samples: int,
    ) -> tuple[float, float]:
        """Compute occupancy penalty along a candidate heading.

        Returns:
            tuple[float, float]: Mean obstacle and pedestrian occupancy along the sample line.
        """
        grid_payload = self._extract_grid_payload(observation)
        if grid_payload is None or np.linalg.norm(direction) < 1e-6:
            return 0.0, 0.0

        grid, meta = grid_payload
        if grid.ndim < 3:
            return 0.0, 0.0
        channel_idx = self._preferred_channel(meta)
        ped_idx = self._grid_channel_index(meta, "pedestrians")
        direction = direction / (np.linalg.norm(direction) + 1e-9)

        samples = np.linspace(base_distance / num_samples, base_distance, num_samples)
        obstacle_vals: list[float] = []
        ped_vals: list[float] = []
        for dist in samples:
            point = robot_pos + direction * dist
            obstacle_vals.append(self._grid_value(point, grid, meta, channel_idx))
            if ped_idx >= 0:
                ped_vals.append(self._grid_value(point, grid, meta, ped_idx))

        obstacle_penalty = float(np.mean(obstacle_vals)) if obstacle_vals else 0.0
        ped_penalty = float(np.mean(ped_vals)) if ped_vals else 0.0
        return obstacle_penalty, ped_penalty

    def _select_safe_heading(
        self,
        robot_pos: np.ndarray,
        base_direction: np.ndarray,
        observation: dict,
        sweep: float,
        num_candidates: int,
        lookahead: float,
        weight: float,
        angle_weight: float,
    ) -> tuple[np.ndarray, float]:
        """Pick heading that balances goal alignment and occupancy clearance.

        Returns:
            tuple[np.ndarray, float]: Chosen direction vector and the associated occupancy penalty.
        """
        if np.linalg.norm(base_direction) < 1e-6 or num_candidates <= 1:
            return base_direction, 0.0

        base_dir = base_direction / (np.linalg.norm(base_direction) + 1e-9)
        angles = np.linspace(-sweep / 2, sweep / 2, num_candidates)
        best_dir = base_dir
        best_cost = float("inf")
        best_penalty = 0.0

        for angle in angles:
            rot = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ],
                dtype=float,
            )
            candidate = rot @ base_dir
            obstacle_penalty, ped_penalty = self._path_penalty(
                robot_pos, candidate, observation, lookahead, max(2, num_candidates)
            )
            penalty = obstacle_penalty + 0.5 * ped_penalty
            angle_cost = abs(angle) / (sweep / 2 if sweep > 0 else 1.0)
            cost = weight * penalty + angle_weight * angle_cost
            if cost < best_cost:
                best_cost = cost
                best_dir = candidate
                best_penalty = penalty

        return best_dir, best_penalty

    def _get_safe_heading(
        self, robot_pos: np.ndarray, base_direction: np.ndarray, observation: dict
    ) -> tuple[np.ndarray, float]:
        """Helper to call _select_safe_heading with config parameters.

        Returns:
            tuple[np.ndarray, float]: Direction vector and occupancy penalty.
        """
        # Type checker can't infer config attribute from mixin class.
        return self._select_safe_heading(  # type: ignore[attr-defined]
            robot_pos,
            base_direction,
            observation,
            sweep=self.config.occupancy_heading_sweep,  # type: ignore[attr-defined]
            num_candidates=self.config.occupancy_candidates,  # type: ignore[attr-defined]
            lookahead=self.config.occupancy_lookahead,  # type: ignore[attr-defined]
            weight=self.config.occupancy_weight,  # type: ignore[attr-defined]
            angle_weight=self.config.occupancy_angle_weight,  # type: ignore[attr-defined]
        )
