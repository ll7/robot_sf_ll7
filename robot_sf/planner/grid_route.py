"""Testing-only occupancy-grid route planner.

This planner recomputes a short 8-connected route over the structured
occupancy grid and tracks the next free waypoint with a bounded unicycle
command. It is intended as a simple topology-aware counterexample to purely
reactive local controllers on static obstacle scenarios.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import pairwise
from math import sqrt
from typing import Any

import numpy as np

from robot_sf.nav.occupancy_grid_utils import ego_to_world
from robot_sf.planner.risk_dwa import _wrap_angle
from robot_sf.planner.socnav import OccupancyAwarePlannerMixin

_DEFAULT_GRID_RESOLUTION = 0.2
_MIN_RESOLUTION = 1e-6
_ROBOT_RADIUS_MARGIN_CELLS = 1


def _finite_or_none(value: Any) -> float | None:
    """Return finite floats for JSON-safe route diagnostics."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


@dataclass
class GridRoutePlannerConfig:
    """Configuration for :class:`GridRoutePlannerAdapter`."""

    max_linear_speed: float = 0.9
    max_angular_speed: float = 1.2
    goal_tolerance: float = 0.25
    heading_gain: float = 1.8
    turn_in_place_angle: float = 0.8
    waypoint_lookahead_cells: int = 5
    waypoint_reached_distance: float = 0.3
    obstacle_threshold: float = 0.5
    obstacle_inflation_cells: int = 1
    clearance_search_cells: int = 5
    stop_distance: float = 0.25
    progress_weight: float = 1.0
    heading_weight: float = 1.0
    clearance_penalty_weight: float = 0.5


class GridRoutePlannerAdapter(OccupancyAwarePlannerMixin):
    """Topology-aware static-obstacle planner over the occupancy grid."""

    _NEIGHBORS: tuple[tuple[int, int, float], ...] = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, sqrt(2.0)),
        (-1, 1, sqrt(2.0)),
        (1, -1, sqrt(2.0)),
        (1, 1, sqrt(2.0)),
    )

    def __init__(self, config: GridRoutePlannerConfig | None = None) -> None:
        """Initialize planner with a static-grid routing configuration."""
        self.config = config or GridRoutePlannerConfig()
        self._last_route_path_key: tuple[Any, ...] | None = None
        self._last_route_path_value: (
            tuple[tuple[tuple[int, int], ...], np.ndarray | None] | None
        ) = None

    def _extract_state(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        """Extract robot and goal state with conservative defaults.

        Returns:
            tuple[np.ndarray, float, np.ndarray, float]:
                Robot position, heading, selected goal point, and robot radius.
        """
        robot_state, goal_state, _ped_state = self._socnav_fields(observation)
        robot_pos = self._as_1d_float(robot_state.get("position", [0.0, 0.0]), pad=2)[:2]
        heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        radius = float(self._as_1d_float(robot_state.get("radius", [0.3]), pad=1)[0])

        goal_next = self._as_1d_float(goal_state.get("next", [0.0, 0.0]), pad=2)[:2]
        goal_current = self._as_1d_float(goal_state.get("current", [0.0, 0.0]), pad=2)[:2]
        if np.linalg.norm(goal_current - robot_pos) > float(self.config.goal_tolerance):
            goal = goal_current
        elif np.linalg.norm(goal_next - robot_pos) > 1e-6:
            goal = goal_next
        else:
            goal = goal_current
        return robot_pos, heading, goal, radius

    def _nominal_command(
        self,
        *,
        robot_pos: np.ndarray,
        heading: float,
        target: np.ndarray,
    ) -> tuple[float, float, float]:
        """Track a local waypoint with bounded unicycle commands.

        Returns:
            tuple[float, float, float]:
                Linear speed, angular speed, and heading error.
        """
        target_vec = target - robot_pos
        target_dist = float(np.linalg.norm(target_vec))
        if target_dist <= float(self.config.waypoint_reached_distance):
            return 0.0, 0.0, 0.0

        desired_heading = float(np.arctan2(target_vec[1], target_vec[0]))
        heading_error = _wrap_angle(desired_heading - heading)
        angular = float(
            np.clip(
                float(self.config.heading_gain) * heading_error,
                -float(self.config.max_angular_speed),
                float(self.config.max_angular_speed),
            )
        )
        if abs(heading_error) >= float(self.config.turn_in_place_angle):
            return 0.0, angular, heading_error

        alignment = max(0.0, np.cos(heading_error))
        linear = float(
            np.clip(
                float(self.config.progress_weight)
                * float(self.config.heading_weight)
                * target_dist
                * alignment,
                0.0,
                float(self.config.max_linear_speed),
            )
        )
        return linear, angular, heading_error

    def _inflate_obstacles(
        self, blocked: np.ndarray, *, inflation_cells: int | None = None
    ) -> np.ndarray:
        """Dilate blocked cells by a small integer radius.

        Returns:
            np.ndarray: Inflated binary occupancy grid.
        """
        radius = max(
            int(
                self.config.obstacle_inflation_cells if inflation_cells is None else inflation_cells
            ),
            0,
        )
        if radius <= 0 or blocked.size == 0:
            return blocked

        inflated = blocked.copy()
        occupied = np.argwhere(blocked)
        for row, col in occupied:
            r0 = max(0, int(row) - radius)
            r1 = min(blocked.shape[0], int(row) + radius + 1)
            c0 = max(0, int(col) - radius)
            c1 = min(blocked.shape[1], int(col) + radius + 1)
            inflated[r0:r1, c0:c1] = True
        return inflated

    def _blocked_grid(
        self, grid: np.ndarray, meta: dict[str, Any], radius: float
    ) -> np.ndarray | None:
        """Build an inflated binary occupancy grid from the preferred channel.

        Returns:
            np.ndarray | None: Inflated obstacle grid, or ``None`` when no usable
            occupancy channel is present.
        """
        channel_idx = self._preferred_channel(meta)
        if channel_idx < 0 or channel_idx >= grid.shape[0]:
            return None
        channel_grid = np.asarray(grid[channel_idx], dtype=float)
        blocked = channel_grid >= float(self.config.obstacle_threshold)
        resolution = float(
            self._as_1d_float(meta.get("resolution", [_DEFAULT_GRID_RESOLUTION]), pad=1)[0]
        )
        dynamic_radius = max(
            int(np.ceil(radius / max(resolution, _MIN_RESOLUTION))) - _ROBOT_RADIUS_MARGIN_CELLS,
            0,
        )
        effective_inflation = max(dynamic_radius, int(self.config.obstacle_inflation_cells))
        return self._inflate_obstacles(blocked, inflation_cells=effective_inflation)

    @staticmethod
    def _nearest_free(
        blocked: np.ndarray, start: tuple[int, int], limit: int
    ) -> tuple[int, int] | None:
        """Find the nearest non-blocked cell around ``start`` within ``limit`` cells.

        Returns:
            tuple[int, int] | None: Nearest free grid cell, or ``None`` when no
            candidate is found inside the search radius.
        """
        row, col = start
        if 0 <= row < blocked.shape[0] and 0 <= col < blocked.shape[1] and not blocked[row, col]:
            return start
        for radius in range(1, max(limit, 1) + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr = row + dr
                    cc = col + dc
                    if rr < 0 or rr >= blocked.shape[0] or cc < 0 or cc >= blocked.shape[1]:
                        continue
                    if blocked[rr, cc]:
                        continue
                    return (rr, cc)
        return None

    @staticmethod
    def _compute_clearance_map(blocked: np.ndarray) -> np.ndarray:
        """BFS clearance map: distance (in cells) from each cell to the nearest obstacle.

        Obstacle cells are seeded at 0 and free cells receive the 4-connected
        BFS distance.  The map is used by :meth:`_astar` to add a small
        ``clearance_penalty_weight / (clearance + 1)`` cost per step so that
        A* naturally picks routes that run through the centre of corridors.

        Returns:
            np.ndarray: Per-cell clearance. Obstacle cells are 0; free cells are >= 1
            when at least one obstacle exists, and remain 0 when the grid has no
            blocked cells.
        """
        rows, cols = blocked.shape
        clearance = np.full(blocked.shape, np.inf, dtype=float)
        clearance[blocked] = 0.0
        visited = blocked.copy()  # obstacle cells are already "visited"

        queue: deque[tuple[int, int]] = deque()
        occ_rows, occ_cols = np.where(blocked)
        for r, c in zip(occ_rows.tolist(), occ_cols.tolist(), strict=True):
            queue.append((int(r), int(c)))

        while queue:
            r, c = queue.popleft()
            dist_next = clearance[r, c] + 1.0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                    visited[nr, nc] = True
                    clearance[nr, nc] = dist_next
                    queue.append((nr, nc))

        return clearance

    @staticmethod
    def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        """Euclidean heuristic for A* search.

        Returns:
            float: Heuristic distance between the cells.
        """
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _astar(
        self,
        blocked: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        clearance_map: np.ndarray | None = None,
    ) -> list[tuple[int, int]]:
        """Compute an 8-connected grid path between start and goal.

        When ``clearance_map`` is provided, each step adds a small
        ``clearance_penalty_weight / (clearance + 1)`` penalty so that A*
        prefers routes that pass through the centre of corridors rather than
        hugging the obstacle boundary.  This is the primary fix for the
        ``narrow_passage`` failure pattern.

        Returns:
            list[tuple[int, int]]: Ordered row/col path, or an empty list when no
            route exists.
        """
        penalty_weight = (
            float(self.config.clearance_penalty_weight) if clearance_map is not None else 0.0
        )
        frontier: list[tuple[float, tuple[int, int]]] = []
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost_so_far: dict[tuple[int, int], float] = {start: 0.0}
        heappush(frontier, (0.0, start))

        while frontier:
            _priority, current = heappop(frontier)
            if current == goal:
                break
            for dr, dc, step_cost in self._NEIGHBORS:
                nxt = (current[0] + dr, current[1] + dc)
                if (
                    nxt[0] < 0
                    or nxt[0] >= blocked.shape[0]
                    or nxt[1] < 0
                    or nxt[1] >= blocked.shape[1]
                ):
                    continue
                if blocked[nxt]:
                    continue
                clearance_penalty = (
                    penalty_weight / (float(clearance_map[nxt]) + 1.0)
                    if clearance_map is not None
                    else 0.0
                )
                new_cost = cost_so_far[current] + step_cost * (1.0 + clearance_penalty)
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self._heuristic(nxt, goal)
                    heappush(frontier, (priority, nxt))
                    came_from[nxt] = current

        if goal not in came_from:
            return []

        path = [goal]
        cursor: tuple[int, int] | None = goal
        while cursor is not None:
            cursor = came_from.get(cursor)
            if cursor is not None:
                path.append(cursor)
        path.reverse()
        return path

    def _grid_to_world(
        self,
        rc: tuple[int, int],
        meta: dict[str, Any],
    ) -> np.ndarray:
        """Convert grid row/col to a world-space cell center.

        Returns:
            np.ndarray: World-space ``(x, y)`` position at the cell center.
        """
        row, col = rc
        origin = self._as_1d_float(meta.get("origin", [0.0, 0.0]), pad=2)[:2]
        resolution = float(self._as_1d_float(meta.get("resolution", [0.2]), pad=1)[0])
        point = np.array(
            [
                origin[0] + (float(col) + 0.5) * resolution,
                origin[1] + (float(row) + 0.5) * resolution,
            ],
            dtype=float,
        )
        use_ego = bool(self._as_1d_float(meta.get("use_ego_frame", [0.0]), pad=1)[0] > 0.5)
        if not use_ego:
            return point
        pose_arr = self._as_1d_float(meta.get("robot_pose", [0.0, 0.0, 0.0]), pad=3)
        pose = ((float(pose_arr[0]), float(pose_arr[1])), float(pose_arr[2]))
        world_x, world_y = ego_to_world(float(point[0]), float(point[1]), pose)
        return np.array([world_x, world_y], dtype=float)

    def _front_clearance(
        self,
        *,
        robot_pos: np.ndarray,
        heading: float,
        grid: np.ndarray,
        meta: dict[str, Any],
    ) -> float:
        """Compute immediate frontal clearance in meters.

        Returns:
            float: Distance to the first occupied cell ahead, or ``inf`` when the
            probe stays clear.
        """
        channel_idx = self._preferred_channel(meta)
        if channel_idx < 0:
            return float("inf")
        resolution = float(self._as_1d_float(meta.get("resolution", [0.2]), pad=1)[0])
        step = max(resolution * 0.75, 0.05)
        direction = np.array([np.cos(heading), np.sin(heading)], dtype=float)
        for distance in np.arange(
            step, max(float(self.config.stop_distance) * 1.5, step) + step, step
        ):
            point = robot_pos + direction * float(distance)
            if self._grid_value(point, grid, meta, channel_idx) >= float(
                self.config.obstacle_threshold
            ):
                return float(distance)
        return float("inf")

    def _route_path_cache_key(
        self,
        *,
        cache_key: Any | None,
        robot_pos: np.ndarray,
        goal: np.ndarray,
        radius: float,
        grid: np.ndarray,
        meta: dict[str, Any],
    ) -> tuple[Any, ...] | None:
        """Build a last-call route cache key for repeated observation consumers.

        Returns:
            tuple[Any, ...] | None: Cache key, or ``None`` when caching is disabled.
        """
        if cache_key is None:
            return None
        return (
            cache_key,
            id(grid),
            id(meta),
            tuple(float(value) for value in np.asarray(robot_pos, dtype=float)[:2]),
            tuple(float(value) for value in np.asarray(goal, dtype=float)[:2]),
            float(radius),
            float(self.config.obstacle_threshold),
            int(self.config.obstacle_inflation_cells),
            int(self.config.clearance_search_cells),
            float(self.config.clearance_penalty_weight),
        )

    def _cache_route_path(
        self,
        key: tuple[Any, ...] | None,
        path: list[tuple[int, int]],
        clearance_map: np.ndarray | None,
    ) -> tuple[list[tuple[int, int]], np.ndarray | None]:
        """Store and return the latest route path result.

        Returns:
            tuple[list[tuple[int, int]], np.ndarray | None]: Route path and clearance map.
        """
        if key is not None:
            self._last_route_path_key = key
            self._last_route_path_value = (tuple(path), clearance_map)
        return path, clearance_map

    def _route_path(
        self,
        *,
        robot_pos: np.ndarray,
        goal: np.ndarray,
        radius: float,
        grid: np.ndarray,
        meta: dict[str, Any],
        cache_key: Any | None = None,
    ) -> tuple[list[tuple[int, int]], np.ndarray | None]:
        """Compute an occupancy-grid route and optional clearance map.

        Returns:
            tuple[list[tuple[int, int]], np.ndarray | None]: Grid-cell route and
            optional clearance map, or ``([], None)`` when no route is available.
        """
        route_cache_key = self._route_path_cache_key(
            cache_key=cache_key,
            robot_pos=robot_pos,
            goal=goal,
            radius=radius,
            grid=grid,
            meta=meta,
        )
        if (
            route_cache_key is not None
            and route_cache_key == self._last_route_path_key
            and self._last_route_path_value is not None
        ):
            cached_path, cached_clearance_map = self._last_route_path_value
            return list(cached_path), cached_clearance_map

        blocked = self._blocked_grid(grid, meta, radius)
        if blocked is None:
            return self._cache_route_path(route_cache_key, [], None)

        start_rc = self._world_to_grid(robot_pos, meta, (blocked.shape[0], blocked.shape[1]))
        goal_rc = self._world_to_grid(goal, meta, (blocked.shape[0], blocked.shape[1]))
        if start_rc is None or goal_rc is None:
            return self._cache_route_path(route_cache_key, [], None)

        free_start = self._nearest_free(blocked, start_rc, int(self.config.clearance_search_cells))
        free_goal = self._nearest_free(blocked, goal_rc, int(self.config.clearance_search_cells))
        if free_start is None or free_goal is None:
            return self._cache_route_path(route_cache_key, [], None)

        clearance_map = (
            self._compute_clearance_map(blocked)
            if float(self.config.clearance_penalty_weight) > 0.0
            else None
        )
        path = self._astar(blocked, free_start, free_goal, clearance_map=clearance_map)
        return self._cache_route_path(route_cache_key, path, clearance_map)

    def _route_target(
        self,
        *,
        robot_pos: np.ndarray,
        goal: np.ndarray,
        radius: float,
        grid: np.ndarray,
        meta: dict[str, Any],
        cache_key: Any | None = None,
    ) -> np.ndarray | None:
        """Resolve the local waypoint target from the occupancy-grid route.

        Returns:
            np.ndarray | None: Selected waypoint target, or ``None`` when no valid
            routed target can be determined.
        """
        path, _clearance_map = self._route_path(
            robot_pos=robot_pos,
            goal=goal,
            radius=radius,
            grid=grid,
            meta=meta,
            cache_key=cache_key,
        )
        if len(path) < 2:
            return goal

        waypoint_idx = min(max(int(self.config.waypoint_lookahead_cells), 1), len(path) - 1)
        return self._grid_to_world(path[waypoint_idx], meta)

    @staticmethod
    def _path_length(path: list[tuple[int, int]], *, stop_index: int | None = None) -> float:
        """Return path length in grid-cell units up to ``stop_index``."""
        if len(path) < 2:
            return 0.0
        end = len(path) - 1 if stop_index is None else max(0, min(int(stop_index), len(path) - 1))
        length = 0.0
        for start, stop in pairwise(path[: end + 1]):
            length += float(np.hypot(stop[0] - start[0], stop[1] - start[1]))
        return length

    @staticmethod
    def _lateral_offset_to_segment(
        point: np.ndarray,
        segment_start: np.ndarray,
        segment_stop: np.ndarray,
    ) -> float | None:
        """Return lateral distance from ``point`` to a world-space segment."""
        segment = segment_stop - segment_start
        length = float(np.linalg.norm(segment))
        if length <= 1e-9:
            return None
        relative = point - segment_start
        return float(abs(segment[0] * relative[1] - segment[1] * relative[0]) / length)

    def _route_corner_distance(
        self,
        path: list[tuple[int, int]],
        *,
        resolution: float,
    ) -> float | None:
        """Return distance to the first meaningful route tangent change."""
        headings: list[float] = []
        for start, stop in pairwise(path):
            headings.append(float(np.arctan2(stop[0] - start[0], stop[1] - start[1])))
        for idx, (prev_heading, next_heading) in enumerate(pairwise(headings), start=1):
            if abs(_wrap_angle(next_heading - prev_heading)) >= 0.35:
                return float(self._path_length(path, stop_index=idx) * resolution)
        return None

    def _route_clearance_diagnostics(
        self,
        *,
        path: list[tuple[int, int]],
        waypoint_idx: int,
        clearance_map: np.ndarray | None,
        resolution: float,
    ) -> tuple[float | None, float | None]:
        """Return center clearance and width estimate at the route waypoint."""
        if clearance_map is None:
            return None, None
        clearance_value = float(clearance_map[path[waypoint_idx]])
        if not np.isfinite(clearance_value):
            return None, None
        center_clearance = clearance_value * resolution
        return float(center_clearance), float(2.0 * center_clearance)

    def _route_geometry_from_path(
        self,
        *,
        path: list[tuple[int, int]],
        clearance_map: np.ndarray | None,
        meta: dict[str, Any],
        robot_pos: np.ndarray,
        heading: float,
    ) -> dict[str, Any]:
        """Build JSON-ready diagnostics from a computed grid route.

        Returns:
            dict[str, Any]: Structured route-corridor diagnostic payload.
        """
        resolution = float(self._as_1d_float(meta.get("resolution", [0.2]), pad=1)[0])
        waypoint_idx = min(max(int(self.config.waypoint_lookahead_cells), 1), len(path) - 1)
        start_world = self._grid_to_world(path[0], meta)
        goal_world = self._grid_to_world(path[-1], meta)
        waypoint_world = self._grid_to_world(path[waypoint_idx], meta)

        tangent_start_idx = waypoint_idx
        tangent_stop_idx = min(waypoint_idx + 1, len(path) - 1)
        if tangent_start_idx == tangent_stop_idx:
            tangent_start_idx = max(0, waypoint_idx - 1)
        tangent_start = self._grid_to_world(path[tangent_start_idx], meta)
        tangent_stop = self._grid_to_world(path[tangent_stop_idx], meta)
        tangent = tangent_stop - tangent_start
        route_tangent_heading = None
        route_heading_error = None
        if float(np.linalg.norm(tangent)) > 1e-9:
            route_tangent_heading = float(np.arctan2(tangent[1], tangent[0]))
            route_heading_error = _wrap_angle(route_tangent_heading - heading)

        center_clearance, corridor_width = self._route_clearance_diagnostics(
            path=path,
            waypoint_idx=waypoint_idx,
            clearance_map=clearance_map,
            resolution=resolution,
        )
        segment_stop_idx = 1
        lateral_offset = self._lateral_offset_to_segment(
            robot_pos,
            start_world,
            self._grid_to_world(path[segment_stop_idx], meta),
        )

        return {
            "route_start_world": [float(start_world[0]), float(start_world[1])],
            "route_goal_world": [float(goal_world[0]), float(goal_world[1])],
            "route_waypoint_world": [float(waypoint_world[0]), float(waypoint_world[1])],
            "route_waypoint_index": int(waypoint_idx),
            "route_path_cell_count": len(path),
            "route_remaining_distance": float(self._path_length(path) * resolution),
            "route_distance_to_waypoint": float(
                self._path_length(path, stop_index=waypoint_idx) * resolution
            ),
            "route_corner_distance": _finite_or_none(
                self._route_corner_distance(path, resolution=resolution)
            ),
            "route_tangent_heading": _finite_or_none(route_tangent_heading),
            "route_heading_error": _finite_or_none(route_heading_error),
            "corridor_center_clearance": _finite_or_none(center_clearance),
            "corridor_width_estimate": _finite_or_none(corridor_width),
            "robot_lateral_offset_to_corridor": _finite_or_none(lateral_offset),
        }

    def route_geometry(self, observation: dict[str, Any]) -> dict[str, Any] | None:
        """Return route-corridor geometry diagnostics without producing a command.

        Returns:
            dict[str, Any] | None: JSON-ready route geometry, or ``None`` when
            structured grid routing is unavailable.
        """
        try:
            robot_pos, heading, goal, radius = self._extract_state(observation)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return None

        payload = self._extract_grid_payload(observation)
        if payload is None:
            return None
        grid, meta = payload
        try:
            path, clearance_map = self._route_path(
                robot_pos=robot_pos,
                goal=goal,
                radius=radius,
                grid=grid,
                meta=meta,
                cache_key=id(observation),
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return None
        if len(path) < 2:
            return None

        return self._route_geometry_from_path(
            path=path,
            clearance_map=clearance_map,
            meta=meta,
            robot_pos=robot_pos,
            heading=heading,
        )

    def route_waypoint(self, observation: dict[str, Any]) -> np.ndarray | None:
        """Resolve a topology-aware waypoint without converting it into a command.

        Returns:
            np.ndarray | None: World-space waypoint target, or ``None`` when the
            structured occupancy-grid route cannot be computed safely.
        """
        try:
            robot_pos, _heading, goal, radius = self._extract_state(observation)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return None

        payload = self._extract_grid_payload(observation)
        if payload is None:
            return None
        grid, meta = payload
        try:
            return self._route_target(
                robot_pos=robot_pos,
                goal=goal,
                radius=radius,
                grid=grid,
                meta=meta,
                cache_key=id(observation),
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return None

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return a bounded ``(v, omega)`` command from grid routing and waypoint tracking."""
        try:
            robot_pos, heading, goal, radius = self._extract_state(observation)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return 0.0, 0.0

        goal_dist = float(np.linalg.norm(goal - robot_pos))
        if goal_dist <= float(self.config.goal_tolerance):
            return 0.0, 0.0

        payload = self._extract_grid_payload(observation)
        if payload is None:
            linear, angular, _heading_error = self._nominal_command(
                robot_pos=robot_pos,
                heading=heading,
                target=goal,
            )
            return linear, angular

        grid, meta = payload
        waypoint = self._route_target(
            robot_pos=robot_pos,
            goal=goal,
            radius=radius,
            grid=grid,
            meta=meta,
            cache_key=id(observation),
        )
        if waypoint is None:
            linear, angular, _heading_error = self._nominal_command(
                robot_pos=robot_pos,
                heading=heading,
                target=goal,
            )
            return linear, angular

        linear, angular, heading_error = self._nominal_command(
            robot_pos=robot_pos,
            heading=heading,
            target=waypoint,
        )

        front_clearance = self._front_clearance(
            robot_pos=robot_pos, heading=heading, grid=grid, meta=meta
        )
        if front_clearance <= float(self.config.stop_distance):
            turn_sign = 1.0 if heading_error >= 0.0 else -1.0
            return 0.0, float(turn_sign * min(float(self.config.max_angular_speed), 0.9))

        return (
            float(np.clip(linear, 0.0, float(self.config.max_linear_speed))),
            float(
                np.clip(
                    angular,
                    -float(self.config.max_angular_speed),
                    float(self.config.max_angular_speed),
                )
            ),
        )


def build_grid_route_config(cfg: dict[str, Any] | None) -> GridRoutePlannerConfig:
    """Build :class:`GridRoutePlannerConfig` from a mapping payload.

    Returns:
        GridRoutePlannerConfig: Parsed planner configuration.
    """
    if not isinstance(cfg, dict):
        return GridRoutePlannerConfig()
    return GridRoutePlannerConfig(
        max_linear_speed=float(cfg.get("max_linear_speed", 0.9)),
        max_angular_speed=float(cfg.get("max_angular_speed", 1.2)),
        goal_tolerance=float(cfg.get("goal_tolerance", 0.25)),
        heading_gain=float(cfg.get("heading_gain", 1.8)),
        turn_in_place_angle=float(cfg.get("turn_in_place_angle", 0.8)),
        waypoint_lookahead_cells=int(cfg.get("waypoint_lookahead_cells", 5)),
        waypoint_reached_distance=float(cfg.get("waypoint_reached_distance", 0.3)),
        obstacle_threshold=float(cfg.get("obstacle_threshold", 0.5)),
        obstacle_inflation_cells=int(cfg.get("obstacle_inflation_cells", 1)),
        clearance_search_cells=int(cfg.get("clearance_search_cells", 5)),
        stop_distance=float(cfg.get("stop_distance", 0.25)),
        progress_weight=float(cfg.get("progress_weight", 1.0)),
        heading_weight=float(cfg.get("heading_weight", 1.0)),
        clearance_penalty_weight=float(cfg.get("clearance_penalty_weight", 0.5)),
    )


__all__ = [
    "GridRoutePlannerAdapter",
    "GridRoutePlannerConfig",
    "build_grid_route_config",
]
