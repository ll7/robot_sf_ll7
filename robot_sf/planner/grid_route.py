"""Testing-only occupancy-grid route planner.

This planner recomputes a short 8-connected route over the structured
occupancy grid and tracks the next free waypoint with a bounded unicycle
command. It is intended as a simple topology-aware counterexample to purely
reactive local controllers on static obstacle scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import sqrt
from typing import Any

import numpy as np

from robot_sf.nav.occupancy_grid_utils import ego_to_world
from robot_sf.planner.risk_dwa import _wrap_angle
from robot_sf.planner.socnav import OccupancyAwarePlannerMixin


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

    def _inflate_obstacles(self, blocked: np.ndarray) -> np.ndarray:
        """Dilate blocked cells by a small integer radius.

        Returns:
            np.ndarray: Inflated binary occupancy grid.
        """
        radius = max(int(self.config.obstacle_inflation_cells), 0)
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
        resolution = float(self._as_1d_float(meta.get("resolution", [0.2]), pad=1)[0])
        dynamic_radius = max(int(np.ceil(radius / max(resolution, 1e-6))) - 1, 0)
        if dynamic_radius > int(self.config.obstacle_inflation_cells):
            original = int(self.config.obstacle_inflation_cells)
            self.config.obstacle_inflation_cells = dynamic_radius
            try:
                return self._inflate_obstacles(blocked)
            finally:
                self.config.obstacle_inflation_cells = original
        return self._inflate_obstacles(blocked)

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
    def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        """Euclidean heuristic for A* search.

        Returns:
            float: Heuristic distance between the cells.
        """
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _astar(
        self, blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Compute an 8-connected grid path between start and goal.

        Returns:
            list[tuple[int, int]]: Ordered row/col path, or an empty list when no
            route exists.
        """
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
                new_cost = cost_so_far[current] + step_cost
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
        blocked = self._blocked_grid(grid, meta, radius)
        if blocked is None:
            linear, angular, _heading_error = self._nominal_command(
                robot_pos=robot_pos,
                heading=heading,
                target=goal,
            )
            return linear, angular

        start_rc = self._world_to_grid(robot_pos, meta, (blocked.shape[0], blocked.shape[1]))
        goal_rc = self._world_to_grid(goal, meta, (blocked.shape[0], blocked.shape[1]))
        if start_rc is None or goal_rc is None:
            linear, angular, _heading_error = self._nominal_command(
                robot_pos=robot_pos,
                heading=heading,
                target=goal,
            )
            return linear, angular

        free_start = self._nearest_free(blocked, start_rc, int(self.config.clearance_search_cells))
        free_goal = self._nearest_free(blocked, goal_rc, int(self.config.clearance_search_cells))
        if free_start is None or free_goal is None:
            return 0.0, 0.0

        path = self._astar(blocked, free_start, free_goal)
        if len(path) >= 2:
            waypoint_idx = min(max(int(self.config.waypoint_lookahead_cells), 1), len(path) - 1)
            waypoint = self._grid_to_world(path[waypoint_idx], meta)
        else:
            waypoint = goal

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
    )


__all__ = ["GridRoutePlannerAdapter", "GridRoutePlannerConfig", "build_grid_route_config"]
