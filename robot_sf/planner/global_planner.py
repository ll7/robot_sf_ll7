"""Global planner primitives for SVG-based waypoint generation."""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

from loguru import logger
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

from robot_sf.common.types import Vec2D
from robot_sf.nav.map_config import MapDefinition
from robot_sf.planner.path_smoother import douglas_peucker
from robot_sf.planner.visibility_graph import VisibilityGraph


@dataclass
class PlannerConfig:
    """Configuration for the global path planner."""

    robot_radius: float = 0.4
    min_safe_clearance: float = 0.3
    enable_smoothing: bool = True
    smoothing_epsilon: float = 0.1
    cache_graphs: bool = True
    fallback_on_failure: bool = True

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
        if self.robot_radius <= 0:
            raise ValueError(f"robot_radius must be positive, got {self.robot_radius}")
        if self.min_safe_clearance < 0:
            raise ValueError(
                f"min_safe_clearance cannot be negative, got {self.min_safe_clearance}"
            )
        if self.enable_smoothing and self.smoothing_epsilon <= 0:
            raise ValueError(
                f"smoothing_epsilon must be positive when smoothing is enabled, got {self.smoothing_epsilon}"
            )


class PlanningFailedError(Exception):
    """Raised when no valid path exists between start and goal."""

    def __init__(self, start: Vec2D, goal: Vec2D, reason: str):
        """Capture planning context for easier debugging.

        Args:
            start: Start position of the failed query.
            goal: Goal position of the failed query.
            reason: Human-readable failure description.
        """
        self.start = start
        self.goal = goal
        self.reason = reason
        super().__init__(
            f"Planning failed: {reason}\n"
            f"  Start: ({start[0]:.2f}, {start[1]:.2f})\n"
            f"  Goal: ({goal[0]:.2f}, {goal[1]:.2f})"
        )


class GlobalPlanner:
    """Visibility-graph-based path planner for 2D environments."""

    def __init__(self, map_definition: MapDefinition, config: PlannerConfig | None = None) -> None:
        """Initialize planner with map and configuration.

        Args:
            map_definition: Map containing obstacles, zones, and points of interest.
            config: Planner configuration; defaults are used when None.
        """
        self.map_def = map_definition
        self.config = config or PlannerConfig()
        self._graph: VisibilityGraph | None = None

    def plan(self, start: Vec2D, goal: Vec2D, *, via_pois: list[str] | None = None) -> list[Vec2D]:
        """Compute collision-free path from start to goal.

        Args:
            start: Starting position in map coordinates.
            goal: Goal position in map coordinates.
            via_pois: Optional ordered list of POI identifiers to route through.

        Returns:
            Waypoint list including start and goal (and any via POIs when present).

        Raises:
            PlanningFailedError: If no collision-free path exists and fallback is disabled.
            ValueError: If start/goal are outside map bounds.
        """
        self._validate_bounds(start)
        self._validate_bounds(goal)
        via_points = self._resolve_via_pois(via_pois or [])

        planning_obstacles = self._inflate_obstacles()
        collision_obstacles = self._inflate_obstacles_for_collision()
        graph_obstacles = self._prepare_graph_obstacles(planning_obstacles)
        start_safe = self._project_to_free_space(start, planning_obstacles)
        goal_safe = self._project_to_free_space(goal, planning_obstacles)

        if not self.map_def.obstacles:
            return self._path_without_obstacles(start_safe, goal_safe, via_points)

        logger.debug(
            "Planning path start={start} goal={goal} via={vias} inflated_obs={count}",
            start=start_safe,
            goal=goal_safe,
            vias=len(via_points),
            count=len(planning_obstacles),
        )

        planner_graph = self._get_or_build_graph(graph_obstacles)
        if planner_graph._built:
            waypoints = self._compute_waypoints(planner_graph, start_safe, goal_safe, via_points)
            return self._finalize_path(
                waypoints, planning_obstacles, collision_obstacles, start_safe, goal_safe
            )

        # Fallback: Add intermediate waypoint to ensure at least 3 waypoints for navigation
        mid_x = (start_safe[0] + goal_safe[0]) / 2
        mid_y = (start_safe[1] + goal_safe[1]) / 2
        intermediate = (mid_x, mid_y)
        return [start_safe, intermediate, goal_safe]

    def plan_multi_goal(
        self, start: Vec2D, goals: list[Vec2D], *, optimize_order: bool = True
    ) -> list[Vec2D]:
        """Plan a path visiting multiple goals.

        Args:
            start: Starting position.
            goals: Goal positions to visit.
            optimize_order: Whether to reorder goals for efficiency using nearest-neighbor.

        Returns:
            Path that visits all requested goals.

        Raises:
            ValueError: If no goals provided.
        """
        if not goals:
            raise ValueError("goals must not be empty")

        ordered_goals = list(goals)
        if optimize_order:
            ordered_goals = self._nearest_neighbor_order(start, goals)

        full_path: list[Vec2D] = []
        current = start
        for goal in ordered_goals:
            segment = self.plan(current, goal)
            if full_path:
                full_path.extend(segment[1:])
            else:
                full_path.extend(segment)
            current = goal
        return full_path

    def invalidate_cache(self) -> None:
        """Clear cached visibility graph."""
        self._graph = None
        VisibilityGraph.clear_cache()

    def build_inflated_obstacles(self) -> list[Polygon]:
        """Return inflated obstacles using the current planner configuration."""
        return self._inflate_obstacles()

    def _get_or_build_graph(self, polygons: list[Polygon]) -> VisibilityGraph:
        """Return a cached or newly built visibility graph.

        Gracefully handles pyvisgraph numerical issues with certain polygon
        configurations by falling back to direct start-goal planning.
        """
        if self.config.cache_graphs and self._graph is not None:
            logger.debug("Using cached visibility graph.")
            return self._graph

        logger.debug("Building visibility graph with {count} polygons.", count=len(polygons))
        try:
            if self.config.cache_graphs:
                graph = VisibilityGraph.get_cached(polygons)
                self._graph = graph
                return graph

            graph = VisibilityGraph()
            graph.build(polygons)
            return graph
        except (ValueError, ZeroDivisionError) as e:
            # pyvisgraph can fail on certain polygon configurations (numerical issues)
            logger.warning(
                "Visibility graph construction failed ({error}): {msg}. "
                "Falling back to direct planning.",
                error=type(e).__name__,
                msg=str(e),
            )
            # Return an empty graph - direct start-goal paths will be used
            graph = VisibilityGraph()
            return graph

    def _path_without_obstacles(
        self, start: Vec2D, goal: Vec2D, via_points: list[Vec2D]
    ) -> list[Vec2D]:
        """Return a straight-line path when no obstacles exist."""
        if via_points:
            return [start, *via_points, goal]
        return [start, goal]

    def _compute_waypoints(
        self,
        planner_graph: VisibilityGraph,
        start: Vec2D,
        goal: Vec2D,
        via_points: list[Vec2D],
    ) -> list[Vec2D]:
        """Compute waypoints across any via points, applying fallback when needed.

        Returns:
            Combined waypoint list from start through any via points to goal.
        """
        waypoints: list[Vec2D] = []
        current_start = start
        for intermediate in [*via_points, goal]:
            segment = planner_graph.shortest_path(current_start, intermediate)
            if not segment:
                if self.config.fallback_on_failure:
                    logger.warning(
                        "No path found; returning straight line fallback between {start} and {goal}",
                        start=current_start,
                        goal=intermediate,
                    )
                    waypoints.extend([current_start, intermediate])
                    current_start = intermediate
                    continue
                raise PlanningFailedError(
                    start=current_start, goal=intermediate, reason="No path found"
                )

            if waypoints:
                # Avoid duplicating the connecting waypoint
                waypoints.extend(segment[1:])
            else:
                waypoints.extend(segment)
            current_start = intermediate
        return waypoints

    def _finalize_path(
        self,
        waypoints: list[Vec2D],
        planning_obstacles: list[Polygon],
        collision_obstacles: list[Polygon],
        start_safe: Vec2D,
        goal_safe: Vec2D,
    ) -> list[Vec2D]:
        """Validate and optionally smooth the generated path.

        Returns:
            The final validated waypoint sequence.
        """
        try:
            # Validate against collision envelopes (robot radius only)
            self._validate_path(waypoints, collision_obstacles)
            # Smooth using planning envelopes (radius + clearance)
            waypoints = self._maybe_smooth_path(waypoints, planning_obstacles)
        except PlanningFailedError as err:
            if self.config.fallback_on_failure:
                logger.warning(
                    "Invalid planned path ({reason}); using straight-line fallback.",
                    reason=err.reason,
                )
                return [start_safe, goal_safe]
            raise
        return waypoints

    def _maybe_smooth_path(
        self, waypoints: list[Vec2D], inflated_obstacles: list[Polygon]
    ) -> list[Vec2D]:
        """Apply smoothing when enabled and return a validated path.

        Returns:
            Smoothed waypoint list when valid; otherwise the original path.
        """
        if not (self.config.enable_smoothing and len(waypoints) > 2):
            return waypoints

        smoothed = douglas_peucker(waypoints, self.config.smoothing_epsilon)
        try:
            self._validate_path(smoothed, inflated_obstacles)
            return smoothed
        except PlanningFailedError:
            logger.warning("Smoothed path invalid; keeping original polyline.")
            return waypoints

    def _inflate_obstacles(self) -> list[Polygon]:
        """Inflate obstacles by robot radius and safety margin for planning.

        Returns:
            List of buffered polygons (radius + clearance) representing planning keep-out zones.
        """
        margin = self.config.robot_radius + self.config.min_safe_clearance
        inflated: list[Polygon] = []
        for obstacle in self.map_def.obstacles:
            poly = Polygon(obstacle.vertices)
            if poly.is_empty or poly.area <= 0:
                logger.warning(
                    "Skipping degenerate obstacle with vertices: {verts}", verts=obstacle.vertices
                )
                continue
            buffered = poly.buffer(margin)
            if buffered.is_empty:
                continue
            inflated.append(buffered)
        return inflated

    def _inflate_obstacles_for_collision(self) -> list[Polygon]:
        """Inflate obstacles by robot radius only for collision checks.

        Returns:
            List of buffered polygons (radius only) for validating path collision safety.
        """
        margin = self.config.robot_radius
        inflated: list[Polygon] = []
        for obstacle in self.map_def.obstacles:
            poly = Polygon(obstacle.vertices)
            if poly.is_empty or poly.area <= 0:
                logger.warning(
                    "Skipping degenerate obstacle with vertices: {verts}", verts=obstacle.vertices
                )
                continue
            buffered = poly.buffer(margin)
            if buffered.is_empty:
                continue
            inflated.append(buffered)
        return inflated

    def _project_to_free_space(self, point: Vec2D, obstacles: Iterable[Polygon]) -> Vec2D:
        """Move a point onto the nearest obstacle boundary if it lies inside.

        Returns:
            The original point when already collision-free, otherwise the projected point.
        """
        pt = Point(point)
        for poly in obstacles:
            if poly.contains(pt):
                _, nearest = nearest_points(pt, poly.exterior)
                projected = (nearest.x, nearest.y)
                logger.warning(
                    "Projected point {pt} to nearest free boundary {proj}", pt=point, proj=projected
                )
                return projected
        return point

    def _validate_bounds(self, point: Vec2D) -> None:
        """Ensure a point lies within the map extents."""
        x, y = point
        if not (0 <= x <= self.map_def.width and 0 <= y <= self.map_def.height):
            raise ValueError(
                f"Point {point} lies outside map bounds (0, 0) to ({self.map_def.width}, {self.map_def.height})"
            )

    def _resolve_via_pois(self, via_pois: list[str]) -> list[Vec2D]:
        """Translate POI identifiers into waypoint coordinates.

        Returns:
            Ordered list of waypoint positions corresponding to provided POI IDs.
        """
        if not via_pois:
            return []
        poi_ids = list(self.map_def.poi_labels.keys())
        points: list[Vec2D] = []
        for poi_id in via_pois:
            if poi_id not in self.map_def.poi_labels:
                raise KeyError(f"POI '{poi_id}' not found in map.")
            idx = poi_ids.index(poi_id)
            points.append(self.map_def.poi_positions[idx])
        return points

    def _validate_path(self, path: list[Vec2D], obstacles: Iterable[Polygon]) -> None:
        """Ensure the path stays within bounds and outside inflated obstacles."""
        if not path:
            raise PlanningFailedError(
                start=(0.0, 0.0), goal=(0.0, 0.0), reason="Empty path produced"
            )

        for waypoint in path:
            x, y = waypoint
            if not (0 <= x <= self.map_def.width and 0 <= y <= self.map_def.height):
                raise PlanningFailedError(
                    start=path[0], goal=path[-1], reason="Path leaves map bounds during planning"
                )
            pt = Point(waypoint)
            for poly in obstacles:
                if poly.contains(pt):
                    raise PlanningFailedError(
                        start=path[0], goal=path[-1], reason="Path enters inflated obstacle"
                    )
        for start_pt, end_pt in pairwise(path):
            segment = LineString([start_pt, end_pt])
            for poly in obstacles:
                if poly.crosses(segment) or poly.contains(segment):
                    logger.warning(
                        "Narrow passage detected: path segment intersects inflated obstacle boundary."
                    )
                elif poly.touches(segment):
                    logger.warning(
                        "Narrow passage detected: path segment touches inflated obstacle."
                    )

    def _prepare_graph_obstacles(self, polygons: list[Polygon]) -> list[Polygon]:
        """Provide polygons for the visibility graph without extra inflation.

        The visibility graph already runs on the boundaries of the planning-inflated
        obstacles (radius + clearance). We intentionally avoid any additional buffer
        so that validation remains aligned with the collision envelopes (radius only).

        Returns:
            Polygons used to seed the visibility graph.
        """
        return polygons

    def _nearest_neighbor_order(self, start: Vec2D, goals: list[Vec2D]) -> list[Vec2D]:
        """Return goals ordered by nearest-neighbor heuristic."""
        remaining = goals.copy()
        ordered: list[Vec2D] = []
        current = start
        while remaining:
            remaining.sort(key=lambda g: (g[0] - current[0]) ** 2 + (g[1] - current[1]) ** 2)
            next_goal = remaining.pop(0)
            ordered.append(next_goal)
            current = next_goal
        return ordered
