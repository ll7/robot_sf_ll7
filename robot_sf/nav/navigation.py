"""Route sampling and waypoint navigation helpers.

This module provides utilities to sample valid robot routes from a map definition and
to track progress along those routes during navigation.
"""

from dataclasses import dataclass, field
from math import atan2, dist
from random import sample

import numpy as np
from loguru import logger
from shapely.geometry import Polygon
from shapely.prepared import PreparedGeometry, prep

from robot_sf.common.types import Vec2D
from robot_sf.nav.free_space_sampling import sample_free_points_in_bounds
from robot_sf.nav.map_config import MapDefinition
from robot_sf.ped_npc.ped_zone import sample_zone
from robot_sf.planner import PlanningError
from robot_sf.planner.visibility_planner import PlanningFailedError

_PLANNER_RETRY_ATTEMPTS = 5


@dataclass(frozen=True)
class NavigationSettings:
    """Settings for route sampling behavior in navigation utilities."""

    waypoint_noise_enabled: bool = False
    waypoint_noise_std: float = 0.0

    def __post_init__(self) -> None:
        """Validate waypoint-noise configuration values."""
        if self.waypoint_noise_std < 0:
            raise ValueError("waypoint_noise_std must be >= 0.0")


def _resolve_navigation_settings(map_def: MapDefinition) -> NavigationSettings:
    """Resolve navigation settings from map metadata with safe defaults.

    Returns:
        NavigationSettings: Effective settings for route sampling.
    """
    configured = getattr(map_def, "_navigation_settings", None)
    if isinstance(configured, NavigationSettings):
        return configured
    return NavigationSettings()


def _apply_waypoint_noise(
    waypoints: list[Vec2D],
    *,
    settings: NavigationSettings,
) -> list[Vec2D]:
    """Apply Gaussian noise to intermediate waypoints when enabled.

    Returns:
        list[Vec2D]: New waypoint list with optional noise on intermediate points.
    """
    if not settings.waypoint_noise_enabled or settings.waypoint_noise_std <= 0.0:
        return list(waypoints)
    if not waypoints:
        return []
    waypoints_array = np.asarray(waypoints, dtype=float)
    noise = np.random.normal(0.0, settings.waypoint_noise_std, size=waypoints_array.shape)
    return [tuple(point) for point in (waypoints_array + noise)]


def _select_spawn_id(map_def: MapDefinition, spawn_id: int | None) -> int:
    """Pick a spawn id, falling back to first available or zero.

    Returns:
        int: Selected spawn identifier.
    """
    if spawn_id is not None:
        return spawn_id
    available_spawns = list(map_def.robot_routes_by_spawn_id.keys())
    return sample(available_spawns, k=1)[0] if available_spawns else 0


def _sample_start_goal_global(
    bounds: tuple[float, float, float, float],
    obstacles: list[PreparedGeometry],
    attempt: int,
) -> tuple[Vec2D, Vec2D] | None:
    """Sample start/goal from free space; return None on failure.

    Returns:
        tuple[Vec2D, Vec2D] | None: Sampled start and goal or None when sampling fails.
    """
    try:
        start, goal = sample_free_points_in_bounds(bounds, 2, obstacle_polygons=obstacles)
    except RuntimeError as exc:
        logger.warning(
            "Global sampling failed on attempt {attempt}: {err}",
            attempt=attempt,
            err=exc,
        )
        return None
    if dist(start, goal) < 1e-9:
        logger.warning(
            "Global sampling produced identical start/goal on attempt {attempt}; resampling.",
            attempt=attempt,
        )
        return None
    return start, goal


def _sample_start_goal_from_routes(
    routes_for_spawn: list,
    obstacles: list[PreparedGeometry],
) -> tuple[Vec2D, Vec2D]:
    """Sample start/goal from route spawn/goal zones.

    Returns:
        tuple[Vec2D, Vec2D]: Sampled start and goal points.
    """
    route_choice = sample(routes_for_spawn, k=1)[0]
    start = sample_zone(route_choice.spawn_zone, 1, obstacle_polygons=obstacles)[0]
    goal = sample_zone(route_choice.goal_zone, 1, obstacle_polygons=obstacles)[0]
    return start, goal


def _plan_with_planner(
    map_def: MapDefinition,
    planner,
    spawn_id: int | None,
    global_sampling: bool,
    prepared_obstacles: list[PreparedGeometry],
) -> tuple[list[Vec2D] | None, int]:
    """Attempt to plan using the configured planner; return route or None plus chosen spawn_id.

    Returns:
        tuple[list[Vec2D] | None, int]: Planned route (or None) and the spawn id used.
    """
    if not map_def.robot_routes_by_spawn_id and not global_sampling:
        msg = "Planner mode enabled but no robot routes are defined on the map."
        raise ValueError(msg)

    spawn_id = _select_spawn_id(map_def, spawn_id)
    routes_for_spawn = map_def.robot_routes_by_spawn_id.get(spawn_id, [])
    if not routes_for_spawn and not global_sampling:
        msg = f"Planner mode: no routes found for spawn_id={spawn_id}"
        raise ValueError(msg)
    if not routes_for_spawn and global_sampling:
        logger.warning(
            "Global sampling enabled but no predefined routes for spawn_id={spawn_id}; "
            "fallback routes will be unavailable.",
            spawn_id=spawn_id,
        )

    bounds = map_def.get_map_bounds()

    for attempt in range(_PLANNER_RETRY_ATTEMPTS):
        sampled = (
            _sample_start_goal_global(
                bounds,
                prepared_obstacles,
                attempt=attempt + 1,
            )
            if global_sampling
            else _sample_start_goal_from_routes(routes_for_spawn, prepared_obstacles)
        )
        if sampled is None:
            continue
        start, goal = sampled
        try:
            planned = planner.plan(start, goal)
        except (PlanningFailedError, PlanningError) as exc:
            start_fmt = f"({start[0]:.2f}, {start[1]:.2f})"
            goal_fmt = f"({goal[0]:.2f}, {goal[1]:.2f})"
            logger.warning(
                "Planner attempt {attempt}/{max_attempts} failed for spawn_id={spawn_id} "
                "start={start} goal={goal}: {err}",
                attempt=attempt + 1,
                max_attempts=_PLANNER_RETRY_ATTEMPTS,
                spawn_id=spawn_id,
                start=start_fmt,
                goal=goal_fmt,
                err=exc,
            )
            continue

        route = planned[0] if isinstance(planned, tuple) else planned
        logger.info(
            "Planner produced route with {n_waypoints} waypoints for spawn_id={spawn_id} "
            "on attempt {attempt}/{max_attempts}",
            n_waypoints=len(route),
            spawn_id=spawn_id,
            attempt=attempt + 1,
            max_attempts=_PLANNER_RETRY_ATTEMPTS,
        )
        return route, spawn_id

    logger.error(
        "Planner failed after {attempts} attempts for spawn_id={spawn_id}; "
        "falling back to predefined route.",
        attempts=_PLANNER_RETRY_ATTEMPTS,
        spawn_id=spawn_id,
    )
    if global_sampling and not map_def.robot_routes_by_spawn_id:
        raise PlanningError(
            "Planner failed with global sampling and no predefined routes available for fallback.",
        )
    return None, spawn_id


@dataclass
class RouteNavigator:
    """Tracks progress along an ordered list of waypoints.

    Attributes:
        waypoints: Ordered list of waypoints the robot must follow.
        waypoint_id: Index of the currently targeted waypoint.
        proximity_threshold: Distance tolerance for considering a waypoint reached.
        pos: Latest known robot position.
        reached_waypoint: Whether the current waypoint was reached on the last update.
    """

    waypoints: list[Vec2D] = field(default_factory=list)
    waypoint_id: int = 0
    proximity_threshold: float = 1.0  # info: should be set to vehicle radius + goal radius
    pos: Vec2D = field(default=(0, 0))
    reached_waypoint: bool = False

    @property
    def reached_destination(self) -> bool:
        """Whether the final waypoint has been reached within the threshold.

        Returns:
            bool: ``True`` when the last waypoint is within ``proximity_threshold`` of
            ``pos`` or when the route is empty.
        """

        return (
            len(self.waypoints) == 0
            or dist(self.waypoints[-1], self.pos) <= self.proximity_threshold
        )

    @property
    def current_waypoint(self) -> Vec2D:
        """Current target waypoint.

        Returns:
            Vec2D: Current waypoint coordinates.
        """
        if not self.waypoints:
            raise ValueError("RouteNavigator has no waypoints; cannot fetch current waypoint.")
        return self.waypoints[self.waypoint_id]

    @property
    def next_waypoint(self) -> Vec2D | None:
        """Next waypoint, if it exists.

        Returns:
            Vec2D | None: Next waypoint coordinates, or ``None`` if at the end.
        """

        return (
            self.waypoints[self.waypoint_id + 1]
            if self.waypoint_id + 1 < len(self.waypoints)
            else None
        )

    @property
    def initial_orientation(self) -> float:
        """Initial heading from the first to the second waypoint.

        Returns:
            float: Orientation in radians.
        """

        return atan2(
            self.waypoints[1][1] - self.waypoints[0][1],
            self.waypoints[1][0] - self.waypoints[0][0],
        )

    def update_position(self, pos: Vec2D):
        """Update the robot position and advance the waypoint if reached.

        Args:
            pos: New robot position.
        """
        if not self.waypoints:
            self.pos = pos
            self.reached_waypoint = False
            return

        reached_waypoint = dist(self.current_waypoint, pos) <= self.proximity_threshold
        if reached_waypoint:
            next_idx = self.waypoint_id + 1
            if self.waypoints:
                self.waypoint_id = min(len(self.waypoints) - 1, next_idx)
            else:
                self.waypoint_id = 0
        self.pos = pos
        self.reached_waypoint = reached_waypoint

    def new_route(self, route: list[Vec2D]):
        """Replace the active route and reset progress.

        Args:
            route: Ordered list of waypoints to follow.
        """
        self.waypoints = route
        self.waypoint_id = 0
        self.reached_waypoint = False


def sample_route(map_def: MapDefinition, spawn_id: int | None = None) -> list[Vec2D]:
    """Sample a concrete waypoint route for a robot spawn.

    Args:
        map_def: Map definition containing predefined routes and zones.
        spawn_id: Optional spawn identifier; chooses a random spawn when ``None``.

    Returns:
        list[Vec2D]: Waypoints including sampled spawn and goal positions.
    """

    planner = getattr(map_def, "_global_planner", None)
    use_planner = getattr(map_def, "_use_planner", False)
    global_sampling = bool(getattr(map_def, "_sample_positions_globally", False))
    prepared_obstacles = get_prepared_obstacles(map_def)
    chosen_spawn = spawn_id

    if use_planner and planner is not None:
        planned_route, chosen_spawn = _plan_with_planner(
            map_def,
            planner,
            spawn_id,
            global_sampling,
            prepared_obstacles,
        )
        if planned_route is not None:
            return planned_route

    # If no spawn_id is provided, choose a random one
    if not map_def.robot_routes_by_spawn_id:
        raise PlanningError("No robot routes available for fallback after planner failure.")
    if chosen_spawn is None:
        available_spawns = list(map_def.robot_routes_by_spawn_id.keys())
        chosen_spawn = sample(available_spawns, k=1)[0]
    elif chosen_spawn not in map_def.robot_routes_by_spawn_id:
        raise PlanningError(
            f"No robot routes for spawn_id={chosen_spawn} to fall back on after planner failure.",
        )

    # Get the routes for the chosen spawn_id
    routes = map_def.robot_routes_by_spawn_id[chosen_spawn]

    # Sample a route from the routes
    route = sample(routes, k=1)[0]

    # Sample an initial spawn and a final goal from the route's spawn and goal zones
    initial_spawn = sample_zone(route.spawn_zone, 1, obstacle_polygons=prepared_obstacles)[0]
    final_goal = sample_zone(route.goal_zone, 1, obstacle_polygons=prepared_obstacles)[0]

    # Construct the route with optional noise on intermediate waypoints only.
    settings = _resolve_navigation_settings(map_def)
    waypoints = _apply_waypoint_noise(route.waypoints, settings=settings)
    route = [initial_spawn, *waypoints, final_goal]

    return route


def get_prepared_obstacles(map_def: MapDefinition) -> list[PreparedGeometry]:
    """Return cached prepared obstacle polygons for a map definition.

    Args:
        map_def: Map definition with polygonal obstacles.

    Returns:
        list[PreparedGeometry]: Prepared shapely polygons, cached on the map.
    """
    prepared: list[PreparedGeometry] | None = getattr(map_def, "_prepared_obstacles", None)
    if prepared is not None:
        return prepared
    polygons = [
        Polygon(obstacle.vertices) for obstacle in map_def.obstacles if len(obstacle.vertices) >= 3
    ]
    prepared = [prep(poly) for poly in polygons if not poly.is_empty]
    map_def._prepared_obstacles = prepared
    return prepared
