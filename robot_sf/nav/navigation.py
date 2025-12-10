"""Route sampling and waypoint navigation helpers.

This module provides utilities to sample valid robot routes from a map definition and
to track progress along those routes during navigation.
"""

from dataclasses import dataclass, field
from math import atan2, dist
from random import randint, sample

from shapely.geometry import Polygon
from shapely.prepared import PreparedGeometry, prep

from robot_sf.common.types import Vec2D
from robot_sf.nav.map_config import MapDefinition
from robot_sf.ped_npc.ped_zone import sample_zone


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

        reached_waypoint = dist(self.current_waypoint, pos) <= self.proximity_threshold
        if reached_waypoint:
            self.waypoint_id = min(len(self.waypoints) - 1, self.waypoint_id + 1)
        self.pos = pos
        self.reached_waypoint = reached_waypoint

    def new_route(self, route: list[Vec2D]):
        """Replace the active route and reset progress.

        Args:
            route: Ordered list of waypoints to follow.
        """

        self.waypoints = route
        self.waypoint_id = 0


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

    if use_planner and planner is not None:
        spawn_idx = (
            spawn_id if spawn_id is not None else randint(0, len(map_def.robot_spawn_zones) - 1)
        )
        goal_idx = randint(0, len(map_def.robot_goal_zones) - 1)
        prepared_obstacles = get_prepared_obstacles(map_def)
        start = sample_zone(
            map_def.robot_spawn_zones[spawn_idx], 1, obstacle_polygons=prepared_obstacles
        )[0]
        goal = sample_zone(
            map_def.robot_goal_zones[goal_idx], 1, obstacle_polygons=prepared_obstacles
        )[0]
        return planner.plan(start, goal)

    # If no spawn_id is provided, choose a random one
    spawn_id = spawn_id if spawn_id is not None else randint(0, map_def.num_start_pos - 1)

    # Get the routes for the chosen spawn_id
    routes = map_def.robot_routes_by_spawn_id[spawn_id]

    # Sample a route from the routes
    route = sample(routes, k=1)[0]

    # Sample an initial spawn and a final goal from the route's spawn and goal zones
    prepared_obstacles = get_prepared_obstacles(map_def)

    initial_spawn = sample_zone(route.spawn_zone, 1, obstacle_polygons=prepared_obstacles)[0]
    final_goal = sample_zone(route.goal_zone, 1, obstacle_polygons=prepared_obstacles)[0]

    # Construct the route
    route = [initial_spawn, *route.waypoints, final_goal]

    # TODO(#254): add noise to the exact waypoint positions to avoid learning routes by heart
    # See: https://github.com/ll7/robot_sf_ll7/issues/254

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
