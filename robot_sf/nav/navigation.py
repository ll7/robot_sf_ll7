from dataclasses import dataclass, field
from math import atan2, dist
from random import randint, sample

from robot_sf.common.types import Vec2D
from robot_sf.nav.map_config import MapDefinition
from robot_sf.ped_npc.ped_zone import sample_zone


@dataclass
class RouteNavigator:
    """
    A class to represent a route navigator.

    Attributes
    ----------
    waypoints : List[Vec2D]
        The list of waypoints.
    waypoint_id : int
        The current waypoint index.
    proximity_threshold : float
        The proximity threshold to consider a waypoint as reached.
    pos : Vec2D
        The current position.
    reached_waypoint : bool
        Whether the current waypoint has been reached.

    Methods
    -------
    reached_destination():
        Checks if the destination has been reached.
    current_waypoint():
        Returns the current waypoint.
    next_waypoint():
        Returns the next waypoint.
    initial_orientation():
        Returns the initial orientation.
    update_position(pos: Vec2D):
        Updates the current position and checks if the current waypoint has been reached.
    new_route(route: List[Vec2D]):
        Sets a new route.
    """

    waypoints: list[Vec2D] = field(default_factory=list)
    waypoint_id: int = 0
    proximity_threshold: float = 1.0  # info: should be set to vehicle radius + goal radius
    pos: Vec2D = field(default=(0, 0))
    reached_waypoint: bool = False

    @property
    def reached_destination(self) -> bool:
        """
        Checks if the destination has been reached.

        Returns
        -------
        bool
            True if the destination has been reached, False otherwise.
        """

        return (
            len(self.waypoints) == 0
            or dist(self.waypoints[-1], self.pos) <= self.proximity_threshold
        )

    @property
    def current_waypoint(self) -> Vec2D:
        """
        Returns the current waypoint.

        Returns
        -------
        Vec2D
            The current waypoint.
        """

        return self.waypoints[self.waypoint_id]

    @property
    def next_waypoint(self) -> Vec2D | None:
        """
        Returns the next waypoint.

        Returns
        -------
        Optional[Vec2D]
            The next waypoint, or None if there is no next waypoint.
        """

        return (
            self.waypoints[self.waypoint_id + 1]
            if self.waypoint_id + 1 < len(self.waypoints)
            else None
        )

    @property
    def initial_orientation(self) -> float:
        """
        Returns the initial orientation.

        Returns
        -------
        float
            The initial orientation.
        """

        return atan2(
            self.waypoints[1][1] - self.waypoints[0][1],
            self.waypoints[1][0] - self.waypoints[0][0],
        )

    def update_position(self, pos: Vec2D):
        """
        Updates the current position and checks if the current waypoint has been reached.

        Parameters
        ----------
        pos : Vec2D
            The new position.
        """

        reached_waypoint = dist(self.current_waypoint, pos) <= self.proximity_threshold
        if reached_waypoint:
            self.waypoint_id = min(len(self.waypoints) - 1, self.waypoint_id + 1)
        self.pos = pos
        self.reached_waypoint = reached_waypoint

    def new_route(self, route: list[Vec2D]):
        """
        Sets a new route.

        Parameters
        ----------
        route : List[Vec2D]
            The new route.
        """

        self.waypoints = route
        self.waypoint_id = 0


def sample_route(map_def: MapDefinition, spawn_id: int | None = None) -> list[Vec2D]:
    """
    Samples a route from the given map definition.

    Parameters
    ----------
    map_def : MapDefinition
        The map definition.
    spawn_id : Optional[int], optional
        The spawn ID, by default None. If None, a random spawn ID is chosen.

    Returns
    -------
    List[Vec2D]
        The sampled route.
    """

    # If no spawn_id is provided, choose a random one
    spawn_id = spawn_id if spawn_id is not None else randint(0, map_def.num_start_pos - 1)

    # Get the routes for the chosen spawn_id
    routes = map_def.robot_routes_by_spawn_id[spawn_id]

    # Sample a route from the routes
    route = sample(routes, k=1)[0]

    # Sample an initial spawn and a final goal from the route's spawn and goal zones
    initial_spawn = sample_zone(route.spawn_zone, 1)[0]
    final_goal = sample_zone(route.goal_zone, 1)[0]

    # Construct the route
    route = [initial_spawn, *route.waypoints, final_goal]

    # TODO(#254): add noise to the exact waypoint positions to avoid learning routes by heart
    # See: https://github.com/ll7/robot_sf_ll7/issues/254

    return route
