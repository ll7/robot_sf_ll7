from math import dist, atan2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

Vec2D = Tuple[float, float]
Zone = Tuple[Vec2D, Vec2D, Vec2D]


@dataclass
class RouteNavigator:
    """
    A class that represents a route navigator for navigating through waypoints.

    Attributes:
        waypoints (List[Vec2D]): The list of waypoints to navigate.
        waypoint_id (int): The index of the current waypoint.
        proximity_threshold (float): The proximity threshold for reaching a waypoint.
        pos (Vec2D): The current position of the navigator.
        reached_waypoint (bool): Indicates whether the current waypoint has been reached.
    """

    waypoints: List[Vec2D] = field(default_factory=list)
    waypoint_id: int = 0
    proximity_threshold: float = 1.0  # info: should be set to vehicle radius + goal radius
    pos: Vec2D = field(default=(0, 0))
    reached_waypoint: bool = False

    @property
    def reached_destination(self) -> bool:
        """
        Check if the destination has been reached.

        Returns:
            bool: True if the destination has been reached, False otherwise.
        """
        return len(self.waypoints) == 0 or \
            dist(self.waypoints[-1], self.pos) <= self.proximity_threshold

    @property
    def current_waypoint(self) -> Vec2D:
        """
        Get the current waypoint.

        Returns:
            Vec2D: The current waypoint.
        """
        return self.waypoints[self.waypoint_id]

    @property
    def next_waypoint(self) -> Optional[Vec2D]:
        """
        Get the next waypoint.

        Returns:
            Optional[Vec2D]: The next waypoint if available, None otherwise.
        """
        return self.waypoints[self.waypoint_id + 1] \
            if self.waypoint_id + 1 < len(self.waypoints) else None

    @property
    def initial_orientation(self) -> float:
        """
        Get the initial orientation of the navigator.

        Returns:
            float: The initial orientation in radians.
        """
        return atan2(self.waypoints[1][1] - self.waypoints[0][1],
                     self.waypoints[1][0] - self.waypoints[0][0])

    def update_position(self, pos: Vec2D):
        """
        Update the position of the navigator.

        Args:
            pos (Vec2D): The new position of the navigator.
        """
        reached_waypoint = dist(self.current_waypoint, pos) <= self.proximity_threshold
        if reached_waypoint:
            self.waypoint_id = min(len(self.waypoints) - 1, self.waypoint_id + 1)
        self.pos = pos
        self.reached_waypoint = reached_waypoint

    def new_route(self, route: List[Vec2D]):
        """
        Set a new route for the navigator.

        Args:
            route (List[Vec2D]): The new route to navigate.
        """
        self.waypoints = route
        self.waypoint_id = 0
