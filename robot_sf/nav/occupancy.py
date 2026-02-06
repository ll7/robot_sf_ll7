"""Lightweight continuous collision checks (no rasterized grid).

This module provides:
* Circle-circle and circle-segment intersection helpers (Numba-jitted).
* ``ContinuousOccupancy`` for simple in-bounds, obstacle-segment, and pedestrian-circle checks.
* ``EgoPedContinuousOccupancy`` for ego pedestrians to check robot collisions.

Notably, this is not an occupancy grid: there is no rasterization, costmap, or spatial index.
Callers poll callbacks for agent/goal/obstacle positions each step and run O(N) checks over
segments/pedestrians. It works for small numbers of obstacles/agents, but is not optimized for
large maps or dense crowds.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numba
import numpy as np

from robot_sf.common.geometry import euclid_dist
from robot_sf.common.types import Circle2D, Line2D, Vec2D

if TYPE_CHECKING:
    from robot_sf.nav.map_config import MapDefinition


@numba.njit(fastmath=True)
def is_circle_circle_intersection(c_1: Circle2D, c_2: Circle2D) -> bool:
    """
    Checks if two circles intersect.

    Parameters
    ----------
    c_1 : Circle2D
        The first circle, represented as a tuple of the center coordinates and
        the radius.
    c_2 : Circle2D
        The second circle, represented as a tuple of the center coordinates and
        the radius.

    Returns
    -------
    bool
        True if the circles intersect, False otherwise.
    """

    # Extract the center coordinates and radius of the first circle
    center_1, radius_1 = c_1

    # Extract the center coordinates and radius of the second circle
    center_2, radius_2 = c_2

    # Calculate the square of the distance between the centers of the circles
    dist_sq = (center_1[0] - center_2[0]) ** 2 + (center_1[1] - center_2[1]) ** 2

    # Calculate the square of the sum of the radii of the circles
    rad_sum_sq = (radius_1 + radius_2) ** 2

    # The circles intersect if the square of the distance between their centers
    # is less than or equal to the square of the sum of their radii
    return dist_sq <= rad_sum_sq


@numba.njit(fastmath=True)
def is_circle_line_intersection(circle: Circle2D, segment: Line2D) -> bool:
    """Simple vector math implementation using quadratic solution formula.

    Returns:
        bool: True if the circle intersects the line segment, False otherwise.
    """
    (c_x, c_y), r = circle
    # Line2D is ((x1, y1), (x2, y2))
    (p1_x, p1_y), (p2_x, p2_y) = segment

    # shift circle's center to the origin (0, 0)
    (p1_x, p1_y), (p2_x, p2_y) = (p1_x - c_x, p1_y - c_y), (p2_x - c_x, p2_y - c_y)

    r_sq = r**2
    norm_p1, norm_p2 = p1_x**2 + p1_y**2, p2_x**2 + p2_y**2

    # edge case: line segment ends inside the circle -> collision!
    if norm_p1 <= r_sq or norm_p2 <= r_sq:
        return True

    # coefficients a, b, c of the quadratic solution formula
    s_x, s_y = p2_x - p1_x, p2_y - p1_y
    t_x, t_y = p1_x, p1_y
    a = s_x**2 + s_y**2
    b = 2 * (s_x * t_x + s_y * t_y)
    c = norm_p1 - r_sq

    # discard cases where infinite line doesn't collide
    disc = b**2 - 4 * a * c
    if disc < 0:
        return False

    # check if collision is actually within the line segment
    disc_root = disc**0.5
    return 0 <= -b - disc_root <= 2 * a or 0 <= -b + disc_root <= 2 * a


def circle_collides_any(circle: Circle2D, others: Iterable[Circle2D]) -> bool:
    """Check whether a circle collides with any other circles.

    Args:
        circle: Circle to test ((x, y), radius).
        others: Iterable of circles to test against.

    Returns:
        bool: True if any intersection is found.
    """
    for other in others:
        if is_circle_circle_intersection(circle, other):
            return True
    return False


def circle_collides_any_lines(circle: Circle2D, segments: Iterable[Line2D] | np.ndarray) -> bool:
    """Check whether a circle collides with any line segments.

    Args:
        circle: Circle to test ((x, y), radius).
        segments: Iterable of Line2D segments or an (N, 4) array of line endpoints.

    Returns:
        bool: True if any intersection is found.
    """
    if isinstance(segments, np.ndarray):
        for s_x, s_y, e_x, e_y in segments:
            if is_circle_line_intersection(circle, ((s_x, s_y), (e_x, e_y))):
                return True
        return False

    for seg in segments:
        try:
            (s_x, s_y), (e_x, e_y) = seg
        except (TypeError, ValueError):
            try:
                s_x, s_y, e_x, e_y = seg  # type: ignore[misc]
            except (TypeError, ValueError):
                continue
        if is_circle_line_intersection(circle, ((s_x, s_y), (e_x, e_y))):
            return True
    return False


@dataclass
class ContinuousOccupancy:
    """
    A class to represent a continuous occupancy.

    Attributes
    ----------
    width : float
        The width of the occupancy.
    height : float
        The height of the occupancy.
    get_agent_coords : Callable[[], Vec2D]
        A function to get the agent coordinates.
    get_goal_coords : Callable[[], Vec2D]
        A function to get the goal coordinates.
    get_obstacle_coords : Callable[[], np.ndarray]
        A function to get the obstacle coordinates.
    get_pedestrian_coords : Callable[[], np.ndarray]
        A function to get the pedestrian coordinates.
    agent_radius : float, optional
        The robot radius, by default 1.0.
    ped_radius : float, optional
        The pedestrian radius, by default 0.4.
    goal_radius : float, optional
        The goal radius, by default 1.0.
    """

    width: float
    height: float
    get_agent_coords: Callable[[], Vec2D]
    get_goal_coords: Callable[[], Vec2D]
    get_obstacle_coords: Callable[[], np.ndarray]
    get_pedestrian_coords: Callable[[], np.ndarray]
    get_dynamic_objects: Callable[[], list[Circle2D]] | None = None
    agent_radius: float = 1.0
    ped_radius: float = 0.4
    goal_radius: float = 1.0

    @property
    def obstacle_coords(self) -> np.ndarray:
        """
        Returns the obstacle coordinates.

        Returns
        -------
        np.ndarray
            The obstacle coordinates.
        """
        return self.get_obstacle_coords()

    @property
    def pedestrian_coords(self) -> np.ndarray:
        """
        Returns the pedestrian coordinates.

        Returns
        -------
        np.ndarray
            The pedestrian coordinates.
        """
        return self.get_pedestrian_coords()

    @property
    def is_obstacle_collision(self) -> bool:
        """
        Checks if there is a collision with an obstacle.

        Returns
        -------
        bool
            True if there is a collision with an obstacle, False otherwise.
        """
        agent_x, agent_y = self.get_agent_coords()
        if not self.is_in_bounds(agent_x, agent_y):
            return True

        collision_distance = self.agent_radius
        circle_agent = ((agent_x, agent_y), collision_distance)
        return circle_collides_any_lines(circle_agent, self.obstacle_coords)

    @property
    def is_pedestrian_collision(self) -> bool:
        """
        Checks if there is a collision with a pedestrian.

        Returns
        -------
        bool
            True if there is a collision with a pedestrian, False otherwise.
        """
        collision_distance = self.agent_radius
        ped_radius = self.ped_radius
        circle_agent = (self.get_agent_coords(), collision_distance)
        ped_circles = (((ped_x, ped_y), ped_radius) for ped_x, ped_y in self.pedestrian_coords)
        return circle_collides_any(circle_agent, ped_circles)

    @property
    def is_dynamic_collision(self) -> bool:
        """Checks collision against arbitrary dynamic objects.

        Returns:
            bool: True if the agent intersects any dynamic object.
        """
        if self.get_dynamic_objects is None:
            return False
        circle_agent = (self.get_agent_coords(), self.agent_radius)
        return circle_collides_any(circle_agent, self.get_dynamic_objects())

    @property
    def is_robot_at_goal(self) -> bool:
        """
        Checks if the robot is at the goal.

        Returns
        -------
        bool
            True if the robot is at the goal, False otherwise.
        """
        agent_circle = (self.get_agent_coords(), self.agent_radius)
        goal_circle = (self.get_goal_coords(), self.goal_radius)
        return is_circle_circle_intersection(agent_circle, goal_circle)

    def is_in_bounds(self, world_x: float, world_y: float) -> bool:
        """
        Checks if a point is within the bounds of the occupancy.
        Parameters
        ----------
        world_x : float
            The x-coordinate of the point.
        world_y : float
            The y-coordinate of the point.

        Returns
        -------
        bool
            True if the point is within the bounds of the occupancy, False otherwise.
        """
        return 0 <= world_x <= self.width and 0 <= world_y <= self.height


@dataclass
class EgoPedContinuousOccupancy(ContinuousOccupancy):
    """
    A class, which extends the continuous occupancy for the ego pedestrian.

    Attributes
    ----------
    width : float
        The width of the occupancy.
    height : float
        The height of the occupancy.
    get_agent_coords : Callable[[], Vec2D]
        A function to get the agent coordinates.
    get_goal_coords : Callable[[], Vec2D]
        A function to get the goal coordinates.
    get_obstacle_coords : Callable[[], np.ndarray]
        A function to get the obstacle coordinates.
    get_pedestrian_coords : Callable[[], np.ndarray]
        A function to get the pedestrian coordinates.
    agent_radius : float, optional
        The robot radius, by default 1.0.
    ped_radius : float, optional
        The pedestrian radius, by default 0.4.
    goal_radius : float, optional
        The goal radius, by default 1.0.
    get_enemy_coords : Optional[Callable[[], Vec2D]]
        The coordinates of the opposing agent.
    enemy_radius : optional, float=1.0
        The radius of the opposing agent.
    """

    get_enemy_coords: Callable[[], Vec2D] | None = None
    enemy_radius: float = 1.0

    @property
    def enemy_coords(self) -> np.ndarray:
        """
        Returns the enemy coordinates.

        Returns
        -------
        np.ndarray
            The enemy coordinates.
        """
        return self.get_enemy_coords()

    @property
    def distance_to_robot(self) -> float:
        """
        Gets the Euclidean distance to the robot.

        Returns
        -------
        float
            Distance to the robot.
        """
        return euclid_dist(self.get_enemy_coords(), self.get_agent_coords())

    @property
    def is_agent_agent_collision(self) -> bool:
        """
        Checks if the agent collided with another agent.

        Returns
        -------
        bool
            True if the agent collided with another agent, False otherwise.
        """
        if self.get_enemy_coords is None:
            return False

        agent_circle = (self.get_agent_coords(), self.agent_radius)
        enemy_circle = (self.get_enemy_coords(), self.enemy_radius)
        return is_circle_circle_intersection(agent_circle, enemy_circle)


def check_quality_of_map_point(map_def: "MapDefinition", point: Vec2D, radius: float = 0.0) -> bool:
    """Return True if a point (or circular footprint) is inside bounds and obstacle-free.

    Args:
        map_def: Map definition containing bounds and obstacles.
        point: (x, y) coordinates to test.
        radius: Optional footprint radius; when <= 0, a small epsilon radius is used.

    Returns:
        bool: True when the point lies within map bounds and does not intersect any obstacle edge.
    """
    x, y = point
    if not (0.0 <= x <= map_def.width and 0.0 <= y <= map_def.height):
        return False

    eff_radius = radius if radius > 0 else 1e-6
    circle: Circle2D = (point, eff_radius)
    obstacle_lines: list[Line2D] = []
    for obstacle in map_def.obstacles:
        obstacle_lines.extend(((line[0], line[1]), (line[2], line[3])) for line in obstacle.lines)
    for bound in map_def.bounds:
        if isinstance(bound, (tuple, list)) and len(bound) == 2:
            obstacle_lines.append(bound)  # type: ignore[list-item]
            continue
        try:
            x1, x2, y1, y2 = bound  # type: ignore[misc]
        except (TypeError, ValueError):
            continue
        obstacle_lines.append(((x1, y1), (x2, y2)))

    return not any(is_circle_line_intersection(circle, seg) for seg in obstacle_lines)
