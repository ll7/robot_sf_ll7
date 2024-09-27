from typing import Callable, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import numba


Vec2D = Tuple[float, float]
Circle2D = Tuple[Vec2D, float]
Line2D = Tuple[Vec2D, Vec2D]

#TODO: REFACTOR IMPORTS TO UTILS FILE -> euclid_dist is defined in range_sensor.py

@numba.njit(fastmath=True)
def euclid_dist(vec_1: Vec2D, vec_2: Vec2D) -> float:
    """
    Calculate Euclidean distance between two 2D vectors.

    Parameters:
    vec_1 (Vec2D): First 2D vector.
    vec_2 (Vec2D): Second 2D vector.

    Returns:
    float: Euclidean distance between vec_1 and vec_2.
    """
    # Subtract corresponding elements of vectors
    # Square the results, sum them, and take square root
    return ((vec_1[0] - vec_2[0])**2 + (vec_1[1] - vec_2[1])**2)**0.5

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
    dist_sq = (center_1[0] - center_2[0])**2 + (center_1[1] - center_2[1])**2

    # Calculate the square of the sum of the radii of the circles
    rad_sum_sq = (radius_1 + radius_2)**2

    # The circles intersect if the square of the distance between their centers
    # is less than or equal to the square of the sum of their radii
    return dist_sq <= rad_sum_sq


@numba.njit(fastmath=True)
def is_circle_line_intersection(circle: Circle2D, segment: Line2D) -> bool:
    """Simple vector math implementation using quadratic solution formula."""
    (c_x, c_y), r = circle
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
    agent_radius: float=1.0
    ped_radius: float=0.4
    goal_radius: float=1.0

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
        for s_x, s_y, e_x, e_y in self.obstacle_coords:
            if is_circle_line_intersection(circle_agent, ((s_x, s_y), (e_x, e_y))):
                return True
        return False

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
        for ped_x, ped_y in self.pedestrian_coords:
            circle_ped = ((ped_x, ped_y), ped_radius)
            if is_circle_circle_intersection(circle_agent, circle_ped):
                return True
        return False

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

    get_enemy_coords: Optional[Callable[[], Vec2D]] = None
    enemy_radius: float=1.0

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
        Gets the euklidean distance to the robot.

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
