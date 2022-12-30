from math import dist
from typing import Callable, Tuple
from dataclasses import dataclass

import numpy as np

from robot_sf.geometry import is_circle_circle_intersection, is_circle_line_intersection


Vec2D = Tuple[float, float]


@dataclass
class ContinuousOccupancy:
    box_size: float
    get_robot_coords: Callable[[], Vec2D]
    get_goal_coords: Callable[[], Vec2D]
    get_obstacle_coords: Callable[[], np.ndarray]
    get_pedestrian_coords: Callable[[], np.ndarray]
    robot_radius: float=1.0
    goal_radius: float=1.0

    @property
    def obstacle_coords(self) -> np.ndarray:
        return self.get_obstacle_coords()[:, :4]

    @property
    def pedestrian_coords(self) -> np.ndarray:
        return self.get_pedestrian_coords()

    @property
    def is_robot_collision(self) -> bool:
        robot_x, robot_y = self.get_robot_coords()
        return self.is_pedestrians_collision(self.robot_radius) or \
            self.is_obstacle_collision(self.robot_radius) or \
            not self.is_in_bounds(robot_x, robot_y)

    @property
    def is_robot_at_goal(self) -> bool:
        return dist(self.get_robot_coords(), self.get_goal_coords()) < self.goal_radius

    def is_obstacle_collision(self, collision_distance: float) -> bool:
        circle_robot = (self.get_robot_coords(), collision_distance)
        for s_x, s_y, e_x, e_y in self.obstacle_coords:
            if is_circle_line_intersection(circle_robot, ((s_x, s_y), (e_x, e_y))):
                return True
        return False

    def is_pedestrians_collision(self, collision_distance: float) -> bool:
        ped_radius = 0.4
        circle_robot = (self.get_robot_coords(), collision_distance)
        for ped_x, ped_y in self.pedestrian_coords:
            circle_ped = ((ped_x, ped_y), ped_radius)
            if is_circle_circle_intersection(circle_robot, circle_ped):
                return True
        return False

    def is_in_bounds(self, world_x: float, world_y: float) -> bool:
        return -self.box_size <= world_x <= self.box_size \
            and -self.box_size <= world_y <= self.box_size
