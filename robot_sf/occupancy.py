from typing import Callable, Tuple
from dataclasses import dataclass

import numpy as np

from robot_sf.geometry import \
    is_circle_circle_intersection, \
    is_circle_line_intersection


Vec2D = Tuple[float, float]


@dataclass
class ContinuousOccupancy:
    width: float
    height: float
    get_robot_coords: Callable[[], Vec2D]
    get_goal_coords: Callable[[], Vec2D]
    get_obstacle_coords: Callable[[], np.ndarray]
    get_pedestrian_coords: Callable[[], np.ndarray]
    robot_radius: float=1.0
    ped_radius: float=0.4
    goal_radius: float=1.0

    @property
    def obstacle_coords(self) -> np.ndarray:
        return self.get_obstacle_coords()

    @property
    def pedestrian_coords(self) -> np.ndarray:
        return self.get_pedestrian_coords()

    @property
    def is_obstacle_collision(self) -> bool:
        robot_x, robot_y = self.get_robot_coords()
        if not self.is_in_bounds(robot_x, robot_y):
            return True

        collision_distance = self.robot_radius
        circle_robot = ((robot_x, robot_y), collision_distance)
        for s_x, s_y, e_x, e_y in self.obstacle_coords:
            if is_circle_line_intersection(circle_robot, ((s_x, s_y), (e_x, e_y))):
                return True
        return False

    @property
    def is_pedestrian_collision(self) -> bool:
        collision_distance = self.robot_radius
        ped_radius = self.ped_radius
        circle_robot = (self.get_robot_coords(), collision_distance)
        for ped_x, ped_y in self.pedestrian_coords:
            circle_ped = ((ped_x, ped_y), ped_radius)
            if is_circle_circle_intersection(circle_robot, circle_ped):
                return True
        return False

    @property
    def is_robot_at_goal(self) -> bool:
        robot_circle = (self.get_robot_coords(), self.robot_radius)
        goal_circle = (self.get_goal_coords(), self.goal_radius)
        return is_circle_circle_intersection(robot_circle, goal_circle)

    def is_in_bounds(self, world_x: float, world_y: float) -> bool:
        return 0 <= world_x <= self.width and 0 <= world_y <= self.height
