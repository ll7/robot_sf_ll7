from typing import Callable, Tuple, List

import numpy as np

from robot_sf.geometry import is_circle_circle_intersection, is_circle_line_intersection


Vec2D = Tuple[float, float]


class ContinuousOccupancy:
    def __init__(self, box_size: float,
                 get_obstacle_coords: Callable[[], np.ndarray],
                 get_pedestrian_coords: Callable[[], np.ndarray]):
        self.box_size = box_size
        self.get_obstacle_coords = get_obstacle_coords
        self.get_pedestrian_coords = get_pedestrian_coords

    @property
    def obstacle_coords(self) -> np.ndarray:
        return self.get_obstacle_coords()[:, :4]

    @property
    def pedestrian_coords(self) -> np.ndarray:
        return self.get_pedestrian_coords()

    def is_collision(self, robot_pos: Vec2D, collision_distance: float) -> bool:
        return self.is_pedestrians_collision(robot_pos, collision_distance) \
            or self.is_obstacle_collision(robot_pos, collision_distance)

    def is_obstacle_collision(self, robot_pos: Vec2D, collision_distance: float) -> bool:
        circle_robot = (robot_pos, collision_distance)
        for s_x, s_y, e_x, e_y in self.obstacle_coords:
            if is_circle_line_intersection(circle_robot, ((s_x, s_y), (e_x, e_y))):
                return True
        return False

    def is_pedestrians_collision(self, robot_pos: Vec2D, collision_distance: float) -> bool:
        ped_radius = 0.4
        circle_robot = (robot_pos, collision_distance)
        for ped_x, ped_y in self.pedestrian_coords:
            circle_ped = ((ped_x, ped_y), ped_radius)
            if is_circle_circle_intersection(circle_robot, circle_ped):
                return True
        return False

    def is_in_bounds(self, world_x: float, world_y: float) -> bool:
        return -self.box_size <= world_x <= self.box_size \
            and -self.box_size <= world_y <= self.box_size

    def position_bounds(self) -> Tuple[List[float], List[float]]:
        low_bound  = [-self.box_size, -self.box_size, -np.pi]
        high_bound = [self.box_size, self.box_size,  np.pi]
        return low_bound, high_bound

    def update_moving_objects(self):
        pass
