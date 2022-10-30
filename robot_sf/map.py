from math import floor, ceil
from typing import Callable, Tuple, List

import numba
import numpy as np

import logging
logging.getLogger('numba').setLevel(logging.WARNING)

from robot_sf.vector import Vec2D


class BinaryOccupancyGrid():
    """Representing a discretized 2D map keeping track of all
    obstacles, pedestrians to simulate interactions with the robot."""

    def __init__(self, map_height: float, map_width: float,
                 map_resolution: float, box_size: float,
                 get_obstacle_coords: Callable[[], np.ndarray],
                 get_pedestrian_coords: Callable[[], np.ndarray]):
        self.box_size = box_size
        self.get_obstacle_coords = get_obstacle_coords
        self.get_pedestrian_coords = get_pedestrian_coords

        self.grid_width = int(ceil(map_width * map_resolution))
        self.grid_height = int(ceil(map_height * map_resolution))

        self.occupancy_obstacles, self.obstacle_coordinates = \
            self._initialize_static_objects()
        self.update_moving_objects()

    @property
    def occupancy_overall(self) -> np.ndarray:
        return np.logical_or(self.occupancy_pedestrians, self.occupancy_obstacles)

    def robot_occupancy(self, robot_pos: Vec2D, coll_distance: float) -> np.ndarray:
        world_x, world_y = robot_pos.as_list
        x, y = self._world_coords_to_grid_cell(world_x, world_y)
        x_step, y_step = self._world_length_to_grid_cell_span(coll_distance)

        occ_shape = (self.grid_width, self.grid_height)
        occupancy = np.zeros(occ_shape, dtype=bool)
        fill_surrounding(occupancy, x, y, x_step, y_step)
        return occupancy

    def is_collision(self, robot_pos: Vec2D, collision_distance: float) -> bool:
        return self.is_pedestrians_collision(robot_pos, collision_distance) \
            or self.is_obstacle_collision(robot_pos, collision_distance)

    def is_obstacle_collision(self, robot_pos: Vec2D, collision_distance: float) -> bool:
        occupancy = self.robot_occupancy(robot_pos, collision_distance)
        return np.logical_and(self.occupancy_obstacles, occupancy).any()

    def is_pedestrians_collision(self, robot_pos: Vec2D, collision_distance: float) -> bool:
        occupancy = self.robot_occupancy(robot_pos, collision_distance)
        return np.logical_and(self.occupancy_pedestrians, occupancy).any()

    def is_in_bounds(self, x: float, y: float) -> bool:
        return -self.box_size <= x <= self.box_size \
            and -self.box_size <= y <= self.box_size

    def position_bounds(self) -> Tuple[List[float], List[float]]:
        low_bound  = [-self.box_size, -self.box_size, -np.pi]
        high_bound = [self.box_size, self.box_size,  np.pi]
        return low_bound, high_bound

    def update_moving_objects(self):
        self.occupancy_pedestrians = self._compute_moving_objects_occupancy()

    def _world_coords_to_grid_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Map coordinates from a continuous 2D world to a discrete 2D grid.
          x: [-box_x, box_x] -> [0, grid_width  - 1],
          y: [-box_y, box_y] -> [0, grid_height - 1]"""
        scaled_x = (x + self.box_size) / (2 * self.box_size) * self.grid_width
        scaled_y = (y + self.box_size) / (2 * self.box_size) * self.grid_height
        grid_x = min(floor(scaled_x), self.grid_width - 1)
        grid_y = min(floor(scaled_y), self.grid_height - 1)
        # info: min() function handles the rare case of an index overflow
        #       when processing x = self.grid_width and/or y = self.grid_height
        return grid_x, grid_y

    def _grid_cell_centroid_to_world_coords(self, x: int, y: int) -> Tuple[float, float]:
        """Map coordinates from a continuous 2D world to a discrete 2D grid.
          x: [0, grid_width  - 1] -> [-box_x, box_x],
          y: [0, grid_height - 1] -> [-box_y, box_y]"""
        world_x = (x + 0.5) / self.grid_width * 2 * self.box_size - self.box_size
        world_y = (y + 0.5) / self.grid_height * 2 * self.box_size - self.box_size
        return world_x, world_y

    def _world_length_to_grid_cell_span(self, length: float) -> Tuple[int, int]:
        """Convert scalar world length to the corresponding,
        rounded grid cells in x and y direction."""
        rel_len = length / (2 * self.box_size)
        return round(rel_len * self.grid_width), round(rel_len * self.grid_height)

    def _initialize_static_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        occ_shape = (self.grid_width, self.grid_height)
        occupancy = np.zeros(occ_shape, dtype=bool)

        all_coords = np.concatenate(self.get_obstacle_coords())
        coords_in_bounds = [pos for pos in all_coords \
                            if pos.size != 0 and self.is_in_bounds(pos[0], pos[1])]
        grid_coords = [self._world_coords_to_grid_cell(pos[0], pos[1])
                       for pos in coords_in_bounds]

        for x, y in grid_coords:
            occupancy[x, y] = 1

        # make the map boundary an "obstacle"
        occupancy[:, 0] = 1
        occupancy[:, -1] = 1
        occupancy[0, :] = 1
        occupancy[-1, :] = 1

        # info: this is not really efficient, but don't worry about
        #       performance, it's only executed once on map creation
        radius = 0.3
        x_step, y_step = self._world_length_to_grid_cell_span(radius)
        x, y = np.where(occupancy)
        eval_points = np.vstack((x, y)).T

        for x, y in eval_points:
            x, y = self._world_coords_to_grid_cell(x, y)
            fill_surrounding(occupancy, x, y, x_step, y_step)

        return occupancy, np.array(coords_in_bounds)

    def _compute_moving_objects_occupancy(self) -> np.ndarray:
        occ_shape = (self.grid_width, self.grid_height)
        occupancy = np.zeros(occ_shape, dtype=bool)

        peds_pos = self.get_pedestrian_coords()
        grid_coords = [
            self._world_coords_to_grid_cell(pos[0], pos[1])
            for pos in peds_pos
            if pos.size != 0 and self.is_in_bounds(pos[0], pos[1])
        ]

        radius = 0.4
        x_step, y_step = self._world_length_to_grid_cell_span(radius)
        for x, y in grid_coords:
            # TODO: add noise to the signal
            fill_surrounding(occupancy, x, y, x_step, y_step)

        return occupancy


@numba.njit(fastmath=True)
def fill_surrounding(occupancy: np.ndarray, pos_x: int, pos_y: int, x_dist: int, y_dist: int):
    # TODO: add noise to the occupancy
    width, height = occupancy.shape
    for x_offset in range(-x_dist, 2 * x_dist + 1):
        for y_offset in range(-y_dist, 2 * y_dist + 1):
            x, y = pos_x + x_offset, pos_y + y_offset
            if 0 <= x < width and 0 <= y < height:
                occupancy[x, y] = 1
