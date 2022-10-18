from math import floor
from typing import Callable, Tuple, List

import numpy as np

from robot_sf.vector import Vec2D


class BinaryOccupancyGrid():
    """Representing a discretized 2D map keeping track of all
    obstacles, pedestrians """

    def __init__(self, map_height: float, map_width: float,
                 map_resolution: float, box_size: float,
                 get_obstacle_coords: Callable[[], np.ndarray],
                 get_pedestrian_coords: Callable[[], np.ndarray]):
        self.map_height = map_height
        self.map_width = map_width
        self.map_resolution = map_resolution
        self.box_size = box_size
        self.get_obstacle_coords = get_obstacle_coords
        self.get_pedestrian_coords = get_pedestrian_coords

        self.grid_width = int(np.ceil(self.map_width * self.map_resolution))
        self.grid_height = int(np.ceil(self.map_height * self.map_resolution))

        self.occupancy_static_objects, self.obstacle_coordinates = \
            self._initialize_static_objects()
        occ_shape = (self.grid_width, self.grid_height)
        self.occupancy_moving_objects = np.zeros(occ_shape, dtype=bool)
        self.update_moving_objects()

    @property
    def occupancy_overall(self) -> np.ndarray:
        return np.logical_or(self.occupancy_moving_objects, self.occupancy_static_objects)

    @property
    def occupancy_overall_xy(self) -> np.ndarray:
        # info: swap (y, x) coordinates back to (x, y) coordinates
        # TODO: remove this dirty hack once the class is refactored
        return self.occupancy_overall[:, ::-1]

    def is_collision(self, robot_pos: Vec2D, collision_distance: float) -> bool:
        return self.is_pedestrians_collision(robot_pos, collision_distance) \
            or self.is_obstacle_collision(robot_pos, collision_distance)

    def is_obstacle_collision(self, robot_pos: Vec2D, collision_distance: float) -> bool:
        occupancy = self._get_robot_occupancy(robot_pos, collision_distance)
        return np.logical_and(self.occupancy_static_objects, occupancy).any()

    def is_pedestrians_collision(self, robot_pos: Vec2D, collision_distance: float) -> bool:
        occupancy = self._get_robot_occupancy(robot_pos, collision_distance)
        return np.logical_and(self.occupancy_moving_objects, occupancy).any()

    def position_bounds(self, margin: float) -> Tuple[List[float], List[float]]:
        # TODO: figure out what "margin" is supposed to achieve
        low_bound  = [-self.box_size, -self.box_size, -np.pi]
        high_bound = [self.box_size, self.box_size,  np.pi]
        return low_bound, high_bound

    def update_moving_objects(self, map_margin: float=1.5):
        self.occupancy_moving_objects = self._compute_moving_objects_occupancy()

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

    def _is_in_bounds(self, x: float, y: float) -> bool:
        # TODO: add a more sophisticated approach to this
        return -self.box_size <= x < self.box_size \
            and -self.box_size <= y < self.box_size

    def _get_robot_occupancy(self, robot_pos: Vec2D, coll_distance: float) -> np.ndarray:
        rob_matrix = np.zeros(self.occupancy_moving_objects.shape, dtype=bool)
        idx = self.convert_world_to_grid_no_error(np.array(robot_pos.as_list)[np.newaxis, :])
        int_radius_step = round(self.map_resolution * coll_distance)
        return fill_surrounding(rob_matrix, int_radius_step, idx)

    def _initialize_static_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        occ_shape = (self.grid_width, self.grid_height)
        occupancy = np.zeros(occ_shape, dtype=bool)

        all_coords = self.get_obstacle_coords()
        coords_in_bounds = [pos for pos in all_coords \
                            if pos.size != 0 and self._is_in_bounds(pos[0], pos[1])]
        grid_coords = [self._world_coords_to_grid_cell(pos[0], pos[1])
                       for pos in coords_in_bounds]

        for x, y in grid_coords:
            occupancy[x, y] = 1

        # make the map boundary an "obstacle"
        occupancy[:, 0] = 1
        occupancy[:, -1] = 1
        occupancy[0, :] = 1
        occupancy[-1, :] = 1

        radius = 0.3
        int_radius_step = round(self.map_resolution * radius)
        eval_points = np.asarray(np.where(occupancy))
        occupancy = fill_surrounding(occupancy, int_radius_step, eval_points)

        return occupancy, np.array(coords_in_bounds)

    def _compute_moving_objects_occupancy(self) -> np.ndarray:
        occ_shape = (self.grid_width, self.grid_height)
        occupancy = np.zeros(occ_shape, dtype=bool)

        peds_pos = self.get_pedestrian_coords()
        grid_coords = [
            self._world_coords_to_grid_cell(pos[0], pos[1])
            for pos in peds_pos
            if pos.size != 0 and self._is_in_bounds(pos[0], pos[1])
        ]

        for x, y in grid_coords:
            occupancy[x, y] = 1

        radius= 0.4
        int_radius_step = round(self.map_resolution * radius)
        eval_points = np.asarray(np.where(occupancy))
        occupancy = fill_surrounding(occupancy, int_radius_step, eval_points, add_noise=True)

        return occupancy

    def check_if_valid_world_coordinates(self, pair, margin=0):
        if isinstance(pair, list):
            pair = np.array(pair)

        if len(pair.shape) < 2:
            pair = pair[np.newaxis, :]

        offset = margin * np.array([self.map_width, self.map_height])
        valid_pairs = np.bitwise_and(
            (pair >= (self.min_val + offset)).all(axis = 1),
            (pair <= (self.max_val - offset)).all(axis = 1))
        if valid_pairs.all():
            return pair
        elif valid_pairs.any():
            return pair[valid_pairs, :]
        else:
            return np.array(False)

    def check_if_valid_grid_index(self,pair):
        for i in range(pair.shape[0]):
            if pair[i,0] < 0 or pair[i,0] > self.grid_size['y'] or pair[i,1] < 0 \
                    or pair[i,1] > self.grid_size['x'] or not \
                    issubclass(pair.dtype.type,np.integer):
                return False
        return True

    def world_coords_to_grid_index(self, pair):
        pair = self.check_if_valid_world_coordinates(pair)
        return self.convert_world_to_grid_no_error(pair)

    def grid_index_to_world_coords(self, pair):
        if not self.check_if_valid_grid_index(pair):
            raise ValueError('Invalid grid indices with the current map!')
        val = np.zeros(pair.shape)
        for i in range(pair.shape[0]):
            val[i, 0] = self.x[0, pair[i, 1]]
            val[i, 1] = self.y[pair[i, 0], 0]
        return val

    def convert_world_to_grid_no_error(self, pair):
        # TODO: this is 100% BS, refactor!!!
        return np.concatenate(
            (np.abs(self.y[:, 0][:, np.newaxis] - pair[:, 1].T).argmin(axis=0)[:, np.newaxis],
             np.abs(self.x[0, :][:, np.newaxis] - pair[:, 0].T).argmin(axis=0)[:, np.newaxis]), axis=1)


def fill_surrounding(matrix, int_radius_step, coords: np.ndarray, add_noise = False):

    def n_closest_fill(x: np.ndarray, pos: Tuple[float, float], d: int, new_fill):
        x_copy = x.copy()
        n_x, n_y = pos
        x_copy[n_x-d:n_x+d+1, n_y-d:n_y+d+1] = new_fill
        # bitwise OR means set bits never get cleaned up
        # TODO: think about whether this is actually the desired behavior
        x = np.logical_or(x, x_copy)
        return x

    window = np.arange(-int_radius_step, int_radius_step + 1)
    msh_grid = np.stack(np.meshgrid(window, window), axis=2)
    bool_subm = np.sqrt(np.square(msh_grid).sum(axis=2)) < int_radius_step

    N = 2 * int_radius_step + 1
    p = np.round(np.exp(-(np.square(msh_grid) / 15).sum(axis=2)), 3)

    for i in np.arange(coords.shape[0]):
        if (coords[i, :] > int_radius_step).all() and (coords[i, :] < matrix.shape[0] - int_radius_step).all():
            bool_subm_i = np.random.random(size=(N, N)) < p if add_noise else bool_subm
            matrix = n_closest_fill(matrix, coords[i, :], int_radius_step, bool_subm_i) 
        else:
            matrix_filled = fill_surrounding_brute(matrix.copy(),int_radius_step,coords[i,0],coords[i,1])
            # TODO: don't overwrite the input array, make this work in-place to avoid allocation
            matrix = np.logical_or(matrix, matrix_filled)

    return matrix


def fill_surrounding_brute(matrix: np.ndarray, int_radius_step, i, j):
    row_eval = np.unique(np.minimum(matrix.shape[0] - 1, np.maximum(0, np.arange(i - int_radius_step, i + 1 + int_radius_step))))
    col_eval = np.unique(np.minimum(matrix.shape[1] - 1, np.maximum(0, np.arange(j - int_radius_step, j + 1 + int_radius_step))))
    eval_indices = np.sqrt((row_eval-i)[:,np.newaxis]**2 + (col_eval-j)[np.newaxis,:]**2) < int_radius_step
    matrix[row_eval[0]:row_eval[-1]+1, col_eval[0]:col_eval[-1]+1] = eval_indices
    return matrix
