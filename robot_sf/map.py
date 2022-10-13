from typing import Callable, Tuple, List

import numpy as np

from robot_sf.vector import Vec2D
from robot_sf.utils.utilities import linspace


class BinaryOccupancyGrid():
    """The class is responsible of creating an object
    representing a discrete map of the environment"""

    def __init__(self, map_height, map_width, map_resolution, box_size: float,
                 get_obstacle_coords: Callable[[], np.ndarray],
                 get_pedestrian_coords: Callable[[], np.ndarray]):
        # If no arguments passed create default map.
        # Use this if you plan to load an existing map from file
        """map_height: height of the map in meters
           map_length: lenght of the map in meters
           map_resolution: number of cells per meters
           grid_size: number of cells along x and y axis
           cell_size: dimension of a single cell expressed in meters"""

        self.box_size = box_size
        self.get_obstacle_coords = get_obstacle_coords
        self.get_pedestrian_coords = get_pedestrian_coords

        self.map_height = map_height
        self.map_width = map_width
        self.map_resolution = map_resolution
        self.grid_size = dict()
        self.grid_size['x'] = int(np.ceil(self.map_width * self.map_resolution))
        self.grid_size['y'] = int(np.ceil(self.map_height * self.map_resolution))

        self.cell_size = dict()
        self.cell_size['x'] = self.map_width/(self.grid_size['x'])
        self.cell_size['y'] = self.map_height/(self.grid_size['y'])
        x = linspace(0 + self.cell_size['x'] / 2, self.map_width - self.cell_size['x'] / 2, self.grid_size['x'])
        y = linspace(0 + self.cell_size['y'] / 2, self.map_height - self.cell_size['y'] / 2, self.grid_size['y'])
        y = np.flip(y)
        self.x, self.y = np.meshgrid(x, y)
        self.min_val = np.array([self.x[0,  0] - self.cell_size['x'] / 2, self.y[-1, 0] - self.cell_size['y'] / 2])
        self.max_val = np.array([self.x[0, -1] + self.cell_size['x'] / 2, self.y[ 0, 0] + self.cell_size['y'] / 2])

        # TODO: refactor the logic to use (x, y) coordinates instead of (y, x) coordinates
        self.occupancy_moving_objects = np.zeros((self.grid_size['y'], self.grid_size['x']), dtype = bool)
        self.occupancy_static_objects = self.occupancy_moving_objects.copy()
        self.occupancy_robot = self.occupancy_moving_objects.copy()
        self.occupancy_overall = self.occupancy_moving_objects.copy()

        self.grid_origin = [0, 0]
        self.add_noise = True
        self.move_map_frame([self.map_width / 2, self.map_height / 2])
        self.initialize_static_objects()
        self.update_moving_objects()

    @property
    def occupancy_overall_xy(self) -> np.ndarray:
        # info: swap (y, x) coordinates back to (x, y) coordinates
        return self.occupancy_overall[:, ::-1]

    def get_robot_occupancy(self, robot_pos: Vec2D, coll_distance: float) -> np.ndarray:
        rob_matrix = np.zeros(self.occupancy_moving_objects.shape, dtype=bool)
        idx = self.convert_world_to_grid_no_error(np.array(robot_pos.as_list)[np.newaxis, :])
        int_radius_step = round(self.map_resolution * coll_distance)
        return fill_surrounding(rob_matrix, int_radius_step, idx)

    def position_bounds(self, margin: float):
        # TODO: figure out what this does

        x_idx_min = round(margin * self.grid_size['x'])
        x_idx_max = round((1 - margin) * self.grid_size['x'])

        y_idx_min = round(margin * self.grid_size['y'])
        y_idx_max = round((1 - margin) * self.grid_size['y'])

        low_bound  = [self.x[0, x_idx_min], self.y[y_idx_min, 0], -np.pi]
        high_bound = [self.x[0, x_idx_max], self.y[y_idx_max, 0],  np.pi]
        return low_bound, high_bound

    def is_collision(self, robot_pos: Vec2D, collision_distance: float):
        return self.is_pedestrians_collision(robot_pos, collision_distance) \
            or self.is_obstacle_collision(robot_pos, collision_distance)

    def is_obstacle_collision(self, robot_pos: Vec2D, collision_distance: float) -> bool:
        occupancy = self.get_robot_occupancy(robot_pos, collision_distance)
        return np.logical_and(self.occupancy_static_objects, occupancy).any()

    def is_pedestrians_collision(self, robot_pos: Vec2D, collision_distance: float) -> bool:
        occupancy = self.get_robot_occupancy(robot_pos, collision_distance)
        return np.logical_and(self.occupancy_moving_objects, occupancy).any()

    def set_fixed_objects_occupancy(self, pair, val):
        # pair must be a (m,2) list or numpy array
        pair = np.array(pair)
        val = np.array(val, dtype=bool)

        # need to search in world coordinates!
        # first check if input is valid
        pair = self.check_if_valid_world_coordinates(pair)
        if not pair.any():
            return

        # now if pair is valid start searching the indexes of the corrisponding x and y values
        idx = self.world_coords_to_grid_index(pair)
        for i in range(idx.shape[0]):
            self.occupancy_static_objects[idx[i, 0], idx[i, 1]] = val[i]

        self.obstacle_coordinates = np.concatenate([
            self.x[self.occupancy_static_objects][:, np.newaxis],
            self.y[self.occupancy_static_objects][:, np.newaxis]], axis=1)

        # make the map boundary an "obstacle"
        self.occupancy_static_objects[:, 0] = True
        self.occupancy_static_objects[:, -1] = True
        self.occupancy_static_objects[0, :] = True
        self.occupancy_static_objects[-1, :] = True
        self.occupancy_overall = self.occupancy_static_objects

    def set_moving_objects_occupancy(self, pair, val):
        # pair must be a (m,2) list or numpy array
        pair = np.array(pair)
        val = np.array(val, dtype=bool)
        val = val if val.shape else val * np.ones((pair.shape[0], 1))

        # need to search in world coordinates!
        # first check if input is valid
        pair = self.check_if_valid_world_coordinates(pair)
        if not pair.any():
            return None

        # now if pair is valid start searching the indexes of the corrisponding x and y values
        idx = self.world_coords_to_grid_index(pair)
        for i in range(idx.shape[0]):
            self.occupancy_moving_objects[idx[i, 0], idx[i, 1]] = val[i]
        self.update_overall_occupancy()

    def update_overall_occupancy(self):
        self.occupancy_overall = np.logical_or(
            self.occupancy_moving_objects, self.occupancy_static_objects)

    def inflate(self, radius: float, fixed_objects_map: bool=False):
        """Fill the occupancy around an object in circle shape"""
        # create a copy of the occupancy matrix
        int_radius_step = round(self.map_resolution * radius)
        if fixed_objects_map:  # problem here!!!!
            eval_points = np.asarray(np.where(self.occupancy_static_objects)).T
            self.occupancy_static_objects = fill_surrounding(
                self.occupancy_static_objects, int_radius_step, eval_points)
            self.update_overall_occupancy()
        else:
            eval_points = np.asarray(np.where(self.occupancy_moving_objects)).T
            self.occupancy_moving_objects = fill_surrounding(
                self.occupancy_moving_objects, int_radius_step, eval_points, add_noise=self.add_noise)
            self.update_overall_occupancy()

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
            if pair[i,0] < 0 or pair[i,0] > self.grid_size['y'] or pair[i,1] < 0 or pair[i,1] > self.grid_size['x'] or not \
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
        return np.concatenate(
            (np.abs(self.y[:, 0][:, np.newaxis] - pair[:, 1].T).argmin(axis=0)[:, np.newaxis],
             np.abs(self.x[0, :][:, np.newaxis] - pair[:, 0].T).argmin(axis=0)[:, np.newaxis]), axis=1)

    def initialize_static_objects(self, map_margin=1.5):
        if self.x[0, 0] == self.cell_size['x'] / 2:
            maps_alignment_offset = map_margin * self.box_size
        else:
            maps_alignment_offset = 0

        # assign fixed obstacles to map
        tmp = [item for item in self.get_obstacle_coords() if item.size != 0]
        obs_coordinates = np.concatenate(tmp)
        new_obs_coordinates = obs_coordinates + np.ones((obs_coordinates.shape[0], 2)) * maps_alignment_offset
        # reset occupancy map
        self.occupancy_static_objects = np.zeros(self.occupancy_static_objects.shape, dtype=bool)
        self.set_fixed_objects_occupancy(new_obs_coordinates, np.ones((obs_coordinates.shape[0], 1), dtype=bool))
        self.inflate(radius=.3, fixed_objects_map=True)

    #update map from pedestrians simulation environment
    def update_moving_objects(self, map_margin: float=1.5):
        # shift peds sim map to top-left quadrant if necessary
        if self.x[0, 0] == self.cell_size['x'] / 2:
            maps_alignment_offset = map_margin * self.box_size
        else:
            maps_alignment_offset = 0

        # get peds states
        peds_pos = self.get_pedestrian_coords()
        # add offset to pedestrians maps (it can also have negative coordinates values)
        n_pedestrians = peds_pos.shape[0]
        peds_new_coordinates = peds_pos + np.ones((n_pedestrians, 2)) * maps_alignment_offset
        # reset occupancy map
        self.occupancy_moving_objects = np.zeros(self.occupancy_moving_objects.shape,dtype = bool)
        # assign pedestrians positions to robotMap
        self.set_moving_objects_occupancy(peds_new_coordinates, np.ones((n_pedestrians,1)))
        self.inflate(radius=.4) # inflate pedestrians only

    def move_map_frame(self, new_position):
        ''' This method will change the position of the grid 
            origin. By default when constructing '''
        self.x = self.x - new_position[0]
        self.y = self.y - new_position[1]
        self.grid_origin = [new_position[0], new_position[1]]

        self.min_val = np.array([self.x[0,  0] - self.cell_size['x'] / 2, self.y[-1, 0] - self.cell_size['y'] / 2])
        self.max_val = np.array([self.x[0, -1] + self.cell_size['x'] / 2, self.y[ 0, 0] + self.cell_size['y'] / 2])


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
            matrix = np.logical_or(matrix,matrix_filled)

    return matrix


def fill_surrounding_brute(matrix: np.ndarray, int_radius_step, i, j):
    row_eval = np.unique(np.minimum(matrix.shape[0] - 1, np.maximum(0, np.arange(i - int_radius_step, i + 1 + int_radius_step))))
    col_eval = np.unique(np.minimum(matrix.shape[1] - 1, np.maximum(0, np.arange(j - int_radius_step, j + 1 + int_radius_step))))
    eval_indices = np.sqrt((row_eval-i)[:,np.newaxis]**2 + (col_eval-j)[np.newaxis,:]**2) < int_radius_step
    matrix[row_eval[0]:row_eval[-1]+1, col_eval[0]:col_eval[-1]+1] = eval_indices
    return matrix
