from typing import Tuple

import numpy as np

from robot_sf.vector import Vec2D
from robot_sf.extenders_py_sf.extender_sim import ExtdSimulator
from robot_sf.utils.utilities import linspace


# TODO: figure out how the raycast is done here and why the range_sensor.py is unused


class BinaryOccupancyGrid():
    """The class is responsible of creating an object
    representing a discrete map of the environment"""

    def __init__(self, map_height=1, map_length=1, map_resolution=10, peds_sim_env=None):
        # If no arguments passed create default map.
        # Use this if you plan to load an existing map from file
        """map_height: height of the map in meters
           map_length: lenght of the map in meters
           map_resolution: number of cells per meters
           grid_size: number of cells along x and y axis
           cell_size: dimension of a single cell expressed in meters"""

        # TODO: make all constructor calls provide a peds_sim_env
        self.peds_sim_env = ExtdSimulator(np.zeros((1, 7))) if peds_sim_env is None else peds_sim_env

        self.map_height = map_height
        self.map_length = map_length
        self.map_resolution = map_resolution
        self.grid_size = dict()
        self.grid_size['x'] = int(np.ceil(self.map_length*self.map_resolution))
        self.grid_size['y'] = int(np.ceil(self.map_height*self.map_resolution))

        self.cell_size = dict()
        self.cell_size['x'] = self.map_length/(self.grid_size['x'])
        self.cell_size['y'] = self.map_height/(self.grid_size['y'])

        x = linspace(0 + self.cell_size['x'] / 2, self.map_length - self.cell_size['x'] / 2, self.grid_size['x'])
        y = linspace(0 + self.cell_size['y'] / 2, self.map_height - self.cell_size['y'] / 2, self.grid_size['y'])
        y = np.flip(y)
        self.x, self.y = np.meshgrid(x, y)
        self.min_val = np.array([self.x[0,  0] - self.cell_size['x'] / 2, self.y[-1, 0] - self.cell_size['y'] / 2])
        self.max_val = np.array([self.x[0, -1] + self.cell_size['x'] / 2, self.y[ 0, 0] + self.cell_size['y'] / 2])

        self.occupancy_moving_objects = np.zeros((self.grid_size['y'], self.grid_size['x']), dtype = bool)
        self.occupancy_moving_objects_raw = self.occupancy_moving_objects.copy()
        self.occupancy_static_objects = self.occupancy_moving_objects.copy()
        self.occupancy_robot = self.occupancy_moving_objects.copy()
        self.occupancy_static_objects_raw = self.occupancy_moving_objects.copy()
        self.occupancy_overall = self.occupancy_moving_objects.copy()

        self.grid_origin = [0, 0]
        self.add_noise = True

    def get_robot_occupancy(self, robot_pos: Vec2D, coll_distance: float) -> np.ndarray:
        rob_matrix = np.zeros(self.occupancy_moving_objects.shape, dtype=bool)
        idx = self.convert_world_to_grid_no_error(np.array(robot_pos.as_list)[np.newaxis, :])
        int_radius_step = round(self.map_resolution * coll_distance)
        return fill_surrounding(rob_matrix, int_radius_step, idx)

    def position_bounds(self, margin: float):
        # TODO: move this into the map class
        x_idx_min = round(margin * self.grid_size['x'])
        x_idx_max = round((1 - margin) * self.grid_size['x'])

        y_idx_min = round(margin * self.grid_size['y'])
        y_idx_max = round((1 - margin) * self.grid_size['y'])

        low_bound  = [self.x[0, x_idx_min], self.y[y_idx_min, 0], -np.pi]
        high_bound = [self.x[0, x_idx_max], self.y[y_idx_max, 0],  np.pi]
        return low_bound, high_bound

    def check_collision(self, robot_pos: Vec2D, collision_distance: float):
        return self.check_pedestrians_collision(robot_pos, collision_distance) \
            or self.check_obstacle_collision(robot_pos, collision_distance)

    def check_obstacle_collision(self, robot_pos: Vec2D, collision_distance: float) -> bool:
        occupancy = self.get_robot_occupancy(robot_pos, collision_distance)
        return np.logical_and(self.occupancy_static_objects, occupancy).any()

    def check_pedestrians_collision(self, robot_pos: Vec2D, collision_distance: float) -> bool:
        occupancy = self.get_robot_occupancy(robot_pos, collision_distance)
        return np.logical_and(self.occupancy_moving_objects, occupancy).any()

    def set_occupancy(self, pair, val, system_type = 'world', fixed_objects_map = False):
        # pair must be a (m,2) list or numpy array
        # system type must be either 'world' or 'grid'

        # TODO: why are there 2 map types? -> 2 classes with same interface ?!

        if system_type == 'world':
            pair = np.array(pair)
        elif system_type == 'grid':
            pair = np.array(pair,dtype=int)
        else:
            raise NameError('Invalid System type')

        val = np.array(val,dtype = bool)
        if not val.shape:
            val = val*np.ones((pair.shape[0],1))

        if system_type == 'world':
            #Need to search in world coordinates!
            #First check if input is valid
            pair = self.check_if_valid_world_coordinates(pair)
            
            if not pair.any():
                return None

            #Now if pair is valid start searching the indexes of the corrisponding x and y values
            idx = self.convert_world_to_grid(pair)
        else:
            if not self.check_if_valid_grid_index(pair):
                raise ValueError('Invalid grid indices with the current map!')
            idx = pair

        if fixed_objects_map:        
            for i in range(idx.shape[0]):
                self.occupancy_static_objects[idx[i,0],idx[i,1]] = val[i]

            self.obstacles_coordinates = self.get_obstacles_coordinates()
            self.occupancy_static_objects[:,0] = True
            self.occupancy_static_objects[:,-1] = True
            self.occupancy_static_objects[0,:] = True
            self.occupancy_static_objects[-1,:] = True
            self.occupancy_overall = self.occupancy_static_objects
        else:
            for i in range(idx.shape[0]):
                self.occupancy_moving_objects[idx[i,0],idx[i,1]] = val[i]
            self.update_overall_occupancy()

    def get_obstacles_coordinates(self):
        return np.concatenate([self.x[self.occupancy_static_objects][:, np.newaxis],
            self.y[self.occupancy_static_objects][:, np.newaxis]], axis=1)

    def update_overall_occupancy(self):
        self.occupancy_overall = np.logical_or(self.occupancy_moving_objects, self.occupancy_static_objects)

    def raycast(self, start_pt, end_pt):
        ''' Here it will be developed the easiest 
        implementation for a raycasting algorithm'''
        #Start point must belong to the map!
        idx_start = self.convert_world_to_grid(start_pt)
        #End point can also not belong to the map. We access to the last available
        idx_end = self.convert_world_to_grid_no_error(end_pt)

        d_row = abs(idx_end[0,1] - idx_start[0,1])
        d_col = abs(idx_end[0,0] - idx_start[0,0])
        
        if d_row == 0 and d_col == 0:
            #Start and end points are coincident
            return idx_start
        elif d_row == 0 and d_col != 0:
            #Handle division by zero
            col_idx = linspace(idx_start[0,0],idx_end[0,0],d_col + 1)
            tmp = np.ones((col_idx.shape[0],2), dtype = int)
            tmp[:,1] = idx_end[0,1]*tmp[:,1]
            tmp[:,0] = col_idx

            return tmp

        else:
            m = (idx_end[0,0] - idx_start[0,0])/(idx_end[0,1] - idx_start[0,1])

            #Get indexes of intercepting ray
            if abs(m) <= 1:
                x = linspace(idx_start[0, 1], idx_end[0, 1], d_row + 1).astype(int) #columns index
                y = np.floor(m * (x - idx_start[0, 1]) + idx_start[0, 0]).astype(int)
                y[y > self.occupancy_moving_objects.shape[0] - 1] = self.occupancy_moving_objects.shape[0] - 1
            elif abs(m) > 1:
                y = linspace(idx_start[0,0], idx_end[0, 0], d_col + 1).astype(int) #rows index
                x = np.floor((y - idx_start[0, 0]) / m + idx_start[0, 1]).astype(int)
                x[x > self.occupancy_moving_objects.shape[1] - 1] = self.occupancy_moving_objects.shape[1] - 1

            indexes = np.zeros((x.shape[0], 2), dtype=int)
            indexes[:, 0] = y
            indexes[:, 1] = x
            return indexes

    def does_ray_collide(self,ray_indexes):
        ''' This method checks if a given input ray
        intercept an obstacle present in the map'''

        #Input ray must be the output of the previous function!
        #so a (m,2) numpy array of ints

        intersections = self.occupancy_overall[ray_indexes[:, 0], ray_indexes[:, 1]]
        if intersections.any():
            idx = np.where(intersections)[0][0]
            return  True, ray_indexes[idx,:], [self.x[0, ray_indexes[idx, 1]], self.y[ray_indexes[idx, 0], 0]]
        return (False, None, None)

    def inflate(self, radius, fixed_objects_map = False):
        ''' Grow in size the obstacles'''
        #create a copy of the occupancy matrix
        int_radius_step = round(self.map_resolution*radius)
        if fixed_objects_map:  # problem here!!!!
            eval_points = np.asarray(np.where(self.occupancy_static_objects)).T
            self.occupancy_static_objects = fill_surrounding(self.occupancy_static_objects,int_radius_step,eval_points)
            self.update_overall_occupancy()
        else:
            eval_points = np.asarray(np.where(self.occupancy_moving_objects)).T
            self.occupancy_moving_objects = fill_surrounding(self.occupancy_moving_objects,int_radius_step,eval_points, add_noise = self.add_noise)
            self.update_overall_occupancy()

    def check_if_valid_world_coordinates(self,pair, margin = 0):
        if isinstance(pair,list):
            pair = np.array(pair)

        if len(pair.shape)<2:
            pair = pair[np.newaxis,:]

        offset = margin*np.array([self.map_length, self.map_height])
        valid_pairs = np.bitwise_and( (pair >= (self.min_val + offset)).all(axis = 1) , (pair <= (self.max_val - offset)).all(axis = 1) )
        if valid_pairs.all():
            return pair
        elif valid_pairs.any():
            return pair[valid_pairs,:]
        else:
            return np.array(False)

    def check_if_valid_grid_index(self,pair):
        for i in range(pair.shape[0]):
            if pair[i,0] < 0 or pair[i,0] > self.grid_size['y'] or pair[i,1] < 0 or pair[i,1] > self.grid_size['x'] or not \
                issubclass(pair.dtype.type,np.integer):
                return False
        return True

    def convert_world_to_grid(self,pair):
        pair = self.check_if_valid_world_coordinates(pair)
        return self.convert_world_to_grid_no_error(pair)

    def convert_grid_to_world(self,pair):
        if not self.check_if_valid_grid_index(pair):
            raise ValueError('Invalid grid indices with the current map!')
        val = np.zeros(pair.shape)
        for i in range(pair.shape[0]):
            val[i,0] = self.x[0,pair[i,1]]
            val[i,1] = self.y[pair[i,0],0]
        return val

    def convert_world_to_grid_no_error(self,pair):
        return np.concatenate((np.abs(self.y[:,0][:,np.newaxis] - pair[:,1].T).argmin(axis = 0)[:,np.newaxis], \
                               np.abs(self.x[0,:][:,np.newaxis] - pair[:,0].T).argmin(axis = 0)[:,np.newaxis]), axis = 1)

    #update map from pedestrians simulation environment
    def update_from_peds_sim(self, map_margin = 1.5, fixed_objects_map = False):
        # shift peds sim map to top-left quadrant if necessary
        if self.x[0,0] == self.cell_size['x']/2:
            maps_alignment_offset = map_margin*self.peds_sim_env.box_size
        else:
            maps_alignment_offset = 0

        if fixed_objects_map:
            # assign fixed obstacles to map
            tmp = [item for item in self.peds_sim_env.env.obstacles if item.size != 0]
            obs_coordinates = np.concatenate(tmp)
            new_obs_coordinates = obs_coordinates + np.ones((obs_coordinates.shape[0], 2)) * maps_alignment_offset
            # reset occupancy map
            self.occupancy_static_objects = np.zeros(self.occupancy_static_objects.shape,dtype = bool)
            self.set_occupancy(new_obs_coordinates, np.ones((obs_coordinates.shape[0],1),dtype = bool), fixed_objects_map = fixed_objects_map)   
            zeros_mat = np.zeros(self.occupancy_static_objects.shape, dtype = bool)
            zeros_mat[::2, ::2] = True
            self.occupancy_static_objects_raw = np.logical_and(self.occupancy_static_objects.copy(), zeros_mat)
            self.inflate(.3, fixed_objects_map=fixed_objects_map)
        else:
            # get peds states
            peds_pos = self.peds_sim_env.current_positions
            # add offset to pedestrians maps (it can also have negative coordinates values)
            n_pedestrians = peds_pos.shape[0]
            peds_new_coordinates = peds_pos + np.ones((n_pedestrians,2))*maps_alignment_offset
            # reset occupancy map
            self.occupancy_moving_objects = np.zeros(self.occupancy_moving_objects.shape,dtype = bool)
            # assign pedestrians positions to robotMap
            self.set_occupancy(peds_new_coordinates, np.ones((n_pedestrians,1)))
            self.occupancy_moving_objects_raw =self.occupancy_moving_objects.copy()
            self.inflate(.4) # inflate pedestrians only

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
