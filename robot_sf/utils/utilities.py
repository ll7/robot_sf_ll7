# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 11:23:20 2020

@author: Enrico Regolin
"""

import numpy as np
from pysocialforce.utils import stateutils


def norm_angles(angles: np.ndarray) -> np.ndarray:
    """Normalize angles between [-pi, pi)"""
    return (angles + np.pi) % (2 * np.pi) -np.pi


# check vectorization of these functions
def line_segment(p0, p1):
    A = p0[:, 1] - p1[:, 1]
    B = p1[:, 0] - p0[:, 0]
    C = p0[:, 0] * p1[:, 1] - p1[:, 0] * p0[:, 1]
    return np.array([A, B, -C]).T


# check vectorization of these functions
def lines_intersection(l_1, l_2, p0_l1, p1_l1, p0_l2, p1_l2):
    x_min_l1 = np.tile(np.minimum(p0_l1[:, 0],p1_l1[:, 0])[:, np.newaxis], (1, l_2.shape[0]))
    x_max_l1 = np.tile(np.maximum(p0_l1[:, 0],p1_l1[:, 0])[:, np.newaxis], (1, l_2.shape[0]))
    y_min_l1 = np.tile(np.minimum(p0_l1[:, 1],p1_l1[:, 1])[:, np.newaxis], (1, l_2.shape[0]))
    y_max_l1 = np.tile(np.maximum(p0_l1[:, 1],p1_l1[:, 1])[:, np.newaxis], (1, l_2.shape[0]))
    x_min_l2 = np.tile(np.minimum(p0_l2[:, 0],p1_l2[:, 0])[np.newaxis, :], (l_1.shape[0], 1))
    x_max_l2 = np.tile(np.maximum(p0_l2[:, 0],p1_l2[:, 0])[np.newaxis, :], (l_1.shape[0], 1))
    y_min_l2 = np.tile(np.minimum(p0_l2[:, 1],p1_l2[:, 1])[np.newaxis, :], (l_1.shape[0], 1))
    y_max_l2 = np.tile(np.maximum(p0_l2[:, 1],p1_l2[:, 1])[np.newaxis, :], (l_1.shape[0], 1))

    d   = l_1[:, 0][:, np.newaxis] * l_2[:, 1][np.newaxis, :] - l_1[:, 1][:, np.newaxis] * l_2[:, 0][np.newaxis, :]
    d_x = l_1[:, 2][:, np.newaxis] * l_2[:, 1][np.newaxis, :] - l_1[:, 1][:, np.newaxis] * l_2[:, 2][np.newaxis, :]
    d_y = l_1[:, 0][:, np.newaxis] * l_2[:, 2][np.newaxis, :] - l_1[:, 2][:, np.newaxis] * l_2[:, 0][np.newaxis, :]

    # TODO: why ignore?! rather fix the data instead!!!
    with np.errstate(divide='ignore', invalid='ignore'):
        x = d_x / d
        y = d_y / d
        nan_mask = np.logical_or.reduce((
            (x < x_min_l1), (x > x_max_l1), (y < y_min_l1), (y > y_max_l1),
            (x < x_min_l2), (x > x_max_l2), (y < y_min_l2), (y > y_max_l2)))

    # TODO: why NaN??? this causes numeric errors
    x[nan_mask] = np.NAN
    y[nan_mask] = np.NAN

    return x, y


def rotate_segment(origin, point, angle):
    """
    Rotate a point counterclockwise by a given array of angles around a given origin.
    The angle should be given in radians.
    """
    o_x, o_y = origin
    p_x, p_y = point

    q_x = o_x + np.cos(angle) * (p_x - o_x) - np.sin(angle) * (p_y - o_y)
    q_y = o_y + np.sin(angle) * (p_x - o_x) + np.cos(angle) * (p_y - o_y)
    return q_x, q_y


def change_direction(p0, p1, current_positions, destinations, view_distance, angles, direction, desired_directions):
    #1. find pedestrians who are headed towards an obstacle (within horizon defined by view_distance)
    l_directions = line_segment(current_positions, destinations)
    l_obstacles  = line_segment(p0, p1)

    R = lines_intersection(l_directions, l_obstacles, current_positions, destinations, p0, p1)
    intersections_coordinates = np.stack((R[0], R[1]))

    distances = intersections_coordinates - (np.repeat(current_positions.T[:, :, None], intersections_coordinates.shape[2], axis=2))
    with np.errstate(invalid='ignore'):
        peds_collision_indices = (np.sqrt((distances**2).sum(axis = 0)) < view_distance).any(axis=1)

    #2. only for these, evaluate trajectories which deviate from original directions,
    #   by changing angles (within predefined angular and distance ranges)

    #3. select obstacle-free direction with least angular deviation,
    #   or (if not available) angle with most distant obstacle  

    collision_states: np.ndarray = current_positions[peds_collision_indices]
    close_targets: np.ndarray = collision_states + view_distance * desired_directions[peds_collision_indices]

    if collision_states.shape[0] == 1:
        # generate array of possible destinations, based on angles considered
        wide_scope = rotate_segment(collision_states.reshape(2), close_targets.reshape(2), angles)
        wide_scope__ = np.stack((wide_scope[0],wide_scope[1]), axis=1)  # vectorize it

        # get all intersections of new array with existing objects
        l_eval_dirs = line_segment(collision_states.reshape(1, 2), wide_scope__)
        R = lines_intersection(l_eval_dirs, l_obstacles, collision_states.reshape(1, 2), wide_scope__, p0, p1)
        intersections_coordinates = np.stack((R[0], R[1]))

        ### calculate distances from intersections
        dist_0 = intersections_coordinates[0] - collision_states.reshape(2)[0]
        dist_1 = intersections_coordinates[1] - collision_states.reshape(2)[1]
        my_dist = np.sqrt((np.stack((dist_0, dist_1))**2).sum(axis=0))
        my_dist = np.minimum(view_distance, my_dist)

        ### choose angles without intersections
        idxs_nan = np.where(np.isnan(my_dist).all(axis=1))[0]  #generated as tuple
        idx_used =  np.argmin(np.absolute(idxs_nan - len(angles) / 2))
        ped_idx = np.where(peds_collision_indices)[0] # indices of directions to be changed

        #generate new destinations
        new_goal = rotate_segment(collision_states.reshape(2), destinations[ped_idx][0, :], angles[idx_used])
        new_direction = ([new_goal[0], new_goal[1]] - collision_states.reshape(2)) / np.linalg.norm(new_goal - collision_states)
        direction[ped_idx] = new_direction

        ### same as above, iteratively
    elif collision_states.shape[0] > 1:
        for i in range(collision_states.shape[0]): 
            wide_scope = rotate_segment(collision_states[i].reshape(2), close_targets[i].reshape(2), angles)
            wide_scope__ = np.stack((wide_scope[0],wide_scope[1]),axis = 1)

            l_eval_dirs = line_segment(collision_states[i][np.newaxis, :], wide_scope__)
            R = lines_intersection(l_eval_dirs, l_obstacles, collision_states[i][np.newaxis,:], wide_scope__, p0, p1)
            intersections_coordinates = np.stack((R[0], R[1]))
            dist_0 = intersections_coordinates[0] - collision_states[i][0]
            dist_1 = intersections_coordinates[1] - collision_states[i][1]
            my_dist = np.sqrt((np.stack((dist_0, dist_1))**2).sum(axis=0))
            my_dist = np.minimum(view_distance, my_dist)

            idxs_nan = np.where(np.isnan(my_dist).all(axis = 1))[0]  #generated as tuple
            # TODO: figure out why argmin occasionally receives empty sequences as argument
            idx_used =  np.argmin(np.absolute(idxs_nan - len(angles) / 2))
            ped_idx = np.where(peds_collision_indices)[0][i]

            new_goal = rotate_segment(collision_states[i].reshape(2), destinations[ped_idx], angles[idx_used])
            new_direction = (new_goal - collision_states[i]) / np.linalg.norm(new_goal - collision_states[i])
            direction[ped_idx] = new_direction

    return direction, peds_collision_indices


def fill_state(coordinate_a, coordinate_b, origin,box_size):
    distance = box_size * 1.1 if origin else box_size * 1.6

    if isinstance(coordinate_b, np.ndarray):
        return build_coordinates_array(coordinate_a, coordinate_b, len(coordinate_b), distance)
    else:
        return build_coordinates_scalar(coordinate_a, coordinate_b, distance)


def build_coordinates_array(coordinate_a, coordinate_b, dim, distance):
    if coordinate_a == 0:
        return np.concatenate((-distance * np.ones([dim, 1]), coordinate_b[:, np.newaxis]) , axis=1)
    elif coordinate_a == 1:
        return np.concatenate((coordinate_b[:, np.newaxis], -distance * np.ones([dim, 1])), axis=1)
    elif coordinate_a == 2:
        return np.concatenate((distance * np.ones([dim, 1]), coordinate_b[:, np.newaxis]), axis=1)
    elif coordinate_a == 3:
        return np.concatenate((coordinate_b[:, np.newaxis], distance * np.ones([dim, 1])), axis=1)


def build_coordinates_scalar(coordinate_a, coordinate_b, distance):
    if coordinate_a == 0:
        return np.array([-distance, coordinate_b])
    elif coordinate_a == 1:
        return np.array([coordinate_b, -distance])
    elif coordinate_a == 2:
        return np.array([distance, coordinate_b])
    elif coordinate_a == 3:
        return np.array([coordinate_b, distance])


# function used to correctly update groups indices after states are removed
def fun_reduce_index(list_of_lists, num):
    for idx_out, a_list in enumerate(list_of_lists):
        for idx, item in enumerate(a_list):
            if item > num:
                a_list[idx] = item-1
        list_of_lists[idx_out] = a_list        
    return list_of_lists


def add_new_group(box_size, max_grp_size, n_pedestrians_actual, group_width_max,
                  group_width_min, tau, average_speed, speed_variance_red):
    # generate states of new group
    square_dim = box_size + 1

    #Initialize new random new group size
    new_grp_size = np.random.randint(2, max_grp_size)
    group_origin_a = np.random.randint(0, 4) #0:left, 1:bottom, 2:right, 3:top
    group_width = (group_width_max-group_width_min)*np.random.random_sample()+ group_width_min

    #Generate pedestrians group position
    group_origin_b = np.random.randint(-square_dim, square_dim) \
        + group_width * np.random.random_sample(new_grp_size) - group_width / 2

    #Choose random destination and delete the origin
    group_destination_a = np.random.choice(np.delete(np.array([0, 1, 2, 3]), group_origin_a))
    group_destination_b = np.random.randint(-square_dim, square_dim) * np.ones((new_grp_size,))                                        

    origin_states      = fill_state(group_origin_a, group_origin_b, True, box_size)
    destination_states = fill_state(group_destination_a, group_destination_b, False, box_size)
 
    new_group_states = np.concatenate((
            origin_states,
            np.zeros(origin_states.shape),
            destination_states,
            tau * np.ones((origin_states.shape[0], 1))
        ), axis=1)
    # define initial speeds of group
    new_group_directions = stateutils.desired_directions(new_group_states)[0]
    random_speeds = np.repeat((average_speed + np.random.randn(new_group_directions.shape[0]) / speed_variance_red)[np.newaxis, :], 2, axis=0).T
    new_group_states[:, 2:4] = np.multiply(new_group_directions, random_speeds)
    # new group indices
    new_group = n_pedestrians_actual+np.arange(new_grp_size)
    return new_group_states, new_group


def add_new_individuals(box_size: int, max_single_peds: int, tau: float,
        average_speed: float = 0.5, speed_variance_red: float = 10):

    square_dim = box_size + 1
    new_pedestrians = np.random.randint(1, max_single_peds)
    new_pedestrians_states = np.zeros((new_pedestrians, 7))

    for i in range(new_pedestrians):
        # randomly generate origin and destination of new pedestrian
        origin_a = np.random.randint(0, 4) #0:left, 1:bottom, 2:right, 3:top
        origin_b = 2 * square_dim * np.random.random_sample() - square_dim
        destination_a = np.random.choice(np.delete(np.array([0, 1, 2, 3]), origin_a))
        destination_b = 2 * square_dim*np.random.random_sample() - square_dim
        # fill i-th row of the list of new pedestrian states
        new_pedestrians_states[i,:2]=fill_state(origin_a, origin_b, True,box_size)
        new_pedestrians_states[i,4:6]=fill_state(destination_a, destination_b, False,box_size)
        new_pedestrians_states[i,-1] = tau
        # add new pedestrian state to list of other pedestrians

    #speeds update
    new_peds_directions = stateutils.desired_directions(new_pedestrians_states)[0]
    #randomize initial speeds
    random_speeds = np.repeat((average_speed + np.random.randn(
        new_peds_directions.shape[0]) / speed_variance_red)[np.newaxis, :], 2, axis=0).T
    new_pedestrians_states[:, 2:4] = np.multiply(new_peds_directions, random_speeds)

    return new_pedestrians_states
