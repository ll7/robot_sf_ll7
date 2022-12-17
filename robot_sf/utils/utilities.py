from typing import Tuple

import numpy as np
from pysocialforce.utils import stateutils


def norm_angles(angles: np.ndarray) -> np.ndarray:
    """Normalize angles between [-pi, pi)"""
    return (angles + np.pi) % (2 * np.pi) -np.pi


# check vectorization of these functions
def line_segment(p_0, p_1):
    seg_a = p_0[:, 1] - p_1[:, 1]
    seg_b = p_1[:, 0] - p_0[:, 0]
    seg_c = p_0[:, 0] * p_1[:, 1] - p_1[:, 0] * p_0[:, 1]
    return np.array([seg_a, seg_b, -seg_c]).T


# check vectorization of these functions
def lines_intersection(l_1, l_2, p0_l1, p1_l1, p0_l2, p1_l2):
    # used for 3 times in ExtdSimulator.change_direction()
    x_min_l1 = np.tile(np.minimum(p0_l1[:, 0],p1_l1[:, 0])[:, np.newaxis], (1, l_2.shape[0]))
    x_max_l1 = np.tile(np.maximum(p0_l1[:, 0],p1_l1[:, 0])[:, np.newaxis], (1, l_2.shape[0]))
    y_min_l1 = np.tile(np.minimum(p0_l1[:, 1],p1_l1[:, 1])[:, np.newaxis], (1, l_2.shape[0]))
    y_max_l1 = np.tile(np.maximum(p0_l1[:, 1],p1_l1[:, 1])[:, np.newaxis], (1, l_2.shape[0]))
    x_min_l2 = np.tile(np.minimum(p0_l2[:, 0],p1_l2[:, 0])[np.newaxis, :], (l_1.shape[0], 1))
    x_max_l2 = np.tile(np.maximum(p0_l2[:, 0],p1_l2[:, 0])[np.newaxis, :], (l_1.shape[0], 1))
    y_min_l2 = np.tile(np.minimum(p0_l2[:, 1],p1_l2[:, 1])[np.newaxis, :], (l_1.shape[0], 1))
    y_max_l2 = np.tile(np.maximum(p0_l2[:, 1],p1_l2[:, 1])[np.newaxis, :], (l_1.shape[0], 1))

    d   = l_1[:, 0][:, np.newaxis] * l_2[:, 1][np.newaxis, :] \
            - l_1[:, 1][:, np.newaxis] * l_2[:, 0][np.newaxis, :]
    d_x = l_1[:, 2][:, np.newaxis] * l_2[:, 1][np.newaxis, :] \
            - l_1[:, 1][:, np.newaxis] * l_2[:, 2][np.newaxis, :]
    d_y = l_1[:, 0][:, np.newaxis] * l_2[:, 2][np.newaxis, :] \
            - l_1[:, 2][:, np.newaxis] * l_2[:, 0][np.newaxis, :]

    # TODO: why ignore?! rather fix the data instead!!!
    with np.errstate(divide='ignore', invalid='ignore'):
        pos_x = d_x / d
        pos_y = d_y / d
        nan_mask = np.logical_or.reduce((
            (pos_x < x_min_l1), (pos_x > x_max_l1), (pos_y < y_min_l1), (pos_y > y_max_l1),
            (pos_x < x_min_l2), (pos_x > x_max_l2), (pos_y < y_min_l2), (pos_y > y_max_l2)))

    # TODO: why NaN??? this causes numeric errors
    pos_x[nan_mask] = np.NAN
    pos_y[nan_mask] = np.NAN

    return pos_x, pos_y


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


def change_direction(p_0, p_1, current_positions, destinations,
                     view_distance, angles, direction, desired_directions) -> Tuple[np.ndarray, np.ndarray]:

    # this handles stuck pedestrians whose force is close to 0
    # TODO: check if this acutally does something (shouldn't be required to run the simulation)

    #1. find pedestrians who are headed towards an obstacle (within horizon defined by view_distance)
    l_directions = line_segment(current_positions, destinations)
    l_obstacles  = line_segment(p_0, p_1)

    intersect = lines_intersection(l_directions, l_obstacles, current_positions, destinations, p_0, p_1)
    intersections_coordinates = np.stack((intersect[0], intersect[1]))

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
        intersect = lines_intersection(
            l_eval_dirs, l_obstacles, collision_states.reshape(1, 2), wide_scope__, p_0, p_1)
        intersections_coordinates = np.stack((intersect[0], intersect[1]))

        ### calculate distances from intersections
        dist_0 = intersections_coordinates[0] - collision_states.reshape(2)[0]
        dist_1 = intersections_coordinates[1] - collision_states.reshape(2)[1]
        my_dist = np.sqrt((np.stack((dist_0, dist_1))**2).sum(axis=0))
        my_dist = np.minimum(view_distance, my_dist)

        ### choose angles without intersections
        idxs_nan = np.where(np.isnan(my_dist).all(axis=1))[0]  #generated as tuple

        if idxs_nan.shape[0] > 0:
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
            intersect = lines_intersection(l_eval_dirs, l_obstacles, collision_states[i][np.newaxis,:], wide_scope__, p_0, p_1)
            intersections_coordinates = np.stack((intersect[0], intersect[1]))
            dist_0 = intersections_coordinates[0] - collision_states[i][0]
            dist_1 = intersections_coordinates[1] - collision_states[i][1]
            my_dist = np.sqrt((np.stack((dist_0, dist_1))**2).sum(axis=0))
            my_dist = np.minimum(view_distance, my_dist)

            idxs_nan = np.where(np.isnan(my_dist).all(axis = 1))[0]  #generated as tuple
            if idxs_nan.shape[0] > 0:
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
    # spawn pedestrians at the edges
    if coordinate_a == 0:
        return np.concatenate((-distance * np.ones([dim, 1]), coordinate_b[:, np.newaxis]) , axis=1)
    elif coordinate_a == 1:
        return np.concatenate((coordinate_b[:, np.newaxis], -distance * np.ones([dim, 1])), axis=1)
    elif coordinate_a == 2:
        return np.concatenate((distance * np.ones([dim, 1]), coordinate_b[:, np.newaxis]), axis=1)
    elif coordinate_a == 3:
        return np.concatenate((coordinate_b[:, np.newaxis], distance * np.ones([dim, 1])), axis=1)
    raise ValueError(f'unknown coordinate {coordinate_a}, needs to be within [0, 3]')


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
    for i, sub_list in enumerate(list_of_lists):
        for j, item in enumerate(sub_list):
            if item > num:
                sub_list[j] = item-1
        list_of_lists[i] = sub_list
    return list_of_lists
