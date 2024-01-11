"""Calculate forces for individuals and groups"""
import re
from math import atan2, exp
from typing import Tuple, List, Protocol, Callable

import numpy as np
from numba import njit

import logging
logging.getLogger('numba').setLevel(logging.WARNING)

from pysocialforce.scene import Line2D, Point2D, PedState
from pysocialforce import logger
from pysocialforce.config import \
    DesiredForceConfig, SocialForceConfig, ObstacleForceConfig, \
    GroupCoherenceForceConfig, GroupGazeForceConfig, GroupReplusiveForceConfig


Force = Callable[[], np.ndarray]


class SimEntitiesProvider(Protocol):
    def get_obstacles(self) -> List[np.ndarray]:
        raise NotImplementedError()

    def get_raw_obstacles(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def peds(self) -> PedState:
        raise NotImplementedError()


class DebuggableForce:
    def __init__(self, force: Force):
        self.force = force

    def __call__(self, debug: bool=False):
        force = self.force()
        if debug:
            force_type = self.camel_to_snake(type(self).__name__)
            logger.debug(f"{force_type}:\n {repr(force)}")
        return force

    @staticmethod
    def camel_to_snake(camel_case_string: str) -> str:
        """Convert CamelCase to snake_case"""
        return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_case_string).lower()


class DesiredForce:
    """Calculates the force between this agent and the next assigned waypoint.
    If the waypoint has been reached, the next waypoint in the list will be
    selected.
    :return: the calculated force
    """

    def __init__(self, config: DesiredForceConfig, peds: PedState):
        self.config = config
        self.peds = peds

    def __call__(self):
        relexation_time: float = self.config.relaxation_time
        goal_threshold = self.config.goal_threshold
        pos = self.peds.pos()
        vel = self.peds.vel()
        goal = self.peds.goal()
        direction, dist = normalize(goal - pos)
        force = np.zeros((self.peds.size(), 2))
        force[dist > goal_threshold] = (
            direction * self.peds.max_speeds.reshape((-1, 1)) - vel.reshape((-1, 2))
        )[dist > goal_threshold, :]
        force[dist <= goal_threshold] = -1.0 * vel[dist <= goal_threshold]
        force /= relexation_time
        return force * self.config.factor


class SocialForce:
    """Calculates the social force between this agent and all the other agents
    belonging to the same scene.
    It iterates over all agents inside the scene, has therefore the complexity
    O(N^2). A better
    agent storing structure in Tscene would fix this. But for small (less than
    10000 agents) scenarios, this is just
    fine.
    :return:  nx2 ndarray the calculated force
    """

    def __init__(self, config: SocialForceConfig, peds: PedState):
        self.config = config
        self.peds = peds

    def __call__(self):
        ped_positions = self.peds.pos()
        ped_velocities = self.peds.vel()
        forces = social_force(
            ped_positions, ped_velocities, self.config.activation_threshold,
            self.config.n, self.config.n_prime, self.config.lambda_importance, self.config.gamma)
        return forces * self.config.factor


@njit(fastmath=True)
def social_force(
        ped_positions: np.ndarray, ped_velocities: np.ndarray, activation_threshold: float,
        n: int, n_prime: int, lambda_importance: float, gamma: float) -> np.ndarray:

    num_peds = ped_positions.shape[0]
    activation_threshold_sq = activation_threshold**2
    forces = np.zeros((num_peds, 2))

    for ped_i in range(num_peds):
        all_pos_diffs = ped_positions[ped_i] - ped_positions
        pos_dists_sq = np.sum(all_pos_diffs**2, axis=1)
        ped_mask = pos_dists_sq <= activation_threshold_sq
        ped_mask[ped_i] = False
        other_ped_ids = np.where(ped_mask)[0]

        pos_diffs = all_pos_diffs[other_ped_ids]
        vel_diffs = ped_velocities[other_ped_ids] - ped_velocities[ped_i]
        force_x, force_y = social_force_single_ped(
            pos_diffs, vel_diffs, n, n_prime, lambda_importance, gamma)
        forces[ped_i, 0] = force_x
        forces[ped_i, 1] = force_y

    return forces


@njit(fastmath=True)
def social_force_single_ped(
        pos_diffs: np.ndarray, vel_diffs: np.ndarray,
        n: int, n_prime: int, lambda_importance: float, gamma: float) -> Point2D:
    force_sum_x, force_sum_y = 0.0, 0.0
    for i in range(pos_diffs.shape[0]):
        force_x, force_y = social_force_ped_ped(
            pos_diffs[i], vel_diffs[i], n, n_prime, lambda_importance, gamma)
        force_sum_x += force_x
        force_sum_y += force_y
    return force_sum_x, force_sum_y


@njit(fastmath=True)
def social_force_ped_ped(
        pos_diff: Point2D, vel_diff: Point2D, n: int, n_prime: int,
        lambda_importance: float, gamma: float) -> Point2D:

    pos_diff_x, pos_diff_y = pos_diff
    vel_diff_x, vel_diff_y = vel_diff
    (diff_dir_x, diff_dir_y), diff_length = norm_vec((pos_diff_x, pos_diff_y))
    interaction_vec_x = lambda_importance * vel_diff_x + diff_dir_x
    interaction_vec_y = lambda_importance * vel_diff_y + diff_dir_y
    interaction_dir, interaction_length = norm_vec((interaction_vec_x, interaction_vec_y))
    interaction_dir_x, interaction_dir_y = interaction_dir

    theta = atan2(interaction_dir[1], interaction_dir[0]) - atan2(diff_dir_y, diff_dir_x)
    theta_sign = 1 if theta >= 0 else -1
    B = gamma * interaction_length + 1e-8

    force_velocity_amount = exp(-1.0 * diff_length / B - (n_prime * B * theta)**2)
    force_angle_amount = -theta_sign * exp(-1.0 * diff_length / B - (n * B * theta)**2)
    force_velocity_x = interaction_dir_x * force_velocity_amount
    force_velocity_y = interaction_dir_y * force_velocity_amount
    force_angle_x = -interaction_dir_y * force_angle_amount
    force_angle_y = interaction_dir_x * force_angle_amount
    return force_velocity_x + force_angle_x, force_velocity_y + force_angle_y


@njit(fastmath=True)
def norm_vec(vec: Point2D) -> Tuple[Point2D, float]:
    if vec[0] == 0 and vec[1] == 0:
        return vec, 0
    vec_len = (vec[0]**2 + vec[1]**2)**0.5
    return (vec[0] / vec_len, vec[1] / vec_len), vec_len


class ObstacleForce:
    """Calculates the force between this agent and the nearest obstacle in this
    scene.
    :return:  the calculated force
    """

    def __init__(self, config: ObstacleForceConfig, sim: SimEntitiesProvider):
        self.config = config
        self.get_obstacles = sim.get_raw_obstacles
        self.get_peds = sim.peds.pos
        self.get_agent_radius = lambda: sim.peds.agent_radius

    def __call__(self) -> np.ndarray:
        """Computes the obstacle forces per pedestrian,
        output shape (num_peds, 2), forces in x/y direction"""

        ped_positions = self.get_peds()
        forces = np.zeros((ped_positions.shape[0], 2))
        obstacles = self.get_obstacles()
        if len(obstacles) == 0:
            return forces

        sigma = self.config.sigma
        threshold = self.config.threshold

        threshold = threshold + self.get_agent_radius() * sigma
        all_obstacle_forces(forces, ped_positions, obstacles, threshold)
        return forces * self.config.factor


@njit(fastmath=True)
def all_obstacle_forces(out_forces: np.ndarray, ped_positions: np.ndarray,
                        obstacles: np.ndarray, ped_radius: float):
    obstacle_segments = obstacles[:, :4]
    ortho_vecs = obstacles[:, 4:]
    num_peds = ped_positions.shape[0]
    num_obstacles = obstacles.shape[0]
    for i in range(num_peds):
        ped_pos = ped_positions[i]
        for j in range(num_obstacles):
            force_x, force_y = obstacle_force(
                obstacle_segments[j], ortho_vecs[j], ped_pos, ped_radius)
            out_forces[i, 0] += force_x
            out_forces[i, 1] += force_y


@njit(fastmath=True)
def obstacle_force(obstacle: Line2D, ortho_vec: Point2D,
                   ped_pos: Point2D, ped_radius: float) -> Tuple[float, float]:
    """The obstacle force between a line segment (= obstacle) and
    a point (= pedestrian's position) is computed as follows:
    1) compute the distance between the line segment and the point
    2) compute the repulsive force, i.e. the partial derivative by x/y of the point
    regarding the virtual potential field denoted as 1 / (2 * dist(line_seg, point)^2)
    3) return the force as separate x/y components

    There are 3 cases to be considered for computing the distance:
    1) obstacle is just a point instead of a line segment
    2) orthogonal projection hits within the obstacle's line segment
    3) orthogonal projection doesn't hit within the obstacle's line segment"""

    coll_dist = 1e-5
    x1, y1, x2, y2 = obstacle
    (x3, y3), (x4, y4) = ped_pos, (ped_pos[0] + ortho_vec[0], ped_pos[1] + ortho_vec[1])

    # handle edge case where the obstacle is just a point
    if (x1, y1) == (x2, y2):
        obst_dist = max(euclid_dist(ped_pos[0], ped_pos[1], x1, y1) - ped_radius, coll_dist)
        dx_obst_dist, dy_obst_dist = der_euclid_dist(ped_pos, (x1, y1), obst_dist)
        return potential_field_force(obst_dist, dx_obst_dist, dy_obst_dist)

    # info: there's always an intersection with the orthogonal vector
    num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    t = num / den
    ortho_hit = 0 <= t <= 1

    # orthogonal vector doesn't hit within segment bounds
    if not ortho_hit:
        d1 = euclid_dist(ped_pos[0], ped_pos[1], x1, y1)
        d2 = euclid_dist(ped_pos[0], ped_pos[1], x2, y2)
        obst_dist = max(min(d1, d2) - ped_radius, coll_dist)
        closer_obst_bound = (x1, y1) if d1 < d2 else (x2, y2)
        dx_obst_dist, dy_obst_dist = der_euclid_dist(ped_pos, closer_obst_bound, obst_dist)
        return potential_field_force(obst_dist, dx_obst_dist, dy_obst_dist)

    # orthogonal vector hits within segment bounds
    cross_x, cross_y = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
    obst_dist = max(euclid_dist(ped_pos[0], ped_pos[1], cross_x, cross_y) - ped_radius, coll_dist)
    dx3_cross_x = (y4 - y3) / den * (x2 - x1)
    dx3_cross_y = (y4 - y3) / den * (y2 - y1)
    dy3_cross_x = (x3 - x4) / den * (x2 - x1)
    dy3_cross_y = (x3 - x4) / den * (y2 - y1)
    dx_obst_dist = ((cross_x - ped_pos[0]) * (dx3_cross_x - 1) \
        + (cross_y - ped_pos[1]) * dx3_cross_y) / obst_dist
    dy_obst_dist = ((cross_x - ped_pos[0]) * dy3_cross_x \
        + (cross_y - ped_pos[1]) * (dy3_cross_y - 1)) / obst_dist
    return potential_field_force(obst_dist, dx_obst_dist, dy_obst_dist)


@njit(fastmath=True)
def potential_field_force(obst_dist: float, dx_obst_dist: float,
                          dy_obst_dist: float) -> Tuple[float, float]:
    der_potential = 1 / pow(obst_dist, 3)
    return der_potential * dx_obst_dist, der_potential * dy_obst_dist


@njit(fastmath=True)
def euclid_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return pow(pow(x2 - x1, 2) + pow(y2 - y1, 2), 0.5)


@njit(fastmath=True)
def euclid_dist_sq(x1: float, y1: float, x2: float, y2: float) -> float:
    return pow(x2 - x1, 2) + pow(y2 - y1, 2)


@njit(fastmath=True)
def der_euclid_dist(p1: Point2D, p2: Point2D, distance: float) -> Tuple[float, float]:
    # info: distance is an expensive operation and therefore pre-computed
    dx1_dist = (p1[0] - p2[0]) / distance
    dy1_dist = (p1[1] - p2[1]) / distance
    return dx1_dist, dy1_dist


class GroupCoherenceForceAlt:
    """ Alternative group coherence force as specified in pedsim_ros"""

    def __init__(self, config: GroupCoherenceForceConfig, peds: PedState):
        self.peds = peds
        self.config = config

    def __call__(self):
        forces = np.zeros((self.peds.size(), 2))
        if not self.peds.has_group():
            return forces

        for group in self.peds.groups:
            threshold = (len(group) - 1) / 2
            member_pos = self.peds.pos()[group, :]
            if len(member_pos) == 0:
                continue

            com = centroid(member_pos)
            force_vec = com - member_pos
            norms = np.linalg.norm(force_vec, axis=1)
            softened_factor = (np.tanh(norms - threshold) + 1) / 2
            forces[group, :] += (force_vec.T * softened_factor).T
        return forces * self.config.factor


class GroupRepulsiveForce:
    """Group repulsive force"""

    def __init__(self, config: GroupReplusiveForceConfig, peds: PedState):
        self.config = config
        self.peds = peds

    def __call__(self):
        threshold = self.config.threshold
        forces = np.zeros((self.peds.size(), 2))
        if not self.peds.has_group():
            return forces

        for group in self.peds.groups:
            if not group:
                continue

            size = len(group)
            member_pos = self.peds.pos()[group, :]
            diff = each_diff(member_pos)  # others - self
            _, norms = normalize(diff)
            diff[norms > threshold, :] = 0
            forces[group, :] += np.sum(diff.reshape((size, -1, 2)), axis=1)

        return forces * self.config.factor


class GroupGazeForceAlt:
    """Group gaze force"""

    def __init__(self, config: GroupGazeForceConfig, peds: PedState):
        self.config = config
        self.peds = peds

    def __call__(self):
        forces = np.zeros((self.peds.size(), 2))

        if not self.peds.has_group():
            return forces

        ped_positions = self.peds.pos()
        directions, dist = desired_directions(self.peds.state)

        for group in self.peds.groups:
            group_size = len(group)
            if group_size <= 1:
                continue
            forces[group, :] = group_gaze_force(
                ped_positions[group, :], directions[group, :], dist[group])

        return forces * self.config.factor


@njit(fastmath=True)
def group_gaze_force(
        member_pos: np.ndarray, member_directions: np.ndarray,
        member_dist: np.ndarray) -> np.ndarray:
    group_size = member_pos.shape[0]
    out_forces = np.zeros((group_size, 2))
    for i in range(group_size):
        # use center of mass without the current agent
        other_member_pos = member_pos[np.arange(group_size) != i, :2]
        mass_center_without_ped = centroid(other_member_pos)
        relative_com_x = mass_center_without_ped[0] - member_pos[i, 0]
        relative_com_y = mass_center_without_ped[1] - member_pos[i, 1]
        com_dir, com_dist = norm_vec((relative_com_x, relative_com_y))
        # angle between walking direction and center of mass
        ped_dir_x, ped_dir_y = member_directions[i]
        element_prod = ped_dir_x * com_dir[0] + ped_dir_y * com_dir[1]
        factor = com_dist * element_prod / member_dist[i]
        force_x, force_y = ped_dir_x * factor, ped_dir_y * factor
        out_forces[i, 0] = force_x
        out_forces[i, 1] = force_y
    return out_forces


@njit
def vec_len_2d(vec_x: float, vec_y: float) -> float:
    return (vec_x**2 + vec_y**2)**0.5


@njit
def normalize(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize nx2 array along the second axis
    input: [n,2] ndarray
    output: (normalized vectors, norm factors)
    """
    num_vecs = vecs.shape[0]
    vec_lengths = np.zeros((num_vecs))
    unit_vecs = np.zeros((num_vecs, 2))

    for i, (vec_x, vec_y) in enumerate(vecs):
        vec_len = vec_len_2d(vec_x, vec_y)
        vec_lengths[i] = vec_len
        if vec_len > 0:
            unit_vecs[i] = vecs[i] / vec_len

    return unit_vecs, vec_lengths


@njit
def desired_directions(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given the current state and destination, compute desired direction."""
    destination_vectors = state[:, 4:6] - state[:, 0:2]
    directions, dist = normalize(destination_vectors)
    return directions, dist


@njit
def vec_diff(vecs: np.ndarray) -> np.ndarray:
    """r_ab
    r_ab := r_a − r_b.
    """
    diff = np.expand_dims(vecs, 1) - np.expand_dims(vecs, 0)
    return diff


def each_diff(vecs: np.ndarray, keepdims=False) -> np.ndarray:
    """
    :param vecs: nx2 array
    :return: diff with diagonal elements removed
    """
    diff = vec_diff(vecs)
    diff = diff[~np.eye(diff.shape[0], dtype=bool), :]
    if keepdims:
        diff = diff.reshape(vecs.shape[0], -1, vecs.shape[1])
    return diff


@njit
def centroid(vecs: np.ndarray) -> Tuple[float, float]:
    """Center-of-mass of a given group as arithmetic mean."""
    num_datapoints = vecs.shape[0]
    centroid_x, centroid_y = 0, 0
    for x, y in vecs:
        centroid_x += x
        centroid_y += y
    centroid_x /= num_datapoints
    centroid_y /= num_datapoints
    return centroid_x, centroid_y
