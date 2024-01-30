"""Calculate forces for individuals and groups"""
import re
from math import atan2, exp
from typing import Tuple, List, Protocol, Callable
import logging

import numpy as np
from numba import njit

from pysocialforce.scene import Line2D, Point2D, PedState
from pysocialforce import logger
from pysocialforce.config import \
    DesiredForceConfig, SocialForceConfig, ObstacleForceConfig, \
    GroupCoherenceForceConfig, GroupGazeForceConfig, GroupReplusiveForceConfig

logging.getLogger('numba').setLevel(logging.WARNING)

Force = Callable[[], np.ndarray]


class SimEntitiesProvider(Protocol):
    """Not implemented!!!"""

    def get_obstacles(self) -> List[np.ndarray]:
        raise NotImplementedError()

    def get_raw_obstacles(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def peds(self) -> PedState:
        raise NotImplementedError()


class DebuggableForce:
    """A wrapper class that adds debugging functionality to a given force."""

    def __init__(self, force: Force):
        self.force = force

    def __call__(self, debug: bool = False):
        """
        Call the wrapped force and optionally log its value for debugging.

        Args:
            debug (bool, optional): Whether to log the value of the force for debugging. Defaults to False.

        Returns:
            The value of the force.
        """
        force = self.force()
        if debug:
            force_type = self.camel_to_snake(type(self).__name__)
            logger.debug(f"{force_type}:\n {repr(force)}")
        return force

    @staticmethod
    def camel_to_snake(camel_case_string: str) -> str:
        """
        Convert a CamelCase string to snake_case.

        Args:
            camel_case_string (str): The CamelCase string to convert.

        Returns:
            The snake_case version of the input string.
        """
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
        """
        Calculate and return the desired force for each pedestrian.

        :return: A numpy array containing the forces for all pedestrians.
        """
        # Relaxation time determines how quickly a pedestrian adapts their velocity
        relexation_time: float = self.config.relaxation_time
        # Threshold distance to consider a goal as reached
        goal_threshold = self.config.goal_threshold
        
        # Get current position, velocity, and goal for each pedestrian
        pos = self.peds.pos()
        vel = self.peds.vel()
        goal = self.peds.goal()
        
        # Calculate normalized direction vector and distance to the goal
        direction, dist = normalize(goal - pos)
        
        # Initialize force array with zeros for each pedestrian
        force = np.zeros((self.peds.size(), 2))
        
        # For pedestrians further than the goal threshold from their goal,
        # calculate the force based on the desired speed and current velocity
        force[dist > goal_threshold] = (
            direction *
            self.peds.max_speeds.reshape((-1, 1)) - vel.reshape((-1, 2))
        )[dist > goal_threshold, :]
        
        # For pedestrians within the goal threshold, apply a braking force
        force[dist <= goal_threshold] = -1.0 * vel[dist <= goal_threshold]
        
        # Divide the force by the relaxation time to get the final force value
        force /= relexation_time
        
        # Multiply by a factor from the configuration to scale the force
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
        ped_positions: np.ndarray,
        ped_velocities: np.ndarray,
        activation_threshold: float,
        n: int, n_prime: int,
        lambda_importance: float,
        gamma: float
) -> np.ndarray:
    """
    Calculates the social force acting on each pedestrian.

    Args:
        ped_positions (np.ndarray): Array of shape (num_peds, 2) representing the positions of pedestrians.
        ped_velocities (np.ndarray): Array of shape (num_peds, 2) representing the velocities of pedestrians.
        activation_threshold (float): Threshold distance for considering the interaction between pedestrians.
        n (int): Exponent for the repulsive force.
        n_prime (int): Exponent for the attractive force.
        lambda_importance (float): Importance factor for the attractive force.
        gamma (float): Scaling factor for the attractive force.

    Returns:
        np.ndarray: Array of shape (num_peds, 2) representing the social forces acting on each pedestrian.
    """
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
        pos_diffs: np.ndarray,
        vel_diffs: np.ndarray,
        n: int, n_prime:
        int, lambda_importance:
        float, gamma: float
) -> Point2D:
    """
    Calculates the social force exerted on a single pedestrian.

    Args:
        pos_diffs (np.ndarray): Array of position differences between the pedestrian and its neighbors.
        vel_diffs (np.ndarray): Array of velocity differences between the pedestrian and its neighbors.
        n (int): Number of neighbors.
        n_prime (int): Number of neighbors in the preferred direction.
        lambda_importance (float): Importance factor for the social force.
        gamma (float): Scaling factor for the social force.

    Returns:
        Point2D: The total social force exerted on the pedestrian in the x and y directions.
    """
    force_sum_x, force_sum_y = 0.0, 0.0
    for i in range(pos_diffs.shape[0]):
        force_x, force_y = social_force_ped_ped(
            pos_diffs[i], vel_diffs[i], n, n_prime, lambda_importance, gamma)
        force_sum_x += force_x
        force_sum_y += force_y
    return force_sum_x, force_sum_y


@njit(fastmath=True)
def social_force_ped_ped(
        pos_diff: Point2D,
        vel_diff: Point2D,
        n: int,
        n_prime: int,
        lambda_importance: float,
        gamma: float
) -> Point2D:
    """
    Calculates the social force between two pedestrians.

    Args:
        pos_diff (Point2D): The position difference between the two pedestrians.
        vel_diff (Point2D): The velocity difference between the two pedestrians.
        n (int): The number of pedestrians.
        n_prime (int): The number of pedestrians prime.
        lambda_importance (float): The importance of the velocity difference.
        gamma (float): The gamma value.

    Returns:
        Point2D: The social force vector between the two pedestrians.
    """
    # Decompose position and velocity differences into components
    pos_diff_x, pos_diff_y = pos_diff
    vel_diff_x, vel_diff_y = vel_diff
    
    # Calculate normalized direction vector and its length from position diff
    (diff_dir_x, diff_dir_y), diff_length = norm_vec((pos_diff_x, pos_diff_y))
    
    # Compute interaction vector based on velocity difference and direction
    interaction_vec_x = lambda_importance * vel_diff_x + diff_dir_x
    interaction_vec_y = lambda_importance * vel_diff_y + diff_dir_y
    
    # Normalize the interaction vector and get its length
    interaction_dir, interaction_length = norm_vec(
        (interaction_vec_x, interaction_vec_y))
    interaction_dir_x, interaction_dir_y = interaction_dir

    # Calculate angle between interaction direction and difference direction
    theta = atan2(interaction_dir[1], interaction_dir[0]
                  ) - atan2(diff_dir_y, diff_dir_x)
    # Determine the sign of theta for force calculation
    theta_sign = 1 if theta >= 0 else -1
    # Calculate B parameter with a small constant to avoid division by zero
    B = gamma * interaction_length + 1e-8

    # Compute the magnitude of the velocity component of the social force
    force_velocity_amount = exp(-1.0 * diff_length /
                                B - (n_prime * B * theta)**2)
    # Compute the magnitude of the angle component of the social force
    force_angle_amount = -theta_sign * \
        exp(-1.0 * diff_length / B - (n * B * theta)**2)
    
    # Calculate the x and y components of the velocity force
    force_velocity_x = interaction_dir_x * force_velocity_amount
    force_velocity_y = interaction_dir_y * force_velocity_amount
    # Calculate the x and y components of the angle force
    force_angle_x = -interaction_dir_y * force_angle_amount
    force_angle_y = interaction_dir_x * force_angle_amount
    
    # Return the total social force as a combination of both components
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
def all_obstacle_forces(
        out_forces: np.ndarray,
        ped_positions: np.ndarray,
        obstacles: np.ndarray,
        ped_radius: float
):
    """
    Calculates the forces exerted by all obstacles on each pedestrian.

    Parameters:
    - out_forces (np.ndarray): Array to store the resulting forces for each pedestrian.
    - ped_positions (np.ndarray): Array of shape (num_peds, 2) containing the positions of all pedestrians.
    - obstacles (np.ndarray): Array of shape (num_obstacles, 6) containing the obstacle line segments and their orthogonal vectors.
    - ped_radius (float): Radius of the pedestrians.

    Returns:
    None
    """
    # Extract obstacle line segments and their orthogonal vectors
    obstacle_segments = obstacles[:, :4]
    ortho_vecs = obstacles[:, 4:]

    # Get the number of pedestrians and obstacles
    num_peds = ped_positions.shape[0]
    num_obstacles = obstacles.shape[0]

    # Iterate over each pedestrian
    for i in range(num_peds):
        ped_pos = ped_positions[i]  # Current pedestrian position

        # Iterate over each obstacle
        for j in range(num_obstacles):
            # Calculate the force exerted by the current obstacle
            force_x, force_y = obstacle_force(
                obstacle_segments[j], ortho_vecs[j], ped_pos, ped_radius)

            # Accumulate forces from all obstacles on the current pedestrian
            out_forces[i, 0] += force_x
            out_forces[i, 1] += force_y


@njit(fastmath=True)
def obstacle_force(obstacle: Line2D,
                   ortho_vec: Point2D,
                   ped_pos: Point2D,
                   ped_radius: float
                   ) -> Tuple[float, float]:
    """
    Calculate the repulsive force exerted by an obstacle on a pedestrian.

    The force is calculated based on the distance from the pedestrian to the
    nearest point on the obstacle and the gradient of the potential field at
    that point. The potential field is inversely proportional to the square of
    the distance to the obstacle.

    Args:
        obstacle: A tuple representing the endpoints (x1, y1, x2, y2) of the
                  line segment that forms the obstacle.
        ortho_vec: A vector orthogonal to the direction of pedestrian movement.
            # TODO Is this correct? Maybe the ortho_vec is orthogonal to the obstacle?
        ped_pos: The current position (x, y) of the pedestrian.
        ped_radius: The radius of the pedestrian (used for collision avoidance).

    Returns:
        A tuple (force_x, force_y) representing the x and y components of the
        repulsive force exerted by the obstacle on the pedestrian.
    """

    # Minimum distance to consider for collision calculations.
    coll_dist = 1e-5

    # Unpack obstacle endpoints and calculate orthogonal projection points.
    x1, y1, x2, y2 = obstacle
    (x3, y3), (x4, y4) = ped_pos, (ped_pos[0] + ortho_vec[0],
                                   ped_pos[1] + ortho_vec[1])

    # Case 1: Obstacle is a single point (no length).
    if (x1, y1) == (x2, y2):
        obst_dist = max(euclid_dist(ped_pos[0], ped_pos[1], x1, y1) -
                        ped_radius, coll_dist)
        dx_obst_dist, dy_obst_dist = der_euclid_dist(ped_pos, (x1, y1),
                                                     obst_dist)
        return potential_field_force(obst_dist, dx_obst_dist, dy_obst_dist)

    # Calculate intersection of orthogonal projection with obstacle line.
    num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    t = num / den
    ortho_hit = 0 <= t <= 1  # Check if intersection is within segment bounds.

    # Case 2: Orthogonal projection does not hit within the obstacle segment.
    if not ortho_hit:
        d1 = euclid_dist(ped_pos[0], ped_pos[1], x1, y1)
        d2 = euclid_dist(ped_pos[0], ped_pos[1], x2, y2)
        obst_dist = max(min(d1, d2) - ped_radius, coll_dist)
        closer_obst_bound = (x1, y1) if d1 < d2 else (x2, y2)
        dx_obst_dist, dy_obst_dist = der_euclid_dist(ped_pos,
                                                     closer_obst_bound,
                                                     obst_dist)
        return potential_field_force(obst_dist, dx_obst_dist, dy_obst_dist)

    # Case 1: Orthogonal projection hits within the obstacle segment.
    cross_x, cross_y = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
    obst_dist = max(euclid_dist(ped_pos[0], ped_pos[1], cross_x, cross_y) -
                    ped_radius, coll_dist)
    # Compute derivatives of the intersection point with respect to ped_pos.
    dx3_cross_x = (y4 - y3) / den * (x2 - x1)
    dx3_cross_y = (y4 - y3) / den * (y2 - y1)
    dy3_cross_x = (x3 - x4) / den * (x2 - x1)
    dy3_cross_y = (x3 - x4) / den * (y2 - y1)
    # Compute derivatives of the obstacle distance with respect to ped_pos.
    dx_obst_dist = ((cross_x - ped_pos[0]) * (dx3_cross_x - 1) +
                    (cross_y - ped_pos[1]) * dx3_cross_y) / obst_dist
    dy_obst_dist = ((cross_x - ped_pos[0]) * dy3_cross_x +
                    (cross_y - ped_pos[1]) * (dy3_cross_y - 1)) / obst_dist
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
    """
    Alternative group coherence force as specified in pedsim_ros
    This class represents the computation of an alternative model for the group 
    coherence force, which is a social force that aims to keep the members of a 
    group together.
    """

    def __init__(self, config: GroupCoherenceForceConfig, peds: PedState):
        """
        Initialize the GroupCoherenceForceAlt object.

        :param config: An instance of GroupCoherenceForceConfig containing the 
            configuration parameters for the coherence force.
        :param peds: An instance of PedState representing the state 
            (position, velocity, etc.) of all pedestrians.
        """
        # Store the pedestrian states which includes positions and velocities.
        self.peds = peds

        # Store the configuration for calculating coherence forces.
        self.config = config

    def __call__(self):
        # Initialize an array to store coherence forces for each pedestrian with zero values.
        forces = np.zeros((self.peds.size(), 2))

        # If no groups exist within the pedestrian data, return the initialized zero-forces array.
        if not self.peds.has_group():
            return forces

        # Iterate over groups of pedestrians to calculate coherence forces.
        for group in self.peds.groups:
            # Set the threshold based on the number of members in the group.
            threshold = (len(group) - 1) / 2

            # Extract the positions of all members in the current group.
            member_pos = self.peds.pos()[group, :]

            # Continue to the next group if the current one has no members.
            if len(member_pos) == 0:
                continue

            # Compute the centroid (center of mass) for the group's positions.
            com = centroid(member_pos)

            # Calculate the vector from individual members to the centroid.
            force_vec = com - member_pos

            # Compute the norms (magnitudes) of those vectors.
            norms = np.linalg.norm(force_vec, axis=1)

            # Apply a softening factor based on the distance from the centroid, using hyperbolic tangent.
            softened_factor = (np.tanh(norms - threshold) + 1) / 2

            # Calculate the actual forces applying the softening factor to the force vectors.
            forces[group, :] += (force_vec.T * softened_factor).T

        # Return the calculated forces scaled by the factor defined in the configuration.
        return forces * self.config.factor


class GroupRepulsiveForce:
    """
    A class representing the computation of group repulsive force which is a social force
    that models the repulsive interaction among different members within a group of pedestrians.
    """

    def __init__(self, config: GroupReplusiveForceConfig, peds: PedState):
        """
        :param config: An instance of GroupReplusiveForceConfig containing configuration parameters for the repulsive force.
        :param peds: An instance of PedState representing the state (position, velocity, etc.) of all pedestrians.
        """
        self.config = config
        self.peds = peds

    def __call__(self):
        # Retrieve the distance threshold from configuration where repulsive force is effective.
        threshold = self.config.threshold
        # Initialize a zero np.array to store repulsive forces for each pedestrian.
        forces = np.zeros((self.peds.size(), 2))

        # If no groups exist in the pedestrian dataset, return the zero-initialized forces array.
        if not self.peds.has_group():
            return forces

        # Iterate over groups of pedestrians.
        for group in self.peds.groups:
            # Skip processing for empty groups.
            if not group:
                continue

            size = len(group)
            member_pos = self.peds.pos()[group, :]

            # Calculate the relative positions among members in the group.
            diff = each_diff(member_pos)  # others - self
            # Normalize the relative position vectors and get their norms.
            _, norms = normalize(diff)
            # Zero out the relative positions of members outside the threshold distance.
            diff[norms > threshold, :] = 0
            # Sum the adjusted relative positions to compute forces for the current group.
            forces[group, :] += np.sum(diff.reshape((size, -1, 2)), axis=1)

        # Multiply the forces by a scaling factor from the configuration and return the result.
        return forces * self.config.factor


class GroupGazeForceAlt:
    """
    Group gaze force
    A class representing the alternative method to compute group gaze force which is 
    a social force that models the interaction within a group of pedestrians.
    """

    def __init__(self, config: GroupGazeForceConfig, peds: PedState):
        """
        :param config: An instance of GroupGazeForceConfig containing
            configuration parameters.
        :param peds: An instance of PedState representing the state of all pedestrians.
        """
        self.config = config
        self.peds = peds

    def __call__(self):
        """
        Calculates and returns the group gaze forces for all pedestrian groups. 
        This method allows an instance of the class to be called as a function.

        :return: A numpy array containing the forces applied to each pedestrian 
            due to group gaze.
        """
        # Initialize a zero matrix for the forces with dimensions equal to the number of pedestrians by 2 (for x and y components).
        forces = np.zeros((self.peds.size(), 2))

        # If there are no groups, return the zero-initialized forces array.
        if not self.peds.has_group():
            return forces

        ped_positions = self.peds.pos()
        # Calculate desired directions and distances for the group gaze force.
        directions, dist = desired_directions(self.peds.state)

        # Iterate over each group in the pedestrian groups.
        for group in self.peds.groups:
            group_size = len(group)
            # If the group size is less than or equal to 1, skip this group as it does not exert group force.
            if group_size <= 1:
                continue
            # Compute the group gaze force for the current group and assign it to the forces array.
            forces[group, :] = group_gaze_force(
                ped_positions[group, :], directions[group, :], dist[group])

        return forces * self.config.factor


@njit(fastmath=True)
def group_gaze_force(
        member_pos: np.ndarray,
        member_directions: np.ndarray,
        member_dist: np.ndarray
) -> np.ndarray:
    """
    Calculates the group gaze force for each member in a group.

    Args:
        member_pos (np.ndarray): Array of shape (group_size, 2) representing the positions of group members.
        member_directions (np.ndarray): Array of shape (group_size, 2) representing the walking directions of group members.
        member_dist (np.ndarray): Array of shape (group_size,) representing the distances between each member and their respective center of mass.

    Returns:
        np.ndarray: Array of shape (group_size, 2) representing the group gaze forces for each member.
    """
    # Determine the number of members in the group based on the array shape.
    group_size = member_pos.shape[0]

    # Initialize a zero array to store the output forces for each member, size Nx2 for N members and 2D forces.
    out_forces = np.zeros((group_size, 2))

    # Iterate over all group members to calculate the force that each should experience.
    for i in range(group_size):
        # Calculate the center of mass of the other members excluding the current member.
        other_member_pos = member_pos[np.arange(group_size) != i, :2]
        mass_center_without_ped = centroid(other_member_pos)

        # Compute the relative vector pointing from the current member's position to this center of mass.
        relative_com_x = mass_center_without_ped[0] - member_pos[i, 0]
        relative_com_y = mass_center_without_ped[1] - member_pos[i, 1]

        # Normalize this vector and get the distance to the center of mass.
        com_dir, com_dist = norm_vec((relative_com_x, relative_com_y))

        # Calculate the dot product between the pedestrian’s direction and the vector
        # pointing towards the center of mass. This will be used to determine
        # the alignment of the pedestrian's gaze with the group's common direction.
        ped_dir_x, ped_dir_y = member_directions[i]
        element_prod = ped_dir_x * com_dir[0] + ped_dir_y * com_dir[1]

        # Weigh the influence by the distance to the center of mass and the
        # aforementioned dot product, normalized by the member’s desired separation distance.
        factor = com_dist * element_prod / member_dist[i]

        # Project the computed factor onto the pedestrian's direction to obtain the force components.
        force_x, force_y = ped_dir_x * factor, ped_dir_y * factor

        # Assign the calculated force to the output array for the current member.
        out_forces[i, 0] = force_x
        out_forces[i, 1] = force_y

    # Return the array containing the forces to be applied to each group member.
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
    """
    Compute the centroid of a set of points in 2D space.

    The centroid is calculated as the arithmetic mean (average) 
    of all the points. It is often referred to as the "center of gravity" or 
    "geometric center" of a two-dimensional shape.

    Parameters:
    vecs (np.ndarray): An array of points where each point is represented as [x, y].

    Returns:
    Tuple[float, float]: A tuple containing the x and y coordinates of the centroid.
    """
    # Check if the array is empty
    if vecs.size == 0 or vecs.shape == (0, ):
        raise ValueError("Input array is empty")

    # Determine the number of data points in the array
    num_datapoints = vecs.shape[0]

    # Initialize sums for x and y coordinates
    centroid_x, centroid_y = 0, 0

    # Sum up all x and y coordinates
    for x, y in vecs:
        centroid_x += x
        centroid_y += y

    # Divide by the number of points to get the average for each dimension
    centroid_x /= num_datapoints
    centroid_y /= num_datapoints

    # Return the centroid coordinates as a tuple
    return centroid_x, centroid_y
