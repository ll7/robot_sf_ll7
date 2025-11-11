from collections.abc import Callable
from dataclasses import dataclass

import numba
import numpy as np
from pysocialforce.scene import PedState

from robot_sf.common.types import Vec2D


@dataclass
class PedRobotForceConfig:
    is_active: bool = True
    robot_radius: float = 1.0
    activation_threshold: float = 2.0
    force_multiplier: float = 10.0


class PedRobotForce:
    """This force represents a behavior that's making the pedestrians
    aware of the robot such that they try to avoid it.

    If the force multiplier is parameterized with a negative factor,
    this force can be used for adverserial trainings as well."""

    def __init__(
        self,
        config: PedRobotForceConfig,
        peds: PedState,
        get_robot_pos: Callable[[], Vec2D],
    ):
        self.config = config
        self.peds = peds
        self.get_robot_pos = get_robot_pos
        self.last_forces = 0.0

    def __call__(self) -> np.ndarray:
        threshold = (
            self.config.activation_threshold + self.peds.agent_radius + self.config.robot_radius
        )
        ped_positions = self.peds.pos()
        robot_pos = self.get_robot_pos()
        forces = np.zeros((self.peds.size(), 2))
        ped_robot_force(forces, ped_positions, robot_pos, threshold)
        forces = forces * self.config.force_multiplier
        self.last_forces = forces
        return forces


@numba.njit(fastmath=True)
def ped_robot_force(
    out_forces: np.ndarray,
    ped_positions: np.ndarray,
    robot_pos: Vec2D,
    threshold: float,
):
    """
    Compute the forces exerted on pedestrians by a robot.

    This function uses the potential field method to compute the forces. The force is
    computed for each pedestrian and stored in the `out_forces` array.

    Parameters
    ----------
    out_forces : np.ndarray
        An array where the computed forces will be stored. The array should have the same
        length as `ped_positions`.
    ped_positions : np.ndarray
        An array of the positions of the pedestrians.
    robot_pos : Vec2D
        The position of the robot.
    threshold : float
        The distance threshold for computing the force. If a pedestrian is farther than
        this distance from the robot, the force is not computed.

    Returns
    -------
    None
    """
    # Iterate over all pedestrians
    for i, ped_pos in enumerate(ped_positions):
        # Compute the Euclidean distance between the pedestrian and the robot
        distance = euclid_dist(robot_pos, ped_pos)
        # If the distance is less than or equal to the threshold
        if distance <= threshold:
            # Compute the derivative of the Euclidean distance
            dx_dist, dy_dist = der_euclid_dist(ped_pos, robot_pos, distance)
            # Compute the force using the potential field method and store it in the
            # `out_forces` array
            out_forces[i] = potential_field_force(distance, dx_dist, dy_dist)


# TODO(#250): REFACTOR TO UTILS FILE -> euclid_dist is defined in range_sensor.py
# TODO(#250): Refactor duplicated `euclid_dist` function to a central utils file.
# See: https://github.com/ll7/robot_sf_ll7/issues/250
@numba.njit(fastmath=True)
def euclid_dist(v_1: Vec2D, v_2: Vec2D) -> float:
    """
    Compute the Euclidean distance between two 2D vectors.

    This function uses the standard formula for Euclidean distance: sqrt((x1 - x2)^2 + (y1 - y2)^2).

    Parameters
    ----------
    v_1 : Vec2D
        The first 2D vector. This is a tuple or list of two numbers representing
        the x and y coordinates.
    v_2 : Vec2D
        The second 2D vector. This is a tuple or list of two numbers representing
        the x and y coordinates.

    Returns
    -------
    float
        The Euclidean distance between `v_1` and `v_2`.
    """
    # Compute the difference in x coordinates and square it
    x_diff_sq = (v_1[0] - v_2[0]) ** 2
    # Compute the difference in y coordinates and square it
    y_diff_sq = (v_1[1] - v_2[1]) ** 2
    # Return the square root of the sum of the squared differences
    return (x_diff_sq + y_diff_sq) ** 0.5


@numba.njit(fastmath=True)
def der_euclid_dist(p1: Vec2D, p2: Vec2D, distance: float) -> Vec2D:
    # info: distance is an expensive operation and therefore pre-computed
    dx1_dist = (p1[0] - p2[0]) / distance
    dy1_dist = (p1[1] - p2[1]) / distance
    return dx1_dist, dy1_dist


@numba.njit(fastmath=True)
def potential_field_force(dist: float, dx_dist: float, dy_dist: float) -> tuple[float, float]:
    der_potential = 1 / pow(dist, 3)
    return der_potential * dx_dist, der_potential * dy_dist
