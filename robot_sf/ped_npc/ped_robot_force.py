from typing import Tuple, Callable
from dataclasses import dataclass

import numpy as np
import numba
from pysocialforce.scene import PedState

Vec2D = Tuple[float, float]


@dataclass
class PedRobotForceConfig:
    is_active: bool = False
    robot_radius: float = 1.0
    activation_threshold: float = 2.0
    force_multiplier: float = 10.0


class PedRobotForce:
    """This force represents a behavior that's making the pedestrians
    aware of the robot such that they try to avoid it.

    If the force multiplier is parameterized with a negative factor,
    this force can be used for adverserial trainings as well."""

    def __init__(self, config: PedRobotForceConfig, peds: PedState,
                 get_robot_pos: Callable[[], Vec2D]):
        self.config = config
        self.peds = peds
        self.get_robot_pos = get_robot_pos
        self.last_forces = 0.0

    def __call__(self) -> np.ndarray:
        threshold = self.config.activation_threshold \
            + self.peds.agent_radius + self.config.robot_radius
        ped_positions = self.peds.pos()
        robot_pos = self.get_robot_pos()
        forces = np.zeros((self.peds.size(), 2))
        ped_robot_force(forces, ped_positions, robot_pos, threshold)
        forces = forces * self.config.force_multiplier
        self.last_forces = forces
        return forces


@numba.njit(fastmath=True)
def ped_robot_force(
        out_forces: np.ndarray, ped_positions: np.ndarray,
        robot_pos: Vec2D, threshold: float):

    for i, ped_pos in enumerate(ped_positions):
        distance = euclid_dist(robot_pos, ped_pos)
        if distance <= threshold:
            dx_dist, dy_dist = der_euclid_dist(ped_pos, robot_pos, distance)
            out_forces[i] = potential_field_force(distance, dx_dist, dy_dist)


@numba.njit(fastmath=True)
def euclid_dist(v_1: Vec2D, v_2: Vec2D) -> float:
    return ((v_1[0] - v_2[0])**2 + (v_1[1] - v_2[1])**2)**0.5


@numba.njit(fastmath=True)
def der_euclid_dist(p1: Vec2D, p2: Vec2D, distance: float) -> Vec2D:
    # info: distance is an expensive operation and therefore pre-computed
    dx1_dist = (p1[0] - p2[0]) / distance
    dy1_dist = (p1[1] - p2[1]) / distance
    return dx1_dist, dy1_dist


@numba.njit(fastmath=True)
def potential_field_force(dist: float, dx_dist: float, dy_dist: float) -> Tuple[float, float]:
    der_potential = 1 / pow(dist, 3)
    return der_potential * dx_dist, der_potential * dy_dist
