"""Robot-aware force model for pedestrian Social Force simulations.

The module exposes a PySocialForce-compatible callable that applies an inverse-cubic
potential field between the robot and each pedestrian inside a configurable activation
radius. Positive multipliers repel pedestrians from the robot; negative multipliers can
be used for adversarial attraction experiments.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numba
import numpy as np
from pysocialforce.scene import PedState

from robot_sf.common.geometry import euclid_dist
from robot_sf.common.types import Vec2D


@dataclass
class PedRobotForceConfig:
    """Configuration for robot-to-pedestrian force computation."""

    is_active: bool = True
    robot_radius: float = 1.0
    activation_threshold: float = 2.0
    force_multiplier: float = 10.0


class PedRobotForce:
    """PySocialForce-compatible robot interaction force for pedestrians.

    The force reads pedestrian positions from ``peds`` at call time and obtains the
    latest robot position through ``get_robot_pos``. The resulting force array has
    shape ``(num_peds, 2)`` and is stored in ``last_forces`` for diagnostics.
    """

    def __init__(
        self,
        config: PedRobotForceConfig,
        peds: PedState,
        get_robot_pos: Callable[[], Vec2D],
        get_ped_response_multipliers: Callable[[], np.ndarray] | None = None,
    ):
        """Create a robot-aware pedestrian force.

        Args:
            config: Force activation, geometry, and scaling parameters.
            peds: PySocialForce pedestrian state backing the current simulation.
            get_robot_pos: Callback returning the robot position in world coordinates.
            get_ped_response_multipliers: Callback returning a float array of shape (num_peds,)
        """
        self.config = config
        self.peds = peds
        self.get_robot_pos = get_robot_pos
        self.get_ped_response_multipliers = get_ped_response_multipliers
        self.last_forces = 0.0

    def __call__(self) -> np.ndarray:
        """Return the latest robot-to-pedestrian forces computed for the simulation step."""
        threshold = (
            self.config.activation_threshold + self.peds.agent_radius + self.config.robot_radius
        )
        ped_positions = self.peds.pos()
        robot_pos = self.get_robot_pos()
        forces = np.zeros((self.peds.size(), 2))
        ped_robot_force(forces, ped_positions, robot_pos, threshold)
        forces = forces * self.config.force_multiplier
        if self.get_ped_response_multipliers is not None:
            multipliers = self.get_ped_response_multipliers()
            if multipliers is not None:
                forces = forces * multipliers[:, np.newaxis]
        self.last_forces = forces
        return forces


@numba.njit(fastmath=True)
def ped_robot_force(
    out_forces: np.ndarray,
    ped_positions: np.ndarray,
    robot_pos: Vec2D,
    threshold: float,
):
    """Compute repulsive forces applied by the robot to each nearby pedestrian.

    Args:
        out_forces: Output array mutated in-place with per-pedestrian force vectors.
        ped_positions: Current pedestrian positions, shape ``(num_peds, 2)``.
        robot_pos: Robot position in world coordinates.
        threshold: Distance cutoff beyond which forces are not applied.

    Notes:
        ``out_forces`` is modified in place and not returned.
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


@numba.njit(fastmath=True)
def der_euclid_dist(p1: Vec2D, p2: Vec2D, distance: float) -> Vec2D:
    # info: distance is an expensive operation and therefore pre-computed
    """Return the derivative of Euclidean distance with respect to ``p1``.

    Args:
        p1: Point whose distance derivative is being evaluated.
        p2: Reference point for the distance calculation.
        distance: Precomputed Euclidean distance between ``p1`` and ``p2``.

    Returns:
        Unit vector pointing from ``p2`` toward ``p1``.

    Notes:
        ``distance`` must be positive; callers precompute it to avoid duplicate
        square-root work inside the numba kernel.
    """
    dx1_dist = (p1[0] - p2[0]) / distance
    dy1_dist = (p1[1] - p2[1]) / distance
    return dx1_dist, dy1_dist


@numba.njit(fastmath=True)
def potential_field_force(dist: float, dx_dist: float, dy_dist: float) -> tuple[float, float]:
    """Compute the inverse-cubic potential-field force for one pedestrian.

    Args:
        dist: Distance from the pedestrian to the robot.
        dx_dist: X component of the distance derivative.
        dy_dist: Y component of the distance derivative.

    Returns:
        Force vector in world-coordinate units.
    """
    der_potential = 1 / pow(dist, 3)
    return der_potential * dx_dist, der_potential * dy_dist
