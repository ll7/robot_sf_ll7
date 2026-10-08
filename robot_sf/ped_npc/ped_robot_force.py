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


@dataclass
class PedRobotForceV2Config(PedRobotForceConfig):
    """Opt-in robot avoidance with an explicit surface-gap onset and steering.

    The legacy ``activation_threshold`` already adds both radii. In this law,
    ``edge_onset`` names that clearance explicitly: its default gives 3.35 m
    centre distance for a 1.0 m robot and a 0.35 m pedestrian. This is a geometric
    setting, not a population-calibrated reaction distance. ``steering_factor``
    scales the lateral term relative to the existing inverse-cubic repulsion.
    """

    law_version: str = "anticipatory_v2"
    edge_onset: float = 2.0
    steering_factor: float = 1.0

    def __post_init__(self) -> None:
        """Reject unsupported versions and invalid new-law parameters."""
        if self.law_version != "anticipatory_v2":
            raise ValueError(f"Unsupported pedestrian-robot law: {self.law_version!r}")
        if not np.isfinite(self.edge_onset) or self.edge_onset <= 0:
            raise ValueError("edge_onset must be finite and positive")
        if not np.isfinite(self.steering_factor) or self.steering_factor < 0:
            raise ValueError("steering_factor must be finite and non-negative")


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
        get_ped_response_multipliers: Callable[[], np.ndarray | None] | None = None,
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
        law = getattr(self.config, "law_version", None)
        if law == "anticipatory_v2":
            contact_distance = self.peds.agent_radius + self.config.robot_radius
            ped_robot_force_v2(
                forces,
                ped_positions,
                self.peds.vel(),
                robot_pos,
                contact_distance,
                self.config.edge_onset,
                self.config.steering_factor,
            )
        elif law is None:
            ped_robot_force(forces, ped_positions, robot_pos, threshold)
        else:
            raise ValueError(f"Unsupported pedestrian-robot law: {law!r}")
        forces = forces * self.config.force_multiplier
        if self.get_ped_response_multipliers is not None:
            multipliers = self.get_ped_response_multipliers()
            if multipliers is not None:
                multipliers = np.asarray(multipliers, dtype=float)
                # Guard against a stale/mismatched multiplier vector (issue #4618 R6):
                # the pedestrian count can change (e.g. an appended ego row), so only
                # scale when the per-pedestrian multipliers line up with the force rows.
                if multipliers.shape[0] == forces.shape[0]:
                    forces = forces * multipliers[:, np.newaxis]
        self.last_forces = forces
        return forces


@numba.njit(nogil=True)
def ped_robot_force_v2(
    out_forces: np.ndarray,
    ped_positions: np.ndarray,
    ped_velocities: np.ndarray,
    robot_pos: Vec2D,
    contact_distance: float,
    edge_onset: float,
    steering_factor: float,
) -> None:
    """Repel and steer approaching pedestrians whose current path intersects the robot.

    Predict against the robot's current position, as for legacy repulsion. Choose
    the nearer passing side; exact head-on ties pass to the pedestrian's left.
    Steering ramps from zero at onset to full strength at contact and vanishes
    for receding motion or a path missing the contact disc. Inverse-cubic strength
    is bounded at contact inside the disc; this does not resolve physical contact.
    """
    threshold = contact_distance + edge_onset
    for i in range(ped_positions.shape[0]):
        dx = ped_positions[i, 0] - robot_pos[0]
        dy = ped_positions[i, 1] - robot_pos[1]
        distance = np.sqrt(dx * dx + dy * dy)
        if distance > threshold or distance == 0.0:
            continue
        strength = 1.0 / max(distance, contact_distance, 1e-6) ** 3
        out_forces[i, 0] = strength * dx / distance
        out_forces[i, 1] = strength * dy / distance
        vx, vy = ped_velocities[i]
        speed = np.sqrt(vx * vx + vy * vy)
        if speed == 0.0:
            continue
        hx, hy = vx / speed, vy / speed
        forward = -dx * hx - dy * hy
        cross = -dy * hx + dx * hy
        if forward <= 0.0 or abs(cross) >= contact_distance:
            continue
        side = -1.0 if cross > 0.0 else 1.0
        urgency = min(1.0, (threshold - distance) / edge_onset)
        threat = 1.0 - abs(cross) / contact_distance
        lateral = side * steering_factor * strength * urgency * threat
        out_forces[i, 0] -= lateral * hy
        out_forces[i, 1] += lateral * hx


@numba.njit(fastmath=True)
def ped_robot_force(
    out_forces: np.ndarray,
    ped_positions: np.ndarray,
    robot_pos: Vec2D,
    threshold: float,
) -> None:
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
