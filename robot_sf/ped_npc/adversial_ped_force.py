"""Adversarial pedestrian force module.

This module implements an attractive force that pulls specific pedestrians
towards a point in front of the robot, simulating adversarial behavior in
pedestrian dynamics. It provides configuration and computation utilities
for modifying pedestrian trajectories in simulation environments.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numba
import numpy as np
from pysocialforce.scene import PedState

from robot_sf.common.geometry import euclid_dist
from robot_sf.common.types import RobotPose, Vec2D

MIN_ATTRACTION_DISTANCE_M = 1.0


@dataclass
class AdversarialPedForceConfig:
    """Configuration for adversarial pedestrian force behavior.

    Attributes
    ----------
    is_active : bool
        Whether the adversarial force is enabled.
    relaxation_time : float
        Time constant for force relaxation in seconds.
    robot_radius : float
        Radius of the robot in meters.
    activation_threshold : float
        Distance threshold for activating the force in meters.
    force_multiplier : float
        Scaling factor applied to computed forces.
    offset : float
        Distance in front of the robot where the attraction point is located.
    target_ped_idx : int | list[int]
        Index or indices of the target pedestrians to apply forces to.
    """

    is_active: bool = False
    relaxation_time: float = 0.5
    robot_radius: float = 1.0
    activation_threshold: float = 50.0
    force_multiplier: float = 4.0
    offset: float = 3.0
    target_ped_idx: int | list[int] = -1


class AdversarialPedForce:
    """Compute attractive forces pulling pedestrians towards a point in front of the robot.

    This class implements adversarial force behavior that modifies pedestrian trajectories
    to pull specific pedestrians towards a designated point in front of the robot.

    Attributes
    ----------
    config : AdversarialPedForceConfig
        Configuration parameters for the adversarial force behavior.
    peds : PedState
        The pedestrian state containing positions, velocities, and other properties.
    get_robot_pose : Callable[[], RobotPose]
        Function that returns the current robot position and orientation.
    last_forces : np.ndarray
        The forces computed in the last call to __call__.
    target_ped_idx : list[int]
        Indices of pedestrians to apply forces to. Even if restricted to one pedestrian,
        grouped forces may pull more pedestrians towards the robot.

    Methods
    -------
    __call__() -> np.ndarray
        Compute and return the adversarial forces for all pedestrians.
    """

    def __init__(
        self,
        config: AdversarialPedForceConfig,
        peds: PedState,
        get_robot_pose: Callable[[], RobotPose],
    ):
        """Initialize the adversarial pedestrian force computer.

        Parameters
        ----------
        config : AdversarialPedForceConfig
            Configuration parameters for the adversarial force behavior.
        peds : PedState
            The pedestrian state containing positions, velocities, and other properties.
        get_robot_pose : Callable[[], RobotPose]
            Function that returns the current robot position and orientation.
        """
        self.config = config
        self.peds = peds
        self.get_robot_pose = get_robot_pose
        self.last_forces = np.zeros((self.peds.size(), 2), dtype=np.float64)
        self.target_ped_idx: int | list[int] = config.target_ped_idx

    def __call__(self) -> np.ndarray:
        """Compute and return adversarial forces for all pedestrians.

        Calculates attractive forces that pull target pedestrians towards a point
        in front of the robot, scaled by the force multiplier in the configuration.

        Returns
        -------
        np.ndarray
            Array of shape (num_peds, 2) containing computed forces for each pedestrian.
        """
        threshold = (
            self.config.activation_threshold + self.peds.agent_radius + self.config.robot_radius
        )

        ped_positions = np.array(self.peds.pos(), dtype=np.float64)
        ped_velocities = np.array(self.peds.vel(), dtype=np.float64)
        ped_max_speeds = np.array(self.peds.max_speeds, dtype=np.float64)
        robot_pose = self.get_robot_pose()
        robot_pos = np.array(robot_pose[0], dtype=np.float64)
        robot_orient = robot_pose[1]
        forces = np.zeros((self.peds.size(), 2), dtype=np.float64)
        num_peds = forces.shape[0]
        if num_peds == 0:
            self.last_forces = forces
            return forces

        if isinstance(self.target_ped_idx, int):
            raw_target_idx = [int(self.target_ped_idx)]
        else:
            raw_target_idx = [int(idx) for idx in self.target_ped_idx]

        target_ped_idx = np.array(
            [idx for idx in raw_target_idx if -num_peds <= idx < num_peds],
            dtype=np.int64,
        )
        if target_ped_idx.size == 0:
            self.last_forces = forces
            return forces

        adversarial_ped_force(
            out_forces=forces,
            relaxation_time=self.config.relaxation_time,
            ped_positions=ped_positions,
            ped_velocities=ped_velocities,
            ped_max_speeds=ped_max_speeds,
            robot_pos=robot_pos,
            robot_orient=robot_orient,
            offset=self.config.offset,
            threshold=threshold,
            target_ped_idx=target_ped_idx,
        )

        forces = forces * self.config.force_multiplier
        self.last_forces = forces.copy()
        return forces


@numba.njit(fastmath=True)
def adversarial_ped_force(  # noqa: PLR0913
    out_forces: np.ndarray,
    relaxation_time: float,
    ped_positions: np.ndarray,
    ped_velocities: np.ndarray,
    ped_max_speeds: np.ndarray,
    robot_pos: Vec2D,
    robot_orient: float,
    offset: float,
    threshold: float,
    target_ped_idx: np.ndarray,
):
    """
    Compute the attractive force pulling the target pedestrian towards a point in front of the robot
      specified by the offset .

    Parameters
    ----------
    out_forces : np.ndarray
        Output array for computed forces (shape: [num_peds, 2]); modified in place.
    relaxation_time : float
        Time constant for the velocity relaxation term in seconds.
    ped_positions : np.ndarray
        Array of pedestrian positions (shape: [num_peds, 2]).
    ped_velocities : np.ndarray
        Array of current pedestrian velocities (shape: [num_peds, 2]).
    ped_max_speeds : np.ndarray
        Maximum speed per pedestrian (shape: [num_peds]).
    robot_pos : Vec2D
        Position of the robot (length-2 array).
    robot_orient : float
        Orientation of the robot in radians.
    offset : float
        Distance in front of the robot to compute the attraction point.
    threshold : float
        Only apply force if pedestrian is within this distance to the attraction point.
    target_ped_idx : np.ndarray
        Indices of pedestrians to apply the force to.
    """
    for idx in target_ped_idx:
        # Calculate attraction point in front of the robot
        attraction_point = np.empty(2)
        attraction_point[0] = robot_pos[0] + offset * np.cos(robot_orient)
        attraction_point[1] = robot_pos[1] + offset * np.sin(robot_orient)

        ped_pos = ped_positions[idx]
        distance = euclid_dist(attraction_point, ped_pos)

        if MIN_ATTRACTION_DISTANCE_M < distance <= threshold:
            # Desired direction
            direction = (attraction_point - ped_pos) / distance

            # Desired velocity toward attraction point
            v_desired = direction * ped_max_speeds[idx]

            # relaxation toward desired velocity
            out_forces[idx] = (v_desired - ped_velocities[idx]) / relaxation_time


AdversialPedForceConfig = AdversarialPedForceConfig
AdversialPedForce = AdversarialPedForce
adversial_ped_force = adversarial_ped_force
