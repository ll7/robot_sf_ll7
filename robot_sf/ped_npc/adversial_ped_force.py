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


@dataclass
class AdversialPedForceConfig:
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
    target_ped_idx : int
        Index of the target pedestrian to apply forces to.
    """

    is_active: bool = False
    relaxation_time: float = 0.5
    robot_radius: float = 1.0
    activation_threshold: float = 50.0
    force_multiplier: float = 3.0
    offset: float = 0.0
    target_ped_idx: int = -1


class AdversialPedForce:
    """Compute attractive forces pulling pedestrians towards a point in front of the robot.

    This class implements adversarial force behavior that modifies pedestrian trajectories
    to pull specific pedestrians towards a designated point in front of the robot.

    Attributes
    ----------
    config : AdversialPedForceConfig
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
        config: AdversialPedForceConfig,
        peds: PedState,
        get_robot_pose: Callable[[], RobotPose],
    ):
        """Initialize the adversarial pedestrian force computer.

        Parameters
        ----------
        config : AdversialPedForceConfig
            Configuration parameters for the adversarial force behavior.
        peds : PedState
            The pedestrian state containing positions, velocities, and other properties.
        get_robot_pose : Callable[[], RobotPose]
            Function that returns the current robot position and orientation.
        """
        self.config = config
        self.peds = peds
        self.get_robot_pose = get_robot_pose
        self.last_forces = 0.0
        self.target_ped_idx = [0, -1]
        """Even if the target_idx restricts to one ped, groups forces may pull more pedestrians
            towards the robot"""

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
        robot_pos = np.array(self.get_robot_pose()[0], dtype=np.float64)
        robot_orient = self.get_robot_pose()[1]
        forces = np.zeros((self.peds.size(), 2))
        if isinstance(self.target_ped_idx, int):
            target_ped_idx = np.array([self.target_ped_idx], dtype=np.int64)
        else:
            target_ped_idx = np.array(self.target_ped_idx, dtype=np.int64)

        adversial_ped_force(
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
        self.last_forces = forces
        return forces


@numba.njit(fastmath=True)
def adversial_ped_force(  # noqa: PLR0913
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
        Output array for computed forces (shape: [num_peds, 2]).
    ped_positions : np.ndarray
        Array of pedestrian positions (shape: [num_peds, 2]).
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

        if distance > 1e-6:  # avoid division by zero
            # Desired direction
            direction = (attraction_point - ped_pos) / distance

            # Desired velocity toward attraction point
            v_desired = direction * ped_max_speeds[idx]

            # relaxation toward desired velocity
            out_forces[idx] = v_desired - ped_velocities[idx] / relaxation_time
