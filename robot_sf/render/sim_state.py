"""Pygame-free state payloads consumed by simulation renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import numpy as np

    from robot_sf.common.types import DifferentialDriveAction, PedPose, RobotPose, Vec2D
    from robot_sf.ped_ego.unicycle_drive import UnicycleAction
    from robot_sf.robot.bicycle_drive import BicycleAction


@dataclass
class VisualizableAction:
    """Action payload rendered alongside an agent pose and goal."""

    pose: RobotPose
    action: DifferentialDriveAction | BicycleAction | UnicycleAction
    goal: Vec2D


@dataclass
class VisualizableSimState:
    """Renderer-friendly snapshot of a discrete simulator timestep."""

    timestep: int
    """The discrete timestep of the simulation."""

    robot_action: VisualizableAction | None
    """The action taken by the robot at this timestep."""

    robot_pose: RobotPose
    """The pose of the robot at this timestep."""

    pedestrian_positions: np.ndarray
    """The positions of pedestrians at this timestep."""

    ray_vecs: np.ndarray
    """The ray vectors associated with the robot's sensors."""

    ped_actions: np.ndarray
    """The actions taken by pedestrians at this timestep."""

    ego_ped_pose: PedPose | None = None
    """The pose of the ego pedestrian at this timestep. Defaults to None."""

    ego_ped_ray_vecs: np.ndarray | None = None
    """The ray vectors associated with the ego pedestrian's sensors. Defaults to None."""

    ego_ped_action: VisualizableAction | None = None
    """The action taken by the ego pedestrian at this timestep. Defaults to None."""

    time_per_step_in_secs: float | None = None
    """The time taken for each step in seconds. Defaults to None."""

    planned_path: list[Vec2D] | None = None
    """Optional planned path waypoints for debugging."""

    observation_image: np.ndarray | None = None
    """Optional image observation payload for observation-space visualization."""

    def __post_init__(self) -> None:
        """Validate and normalize renderer state defaults."""
        if self.time_per_step_in_secs is None:
            logger.warning("time_per_step_in_secs is None, defaulting to 0.1s.")
            self.time_per_step_in_secs = 0.1
