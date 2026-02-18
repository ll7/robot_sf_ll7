"""Adapters to integrate the classic global planner with RobotEnv and local planners."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.planner.classic_global_planner import ClassicGlobalPlanner, ClassicPlannerConfig
from robot_sf.planner.kinematics_model import (
    BicycleDriveKinematicsModel,
    DifferentialDriveKinematicsModel,
    KinematicsModel,
)
from robot_sf.robot.bicycle_drive import BicycleDriveRobot
from robot_sf.robot.differential_drive import DifferentialDriveRobot

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gymnasium import spaces

    from robot_sf.nav.map_config import MapDefinition


def attach_classic_global_planner(
    map_def: MapDefinition,
    planner_config: ClassicPlannerConfig | None = None,
) -> ClassicGlobalPlanner:
    """Attach a ClassicGlobalPlanner to a map definition for route sampling.

    When attached, :func:`robot_sf.nav.navigation.sample_route` will invoke the planner
    for spawn/goal samples instead of using the pre-authored waypoints on the map.

    Args:
        map_def: Map definition to mutate with planner metadata.
        planner_config: Optional planner configuration; defaults to ``ClassicPlannerConfig()``.

    Returns:
        ClassicGlobalPlanner: The planner instance attached to ``map_def``.
    """
    planner = ClassicGlobalPlanner(map_def, config=planner_config or ClassicPlannerConfig())
    map_def._global_planner = planner
    map_def._use_planner = True
    return planner


@dataclass
class PlannerActionAdapter:
    """Convert planner (linear, angular) commands into the environment action space."""

    robot: BicycleDriveRobot | DifferentialDriveRobot
    action_space: spaces.Box
    time_step: float
    kinematics_model: KinematicsModel | None = None
    last_kinematics_diagnostics: dict[str, Any] | None = None

    def from_velocity_command(self, command: Iterable[float]) -> np.ndarray:
        """Map a (v, w) command into the simulator action space and clip to limits.

        Returns:
            np.ndarray: Action formatted for the robot's configured action space.
        """
        linear_target, angular_target = command
        kinematics_model = self.kinematics_model or self._default_kinematics_model()
        projected = kinematics_model.project((float(linear_target), float(angular_target)))
        self.last_kinematics_diagnostics = kinematics_model.diagnostics(
            (float(linear_target), float(angular_target)),
            projected,
        )
        linear_target, angular_target = projected
        if isinstance(self.robot, BicycleDriveRobot):
            return self._bicycle_action(linear_target, angular_target)
        if isinstance(self.robot, DifferentialDriveRobot):
            return self._differential_action(linear_target, angular_target)
        msg = f"Unsupported robot type for planner adapter: {type(self.robot)}"
        raise ValueError(msg)

    def _default_kinematics_model(self) -> KinematicsModel:
        """Infer a default kinematics model from the attached robot type.

        Returns:
            KinematicsModel: Contract implementation for the configured drivetrain.
        """
        if isinstance(self.robot, BicycleDriveRobot):
            cfg = self.robot.config
            return BicycleDriveKinematicsModel(
                min_velocity=cfg.min_velocity,
                max_velocity=cfg.max_velocity,
                max_angular_speed=cfg.max_steer,
            )
        if isinstance(self.robot, DifferentialDriveRobot):
            cfg = self.robot.config
            return DifferentialDriveKinematicsModel(
                max_linear_speed=cfg.max_linear_speed,
                max_angular_speed=cfg.max_angular_speed,
                allow_backwards=cfg.allow_backwards,
            )
        msg = f"Unsupported robot type for planner adapter: {type(self.robot)}"
        raise ValueError(msg)

    def _bicycle_action(self, linear_target: float, angular_target: float) -> np.ndarray:
        """Compute acceleration/steering commands for a bicycle-drive robot.

        Returns:
            np.ndarray: Clipped acceleration and steering command.
        """
        config = self.robot.config
        current_speed, _ = self.robot.current_speed

        target_speed = float(np.clip(linear_target, config.min_velocity, config.max_velocity))
        accel = (target_speed - current_speed) / max(self.time_step, 1e-6)
        accel = float(np.clip(accel, -config.max_accel, config.max_accel))

        if abs(target_speed) < 1e-6:
            steer = 0.0
        else:
            steer = math.atan(
                angular_target * config.wheelbase / max(abs(target_speed), 1e-6)
            ) * np.sign(target_speed)
        steer = float(np.clip(steer, -config.max_steer, config.max_steer))

        action = np.array([accel, steer], dtype=np.float32)
        return np.clip(action, self.action_space.low, self.action_space.high)

    def _differential_action(self, linear_target: float, angular_target: float) -> np.ndarray:
        """Compute linear/angular deltas for a differential-drive robot.

        Returns:
            np.ndarray: Clipped delta-linear and delta-angular command.
        """
        config = self.robot.config
        current_linear, current_angular = self.robot.current_speed
        target_linear = float(np.clip(linear_target, 0.0, config.max_linear_speed))
        target_angular = float(
            np.clip(angular_target, -config.max_angular_speed, config.max_angular_speed)
        )
        d_linear = target_linear - current_linear
        d_angular = target_angular - current_angular
        action = np.array([d_linear, d_angular], dtype=np.float32)
        return np.clip(action, self.action_space.low, self.action_space.high)
