"""Adapters to integrate the classic global planner with RobotEnv and local planners."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
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

    def from_velocity_command(
        self, command: Iterable[float], *, safety_intervention: bool = False
    ) -> np.ndarray:
        """Map a (v, w) command into the simulator action space and clip to limits.

        Returns:
            np.ndarray: Action formatted for the robot's configured action space.
        """
        linear_target, angular_target = command
        float_cmd = (float(linear_target), float(angular_target))
        kinematics_model = self.kinematics_model or self._default_kinematics_model()
        if self.kinematics_model is None:
            self.kinematics_model = kinematics_model
        if safety_intervention and isinstance(kinematics_model, BicycleDriveKinematicsModel):
            kinematics_model = replace(kinematics_model, creep_speed=0.0)
        projected = kinematics_model.project(float_cmd)
        self.last_kinematics_diagnostics = kinematics_model.diagnostics(
            float_cmd,
            projected,
        )
        linear_target, angular_target = projected
        if isinstance(self.robot, BicycleDriveRobot):
            self.last_kinematics_diagnostics.update(
                safety_intervention=safety_intervention,
                creep_applied=(
                    0.0 <= float_cmd[0] < 1e-3
                    and abs(float_cmd[1]) >= math.radians(1.0)
                    and projected[0] > float_cmd[0]
                ),
            )
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
            # Bicycle max angular speed derives from max steering at max velocity.
            max_angular_speed = (
                cfg.max_velocity * math.tan(cfg.max_steer) / max(cfg.wheelbase, 1e-6)
            )
            return BicycleDriveKinematicsModel(
                max_velocity=cfg.max_velocity,
                max_angular_speed=max_angular_speed,
                allow_backwards=cfg.allow_backwards,
                max_curvature=math.tan(cfg.max_steer) / cfg.wheelbase,
                creep_speed=cfg.creep_speed,
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
        dt = max(float(self.time_step), 1e-6)
        accel = (target_speed - current_speed) / dt
        accel = float(np.clip(accel, -config.max_decel, config.max_accel))
        accel = float(np.clip(accel, self.action_space.low[0], self.action_space.high[0]))
        achievable_speed = float(
            np.clip(current_speed + dt * accel, config.min_velocity, config.max_velocity)
        )
        yaw_limit = abs(achievable_speed) * math.tan(config.max_steer) / config.wheelbase
        achievable_yaw = float(np.clip(angular_target, -yaw_limit, yaw_limit))
        if abs(achievable_speed) < 1e-6:
            steer = 0.0
        else:
            steer = math.atan(achievable_yaw * config.wheelbase / achievable_speed)
        if self.last_kinematics_diagnostics is not None:
            self.last_kinematics_diagnostics.update(
                speed_achievable=achievable_speed,
                yaw_achievable=achievable_yaw,
                acceleration_limited=not math.isclose(achievable_speed, target_speed),
                yaw_limited=not math.isclose(achievable_yaw, angular_target),
            )
        steer = float(np.clip(steer, -config.max_steer, config.max_steer))

        action = np.array([accel, steer], dtype=np.float32)
        return np.clip(action, self.action_space.low, self.action_space.high)

    def _differential_action(self, linear_target: float, angular_target: float) -> np.ndarray:
        """Compute linear/angular accelerations for a differential-drive robot.

        Returns:
            np.ndarray: Clipped linear and angular acceleration command.
        """
        config = self.robot.config
        current_linear, current_angular = self.robot.current_speed
        target_linear = float(
            np.clip(linear_target, config.min_linear_speed, config.max_linear_speed)
        )
        target_angular = float(
            np.clip(angular_target, -config.max_angular_speed, config.max_angular_speed)
        )
        dt = max(float(self.time_step), 1e-6)
        linear_accel = (target_linear - current_linear) / dt
        angular_accel = (target_angular - current_angular) / dt
        action = np.array([linear_accel, angular_accel], dtype=np.float32)
        return np.clip(action, self.action_space.low, self.action_space.high)


__all__ = [
    "PlannerActionAdapter",
    "attach_classic_global_planner",
]
