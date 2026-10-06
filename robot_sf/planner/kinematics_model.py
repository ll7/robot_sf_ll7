"""Kinematics model contract used by planner/runtime command wiring."""

from __future__ import annotations

import math
from dataclasses import InitVar, dataclass
from typing import Any, Protocol

import numpy as np
from loguru import logger

from robot_sf.robot.reverse_drive import validate_reverse_settings

Command2D = tuple[float, float]


class KinematicsModel(Protocol):
    """Runtime contract for command feasibility and projection."""

    name: str

    def is_feasible(self, command: Command2D) -> bool:
        """Return whether a command is natively feasible for this model."""

    def project(self, command: Command2D) -> Command2D:
        """Project/clip a command into the feasible set."""

    def diagnostics(self, command: Command2D, projected: Command2D) -> dict[str, Any]:
        """Return diagnostics payload for metadata and debugging."""


def _build_diagnostics(
    model: KinematicsModel,
    command: Command2D,
    projected: Command2D,
) -> dict[str, Any]:
    """Build a standard command-projection diagnostics payload.

    Returns:
        dict[str, Any]: Structured diagnostics for command adaptation.
    """
    return {
        "kinematics_model": model.name,
        "feasible_native": bool(model.is_feasible(command)),
        "projection_applied": command != projected,
        "command_in": [float(command[0]), float(command[1])],
        "command_projected": [float(projected[0]), float(projected[1])],
    }


@dataclass(frozen=True)
class DifferentialDriveKinematicsModel:
    """Differential-drive command feasibility in (v, omega) space."""

    max_linear_speed: float
    max_angular_speed: float
    allow_backwards: bool = False
    name: str = "differential_drive"
    limited_reverse: InitVar[bool] = False
    max_reverse_speed: InitVar[float] = 0.5

    def __post_init__(self, limited_reverse: bool, max_reverse_speed: float) -> None:
        """Validate the opt-in reverse bounds without changing legacy serialization."""
        validate_reverse_settings(limited_reverse, max_reverse_speed)
        object.__setattr__(self, "limited_reverse", limited_reverse)
        object.__setattr__(self, "max_reverse_speed", float(max_reverse_speed))

    def is_feasible(self, command: Command2D) -> bool:
        """Check whether ``(v, omega)`` is within configured bounds.

        Returns:
            bool: ``True`` when command is already feasible.
        """
        v, omega = command
        min_linear = (
            -self.max_reverse_speed
            if self.limited_reverse
            else -self.max_linear_speed
            if self.allow_backwards
            else 0.0
        )
        return bool(
            min_linear <= v <= self.max_linear_speed
            and -self.max_angular_speed <= omega <= self.max_angular_speed
        )

    def project(self, command: Command2D) -> Command2D:
        """Clip ``(v, omega)`` into configured differential-drive limits.

        Returns:
            Command2D: Projected command in feasible set.
        """
        v, omega = command
        min_linear = (
            -self.max_reverse_speed
            if self.limited_reverse
            else -self.max_linear_speed
            if self.allow_backwards
            else 0.0
        )
        return (
            float(np.clip(v, min_linear, self.max_linear_speed)),
            float(np.clip(omega, -self.max_angular_speed, self.max_angular_speed)),
        )

    def diagnostics(self, command: Command2D, projected: Command2D) -> dict[str, Any]:
        """Build projection diagnostics payload for metadata and debugging.

        Returns:
            dict[str, Any]: Structured diagnostics for command adaptation.
        """
        return _build_diagnostics(self, command, projected)


@dataclass(frozen=True)
class BicycleDriveKinematicsModel:
    """Bicycle-drive feasibility in (v, omega) planning command space."""

    max_velocity: float
    max_angular_speed: float
    allow_backwards: bool = False
    name: str = "bicycle_drive"
    max_curvature: float | None = None
    creep_speed: float = 0.0
    limited_reverse: InitVar[bool] = False
    max_reverse_speed: InitVar[float] = 0.5

    def __post_init__(self, limited_reverse: bool, max_reverse_speed: float) -> None:
        """Require physical curvature and an explicit nonnegative creep speed."""
        validate_reverse_settings(limited_reverse, max_reverse_speed)
        object.__setattr__(self, "limited_reverse", limited_reverse)
        object.__setattr__(self, "max_reverse_speed", float(max_reverse_speed))
        if (
            self.max_curvature is None
            or not math.isfinite(self.max_curvature)
            or self.max_curvature < 0
        ):
            raise ValueError("bicycle max_curvature must be nonnegative: tan(max_steer)/wheelbase")
        if not math.isfinite(self.creep_speed) or self.creep_speed < 0:
            raise ValueError("bicycle creep_speed must be finite and nonnegative")

    @property
    def curvature_limit(self) -> float:
        """Return the explicitly supplied physical tan(max_steer)/wheelbase."""
        assert self.max_curvature is not None  # validated at construction
        return self.max_curvature

    @property
    def min_velocity(self) -> float:
        """Return the minimum feasible linear velocity.

        Returns:
            float: Negative max speed when backwards motion is allowed, otherwise ``0.0``.
        """
        if self.limited_reverse:
            return -self.max_reverse_speed
        return -self.max_velocity if self.allow_backwards else 0.0

    def is_feasible(self, command: Command2D) -> bool:
        """Check whether command is inside bicycle planning bounds.

        Returns:
            bool: ``True`` when command is already feasible.
        """
        v, omega = command
        return bool(
            self.min_velocity <= v <= self.max_velocity
            and abs(omega) <= min(self.max_angular_speed, abs(v) * self.curvature_limit)
        )

    def project(self, command: Command2D) -> Command2D:
        """Project with speed priority onto the coupled bicycle cone.

        Returns:
            Command2D: Physically feasible speed and yaw command.
        """
        return self.project_with_creep_info(command)[0]

    def project_with_creep_info(self, command: Command2D) -> tuple[Command2D, bool]:
        """Project with speed priority and report the optional creep branch.

        This clips yaw at the bounded requested speed, rather than finding a
        Euclidean nearest point. Creep is disabled by default; a zero/zero stop
        stays stopped. Creep must never apply under a safety intervention:
        callers carrying a veto must use a creep-disabled model, as the robot
        adapter does when ``safety_intervention=True``.

        Returns:
            Projected command and whether the optional creep branch raised speed.
        """
        v, omega = command
        creep_applied = False
        if self.creep_speed > v and 0.0 <= v < 1e-3 and abs(omega) >= math.radians(1.0):
            creep_velocity = min(self.creep_speed, self.max_velocity)
            creep_applied = creep_velocity > v
            v = creep_velocity
        v = float(np.clip(v, self.min_velocity, self.max_velocity))
        yaw_limit = min(self.max_angular_speed, abs(v) * self.curvature_limit)
        return (v, float(np.clip(omega, -yaw_limit, yaw_limit))), creep_applied

    def diagnostics(self, command: Command2D, projected: Command2D) -> dict[str, Any]:
        """Build projection diagnostics payload for metadata and debugging.

        Returns:
            dict[str, Any]: Structured diagnostics for command adaptation.
        """
        return _build_diagnostics(self, command, projected)


@dataclass(frozen=True)
class HolonomicPassthroughKinematicsModel:
    """Passthrough model for already-feasible holonomic command outputs."""

    name: str = "holonomic"

    def is_feasible(self, command: Command2D) -> bool:
        """Treat all commands as feasible for passthrough holonomic usage.

        Returns:
            bool: Always ``True``.
        """
        del command
        return True

    def project(self, command: Command2D) -> Command2D:
        """Return command unchanged for holonomic passthrough behavior.

        Returns:
            Command2D: Original command tuple.
        """
        return float(command[0]), float(command[1])

    def diagnostics(self, command: Command2D, projected: Command2D) -> dict[str, Any]:
        """Return passthrough diagnostics payload.

        Returns:
            dict[str, Any]: Diagnostics marking passthrough semantics.
        """
        return _build_diagnostics(self, command, projected)


def resolve_benchmark_kinematics_model(
    *,
    robot_kinematics: str | None,
    command_limits: dict[str, Any] | None = None,
) -> KinematicsModel:
    """Resolve a kinematics model for benchmark planner command projection.

    Returns:
        KinematicsModel: Contract implementation matching the runtime robot mode.
    """
    limits = command_limits or {}
    kinematics = str(robot_kinematics or "differential_drive").strip().lower()
    if kinematics == "bicycle_drive":
        max_velocity = float(limits.get("max_velocity", limits.get("v_max", 2.0)))
        max_angular = float(limits.get("max_angular_speed", limits.get("omega_max", 1.0)))
        return BicycleDriveKinematicsModel(
            max_velocity=float(limits.get("bicycle_max_velocity", max_velocity)),
            max_angular_speed=float(limits.get("bicycle_max_angular_speed", max_angular)),
            allow_backwards=bool(limits.get("allow_backwards", False)),
            max_curvature=limits.get("bicycle_max_curvature"),
            limited_reverse=limits.get("limited_reverse", False),
            max_reverse_speed=limits.get("max_reverse_speed", 0.5),
        )
    if kinematics in {"holonomic", "omni", "omnidirectional"}:
        return HolonomicPassthroughKinematicsModel()
    if kinematics != "differential_drive":
        logger.warning(
            "Unknown robot kinematics '{}' resolved as fallback '{}'.",
            kinematics,
            "differential_drive",
        )
    max_linear = float(limits.get("max_linear_speed", limits.get("v_max", 2.0)))
    max_angular = float(limits.get("max_angular_speed", limits.get("omega_max", 1.0)))
    return DifferentialDriveKinematicsModel(
        max_linear_speed=max_linear,
        max_angular_speed=max_angular,
        allow_backwards=bool(limits.get("allow_backwards", False)),
        limited_reverse=limits.get("limited_reverse", False),
        max_reverse_speed=limits.get("max_reverse_speed", 0.5),
    )


__all__ = [
    "BicycleDriveKinematicsModel",
    "Command2D",
    "DifferentialDriveKinematicsModel",
    "HolonomicPassthroughKinematicsModel",
    "KinematicsModel",
    "resolve_benchmark_kinematics_model",
]
