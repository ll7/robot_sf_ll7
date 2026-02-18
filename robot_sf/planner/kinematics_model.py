"""Kinematics model contract used by planner/runtime command wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

Command2D = tuple[float, float]


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


class KinematicsModel(Protocol):
    """Runtime contract for command feasibility and projection."""

    name: str

    def is_feasible(self, command: Command2D) -> bool:
        """Return whether a command is natively feasible for this model."""

    def project(self, command: Command2D) -> Command2D:
        """Project/clip a command into the feasible set."""

    def diagnostics(self, command: Command2D, projected: Command2D) -> dict[str, Any]:
        """Return diagnostics payload for metadata and debugging."""


@dataclass(frozen=True)
class DifferentialDriveKinematicsModel:
    """Differential-drive command feasibility in (v, omega) space."""

    max_linear_speed: float
    max_angular_speed: float
    allow_backwards: bool = False
    name: str = "differential_drive"

    def is_feasible(self, command: Command2D) -> bool:
        """Check whether ``(v, omega)`` is within configured bounds.

        Returns:
            bool: ``True`` when command is already feasible.
        """
        v, omega = command
        min_linear = -self.max_linear_speed if self.allow_backwards else 0.0
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
        min_linear = -self.max_linear_speed if self.allow_backwards else 0.0
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

    @property
    def min_velocity(self) -> float:
        """Return the minimum feasible linear velocity.

        Returns:
            float: Negative max speed when backwards motion is allowed, otherwise ``0.0``.
        """
        return -self.max_velocity if self.allow_backwards else 0.0

    def is_feasible(self, command: Command2D) -> bool:
        """Check whether command is inside bicycle planning bounds.

        Returns:
            bool: ``True`` when command is already feasible.
        """
        v, omega = command
        return bool(
            self.min_velocity <= v <= self.max_velocity
            and -self.max_angular_speed <= omega <= self.max_angular_speed
        )

    def project(self, command: Command2D) -> Command2D:
        """Clip command to bicycle velocity and angular limits.

        Returns:
            Command2D: Projected command in feasible set.
        """
        v, omega = command
        return (
            float(np.clip(v, self.min_velocity, self.max_velocity)),
            float(np.clip(omega, -self.max_angular_speed, self.max_angular_speed)),
        )

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
            max_velocity=max_velocity,
            max_angular_speed=max_angular,
            allow_backwards=bool(limits.get("allow_backwards", False)),
        )
    if kinematics in {"holonomic", "omni", "omnidirectional"}:
        return HolonomicPassthroughKinematicsModel()
    max_linear = float(limits.get("max_linear_speed", limits.get("v_max", 2.0)))
    max_angular = float(limits.get("max_angular_speed", limits.get("omega_max", 1.0)))
    return DifferentialDriveKinematicsModel(
        max_linear_speed=max_linear,
        max_angular_speed=max_angular,
        allow_backwards=bool(limits.get("allow_backwards", False)),
    )
