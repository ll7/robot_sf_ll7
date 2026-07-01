"""Selectable kinematic robot dynamics models.

This module intentionally covers kinematic motion only. Tire slip, load transfer,
and lean dynamics are deferred to a higher-fidelity model family.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, cos, sin, tan
from typing import TYPE_CHECKING, Protocol

from robot_sf.common.math_utils import clip_scalar, wrap_angle_pi

if TYPE_CHECKING:
    from collections.abc import Sequence

KINEMATIC_DYNAMICS_NAMES = (
    "holonomic_disc",
    "differential_drive",
    "unicycle",
    "kinematic_bicycle",
)


@dataclass(frozen=True)
class RobotDynamicsState:
    """Pose and velocity state shared by the issue-3976 kinematic models."""

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    linear_speed: float = 0.0
    angular_speed: float = 0.0
    steering_angle: float = 0.0

    @property
    def pose(self) -> tuple[tuple[float, float], float]:
        """Return the existing repository robot-pose representation."""

        return ((self.x, self.y), self.heading)


class RobotDynamicsModel(Protocol):
    """Common stepping interface for kinematic robot dynamics models."""

    name: str

    def step(
        self,
        state: RobotDynamicsState,
        control: Sequence[float],
        dt: float,
    ) -> RobotDynamicsState:
        """Advance ``state`` by ``dt`` seconds using ``control``."""


def _positive_dt(dt: float) -> float:
    value = float(dt)
    if value <= 0:
        raise ValueError("dt must be positive.")
    return value


def _control_pair(control: Sequence[float], model_name: str) -> tuple[float, float]:
    if len(control) != 2:
        raise ValueError(f"{model_name} control must contain exactly two values.")
    return float(control[0]), float(control[1])


@dataclass(frozen=True)
class HolonomicDiscDynamics:
    """Holonomic disc model with world-frame ``(vx, vy)`` velocity control."""

    max_speed: float = 2.0
    name: str = "holonomic_disc"

    def __post_init__(self) -> None:
        """Validate holonomic-disc velocity bounds."""

        if self.max_speed <= 0:
            raise ValueError("max_speed must be positive.")

    def step(
        self,
        state: RobotDynamicsState,
        control: Sequence[float],
        dt: float,
    ) -> RobotDynamicsState:
        """Advance a holonomic-disc state from world-frame velocity.

        Returns:
            Updated robot dynamics state.
        """

        dt = _positive_dt(dt)
        vx, vy = _control_pair(control, self.name)
        speed = (vx * vx + vy * vy) ** 0.5
        if speed > self.max_speed:
            scale = self.max_speed / max(speed, 1e-12)
            vx *= scale
            vy *= scale
            speed = self.max_speed
        heading = state.heading if speed <= 1e-12 else atan2(vy, vx)
        return replace(
            state,
            x=state.x + vx * dt,
            y=state.y + vy * dt,
            heading=wrap_angle_pi(heading),
            linear_speed=speed,
            angular_speed=0.0,
            steering_angle=0.0,
        )


@dataclass(frozen=True)
class DifferentialDriveDynamics:
    """Differential-drive model with left/right wheel angular-velocity control."""

    wheel_radius: float = 0.05
    track_width: float = 0.3
    max_wheel_angular_speed: float = 40.0
    name: str = "differential_drive"

    def __post_init__(self) -> None:
        """Validate differential-drive geometry and wheel-speed bounds."""

        if self.wheel_radius <= 0:
            raise ValueError("wheel_radius must be positive.")
        if self.track_width <= 0:
            raise ValueError("track_width must be positive.")
        if self.max_wheel_angular_speed <= 0:
            raise ValueError("max_wheel_angular_speed must be positive.")

    def step(
        self,
        state: RobotDynamicsState,
        control: Sequence[float],
        dt: float,
    ) -> RobotDynamicsState:
        """Advance a differential-drive state from wheel angular velocities.

        Returns:
            Updated robot dynamics state.
        """

        dt = _positive_dt(dt)
        left, right = _control_pair(control, self.name)
        left = clip_scalar(left, -self.max_wheel_angular_speed, self.max_wheel_angular_speed)
        right = clip_scalar(right, -self.max_wheel_angular_speed, self.max_wheel_angular_speed)
        linear_speed = self.wheel_radius * (left + right) / 2.0
        angular_speed = self.wheel_radius * (right - left) / self.track_width
        midpoint_heading = state.heading + angular_speed * dt / 2.0
        return replace(
            state,
            x=state.x + linear_speed * cos(midpoint_heading) * dt,
            y=state.y + linear_speed * sin(midpoint_heading) * dt,
            heading=wrap_angle_pi(state.heading + angular_speed * dt),
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            steering_angle=0.0,
        )


@dataclass(frozen=True)
class UnicycleDynamics:
    """Unicycle model with ``(linear_speed, angular_speed)`` velocity control."""

    max_linear_speed: float = 2.0
    max_angular_speed: float = 1.0
    name: str = "unicycle"

    def __post_init__(self) -> None:
        """Validate unicycle velocity bounds."""

        if self.max_linear_speed <= 0:
            raise ValueError("max_linear_speed must be positive.")
        if self.max_angular_speed <= 0:
            raise ValueError("max_angular_speed must be positive.")

    def step(
        self,
        state: RobotDynamicsState,
        control: Sequence[float],
        dt: float,
    ) -> RobotDynamicsState:
        """Advance a unicycle state from linear and angular velocity.

        Returns:
            Updated robot dynamics state.
        """

        dt = _positive_dt(dt)
        linear_speed, angular_speed = _control_pair(control, self.name)
        linear_speed = clip_scalar(linear_speed, -self.max_linear_speed, self.max_linear_speed)
        angular_speed = clip_scalar(angular_speed, -self.max_angular_speed, self.max_angular_speed)
        midpoint_heading = state.heading + angular_speed * dt / 2.0
        return replace(
            state,
            x=state.x + linear_speed * cos(midpoint_heading) * dt,
            y=state.y + linear_speed * sin(midpoint_heading) * dt,
            heading=wrap_angle_pi(state.heading + angular_speed * dt),
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            steering_angle=0.0,
        )


@dataclass(frozen=True)
class KinematicBicycleDynamics:
    """Kinematic bicycle model with ``(linear_speed, steering_angle)`` control."""

    wheelbase: float = 1.0
    max_linear_speed: float = 3.0
    max_steering_angle: float = 0.78
    name: str = "kinematic_bicycle"

    def __post_init__(self) -> None:
        """Validate kinematic-bicycle geometry and command bounds."""

        if self.wheelbase <= 0:
            raise ValueError("wheelbase must be positive.")
        if self.max_linear_speed <= 0:
            raise ValueError("max_linear_speed must be positive.")
        if self.max_steering_angle <= 0:
            raise ValueError("max_steering_angle must be positive.")

    def step(
        self,
        state: RobotDynamicsState,
        control: Sequence[float],
        dt: float,
    ) -> RobotDynamicsState:
        """Advance a kinematic-bicycle state from speed and steering angle.

        Returns:
            Updated robot dynamics state.
        """

        dt = _positive_dt(dt)
        linear_speed, steering_angle = _control_pair(control, self.name)
        linear_speed = clip_scalar(linear_speed, -self.max_linear_speed, self.max_linear_speed)
        steering_angle = clip_scalar(
            steering_angle,
            -self.max_steering_angle,
            self.max_steering_angle,
        )
        angular_speed = linear_speed * tan(steering_angle) / self.wheelbase
        return replace(
            state,
            x=state.x + linear_speed * cos(state.heading) * dt,
            y=state.y + linear_speed * sin(state.heading) * dt,
            heading=wrap_angle_pi(state.heading + angular_speed * dt),
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            steering_angle=steering_angle,
        )


def build_robot_dynamics(name: str, **kwargs: float) -> RobotDynamicsModel:
    """Create a kinematic dynamics model by stable issue-3976 name.

    Returns:
        Robot dynamics model implementing the common ``step`` interface.
    """

    normalized = str(name).strip().lower()
    if normalized == "holonomic_disc":
        return HolonomicDiscDynamics(**kwargs)
    if normalized == "differential_drive":
        return DifferentialDriveDynamics(**kwargs)
    if normalized == "unicycle":
        return UnicycleDynamics(**kwargs)
    if normalized == "kinematic_bicycle":
        return KinematicBicycleDynamics(**kwargs)
    raise ValueError(
        f"Unknown robot dynamics model {name!r}; expected one of {KINEMATIC_DYNAMICS_NAMES}."
    )


__all__ = [
    "KINEMATIC_DYNAMICS_NAMES",
    "DifferentialDriveDynamics",
    "HolonomicDiscDynamics",
    "KinematicBicycleDynamics",
    "RobotDynamicsModel",
    "RobotDynamicsState",
    "UnicycleDynamics",
    "build_robot_dynamics",
]
