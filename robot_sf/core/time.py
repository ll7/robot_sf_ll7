"""Finite, serializable time and planar-twist primitives for ``core_contract.v1``."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

CORE_TIME_SCHEMA_VERSION = "core_time.v1"


def _finite_float(value: Any, field_name: str) -> float:
    """Normalize one finite real value and reject booleans.

    Returns:
        float: The normalized finite value.
    """

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a real number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a real number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _step_index(value: Any, field_name: str = "step_index") -> int:
    """Normalize one non-negative discrete simulation step.

    Returns:
        int: The validated step index.
    """

    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """Require a mapping for a serialized nested value.

    Returns:
        Mapping[str, Any]: The validated mapping.
    """

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _strict_keys(value: Mapping[str, Any], expected: set[str], field_name: str) -> None:
    """Reject unknown or missing serialized fields."""

    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{field_name} keys mismatch: missing={missing}, extra={extra}")


@dataclass(frozen=True, slots=True)
class SimTime:
    """Discrete simulation time.

    ``step_index`` is a zero-based simulation step. ``seconds`` is elapsed
    simulation time in seconds and is expected to be derived from the active
    fixed step duration rather than wall-clock time.
    """

    step_index: int
    seconds: float

    schema_version: ClassVar[str] = CORE_TIME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate and normalize the time coordinates."""

        object.__setattr__(self, "step_index", _step_index(self.step_index))
        seconds = _finite_float(self.seconds, "seconds")
        if seconds < 0.0:
            raise ValueError("seconds must be non-negative")
        object.__setattr__(self, "seconds", seconds)

    @classmethod
    def from_step(cls, step_index: int, dt_s: float) -> SimTime:
        """Construct time from a step index and positive fixed step duration.

        Returns:
            SimTime: The corresponding elapsed simulation time.
        """

        step = _step_index(step_index)
        duration = _finite_float(dt_s, "dt_s")
        if duration <= 0.0:
            raise ValueError("dt_s must be positive")
        return cls(step, step * duration)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SimTime:
        """Parse a strict serialized time record.

        Returns:
            SimTime: The validated time value.
        """

        mapping = _mapping(value, "sim_time")
        _strict_keys(mapping, {"schema_version", "step_index", "seconds"}, "sim_time")
        if mapping["schema_version"] != cls.schema_version:
            raise ValueError(f"schema_version must be {cls.schema_version!r}")
        return cls(step_index=mapping["step_index"], seconds=mapping["seconds"])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe versioned time record."""

        return {
            "schema_version": self.schema_version,
            "step_index": self.step_index,
            "seconds": self.seconds,
        }

    def advance(self, dt_s: float) -> SimTime:
        """Advance one discrete step using a positive simulation duration.

        Returns:
            SimTime: The next decision-point time.
        """

        duration = _finite_float(dt_s, "dt_s")
        if duration <= 0.0:
            raise ValueError("dt_s must be positive")
        return SimTime(self.step_index + 1, self.seconds + duration)

    @property
    def time_s(self) -> float:
        """Return elapsed time using the repository's common ``*_s`` naming."""

        return self.seconds


@dataclass(frozen=True, slots=True)
class Twist2D:
    """Planar velocity in a declared world frame.

    ``vx`` and ``vy`` are signed linear velocities in metres per second and
    ``omega`` is signed angular velocity in radians per second.  The containing
    :class:`~robot_sf.core.contract.ActorState` declares the coordinate frame.
    """

    vx: float
    vy: float
    omega: float = 0.0

    schema_version: ClassVar[str] = CORE_TIME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate all velocity components as finite values."""

        for field_name in ("vx", "vy", "omega"):
            object.__setattr__(
                self, field_name, _finite_float(getattr(self, field_name), field_name)
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Twist2D:
        """Parse a strict serialized planar twist.

        Returns:
            Twist2D: The validated planar twist.
        """

        mapping = _mapping(value, "twist_2d")
        _strict_keys(mapping, {"schema_version", "vx", "vy", "omega"}, "twist_2d")
        if mapping["schema_version"] != cls.schema_version:
            raise ValueError(f"schema_version must be {cls.schema_version!r}")
        return cls(vx=mapping["vx"], vy=mapping["vy"], omega=mapping["omega"])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe versioned planar twist."""

        return {
            "schema_version": self.schema_version,
            "vx": self.vx,
            "vy": self.vy,
            "omega": self.omega,
        }

    @property
    def velocity_xy(self) -> tuple[float, float]:
        """Return the linear velocity in metres per second."""

        return (self.vx, self.vy)

    @property
    def angular_velocity_rad_s(self) -> float:
        """Return angular velocity in radians per second."""

        return self.omega


__all__ = ["CORE_TIME_SCHEMA_VERSION", "SimTime", "Twist2D"]
