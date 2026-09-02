"""Additive ``core_contract.v1`` aliases and simulation-state value objects.

This module is an integration boundary, not a second implementation of the
domain contracts.  Existing frame, pose, observation, force, transition, and
episode classes are re-exported by identity.  Only the small missing values
(``ActorState`` plus the time/twist primitives) are defined here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Final, TypeAlias

from robot_sf.adversarial.config import Pose2D
from robot_sf.benchmark.types import EpisodeRecord
from robot_sf.prediction.oracle_transition_trace import (
    ForceComponentRecord,
    ForceComponents,
    OracleTransitionTraceV1,
)
from robot_sf.sensor.pedestrian_tracking import (
    PedestrianCoordinateFrame,
    PedestrianObservationSnapshot,
)

from . import CORE_CONTRACT_VERSION
from .time import SimTime, Twist2D

ActorId: TypeAlias = str  # noqa: UP040 - Python 3.11 support
TrackId: TypeAlias = str  # noqa: UP040 - Python 3.11 support
WorldFrame = PedestrianCoordinateFrame
ObservationSnapshot = PedestrianObservationSnapshot
ForceComponent = ForceComponentRecord
ForceBreakdown = ForceComponents
TransitionRecord = OracleTransitionTraceV1

CORE_FIELD_UNITS: dict[str, str] = {
    "pose.x": "m",
    "pose.y": "m",
    "pose.theta": "rad",
    "twist.vx": "m/s",
    "twist.vy": "m/s",
    "twist.omega": "rad/s",
    "time.seconds": "s",
}

# This is deliberately a tuple of stage names rather than a runtime pipeline.
# Consumers can type and document fixed-step boundaries without implying that
# this additive contract owns simulator execution or stage transitions.
DT_DECOMPOSITION_STAGE_ORDER: Final[tuple[str, ...]] = (
    "start_of_step_state",
    "post_behaviour_pedestrian_state",
    "force_evaluation_state",
    "component_forces",
    "final_pre_cap_force",
    "uncapped_velocity",
    "applied_capped_velocity",
    "integrated_state",
    "observation",
    "recorded_transition",
)


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


def _required_text(value: Any, field_name: str) -> str:
    """Normalize one non-empty identity/provenance string.

    Returns:
        str: The stripped identity or provenance value.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _parse_frame(value: Any) -> WorldFrame:
    """Parse one existing coordinate-frame enum without introducing an alias.

    Returns:
        WorldFrame: The validated existing frame enum member.
    """

    candidate = getattr(value, "value", value)
    try:
        return WorldFrame(candidate)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(frame.value for frame in WorldFrame)
        raise ValueError(f"coordinate_frame must be one of: {allowed}") from exc


def _parse_pose(value: Any) -> Pose2D:
    """Validate and normalize an existing ``Pose2D`` value.

    Returns:
        Pose2D: A finite value of the existing pose type.
    """

    if not isinstance(value, Pose2D):
        raise TypeError("pose must be the existing Pose2D contract")
    return Pose2D(
        _finite_float(value.x, "pose.x"),
        _finite_float(value.y, "pose.y"),
        _finite_float(value.theta, "pose.theta"),
    )


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """Require a mapping for a serialized nested value.

    Returns:
        Mapping[str, Any]: The validated mapping.
    """

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _strict_keys(value: Mapping[str, Any], expected: set[str], field_name: str) -> None:
    """Reject unknown or missing versioned fields."""

    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{field_name} keys mismatch: missing={missing}, extra={extra}")


@dataclass(frozen=True, slots=True)
class ActorState:
    """One frame-explicit actor state at a simulation decision point.

    ``pose`` is in metres/radians, ``twist`` is in metres per second and
    radians per second, and ``time`` carries both the discrete step and elapsed
    simulation seconds. ``actor_id`` is the source identity; ``track_id`` is an
    optional observation-track identity and is deliberately kept separate.
    ``valid`` describes state usability, while ``source_identity`` records
    where the state came from.  No simulator/oracle identity is inferred.
    """

    actor_id: ActorId
    pose: Pose2D
    twist: Twist2D
    time: SimTime
    track_id: TrackId | None = None
    coordinate_frame: WorldFrame | str = WorldFrame.GLOBAL_XY
    valid: bool = True
    source_identity: str = "unspecified"

    schema_version: ClassVar[str] = CORE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        """Validate identity, frame, finite values, and validity semantics."""

        object.__setattr__(self, "actor_id", _required_text(self.actor_id, "actor_id"))
        if self.track_id is not None:
            object.__setattr__(self, "track_id", _required_text(self.track_id, "track_id"))
        object.__setattr__(self, "pose", _parse_pose(self.pose))
        if type(self.twist) is not Twist2D:
            raise TypeError("twist must be Twist2D")
        if type(self.time) is not SimTime:
            raise TypeError("time must be SimTime")
        object.__setattr__(self, "coordinate_frame", _parse_frame(self.coordinate_frame))
        if type(self.valid) is not bool:
            raise TypeError("valid must be a bool")
        object.__setattr__(
            self,
            "source_identity",
            _required_text(self.source_identity, "source_identity"),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActorState:
        """Parse a strict, versioned actor state.

        Returns:
            ActorState: The validated actor state.
        """

        mapping = _mapping(value, "actor_state")
        expected = {
            "schema_version",
            "actor_id",
            "track_id",
            "pose",
            "twist",
            "time",
            "coordinate_frame",
            "valid",
            "source_identity",
        }
        _strict_keys(mapping, expected, "actor_state")
        if mapping["schema_version"] != cls.schema_version:
            raise ValueError(f"schema_version must be {cls.schema_version!r}")
        pose_mapping = _mapping(mapping["pose"], "pose")
        _strict_keys(pose_mapping, {"x", "y", "theta"}, "pose")
        return cls(
            actor_id=mapping["actor_id"],
            track_id=mapping["track_id"],
            pose=Pose2D(
                x=pose_mapping["x"],
                y=pose_mapping["y"],
                theta=pose_mapping["theta"],
            ),
            twist=Twist2D.from_dict(_mapping(mapping["twist"], "twist")),
            time=SimTime.from_dict(_mapping(mapping["time"], "time")),
            coordinate_frame=mapping["coordinate_frame"],
            valid=mapping["valid"],
            source_identity=mapping["source_identity"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, versioned actor-state record."""

        return {
            "schema_version": self.schema_version,
            "actor_id": self.actor_id,
            "track_id": self.track_id,
            "pose": {
                "x": self.pose.x,
                "y": self.pose.y,
                "theta": self.pose.theta,
            },
            "twist": self.twist.to_dict(),
            "time": self.time.to_dict(),
            "coordinate_frame": self.coordinate_frame.value,
            "valid": self.valid,
            "source_identity": self.source_identity,
        }

    @property
    def frame(self) -> WorldFrame:
        """Return the declared world frame."""

        return self.coordinate_frame

    @property
    def position_xy(self) -> tuple[float, float]:
        """Return the planar position in metres."""

        return (self.pose.x, self.pose.y)

    @property
    def velocity_xy(self) -> tuple[float, float]:
        """Return the planar velocity in metres per second."""

        return self.twist.velocity_xy

    @property
    def step_index(self) -> int:
        """Return the discrete decision-point step."""

        return self.time.step_index

    @property
    def timestamp_s(self) -> float:
        """Return elapsed simulation time in seconds."""

        return self.time.seconds


__all__ = [
    "CORE_CONTRACT_VERSION",
    "CORE_FIELD_UNITS",
    "DT_DECOMPOSITION_STAGE_ORDER",
    "ActorId",
    "ActorState",
    "EpisodeRecord",
    "ForceBreakdown",
    "ForceComponent",
    "ObservationSnapshot",
    "Pose2D",
    "SimTime",
    "TrackId",
    "TransitionRecord",
    "Twist2D",
    "WorldFrame",
]
