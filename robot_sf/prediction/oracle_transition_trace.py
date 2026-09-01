"""Strict oracle-only transition traces for pedestrian goal-belief research.

These records are labels and upper-bound diagnostics.  They are deliberately separate
from :mod:`robot_sf.prediction.goal_belief_contract`, whose constructor accepts only
typed actor observations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from robot_sf.prediction._contract_utils import (
    canonical_json,
    reject_unknown_keys,
    require_finite,
    require_non_negative,
    require_step_index,
    require_text,
    require_xy,
    stable_digest,
)

ORACLE_TRANSITION_TRACE_SCHEMA_VERSION = "oracle_transition_trace.v1"
SIMULATOR_TIMING_PROVENANCE = "robot_sf.sim.simulator.Simulator.step_once"
SIMULATOR_TIMING_ORDER = (
    "state_t",
    "behavior.step",
    "goal_after_behavior_t",
    "compute_force_t",
    "apply_model_variant_or_residual",
    "integrate",
    "state_t+1",
)


class TransitionBoundaryKind(StrEnum):
    """State boundary captured by an oracle transition."""

    PRE_BEHAVIOR = "pre_behavior"
    POST_BEHAVIOR_PRE_FORCE = "post_behavior_pre_force"
    POST_INTEGRATION = "post_integration"


class GoalChangeKind(StrEnum):
    """Typed goal/route transition label."""

    NONE = "none"
    WAYPOINT_ADVANCE = "waypoint_advance"
    REDIRECT = "redirect"
    ARRIVAL = "arrival"
    RESPAWN = "respawn"
    STOP = "stop"
    RESTART = "restart"
    UNKNOWN = "unknown"


class SpeedCapStatus(StrEnum):
    """Whether the pedestrian speed cap was applied for this transition."""

    NOT_APPLIED = "not_applied"
    APPLIED = "applied"
    UNKNOWN = "unknown"


def _parse_enum(enum_type: type[StrEnum], value: Any, field_name: str) -> StrEnum:
    """Parse an external enum value with a field-specific failure.

    Returns:
        The parsed enum member.
    """
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """Require an object for a nested trace record.

    Returns:
        The validated mapping.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return value


def _optional_xy(value: Any, field_name: str) -> tuple[float, float] | None:
    """Parse a nullable finite two-dimensional vector.

    Returns:
        A finite vector or ``None`` when the field is unavailable.
    """
    return None if value is None else require_xy(value, field_name)


def validate_simulator_timing_order(observed: Sequence[str]) -> tuple[str, ...]:
    """Require the exact order implemented by ``Simulator.step_once``.

    Returns:
        The validated timing-order tuple.
    """
    actual = tuple(observed)
    if actual != SIMULATOR_TIMING_ORDER:
        raise ValueError(
            "simulator timing order mismatch: "
            f"expected {SIMULATOR_TIMING_ORDER!r}, observed {actual!r}"
        )
    return actual


@dataclass(frozen=True, slots=True)
class TransitionBoundary:
    """Typed pedestrian state at one simulator transition boundary."""

    boundary: TransitionBoundaryKind
    timestamp_s: float
    step_index: int
    position_xy: tuple[float, float]
    velocity_xy: tuple[float, float]
    active_goal_xy: tuple[float, float]
    route_waypoint_index: int | None = None
    goal_threshold_reached: bool | None = None

    def __post_init__(self) -> None:
        """Validate finite state and boundary-specific identity."""
        if not isinstance(self.boundary, TransitionBoundaryKind):
            raise TypeError("boundary must be TransitionBoundaryKind")
        object.__setattr__(self, "timestamp_s", require_finite(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "step_index", require_step_index(self.step_index, "step_index"))
        object.__setattr__(self, "position_xy", require_xy(self.position_xy, "position_xy"))
        object.__setattr__(self, "velocity_xy", require_xy(self.velocity_xy, "velocity_xy"))
        object.__setattr__(
            self, "active_goal_xy", require_xy(self.active_goal_xy, "active_goal_xy")
        )
        if self.route_waypoint_index is not None:
            object.__setattr__(
                self,
                "route_waypoint_index",
                require_step_index(self.route_waypoint_index, "route_waypoint_index"),
            )
        if (
            self.goal_threshold_reached is not None
            and type(self.goal_threshold_reached) is not bool
        ):
            raise TypeError("goal_threshold_reached must be bool or None")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe transition boundary."""
        return {
            "boundary": self.boundary.value,
            "timestamp_s": self.timestamp_s,
            "step_index": self.step_index,
            "position_xy": list(self.position_xy),
            "velocity_xy": list(self.velocity_xy),
            "active_goal_xy": list(self.active_goal_xy),
            "route_waypoint_index": self.route_waypoint_index,
            "goal_threshold_reached": self.goal_threshold_reached,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TransitionBoundary:
        """Parse a strict transition boundary.

        Returns:
            A validated transition boundary.
        """
        allowed = {
            "boundary",
            "timestamp_s",
            "step_index",
            "position_xy",
            "velocity_xy",
            "active_goal_xy",
            "route_waypoint_index",
            "goal_threshold_reached",
        }
        reject_unknown_keys(value, allowed, "transition_boundary")
        if set(value) != allowed:
            raise ValueError("transition_boundary is missing a required field")
        return cls(
            boundary=_parse_enum(TransitionBoundaryKind, value["boundary"], "boundary"),
            timestamp_s=value["timestamp_s"],
            step_index=value["step_index"],
            position_xy=require_xy(value["position_xy"], "position_xy"),
            velocity_xy=require_xy(value["velocity_xy"], "velocity_xy"),
            active_goal_xy=require_xy(value["active_goal_xy"], "active_goal_xy"),
            route_waypoint_index=value["route_waypoint_index"],
            goal_threshold_reached=value["goal_threshold_reached"],
        )


@dataclass(frozen=True, slots=True)
class ForceComponents:
    """Typed force-stage placeholders; ``None`` means not instrumented."""

    social_force_xy: tuple[float, float] | None = None
    goal_force_xy: tuple[float, float] | None = None
    obstacle_force_xy: tuple[float, float] | None = None
    pedestrian_robot_force_xy: tuple[float, float] | None = None
    residual_force_xy: tuple[float, float] | None = None
    total_force_xy: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Validate each optional force vector without manufacturing placeholders."""
        for field_name in (
            "social_force_xy",
            "goal_force_xy",
            "obstacle_force_xy",
            "pedestrian_robot_force_xy",
            "residual_force_xy",
            "total_force_xy",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, require_xy(value, field_name))

    def to_dict(self) -> dict[str, Any]:
        """Return all typed force stages with unavailable values as JSON null."""
        return {
            field_name: list(value) if (value := getattr(self, field_name)) is not None else None
            for field_name in (
                "social_force_xy",
                "goal_force_xy",
                "obstacle_force_xy",
                "pedestrian_robot_force_xy",
                "residual_force_xy",
                "total_force_xy",
            )
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ForceComponents:
        """Parse strict typed force placeholders.

        Returns:
            A validated force-stage record.
        """
        allowed = {
            "social_force_xy",
            "goal_force_xy",
            "obstacle_force_xy",
            "pedestrian_robot_force_xy",
            "residual_force_xy",
            "total_force_xy",
        }
        reject_unknown_keys(value, allowed, "force_components")
        if set(value) != allowed:
            raise ValueError("force_components is missing a required field")
        return cls(
            **{field_name: _optional_xy(value[field_name], field_name) for field_name in allowed}
        )


@dataclass(frozen=True, slots=True)
class DynamicsParameters:
    """Simulator parameters needed to interpret an oracle force label."""

    preferred_speed_mps: float | None = None
    relaxation_time_s: float | None = None
    desired_force_factor: float | None = None
    goal_threshold_m: float | None = None
    goal_threshold_reached: bool | None = None

    def __post_init__(self) -> None:
        """Validate optional parameter fields; absent instrumentation stays explicit."""
        for field_name in (
            "preferred_speed_mps",
            "desired_force_factor",
            "goal_threshold_m",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, require_non_negative(value, field_name))
        if self.relaxation_time_s is not None:
            object.__setattr__(
                self,
                "relaxation_time_s",
                require_finite(self.relaxation_time_s, "relaxation_time_s"),
            )
            if self.relaxation_time_s <= 0.0:
                raise ValueError("relaxation_time_s must be positive")
        if (
            self.goal_threshold_reached is not None
            and type(self.goal_threshold_reached) is not bool
        ):
            raise TypeError("goal_threshold_reached must be bool or None")

    def to_dict(self) -> dict[str, Any]:
        """Return simulator parameter provenance."""
        return {
            "preferred_speed_mps": self.preferred_speed_mps,
            "relaxation_time_s": self.relaxation_time_s,
            "desired_force_factor": self.desired_force_factor,
            "goal_threshold_m": self.goal_threshold_m,
            "goal_threshold_reached": self.goal_threshold_reached,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DynamicsParameters:
        """Parse strict simulator parameter provenance.

        Returns:
            A validated dynamics-parameter record.
        """
        allowed = {
            "preferred_speed_mps",
            "relaxation_time_s",
            "desired_force_factor",
            "goal_threshold_m",
            "goal_threshold_reached",
        }
        reject_unknown_keys(value, allowed, "dynamics")
        if set(value) != allowed:
            raise ValueError("dynamics is missing a required field")
        return cls(**{field_name: value[field_name] for field_name in allowed})


@dataclass(frozen=True, slots=True)
class SpeedCap:
    """Speed-cap status and optional applied limit for one transition."""

    status: SpeedCapStatus
    max_speed_mps: float | None = None

    def __post_init__(self) -> None:
        """Validate speed-cap provenance."""
        if not isinstance(self.status, SpeedCapStatus):
            raise TypeError("status must be SpeedCapStatus")
        if self.max_speed_mps is not None:
            object.__setattr__(
                self, "max_speed_mps", require_non_negative(self.max_speed_mps, "max_speed_mps")
            )
        if self.status is SpeedCapStatus.APPLIED and self.max_speed_mps is None:
            raise ValueError("applied speed cap requires max_speed_mps")

    def to_dict(self) -> dict[str, Any]:
        """Return speed-cap provenance."""
        return {"status": self.status.value, "max_speed_mps": self.max_speed_mps}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SpeedCap:
        """Parse strict speed-cap provenance.

        Returns:
            A validated speed-cap record.
        """
        allowed = {"status", "max_speed_mps"}
        reject_unknown_keys(value, allowed, "speed_cap")
        if set(value) != allowed:
            raise ValueError("speed_cap is missing a required field")
        return cls(
            status=_parse_enum(SpeedCapStatus, value["status"], "speed_cap.status"),
            max_speed_mps=value["max_speed_mps"],
        )


@dataclass(frozen=True, slots=True)
class OracleTransitionTraceV1:
    """Immutable oracle trace with explicit pre/post behavior timing."""

    episode_id: str
    transition_id: str
    transition_step_index: int
    simulator_pedestrian_id: str
    actor_track_id: str | None
    backend: str
    pre_behavior: TransitionBoundary
    post_behavior_pre_force: TransitionBoundary
    post_integration: TransitionBoundary
    force_components: ForceComponents
    dynamics: DynamicsParameters
    speed_cap: SpeedCap
    goal_change_kind: GoalChangeKind
    timing_provenance: str = SIMULATOR_TIMING_PROVENANCE
    reset_provenance: str | None = None
    schema_version: str = field(default=ORACLE_TRANSITION_TRACE_SCHEMA_VERSION)

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Validate boundary sequence and keep oracle linkage explicit."""
        if self.schema_version != ORACLE_TRANSITION_TRACE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {ORACLE_TRANSITION_TRACE_SCHEMA_VERSION}")
        require_text(self.episode_id, "episode_id")
        require_text(self.transition_id, "transition_id")
        object.__setattr__(
            self,
            "transition_step_index",
            require_step_index(self.transition_step_index, "transition_step_index"),
        )
        require_text(self.simulator_pedestrian_id, "simulator_pedestrian_id")
        if self.actor_track_id is not None:
            require_text(self.actor_track_id, "actor_track_id")
        require_text(self.backend, "backend")
        require_text(self.timing_provenance, "timing_provenance")
        if type(self.pre_behavior) is not TransitionBoundary:
            raise TypeError("pre_behavior must be TransitionBoundary")
        if type(self.post_behavior_pre_force) is not TransitionBoundary:
            raise TypeError("post_behavior_pre_force must be TransitionBoundary")
        if type(self.post_integration) is not TransitionBoundary:
            raise TypeError("post_integration must be TransitionBoundary")
        if self.pre_behavior.boundary is not TransitionBoundaryKind.PRE_BEHAVIOR:
            raise ValueError("pre_behavior has the wrong boundary kind")
        if (
            self.post_behavior_pre_force.boundary
            is not TransitionBoundaryKind.POST_BEHAVIOR_PRE_FORCE
        ):
            raise ValueError("post_behavior_pre_force has the wrong boundary kind")
        if self.post_integration.boundary is not TransitionBoundaryKind.POST_INTEGRATION:
            raise ValueError("post_integration has the wrong boundary kind")
        if self.pre_behavior.step_index != self.transition_step_index:
            raise ValueError("pre_behavior step does not match transition_step_index")
        if self.post_behavior_pre_force.step_index != self.transition_step_index:
            raise ValueError("post_behavior_pre_force step does not match transition_step_index")
        if self.post_integration.step_index != self.transition_step_index + 1:
            raise ValueError("post_integration must be the next transition step")
        if not math.isclose(
            self.pre_behavior.timestamp_s,
            self.post_behavior_pre_force.timestamp_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("behavior update must not change transition timestamp")
        if self.post_integration.timestamp_s <= self.pre_behavior.timestamp_s:
            raise ValueError("post_integration must be later than pre_behavior")
        if not isinstance(self.force_components, ForceComponents):
            raise TypeError("force_components must be ForceComponents")
        if not isinstance(self.dynamics, DynamicsParameters):
            raise TypeError("dynamics must be DynamicsParameters")
        if not isinstance(self.speed_cap, SpeedCap):
            raise TypeError("speed_cap must be SpeedCap")
        if not isinstance(self.goal_change_kind, GoalChangeKind):
            raise TypeError("goal_change_kind must be GoalChangeKind")
        if self.goal_change_kind is GoalChangeKind.WAYPOINT_ADVANCE:
            if self.pre_behavior.active_goal_xy == self.post_behavior_pre_force.active_goal_xy:
                raise ValueError("waypoint_advance must change the active goal")
        elif (
            self.goal_change_kind is GoalChangeKind.NONE
            and self.pre_behavior.active_goal_xy != self.post_behavior_pre_force.active_goal_xy
        ):
            raise ValueError("goal_change_kind none cannot change the active goal")
        if self.reset_provenance is not None:
            require_text(self.reset_provenance, "reset_provenance")

    def to_dict(self) -> dict[str, Any]:
        """Return an oracle-only JSON payload with timing and identity provenance."""
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "transition_id": self.transition_id,
            "transition_step_index": self.transition_step_index,
            "simulator_pedestrian_id": self.simulator_pedestrian_id,
            "actor_track_id": self.actor_track_id,
            "backend": self.backend,
            "timing": {
                "provenance": self.timing_provenance,
                "order": list(SIMULATOR_TIMING_ORDER),
            },
            "pre_behavior": self.pre_behavior.to_dict(),
            "post_behavior_pre_force": self.post_behavior_pre_force.to_dict(),
            "post_integration": self.post_integration.to_dict(),
            "force_components": self.force_components.to_dict(),
            "dynamics": self.dynamics.to_dict(),
            "speed_cap": self.speed_cap.to_dict(),
            "goal_change_kind": self.goal_change_kind.value,
            "reset_provenance": self.reset_provenance,
        }

    def to_json(self) -> str:
        """Return RFC 8785 canonical oracle JSON."""
        return canonical_json(self.to_dict())

    @property
    def content_digest(self) -> str:
        """Return the deterministic SHA-256 digest of the oracle payload."""
        return stable_digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OracleTransitionTraceV1:
        """Parse a strict oracle trace and reject unknown versioned keys.

        Returns:
            A validated oracle transition trace.
        """
        allowed = {
            "schema_version",
            "episode_id",
            "transition_id",
            "transition_step_index",
            "simulator_pedestrian_id",
            "actor_track_id",
            "backend",
            "timing",
            "pre_behavior",
            "post_behavior_pre_force",
            "post_integration",
            "force_components",
            "dynamics",
            "speed_cap",
            "goal_change_kind",
            "reset_provenance",
        }
        reject_unknown_keys(value, allowed, "oracle_transition_trace")
        if set(value) != allowed:
            raise ValueError("oracle_transition_trace is missing a required field")
        timing = _mapping(value["timing"], "timing")
        reject_unknown_keys(timing, {"provenance", "order"}, "timing")
        if set(timing) != {"provenance", "order"}:
            raise ValueError("timing is missing a required field")
        validate_simulator_timing_order(timing["order"])
        return cls(
            schema_version=value["schema_version"],
            episode_id=value["episode_id"],
            transition_id=value["transition_id"],
            transition_step_index=value["transition_step_index"],
            simulator_pedestrian_id=value["simulator_pedestrian_id"],
            actor_track_id=value["actor_track_id"],
            backend=value["backend"],
            timing_provenance=timing["provenance"],
            pre_behavior=TransitionBoundary.from_dict(
                _mapping(value["pre_behavior"], "pre_behavior")
            ),
            post_behavior_pre_force=TransitionBoundary.from_dict(
                _mapping(value["post_behavior_pre_force"], "post_behavior_pre_force")
            ),
            post_integration=TransitionBoundary.from_dict(
                _mapping(value["post_integration"], "post_integration")
            ),
            force_components=ForceComponents.from_dict(
                _mapping(value["force_components"], "force_components")
            ),
            dynamics=DynamicsParameters.from_dict(_mapping(value["dynamics"], "dynamics")),
            speed_cap=SpeedCap.from_dict(_mapping(value["speed_cap"], "speed_cap")),
            goal_change_kind=_parse_enum(
                GoalChangeKind, value["goal_change_kind"], "goal_change_kind"
            ),
            reset_provenance=value["reset_provenance"],
        )


__all__ = [
    "ORACLE_TRANSITION_TRACE_SCHEMA_VERSION",
    "SIMULATOR_TIMING_ORDER",
    "SIMULATOR_TIMING_PROVENANCE",
    "DynamicsParameters",
    "ForceComponents",
    "GoalChangeKind",
    "OracleTransitionTraceV1",
    "SpeedCap",
    "SpeedCapStatus",
    "TransitionBoundary",
    "TransitionBoundaryKind",
    "validate_simulator_timing_order",
]
