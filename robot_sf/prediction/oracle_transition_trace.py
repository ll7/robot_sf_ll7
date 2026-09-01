"""Strict oracle-only transition traces for pedestrian goal-belief research.

These records are labels and upper-bound diagnostics.  They are deliberately separate
from :mod:`robot_sf.prediction.goal_belief_contract`, whose constructor accepts only
typed actor observations.
"""

# The package supports Python 3.11, while the repository-wide Ruff target is
# Python 3.12. Keep the TypeVar spelling below until the minimum is raised.
# ruff: noqa: UP047

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

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

EnumT = TypeVar("EnumT", bound=StrEnum)


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


class ForceOperationKind(StrEnum):
    """How a force stage changes the preceding force value."""

    NOT_APPLIED = "not_applied"
    ADDITIVE = "additive"
    REPLACEMENT = "replacement"
    TRANSFORMED = "transformed"
    UNKNOWN = "unknown"


class ExactInverseReason(StrEnum):
    """Reason a transition cannot support an exact inverse-force label."""

    HOLD_VELOCITY_RESET = "hold_velocity_reset"
    RESPAWN_REPOSITION = "respawn_reposition"
    POPULATION_CHANGE = "population_change"
    UNMODELED_CONTROLLER_MUTATION = "unmodeled_controller_mutation"
    FORCE_STAGE_UNINSTRUMENTED = "force_stage_uninstrumented"
    ROBOT_FORCE_STATE_UNAVAILABLE = "robot_force_state_unavailable"
    SPEED_CAP_UNKNOWN = "speed_cap_unknown"
    OTHER = "other"


def _parse_enum(enum_type: type[EnumT], value: Any, field_name: str) -> EnumT:
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


def _sum_xy(vectors: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Sum finite two-dimensional vectors.

    Returns:
        The component-wise vector sum.
    """
    return (sum(vector[0] for vector in vectors), sum(vector[1] for vector in vectors))


def _add_xy(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    """Add two two-dimensional vectors.

    Returns:
        The component-wise vector sum.
    """
    return (left[0] + right[0], left[1] + right[1])


@dataclass(frozen=True, slots=True)
class RobotForceState:
    """One robot state snapshot available to pedestrian-force evaluation."""

    robot_index: int
    position_xy: tuple[float, float]
    heading_rad: float
    velocity_xy: tuple[float, float] | None = None
    angular_velocity_rad_s: float | None = None

    def __post_init__(self) -> None:
        """Validate the finite robot state without introducing a robot identity claim."""
        object.__setattr__(self, "robot_index", require_step_index(self.robot_index, "robot_index"))
        object.__setattr__(self, "position_xy", require_xy(self.position_xy, "position_xy"))
        object.__setattr__(self, "heading_rad", require_finite(self.heading_rad, "heading_rad"))
        if self.velocity_xy is not None:
            object.__setattr__(self, "velocity_xy", require_xy(self.velocity_xy, "velocity_xy"))
        if self.angular_velocity_rad_s is not None:
            object.__setattr__(
                self,
                "angular_velocity_rad_s",
                require_finite(self.angular_velocity_rad_s, "angular_velocity_rad_s"),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the force-time robot state."""
        return {
            "robot_index": self.robot_index,
            "position_xy": list(self.position_xy),
            "heading_rad": self.heading_rad,
            "velocity_xy": list(self.velocity_xy) if self.velocity_xy is not None else None,
            "angular_velocity_rad_s": self.angular_velocity_rad_s,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RobotForceState:
        """Parse one strict force-time robot state.

        Returns:
            A validated robot state snapshot.
        """
        allowed = {
            "robot_index",
            "position_xy",
            "heading_rad",
            "velocity_xy",
            "angular_velocity_rad_s",
        }
        reject_unknown_keys(value, allowed, "robot_force_state")
        if set(value) != allowed:
            raise ValueError("robot_force_state is missing a required field")
        return cls(
            robot_index=value["robot_index"],
            position_xy=require_xy(value["position_xy"], "position_xy"),
            heading_rad=value["heading_rad"],
            velocity_xy=_optional_xy(value["velocity_xy"], "velocity_xy"),
            angular_velocity_rad_s=value["angular_velocity_rad_s"],
        )


@dataclass(frozen=True, slots=True)
class ForceTimeRobotState:
    """Immutable robot-state collection supplied to the force stage."""

    robot_states: tuple[RobotForceState, ...] = ()

    def __post_init__(self) -> None:
        """Validate row identity and canonical ordering for the force-time snapshot."""
        states = tuple(self.robot_states)
        if any(type(state) is not RobotForceState for state in states):
            raise TypeError("robot_states must contain RobotForceState values")
        indices = [state.robot_index for state in states]
        if len(set(indices)) != len(indices):
            raise ValueError("robot_states must not duplicate robot_index")
        if indices != sorted(indices):
            raise ValueError("robot_states must be ordered by robot_index")
        object.__setattr__(self, "robot_states", states)

    def to_dict(self) -> dict[str, Any]:
        """Return the force-time robot-state payload."""
        return {"robot_states": [state.to_dict() for state in self.robot_states]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ForceTimeRobotState:
        """Parse a strict force-time robot-state payload.

        Returns:
            A validated force-time robot-state collection.
        """
        reject_unknown_keys(value, {"robot_states"}, "force_time_robot_state")
        if set(value) != {"robot_states"}:
            raise ValueError("force_time_robot_state is missing robot_states")
        raw_states = value["robot_states"]
        if not isinstance(raw_states, Sequence) or isinstance(raw_states, (str, bytes)):
            raise TypeError("force_time_robot_state.robot_states must be an array")
        return cls(
            tuple(
                RobotForceState.from_dict(_mapping(item, "robot_force_state"))
                for item in raw_states
            )
        )


@dataclass(frozen=True, slots=True)
class ControllerMutationFlags:
    """Controller mutations that can invalidate a raw finite-difference inverse label."""

    goal_redirected: bool = False
    hold_velocity_reset: bool = False
    respawn_reposition: bool = False
    population_changed: bool = False
    controller_jump_modelled: bool = False

    def __post_init__(self) -> None:
        """Require explicit boolean mutation flags."""
        for field_name in (
            "goal_redirected",
            "hold_velocity_reset",
            "respawn_reposition",
            "population_changed",
            "controller_jump_modelled",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be bool")

    def to_dict(self) -> dict[str, Any]:
        """Return controller mutation provenance."""
        return {
            "goal_redirected": self.goal_redirected,
            "hold_velocity_reset": self.hold_velocity_reset,
            "respawn_reposition": self.respawn_reposition,
            "population_changed": self.population_changed,
            "controller_jump_modelled": self.controller_jump_modelled,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ControllerMutationFlags:
        """Parse strict controller mutation flags.

        Returns:
            Validated controller mutation flags.
        """
        allowed = {
            "goal_redirected",
            "hold_velocity_reset",
            "respawn_reposition",
            "population_changed",
            "controller_jump_modelled",
        }
        reject_unknown_keys(value, allowed, "controller_mutation_flags")
        if set(value) != allowed:
            raise ValueError("controller_mutation_flags is missing a required field")
        return cls(**{field_name: value[field_name] for field_name in allowed})


@dataclass(frozen=True, slots=True)
class ForceStageResult:
    """Operation and typed result for a post-registry force stage."""

    operation_kind: ForceOperationKind = ForceOperationKind.NOT_APPLIED
    operation: str | None = None
    result_force_xy: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Validate stage semantics without manufacturing absent force values."""
        if not isinstance(self.operation_kind, ForceOperationKind):
            raise TypeError("operation_kind must be ForceOperationKind")
        if self.operation is not None:
            require_text(self.operation, "operation")
        if self.result_force_xy is not None:
            object.__setattr__(
                self, "result_force_xy", require_xy(self.result_force_xy, "result_force_xy")
            )
        if self.operation_kind is ForceOperationKind.NOT_APPLIED and (
            self.operation is not None or self.result_force_xy is not None
        ):
            raise ValueError("not_applied force stages must omit operation and result")
        if self.operation_kind is not ForceOperationKind.NOT_APPLIED and self.operation is None:
            raise ValueError("applied force stages must name their operation")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe force-stage result."""
        return {
            "operation_kind": self.operation_kind.value,
            "operation": self.operation,
            "result_force_xy": (
                list(self.result_force_xy) if self.result_force_xy is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ForceStageResult:
        """Parse a strict force-stage result.

        Returns:
            A validated force-stage operation and result.
        """
        allowed = {"operation_kind", "operation", "result_force_xy"}
        reject_unknown_keys(value, allowed, "force_stage_result")
        if set(value) != allowed:
            raise ValueError("force_stage_result is missing a required field")
        return cls(
            operation_kind=_parse_enum(
                ForceOperationKind, value["operation_kind"], "operation_kind"
            ),
            operation=value["operation"],
            result_force_xy=_optional_xy(value["result_force_xy"], "result_force_xy"),
        )


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
    force_time_robot_state: ForceTimeRobotState | None = None
    mutation_flags: ControllerMutationFlags | None = None

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
        if self.boundary is TransitionBoundaryKind.POST_BEHAVIOR_PRE_FORCE:
            if self.force_time_robot_state is None:
                raise ValueError("post_behavior_pre_force requires force_time_robot_state")
            if self.mutation_flags is None:
                raise ValueError("post_behavior_pre_force requires mutation_flags")
        elif self.force_time_robot_state is not None or self.mutation_flags is not None:
            raise ValueError(
                "force-time state and mutation flags belong to post_behavior_pre_force"
            )

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
            "force_time_robot_state": (
                self.force_time_robot_state.to_dict()
                if self.force_time_robot_state is not None
                else None
            ),
            "mutation_flags": (
                self.mutation_flags.to_dict() if self.mutation_flags is not None else None
            ),
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
            "force_time_robot_state",
            "mutation_flags",
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
            force_time_robot_state=(
                None
                if value["force_time_robot_state"] is None
                else ForceTimeRobotState.from_dict(
                    _mapping(value["force_time_robot_state"], "force_time_robot_state")
                )
            ),
            mutation_flags=(
                None
                if value["mutation_flags"] is None
                else ControllerMutationFlags.from_dict(
                    _mapping(value["mutation_flags"], "mutation_flags")
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class ForceComponents:
    """Typed force stages; ``None`` means the instrumentation is unavailable."""

    social_force_xy: tuple[float, float] | None = None
    goal_force_xy: tuple[float, float] | None = None
    obstacle_force_xy: tuple[float, float] | None = None
    pedestrian_robot_force_xy: tuple[float, float] | None = None
    group_force_xy: tuple[float, float] | None = None
    registry_total_force_xy: tuple[float, float] | None = None
    residual_operation: ForceStageResult = field(default_factory=ForceStageResult)
    model_variant_operation: ForceStageResult = field(default_factory=ForceStageResult)
    final_pre_cap_force_xy: tuple[float, float] | None = None
    uncapped_velocity_xy: tuple[float, float] | None = None
    applied_velocity_xy: tuple[float, float] | None = None

    def __post_init__(self) -> None:  # noqa: C901
        """Validate components and any available vector-sum invariant."""
        for field_name in (
            "social_force_xy",
            "goal_force_xy",
            "obstacle_force_xy",
            "pedestrian_robot_force_xy",
            "group_force_xy",
            "registry_total_force_xy",
            "final_pre_cap_force_xy",
            "uncapped_velocity_xy",
            "applied_velocity_xy",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, require_xy(value, field_name))
        if type(self.residual_operation) is not ForceStageResult:
            raise TypeError("residual_operation must be ForceStageResult")
        if type(self.model_variant_operation) is not ForceStageResult:
            raise TypeError("model_variant_operation must be ForceStageResult")
        nominal_components = (
            self.social_force_xy,
            self.goal_force_xy,
            self.obstacle_force_xy,
            self.pedestrian_robot_force_xy,
            self.group_force_xy,
        )
        if self.registry_total_force_xy is not None and all(
            component is not None for component in nominal_components
        ):
            nominal_sum = _sum_xy(
                tuple(component for component in nominal_components if component is not None)
            )
            if not all(
                math.isclose(nominal_sum[index], self.registry_total_force_xy[index], abs_tol=1e-9)
                for index in (0, 1)
            ):
                raise ValueError("registry_total_force_xy must equal the nominal component sum")

        expected_final = self.registry_total_force_xy
        if (
            expected_final is not None
            and self.residual_operation.operation_kind is ForceOperationKind.ADDITIVE
            and self.residual_operation.result_force_xy is not None
        ):
            expected_final = _add_xy(expected_final, self.residual_operation.result_force_xy)
        if (
            expected_final is not None
            and self.model_variant_operation.operation_kind is ForceOperationKind.ADDITIVE
            and self.model_variant_operation.result_force_xy is not None
        ):
            expected_final = _add_xy(expected_final, self.model_variant_operation.result_force_xy)
        elif (
            expected_final is not None
            and self.model_variant_operation.operation_kind
            in {ForceOperationKind.REPLACEMENT, ForceOperationKind.TRANSFORMED}
            and self.model_variant_operation.result_force_xy is not None
        ):
            expected_final = self.model_variant_operation.result_force_xy
        if (
            expected_final is not None
            and self.final_pre_cap_force_xy is not None
            and self.residual_operation.operation_kind is not ForceOperationKind.UNKNOWN
            and self.model_variant_operation.operation_kind is not ForceOperationKind.UNKNOWN
            and not all(
                math.isclose(
                    expected_final[index], self.final_pre_cap_force_xy[index], abs_tol=1e-9
                )
                for index in (0, 1)
            )
        ):
            raise ValueError("final_pre_cap_force_xy does not match the recorded force stages")

    def to_dict(self) -> dict[str, Any]:
        """Return all typed force stages with unavailable values as JSON null."""
        return {
            field_name: list(value) if (value := getattr(self, field_name)) is not None else None
            for field_name in (
                "social_force_xy",
                "goal_force_xy",
                "obstacle_force_xy",
                "pedestrian_robot_force_xy",
                "group_force_xy",
                "registry_total_force_xy",
                "final_pre_cap_force_xy",
                "uncapped_velocity_xy",
                "applied_velocity_xy",
            )
        } | {
            "residual_operation": self.residual_operation.to_dict(),
            "model_variant_operation": self.model_variant_operation.to_dict(),
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
            "group_force_xy",
            "registry_total_force_xy",
            "residual_operation",
            "model_variant_operation",
            "final_pre_cap_force_xy",
            "uncapped_velocity_xy",
            "applied_velocity_xy",
        }
        reject_unknown_keys(value, allowed, "force_components")
        if set(value) != allowed:
            raise ValueError("force_components is missing a required field")
        return cls(
            social_force_xy=_optional_xy(value["social_force_xy"], "social_force_xy"),
            goal_force_xy=_optional_xy(value["goal_force_xy"], "goal_force_xy"),
            obstacle_force_xy=_optional_xy(value["obstacle_force_xy"], "obstacle_force_xy"),
            pedestrian_robot_force_xy=_optional_xy(
                value["pedestrian_robot_force_xy"], "pedestrian_robot_force_xy"
            ),
            group_force_xy=_optional_xy(value["group_force_xy"], "group_force_xy"),
            registry_total_force_xy=_optional_xy(
                value["registry_total_force_xy"], "registry_total_force_xy"
            ),
            residual_operation=ForceStageResult.from_dict(
                _mapping(value["residual_operation"], "residual_operation")
            ),
            model_variant_operation=ForceStageResult.from_dict(
                _mapping(value["model_variant_operation"], "model_variant_operation")
            ),
            final_pre_cap_force_xy=_optional_xy(
                value["final_pre_cap_force_xy"], "final_pre_cap_force_xy"
            ),
            uncapped_velocity_xy=_optional_xy(
                value["uncapped_velocity_xy"], "uncapped_velocity_xy"
            ),
            applied_velocity_xy=_optional_xy(value["applied_velocity_xy"], "applied_velocity_xy"),
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
    """Oracle speed-cap truth, distinct from actor-side cap uncertainty."""

    status: SpeedCapStatus
    max_speed_mps: float | None = None
    uncapped_speed_mps: float | None = None
    applied_speed_mps: float | None = None

    def __post_init__(self) -> None:
        """Validate speed-cap provenance."""
        if not isinstance(self.status, SpeedCapStatus):
            raise TypeError("status must be SpeedCapStatus")
        if self.max_speed_mps is not None:
            object.__setattr__(
                self, "max_speed_mps", require_non_negative(self.max_speed_mps, "max_speed_mps")
            )
        for field_name in ("uncapped_speed_mps", "applied_speed_mps"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, require_non_negative(value, field_name))
        if self.status is SpeedCapStatus.APPLIED and self.max_speed_mps is None:
            raise ValueError("applied speed cap requires max_speed_mps")
        if (
            self.status is SpeedCapStatus.APPLIED
            and self.applied_speed_mps is not None
            and self.max_speed_mps is not None
            and self.applied_speed_mps > self.max_speed_mps + 1e-9
        ):
            raise ValueError("applied speed must not exceed max_speed_mps")

    @property
    def speed_cap_active(self) -> bool | None:
        """Return exact cap truth, or ``None`` when the simulator did not expose it."""
        return {
            SpeedCapStatus.NOT_APPLIED: False,
            SpeedCapStatus.APPLIED: True,
            SpeedCapStatus.UNKNOWN: None,
        }[self.status]

    def to_dict(self) -> dict[str, Any]:
        """Return speed-cap provenance."""
        return {
            "status": self.status.value,
            "speed_cap_active": self.speed_cap_active,
            "max_speed_mps": self.max_speed_mps,
            "uncapped_speed_mps": self.uncapped_speed_mps,
            "applied_speed_mps": self.applied_speed_mps,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SpeedCap:
        """Parse strict speed-cap provenance.

        Returns:
            A validated speed-cap record.
        """
        allowed = {
            "status",
            "speed_cap_active",
            "max_speed_mps",
            "uncapped_speed_mps",
            "applied_speed_mps",
        }
        reject_unknown_keys(value, allowed, "speed_cap")
        if set(value) != allowed:
            raise ValueError("speed_cap is missing a required field")
        status = _parse_enum(SpeedCapStatus, value["status"], "speed_cap.status")
        expected_active = {
            SpeedCapStatus.NOT_APPLIED: False,
            SpeedCapStatus.APPLIED: True,
            SpeedCapStatus.UNKNOWN: None,
        }[status]
        if value["speed_cap_active"] is not expected_active:
            raise ValueError("speed_cap_active does not match speed_cap.status")
        return cls(
            status=status,
            max_speed_mps=value["max_speed_mps"],
            uncapped_speed_mps=value["uncapped_speed_mps"],
            applied_speed_mps=value["applied_speed_mps"],
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
    exact_inverse_eligible: bool = False
    exact_inverse_reasons: tuple[ExactInverseReason, ...] = (
        ExactInverseReason.FORCE_STAGE_UNINSTRUMENTED,
    )
    timing_provenance: str = SIMULATOR_TIMING_PROVENANCE
    reset_provenance: str | None = None
    schema_version: str = field(default=ORACLE_TRANSITION_TRACE_SCHEMA_VERSION)

    def __post_init__(self) -> None:  # noqa: C901, PLR0912, PLR0915
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
        if type(self.exact_inverse_eligible) is not bool:
            raise TypeError("exact_inverse_eligible must be bool")
        reasons = tuple(self.exact_inverse_reasons)
        if any(type(reason) is not ExactInverseReason for reason in reasons):
            raise TypeError("exact_inverse_reasons must contain ExactInverseReason values")
        if len(set(reasons)) != len(reasons):
            raise ValueError("exact_inverse_reasons must be unique")
        reasons = tuple(sorted(reasons, key=lambda reason: reason.value))
        if self.exact_inverse_eligible and reasons:
            raise ValueError("eligible inverse traces must not carry ineligibility reasons")
        if not self.exact_inverse_eligible and not reasons:
            raise ValueError("ineligible inverse traces must name at least one reason")
        object.__setattr__(self, "exact_inverse_reasons", reasons)
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
        mutations = self.post_behavior_pre_force.mutation_flags
        assert mutations is not None
        required_reasons = set()
        if mutations.hold_velocity_reset:
            required_reasons.add(ExactInverseReason.HOLD_VELOCITY_RESET)
        if mutations.respawn_reposition:
            required_reasons.add(ExactInverseReason.RESPAWN_REPOSITION)
        if mutations.population_changed:
            required_reasons.add(ExactInverseReason.POPULATION_CHANGE)
        if required_reasons and not mutations.controller_jump_modelled:
            if self.exact_inverse_eligible:
                raise ValueError(
                    "controller mutations require an explicit jump model before inverse eligibility"
                )
            if not required_reasons.issubset(reasons):
                raise ValueError(
                    "ineligible inverse trace must explain each unmodeled controller mutation"
                )
        if self.exact_inverse_eligible:
            if any(
                value is None
                for value in (
                    self.force_components.registry_total_force_xy,
                    self.force_components.final_pre_cap_force_xy,
                    self.force_components.uncapped_velocity_xy,
                    self.force_components.applied_velocity_xy,
                )
            ):
                raise ValueError(
                    "eligible inverse traces require complete force and velocity stages"
                )
            if self.speed_cap.status is SpeedCapStatus.UNKNOWN:
                raise ValueError("eligible inverse traces require oracle speed-cap truth")
            if any(
                stage.operation_kind is ForceOperationKind.UNKNOWN
                for stage in (
                    self.force_components.residual_operation,
                    self.force_components.model_variant_operation,
                )
            ):
                raise ValueError("eligible inverse traces require known force-stage operations")

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
            "exact_inverse_eligible": self.exact_inverse_eligible,
            "exact_inverse_reasons": [reason.value for reason in self.exact_inverse_reasons],
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
            "exact_inverse_eligible",
            "exact_inverse_reasons",
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
            exact_inverse_eligible=value["exact_inverse_eligible"],
            exact_inverse_reasons=tuple(
                _parse_enum(ExactInverseReason, reason, "exact_inverse_reasons[]")
                for reason in value["exact_inverse_reasons"]
            ),
            reset_provenance=value["reset_provenance"],
        )


__all__ = [
    "ORACLE_TRANSITION_TRACE_SCHEMA_VERSION",
    "SIMULATOR_TIMING_ORDER",
    "SIMULATOR_TIMING_PROVENANCE",
    "ControllerMutationFlags",
    "DynamicsParameters",
    "ExactInverseReason",
    "ForceComponents",
    "ForceOperationKind",
    "ForceStageResult",
    "ForceTimeRobotState",
    "GoalChangeKind",
    "OracleTransitionTraceV1",
    "RobotForceState",
    "SpeedCap",
    "SpeedCapStatus",
    "TransitionBoundary",
    "TransitionBoundaryKind",
    "validate_simulator_timing_order",
]
