"""Deterministic, generation-only maneuver portfolio for local arbitration.

The portfolio is deliberately a small adapter around the repository's existing
trajectory contracts.  It uses :class:`~robot_sf.research.collision_risk.CandidateAction`
for risk input, :class:`~robot_sf.robot.dynamics.UnicycleDynamics` for every rollout
step, :func:`~robot_sf.benchmark.actuator_feasibility.evaluate_actuator_feasibility`
for the canonical actuator gate, and the route geometry/local-goal helpers from
``robot_sf.nav.global_route``.  It does not estimate pedestrian risk, rank candidates,
or register a planner.

The executable control convention is the one used by ``UnicycleDynamics``:
``(linear_speed_mps, angular_speed_radps)``.  A candidate contains one command for
each interval and one state for each interval boundary.  The first command is
checked against the supplied initial state, so a returned candidate cannot hide an
unexecutable first step behind a clipped dynamics model.  The rollout applies each
command immediately; configured command/brake latency is reported as a diagnostic
assumption and is not simulated by this generation-only adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from shapely.geometry import LineString, Point

from robot_sf.benchmark.actuator_feasibility import (
    VERDICT_ACTUATOR_FEASIBLE,
    ActuatorLimitsConfig,
    evaluate_actuator_feasibility,
)
from robot_sf.common.math_utils import wrap_angle_pi
from robot_sf.nav.global_route import (
    AdaptiveLocalGoalConfig,
    RouteGeometry,
    RouteProjection,
    RouteProjectionHint,
)
from robot_sf.nav.occupancy import check_quality_of_map_point
from robot_sf.research.collision_risk import CandidateAction
from robot_sf.robot.dynamics import RobotDynamicsState, UnicycleDynamics

if TYPE_CHECKING:
    from robot_sf.nav.map_config import MapDefinition


MANEUVER_CANDIDATE_SCHEMA_VERSION = "maneuver_candidate.v1"
REJECTION_REASONS = (
    "invalid_initial_state",
    "invalid_route_projection",
    "actuator_limit",
    "static_collision",
    "no_left_corridor",
    "no_right_corridor",
    "static_geometry_unavailable",
    "pass_heading_ineligible",
    "pass_side_mismatch",
    "rollout_nonfinite",
    "candidate_cap",
)

_PASS_HEADING_TOLERANCE_RAD = math.pi / 3.0

Control = tuple[float, float]
"""Executable ``(linear_speed_mps, angular_speed_radps)`` command."""

JSONScalar = str | int | float | bool | None


class ManeuverId(StrEnum):
    """Stable semantic identifiers consumed by downstream arbitration."""

    ROUTE_FOLLOW = "route_follow"
    PASS_LEFT = "pass_left"
    PASS_RIGHT = "pass_right"
    YIELD_CREEP = "yield_creep"
    CONTROLLED_STOP = "controlled_stop"


class StaticVerifier(Protocol):
    """Protocol for a canonical static-geometry feasibility adapter.

    The callback returns ``True`` when the complete trajectory is statically
    feasible.  It may optionally return a mapping/object with ``feasible`` or
    ``blocked`` and ``min_clearance_m`` fields.  A malformed callback result is
    treated as blocked (fail closed).
    """

    def __call__(
        self,
        states: np.ndarray,
        *,
        robot_radius_m: float,
        clearance_margin_m: float,
    ) -> bool | Mapping[str, object]:
        """Return whether the complete state trajectory is statically feasible."""


@dataclass(frozen=True, slots=True)
class ManeuverPortfolioConfig:
    """Validated bounds for deterministic maneuver generation.

    The defaults match the risk estimator's standard ``dt``/horizon and the
    provisional actuator limits already used by the benchmark feasibility owner.
    ``actuator_limits`` can be replaced with a runtime-derived configuration;
    generation then fails closed if the duplicated motion bounds exceed it.
    """

    dt_s: float = 0.1
    horizon_steps: int = 20
    max_total_candidates: int = 12
    max_route_follow_candidates: int = 3
    max_pass_left_candidates: int = 2
    max_pass_right_candidates: int = 2
    max_yield_candidates: int = 1
    max_stop_candidates: int = 1
    route_heading_samples_rad: tuple[float, ...] = (-0.15, 0.0, 0.15)
    route_speed_samples_mps: tuple[float, ...] = (0.8, 1.0)
    pass_lateral_offsets_m: tuple[float, ...] = (0.75,)
    pass_rejoin_fraction: float = 0.65
    creep_speed_mps: float = 0.2
    yield_deceleration_mps2: float = 0.6
    stop_deceleration_mps2: float = 1.5
    max_linear_acceleration_mps2: float = 1.0
    max_angular_acceleration_radps2: float = 0.5
    max_jerk_mps3: float = 4.0
    max_angular_jerk_radps3: float | None = None
    max_linear_speed_mps: float = 2.0
    max_angular_speed_radps: float = 1.0
    static_clearance_margin_m: float = 0.05
    robot_radius_m: float = 0.3
    actuator_limits: ActuatorLimitsConfig = field(default_factory=ActuatorLimitsConfig)
    candidate_id_version: str = MANEUVER_CANDIDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Validate finite ranges and agreement with canonical actuator limits."""
        finite_positive = (
            "dt_s",
            "yield_deceleration_mps2",
            "stop_deceleration_mps2",
            "max_linear_acceleration_mps2",
            "max_angular_acceleration_radps2",
            "max_jerk_mps3",
            "max_linear_speed_mps",
            "max_angular_speed_radps",
        )
        for name in finite_positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.max_angular_jerk_radps3 is not None and (
            not math.isfinite(float(self.max_angular_jerk_radps3))
            or float(self.max_angular_jerk_radps3) <= 0.0
        ):
            raise ValueError("max_angular_jerk_radps3 must be finite and > 0")
        if (
            isinstance(self.horizon_steps, bool)
            or not isinstance(self.horizon_steps, int)
            or self.horizon_steps <= 0
        ):
            raise ValueError("horizon_steps must be a positive integer")
        for name in (
            "max_total_candidates",
            "max_route_follow_candidates",
            "max_pass_left_candidates",
            "max_pass_right_candidates",
            "max_yield_candidates",
            "max_stop_candidates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_total_candidates < 1 or self.max_stop_candidates < 1:
            raise ValueError("max_total_candidates and max_stop_candidates must be >= 1")
        if self.max_stop_candidates != 1:
            raise ValueError("max_stop_candidates must be exactly 1")
        if not self.candidate_id_version or not isinstance(self.candidate_id_version, str):
            raise ValueError("candidate_id_version must be a non-empty string")
        if not math.isfinite(float(self.creep_speed_mps)) or self.creep_speed_mps < 0.0:
            raise ValueError("creep_speed_mps must be finite and >= 0")
        for name in ("route_heading_samples_rad", "route_speed_samples_mps"):
            values = tuple(float(value) for value in getattr(self, name))
            if not values or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain finite values")
            object.__setattr__(self, name, values)
        offsets = tuple(float(value) for value in self.pass_lateral_offsets_m)
        if not offsets or not all(math.isfinite(value) and value > 0.0 for value in offsets):
            raise ValueError("pass_lateral_offsets_m must contain finite positive values")
        object.__setattr__(self, "pass_lateral_offsets_m", offsets)
        if (
            not math.isfinite(self.pass_rejoin_fraction)
            or not 0.0 < self.pass_rejoin_fraction <= 1.0
        ):
            raise ValueError("pass_rejoin_fraction must be finite and in (0, 1]")
        if (
            not math.isfinite(self.static_clearance_margin_m)
            or self.static_clearance_margin_m < 0.0
        ):
            raise ValueError("static_clearance_margin_m must be finite and >= 0")
        if not math.isfinite(self.robot_radius_m) or self.robot_radius_m <= 0.0:
            raise ValueError("robot_radius_m must be finite and > 0")
        # The local config may be more conservative than the canonical owner, but
        # silently exceeding a supplied canonical limit would make the candidate
        # executable only on paper.
        canonical = self.actuator_limits
        if self.max_linear_acceleration_mps2 > canonical.max_accel_mps2 + 1.0e-12:
            raise ValueError("max_linear_acceleration_mps2 exceeds actuator_limits.max_accel_mps2")
        if self.yield_deceleration_mps2 > canonical.max_decel_mps2 + 1.0e-12:
            raise ValueError("yield_deceleration_mps2 exceeds actuator_limits.max_decel_mps2")
        if self.stop_deceleration_mps2 > canonical.max_decel_mps2 + 1.0e-12:
            raise ValueError("stop_deceleration_mps2 exceeds actuator_limits.max_decel_mps2")
        if self.max_angular_speed_radps > canonical.max_yaw_rate_radps + 1.0e-12:
            raise ValueError("max_angular_speed_radps exceeds actuator_limits.max_yaw_rate_radps")
        if self.max_angular_acceleration_radps2 > canonical.max_steering_rate_radps + 1.0e-12:
            raise ValueError(
                "max_angular_acceleration_radps2 exceeds actuator_limits.max_steering_rate_radps"
            )
        for value in self.route_speed_samples_mps:
            if value < 0.0 or value > self.max_linear_speed_mps + 1.0e-12:
                raise ValueError("route_speed_samples_mps must be within linear speed bounds")
        if self.creep_speed_mps > self.max_linear_speed_mps + 1.0e-12:
            raise ValueError("creep_speed_mps exceeds max_linear_speed_mps")

    @property
    def angular_jerk_limit(self) -> float:
        """Return the configured angular jerk bound, defaulting to linear jerk."""

        return float(self.max_angular_jerk_radps3 or self.max_jerk_mps3)


@dataclass(frozen=True, slots=True)
class ManeuverCandidate:
    """One complete executable candidate and its semantic metadata."""

    candidate_id: str
    maneuver: ManeuverId
    action: CandidateAction
    controls: tuple[Control, ...]
    states: np.ndarray
    generation_source: str
    generation_rank: int
    metadata: Mapping[str, JSONScalar]

    def __post_init__(self) -> None:
        """Freeze arrays and validate the wrapper's alignment contract."""
        if not self.candidate_id or not isinstance(self.candidate_id, str):
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(self.maneuver, ManeuverId):
            raise TypeError("maneuver must be a ManeuverId")
        if isinstance(self.generation_rank, bool) or self.generation_rank < 0:
            raise ValueError("generation_rank must be non-negative")
        controls = tuple((float(control[0]), float(control[1])) for control in self.controls)
        if not all(math.isfinite(value) for control in controls for value in control):
            raise ValueError("controls must be finite")
        array = _immutable_array(self.states)
        if array.ndim != 2 or array.shape[1] < 5 or array.shape[0] != len(controls) + 1:
            raise ValueError("states must have shape (len(controls) + 1, >=5)")
        if not np.all(np.isfinite(array)):
            raise ValueError("states must be finite")
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "states", array)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        waypoints = _immutable_array(self.action.waypoints)
        if waypoints.shape != (array.shape[0], 2) or not np.array_equal(waypoints, array[:, :2]):
            raise ValueError("action waypoints must exactly match state positions")
        object.__setattr__(
            self,
            "action",
            CandidateAction(
                action_id=self.action.action_id,
                waypoints=waypoints,
                representation=self.action.representation,
            ),
        )

    @property
    def first_control(self) -> Control:
        """Return the command emitted at the current planning step."""

        return self.controls[0]

    @property
    def dt_s(self) -> float:
        """Return the common timestep recorded in metadata."""

        return float(self.metadata["dt_s"])

    @property
    def horizon_steps(self) -> int:
        """Return the common horizon recorded in metadata."""

        return int(self.metadata["horizon_steps"])


@dataclass(frozen=True, slots=True)
class ManeuverGenerationReport:
    """Bounded diagnostics for one portfolio generation cycle."""

    schema_version: str
    candidate_set_id: str
    route_hash: str
    initial_robot_state: tuple[float, ...]
    local_goal: tuple[float, float] | None
    requested_maneuvers: tuple[str, ...]
    per_maneuver_attempted_count: Mapping[str, int]
    per_maneuver_valid_count: Mapping[str, int]
    per_reason_rejection_count: Mapping[str, int]
    ordered_candidate_ids: tuple[str, ...]
    candidate_first_controls: Mapping[str, Control]
    stop_complete_within_horizon: Mapping[str, bool]
    static_min_clearance_m: Mapping[str, float | None]
    generation_duration_ms: float
    truncated: bool
    truncation_rule: str | None
    route_projection_status: str

    def as_dict(self) -> dict[str, object]:
        """Return the report in a JSON-safe form."""

        return {
            "schema_version": self.schema_version,
            "candidate_set_id": self.candidate_set_id,
            "route_hash": self.route_hash,
            "initial_robot_state": list(self.initial_robot_state),
            "local_goal": list(self.local_goal) if self.local_goal is not None else None,
            "requested_maneuvers": list(self.requested_maneuvers),
            "per_maneuver_attempted_count": dict(self.per_maneuver_attempted_count),
            "per_maneuver_valid_count": dict(self.per_maneuver_valid_count),
            "per_reason_rejection_count": dict(self.per_reason_rejection_count),
            "ordered_candidate_ids": list(self.ordered_candidate_ids),
            "candidate_id_to_first_control": {
                key: list(value) for key, value in self.candidate_first_controls.items()
            },
            "candidate_id_to_stop_complete_within_horizon": dict(self.stop_complete_within_horizon),
            "candidate_id_to_static_min_clearance": dict(self.static_min_clearance_m),
            "generation_duration_ms": self.generation_duration_ms,
            "truncated": self.truncated,
            "truncation_rule": self.truncation_rule,
            "route_projection_status": self.route_projection_status,
        }


@dataclass(frozen=True, slots=True)
class ManeuverGenerationResult:
    """Candidates plus diagnostics, with sequence-like convenience methods."""

    candidates: tuple[ManeuverCandidate, ...]
    report: ManeuverGenerationReport

    def __iter__(self) -> Iterator[ManeuverCandidate]:
        """Iterate over candidates for callers that only need the portfolio.

        Returns:
            Iterator over retained candidates.
        """

        return iter(self.candidates)

    def __len__(self) -> int:
        """Return the number of retained candidates."""

        return len(self.candidates)

    def __getitem__(self, index: int) -> ManeuverCandidate:
        """Index retained candidates.

        Returns:
            Maneuver candidate at ``index``.
        """

        return self.candidates[index]


@dataclass(frozen=True, slots=True)
class _StaticCheck:
    feasible: bool
    min_clearance_m: float | None


@dataclass(frozen=True, slots=True)
class _Attempt:
    maneuver: ManeuverId
    source: str
    rank: int
    controls: tuple[Control, ...]
    states: np.ndarray
    metadata: Mapping[str, JSONScalar]


def _immutable_array(values: object) -> np.ndarray:
    """Return an ndarray backed by immutable bytes.

    ``setflags(write=False)`` alone protects ordinary callers but can be reversed
    when the array owns its memory.  A bytes-backed view makes the read-only
    contract durable for arrays exposed through candidate actions and states.
    """

    source = np.asarray(values, dtype=float)
    raw = np.ascontiguousarray(source, dtype=float).tobytes()
    result = np.frombuffer(raw, dtype=float).reshape(source.shape)
    return result


def _finite_state(
    initial_state: RobotDynamicsState | Sequence[float] | Mapping[str, object],
) -> RobotDynamicsState:
    """Coerce supported state forms and fail closed on non-finite values.

    Returns:
        Canonical finite robot dynamics state.
    """

    if isinstance(initial_state, RobotDynamicsState):
        values = (
            initial_state.x,
            initial_state.y,
            initial_state.heading,
            initial_state.linear_speed,
            initial_state.angular_speed,
            initial_state.steering_angle,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("initial_state must be finite")
        if initial_state.linear_speed < 0.0:
            raise ValueError("initial_state.linear_speed must be non-negative")
        return initial_state
    if isinstance(initial_state, Mapping):
        try:
            state = RobotDynamicsState(
                x=float(initial_state["x"]),
                y=float(initial_state["y"]),
                heading=float(initial_state["heading"]),
                linear_speed=float(
                    initial_state.get("linear_speed", initial_state.get("speed", 0.0))
                ),
                angular_speed=float(
                    initial_state.get("angular_speed", initial_state.get("yaw_rate", 0.0))
                ),
                steering_angle=float(initial_state.get("steering_angle", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("initial_state mapping is missing finite state fields") from exc
        return _finite_state(state)
    array = np.asarray(initial_state, dtype=float).reshape(-1)
    if array.size not in (5, 6) or not np.all(np.isfinite(array)):
        raise ValueError("initial_state must contain 5 or 6 finite values")
    state = RobotDynamicsState(
        x=float(array[0]),
        y=float(array[1]),
        heading=float(array[2]),
        linear_speed=float(array[3]),
        angular_speed=float(array[4]),
        steering_angle=float(array[5]) if array.size == 6 else 0.0,
    )
    return _finite_state(state)


def _route_tangent(route: RouteGeometry, projection: RouteProjection) -> tuple[np.ndarray, float]:
    """Return the directed route tangent and heading for a valid projection."""

    if not projection.is_valid or projection.segment_index is None:
        raise ValueError("route projection is not valid")
    start, end = route.sections[projection.segment_index]
    tangent = np.asarray((end[0] - start[0], end[1] - start[1]), dtype=float)
    length = float(np.linalg.norm(tangent))
    if length <= 0.0 or not math.isfinite(length):
        raise ValueError("route tangent is not finite")
    tangent /= length
    return tangent, float(math.atan2(tangent[1], tangent[0]))


def _rate_limited_profile(
    initial: float,
    targets: Sequence[float],
    *,
    dt_s: float,
    max_accel: float,
    max_decel: float,
    max_jerk: float,
    lower: float,
    upper: float,
) -> tuple[float, ...]:
    """Build a command profile with bounded acceleration and jerk.

    The profile is a command sequence; the actual state rollout remains owned by
    ``UnicycleDynamics``.  A zero previous acceleration is used for the first
    command, which makes the initial transition explicit and deterministic.

    Returns:
        Bounded command values aligned with ``targets``.
    """

    value = float(initial)
    acceleration = 0.0
    result: list[float] = []
    jerk_step = float(max_jerk) * float(dt_s)
    for target_value in targets:
        target = float(np.clip(float(target_value), lower, upper))
        desired = (target - value) / dt_s
        bounded = float(np.clip(desired, -max_decel, max_accel))
        bounded = float(np.clip(bounded, acceleration - jerk_step, acceleration + jerk_step))
        bounded = float(np.clip(bounded, -max_decel, max_accel))
        # Respect the command bounds while retaining the jerk-limited transition.
        # In particular, never jump to the target with a larger acceleration just
        # because the target lies inside the next interval.  At a speed boundary,
        # reducing the final acceleration is the physically valid projection.
        bounded = float(
            np.clip(
                bounded,
                (lower - value) / dt_s,
                (upper - value) / dt_s,
            )
        )
        next_value = float(np.clip(value + bounded * dt_s, lower, upper))
        # Do not snap to an interior target after jerk limiting: that would turn
        # a small final acceleration into an instantaneous jerk discontinuity.
        # The next interval converges to the target through the same bounded
        # profile, while the command bounds remain hard.
        result.append(next_value)
        acceleration = (next_value - value) / dt_s
        value = next_value
    return tuple(result)


def _jerk_limited_braking_profile(  # noqa: C901
    initial: float,
    *,
    horizon_steps: int,
    dt_s: float,
    max_decel: float,
    max_jerk: float,
) -> tuple[float, ...]:
    """Return a non-negative braking profile with a jerk-safe stop transition.

    Braking starts from the observed speed and zero acceleration.  At each step
    the strongest admissible deceleration is selected subject to the speed that
    remains necessary to ramp acceleration back to zero.  The look-ahead is a
    one-dimensional kinematic bound, not a second rollout model; states are
    still produced exclusively by ``UnicycleDynamics``.
    """

    value = max(float(initial), 0.0)
    acceleration = 0.0
    jerk_step = float(max_jerk) * float(dt_s)
    speed_margin = max(1.0e-7, float(max_jerk) * float(dt_s) * float(dt_s) * 1.0e-3)
    result: list[float] = []

    def future_speed_need(candidate_accel: float) -> float:
        """Return speed lost while ramping acceleration back to zero.

        Returns:
            Speed consumed by the jerk-limited recovery ramp.
        """

        future_accel = float(candidate_accel)
        needed = 0.0
        for _ in range(256):
            future_accel = min(0.0, future_accel + jerk_step)
            needed += max(0.0, -future_accel) * dt_s
            if future_accel >= -1.0e-12:
                break
        return needed

    for _ in range(horizon_steps):
        if value <= 1.0e-12:
            value = 0.0
            acceleration = 0.0
            result.append(value)
            continue
        lower = max(-float(max_decel), acceleration - jerk_step)
        upper = min(0.0, acceleration + jerk_step)

        def feasible(current_value: float, candidate_accel: float) -> bool:
            next_value = current_value + candidate_accel * dt_s
            if next_value < -1.0e-12:
                return False
            needed = future_speed_need(candidate_accel)
            if next_value <= speed_margin:
                return next_value + 1.0e-12 >= needed
            return next_value + 1.0e-12 >= needed + speed_margin

        if not feasible(value, upper):
            # This should only occur for a round-off boundary case after the
            # look-ahead margin.  Keep the jerk-safe edge of the interval; a
            # velocity command may be held, but acceleration must not jump.
            candidate_accel = upper
        elif feasible(value, lower):
            candidate_accel = lower
        else:
            # Feasibility is monotone in acceleration over [lower, upper].
            lo, hi = lower, upper
            for _ in range(60):
                mid = (lo + hi) / 2.0
                if feasible(value, mid):
                    hi = mid
                else:
                    lo = mid
            candidate_accel = hi
        next_value = max(0.0, value + candidate_accel * dt_s)
        result.append(float(next_value))
        acceleration = (next_value - value) / dt_s
        value = next_value
    return tuple(result)


def _smooth_heading_targets(
    *,
    initial_heading: float,
    desired_heading: float,
    horizon_steps: int,
    dt_s: float,
) -> tuple[float, ...]:
    """Return a constant angular-velocity target that reaches a desired heading."""

    error = float(wrap_angle_pi(desired_heading - initial_heading))
    horizon_s = max(float(horizon_steps) * float(dt_s), float(dt_s))
    return (error / horizon_s,) * horizon_steps


def _smoothstep(value: float) -> float:
    """Return the cubic smoothstep value for a clipped fraction."""

    fraction = float(np.clip(value, 0.0, 1.0))
    return fraction * fraction * (3.0 - 2.0 * fraction)


def _pass_reference_controls(
    route: RouteGeometry,
    *,
    projection: RouteProjection,
    tangent: np.ndarray,
    lateral_offset_m: float,
    local_goal: tuple[float, float],
    config: ManeuverPortfolioConfig,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Build route-relative pass targets with an explicit lateral rejoin.

    The reference is only a geometric target.  It is converted to bounded
    velocity commands and then rolled out by ``UnicycleDynamics``.  The lateral
    profile rises to the requested side offset and returns to the route frame
    before the horizon ends, so pass trajectories have a real rejoin phase.

    Returns:
        Route-relative linear and angular target sequences.
    """

    if projection.arc_length_m is None:
        raise ValueError("invalid_route_projection")
    horizon = config.horizon_steps
    normal = np.asarray((-tangent[1], tangent[0]), dtype=float)
    initial_arc = float(projection.arc_length_m)
    local_goal_projection = route.project(local_goal)
    goal_arc = (
        float(local_goal_projection.arc_length_m)
        if local_goal_projection.is_valid and local_goal_projection.arc_length_m is not None
        else route.total_length_m
    )
    speed = max(float(config.creep_speed_mps), min(config.route_speed_samples_mps))
    progress = min(speed * config.dt_s * horizon, max(goal_arc - initial_arc, 0.0))
    # ``pass_rejoin_fraction`` is the portion reserved for returning to the
    # route, so the lateral offset reaches its peak before that phase begins.
    peak_fraction = float(np.clip(1.0 - config.pass_rejoin_fraction, 0.2, 0.8))
    reference = np.empty((horizon + 1, 2), dtype=float)
    for index in range(horizon + 1):
        fraction = index / max(horizon, 1)
        arc = min(initial_arc + progress * fraction, route.total_length_m)
        route_point = np.asarray(route.point_at_arc_length(arc), dtype=float)
        if fraction <= peak_fraction:
            side_fraction = _smoothstep(fraction / peak_fraction)
        else:
            side_fraction = 1.0 - _smoothstep(
                (fraction - peak_fraction) / max(1.0 - peak_fraction, 1.0e-9)
            )
        reference[index] = route_point + float(lateral_offset_m) * side_fraction * normal
    deltas = np.diff(reference, axis=0)
    target_speeds = tuple(float(np.linalg.norm(delta) / config.dt_s) for delta in deltas)
    headings = np.unwrap(np.arctan2(deltas[:, 1], deltas[:, 0]))
    previous_heading = float(np.arctan2(tangent[1], tangent[0]))
    target_angular_speeds: list[float] = []
    for heading in headings:
        target_angular_speeds.append(float((heading - previous_heading) / config.dt_s))
        previous_heading = float(heading)
    # The finite-difference reference naturally approaches the route tangent at
    # the final point and can therefore turn the command back toward the passing
    # side too early.  A bounded side-opposing bias keeps the post-peak phase a
    # genuine rejoin while the shared limiter enforces angular acceleration/jerk.
    rejoin_bias = 0.5 if lateral_offset_m >= 0.0 else -0.5
    for index in range(horizon):
        if index / max(horizon, 1) > peak_fraction:
            target_angular_speeds[index] -= rejoin_bias
    return target_speeds, tuple(target_angular_speeds)


def _rollout(
    initial_state: RobotDynamicsState,
    controls: tuple[Control, ...],
    *,
    dt_s: float,
    dynamics: UnicycleDynamics,
) -> np.ndarray:
    """Roll out controls through the canonical unicycle dynamics owner.

    Returns:
        State boundaries with one row for each command interval boundary.
    """

    states = np.empty((len(controls) + 1, 5), dtype=float)
    current = initial_state
    states[0] = (
        current.x,
        current.y,
        wrap_angle_pi(current.heading),
        current.linear_speed,
        current.angular_speed,
    )
    for index, control in enumerate(controls, start=1):
        current = dynamics.step(current, control, dt_s)
        states[index] = (
            current.x,
            current.y,
            wrap_angle_pi(current.heading),
            current.linear_speed,
            current.angular_speed,
        )
    if not np.all(np.isfinite(states)):
        raise ValueError("rollout_nonfinite")
    return states


def _invoke_static_verifier(  # noqa: C901
    verifier: Callable[..., object],
    states: np.ndarray,
    *,
    robot_radius_m: float,
    clearance_margin_m: float,
) -> _StaticCheck:
    """Invoke a static checker with a small compatibility adapter.

    Returns:
        Normalized feasibility and minimum-clearance result.
    """

    try:
        raw = verifier(
            states,
            robot_radius_m=robot_radius_m,
            clearance_margin_m=clearance_margin_m,
        )
    except TypeError:
        try:
            raw = verifier(states, robot_radius_m + clearance_margin_m)
        except TypeError:
            try:
                raw = verifier(states)
            except Exception:  # noqa: BLE001 - static checks fail closed.
                return _StaticCheck(False, None)
        except Exception:  # noqa: BLE001 - static checks fail closed.
            return _StaticCheck(False, None)
    except Exception:  # noqa: BLE001 - static checks fail closed.
        return _StaticCheck(False, None)

    min_clearance: float | None = None
    if isinstance(raw, (bool, np.bool_)):
        return _StaticCheck(bool(raw), None)
    if isinstance(raw, Mapping):
        feasible = raw.get("feasible")
        blocked = raw.get("blocked")
        if isinstance(raw.get("min_clearance_m"), (int, float)):
            value = float(raw["min_clearance_m"])
            min_clearance = value if math.isfinite(value) else None
        if isinstance(feasible, (bool, np.bool_)):
            return _StaticCheck(bool(feasible), min_clearance)
        if isinstance(blocked, (bool, np.bool_)):
            return _StaticCheck(not bool(blocked), min_clearance)
        return _StaticCheck(False, min_clearance)
    feasible = getattr(raw, "feasible", None)
    blocked = getattr(raw, "blocked", None)
    raw_clearance = getattr(raw, "min_clearance_m", None)
    if isinstance(raw_clearance, (int, float)) and math.isfinite(float(raw_clearance)):
        min_clearance = float(raw_clearance)
    if isinstance(feasible, (bool, np.bool_)):
        return _StaticCheck(bool(feasible), min_clearance)
    if isinstance(blocked, (bool, np.bool_)):
        return _StaticCheck(not bool(blocked), min_clearance)
    return _StaticCheck(False, min_clearance)


def _obstacle_verifier(  # noqa: C901
    static_geometry: object,
    *,
    robot_radius_m: float,
    clearance_margin_m: float,
) -> Callable[..., object]:
    """Adapt a ``MapDefinition`` or obstacle collection to a static verifier.

    Point checks are delegated to the repository's canonical occupancy helper;
    line checks use the same canonical circle/segment primitive.  The adapter is
    intentionally conservative and samples every state boundary, preserving the
    generation-only scope.

    Returns:
        Callback that checks complete trajectories against the supplied geometry.
    """

    map_like = hasattr(static_geometry, "width") and hasattr(static_geometry, "obstacles")
    map_definition = cast("MapDefinition", static_geometry) if map_like else None
    if map_like:
        obstacles = tuple(getattr(map_definition, "obstacles", ()))
    else:
        try:
            obstacles = tuple(static_geometry)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("static_geometry must be a MapDefinition or obstacle iterable") from exc

    # Use the canonical obstacle polygon where available.  For legacy line-only
    # obstacles, retain their exact line geometry rather than checking sampled
    # state points.  The swept LineString query below catches thin walls between
    # state boundaries and polygon interiors alike.
    geometries: list[object] = []
    for obstacle in obstacles:
        geometry = getattr(obstacle, "geometry", None)
        if geometry is not None and not getattr(geometry, "is_empty", True):
            geometries.append(geometry)
            continue
        obstacle_lines = getattr(obstacle, "lines", obstacle)
        for line in obstacle_lines:
            try:
                if len(line) == 4:
                    x1, x2, y1, y2 = (float(value) for value in line)
                    first, second = (x1, y1), (x2, y2)
                else:
                    first_raw, second_raw = line
                    first = (float(first_raw[0]), float(first_raw[1]))
                    second = (float(second_raw[0]), float(second_raw[1]))
                geometries.append(LineString((first, second)))
            except (TypeError, ValueError, IndexError):
                raise ValueError("static obstacle lines must be finite 2D segments") from None
    if map_definition is not None:
        for bound in getattr(map_definition, "bounds", ()):
            try:
                x1, x2, y1, y2 = (float(value) for value in bound)
                geometries.append(LineString(((x1, y1), (x2, y2))))
            except (TypeError, ValueError):
                continue

    def _swept_check(states: np.ndarray, **_: object) -> Mapping[str, object]:
        radius = float(robot_radius_m + clearance_margin_m)
        min_clearance = float("inf")
        if map_definition is not None:
            for row in states:
                if not check_quality_of_map_point(
                    map_definition,
                    (float(row[0]), float(row[1])),
                    radius,
                ):
                    return {"feasible": False, "min_clearance_m": 0.0}
        for index in range(states.shape[0]):
            point = Point(float(states[index, 0]), float(states[index, 1]))
            for geometry in geometries:
                min_clearance = min(min_clearance, float(point.distance(geometry) - radius))
            if index == 0:
                continue
            previous = states[index - 1, :2]
            current = states[index, :2]
            sweep = LineString(
                (
                    tuple(float(value) for value in previous),
                    tuple(float(value) for value in current),
                )
            )
            for geometry in geometries:
                min_clearance = min(min_clearance, float(sweep.distance(geometry) - radius))
        if not math.isfinite(min_clearance):
            min_clearance = None
        return {
            "feasible": min_clearance is None or min_clearance >= -1.0e-12,
            "min_clearance_m": min_clearance,
        }

    return _swept_check


def _candidate_id(config: ManeuverPortfolioConfig, maneuver: ManeuverId, rank: int) -> str:
    """Return a deterministic candidate identifier."""

    return f"{config.candidate_id_version}:{maneuver.value}:{rank}"


def _candidate_set_id(
    route: RouteGeometry,
    state: RobotDynamicsState,
    config: ManeuverPortfolioConfig,
    local_goal: tuple[float, float] | None,
    projection_hint: RouteProjectionHint | None = None,
) -> str:
    """Return a stable planning-cycle identity without process-local values."""

    payload = {
        "schema": config.candidate_id_version,
        "route_hash": route.route_hash,
        "state": [
            float(state.x),
            float(state.y),
            float(state.heading),
            float(state.linear_speed),
            float(state.angular_speed),
        ],
        "dt_s": float(config.dt_s),
        "horizon_steps": int(config.horizon_steps),
        "local_goal": list(local_goal) if local_goal is not None else None,
        "projection_hint": (
            {
                "previous_s_m": projection_hint.previous_s_m,
                "previous_segment_index": projection_hint.previous_segment_index,
                "max_forward_jump_m": projection_hint.max_forward_jump_m,
                "max_backtrack_m": projection_hint.max_backtrack_m,
            }
            if projection_hint is not None
            else None
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _build_controls(
    state: RobotDynamicsState,
    *,
    target_speeds: Sequence[float],
    target_angular_speeds: Sequence[float],
    config: ManeuverPortfolioConfig,
    linear_deceleration_mps2: float | None = None,
    braking_profile: bool = False,
) -> tuple[Control, ...]:
    """Build one aligned velocity command sequence with rate and jerk limits.

    Returns:
        Executable linear/angular velocity commands.
    """

    max_deceleration = min(
        float(linear_deceleration_mps2 or config.stop_deceleration_mps2),
        config.actuator_limits.max_decel_mps2,
    ) * (1.0 - 1.0e-9)
    max_acceleration = float(config.max_linear_acceleration_mps2) * (1.0 - 1.0e-9)
    if braking_profile:
        speeds = _jerk_limited_braking_profile(
            state.linear_speed,
            horizon_steps=len(target_speeds),
            dt_s=config.dt_s,
            max_decel=max_deceleration,
            max_jerk=config.max_jerk_mps3,
        )
    else:
        speeds = _rate_limited_profile(
            state.linear_speed,
            target_speeds,
            dt_s=config.dt_s,
            max_accel=max_acceleration,
            max_decel=max_deceleration,
            max_jerk=config.max_jerk_mps3,
            lower=0.0,
            upper=config.max_linear_speed_mps,
        )
    angular = _rate_limited_profile(
        state.angular_speed,
        target_angular_speeds,
        dt_s=config.dt_s,
        max_accel=float(config.max_angular_acceleration_radps2) * (1.0 - 1.0e-9),
        max_decel=float(config.max_angular_acceleration_radps2) * (1.0 - 1.0e-9),
        max_jerk=config.angular_jerk_limit,
        lower=-config.max_angular_speed_radps,
        upper=config.max_angular_speed_radps,
    )
    controls = tuple((float(v), float(w)) for v, w in zip(speeds, angular, strict=True))
    if not all(
        0.0 <= v <= config.max_linear_speed_mps + 1.0e-12
        and abs(w) <= config.max_angular_speed_radps + 1.0e-12
        for v, w in controls
    ):
        raise ValueError("actuator_limit")
    return controls


def _static_and_actuator_check(
    states: np.ndarray,
    *,
    static_verifier: Callable[..., object] | None,
    config: ManeuverPortfolioConfig,
) -> tuple[bool, str | None, float | None]:
    """Apply static and canonical actuator gates to one rollout.

    Returns:
        Feasibility, stable rejection reason, and optional clearance.
    """

    static_result = _StaticCheck(True, None)
    if static_verifier is not None:
        static_result = _invoke_static_verifier(
            static_verifier,
            states,
            robot_radius_m=config.robot_radius_m,
            clearance_margin_m=config.static_clearance_margin_m,
        )
        if not static_result.feasible:
            return False, "static_collision", static_result.min_clearance_m

    headings = states[:, 2]
    velocities = np.column_stack((states[:, 3] * np.cos(headings), states[:, 3] * np.sin(headings)))
    hazard_clearance = static_result.min_clearance_m
    if hazard_clearance is None:
        hazard_clearance = 1.0e6
    report = evaluate_actuator_feasibility(
        robot_positions=states[:, :2],
        robot_velocities=velocities,
        robot_headings=headings,
        robot_angular_velocities=states[:, 4],
        dt_s=config.dt_s,
        hazard_clearance_m=float(hazard_clearance),
        config=config.actuator_limits,
    )
    if report.verdict != VERDICT_ACTUATOR_FEASIBLE:
        return False, "actuator_limit", static_result.min_clearance_m
    # Jerk is not a field in the canonical feasibility schema, so retain this
    # one additional hard check for the explicit portfolio contract.
    linear_accel = np.diff(states[:, 3]) / config.dt_s
    angular_accel = np.diff(states[:, 4]) / config.dt_s
    if linear_accel.size:
        linear_jerk = np.diff(np.concatenate(([0.0], linear_accel))) / config.dt_s
        angular_jerk = np.diff(np.concatenate(([0.0], angular_accel))) / config.dt_s
        if np.max(np.abs(linear_jerk)) > config.max_jerk_mps3 + 1.0e-8:
            return False, "actuator_limit", static_result.min_clearance_m
        if np.max(np.abs(angular_jerk)) > config.angular_jerk_limit + 1.0e-8:
            return False, "actuator_limit", static_result.min_clearance_m
    return True, None, static_result.min_clearance_m


def _attempt_metadata(
    *,
    config: ManeuverPortfolioConfig,
    route: RouteGeometry,
    local_goal: tuple[float, float] | None,
    stop_complete: bool,
    **extra: JSONScalar,
) -> dict[str, JSONScalar]:
    """Build bounded JSON-safe candidate metadata.

    Returns:
        JSON-safe metadata mapping for one attempted candidate.
    """

    metadata: dict[str, JSONScalar] = {
        "schema_version": MANEUVER_CANDIDATE_SCHEMA_VERSION,
        "dt_s": float(config.dt_s),
        "horizon_steps": int(config.horizon_steps),
        "route_hash": route.route_hash,
        "local_goal_x": local_goal[0] if local_goal is not None else None,
        "local_goal_y": local_goal[1] if local_goal is not None else None,
        "stop_complete_within_horizon": bool(stop_complete),
        "generation_mode": "generation_only",
    }
    metadata.update(extra)
    return metadata


def _make_candidate(
    attempt: _Attempt,
    *,
    candidate_id: str,
    config: ManeuverPortfolioConfig,
) -> ManeuverCandidate:
    """Wrap a validated rollout in the canonical risk action.

    Returns:
        Immutable maneuver candidate compatible with the risk estimator.
    """

    waypoints = _immutable_array(attempt.states[:, :2])
    action = CandidateAction(
        action_id=candidate_id,
        waypoints=waypoints,
        representation="maneuver_portfolio",
    )
    action.as_array(horizon_steps=config.horizon_steps)
    return ManeuverCandidate(
        candidate_id=candidate_id,
        maneuver=attempt.maneuver,
        action=action,
        controls=attempt.controls,
        states=attempt.states,
        generation_source=attempt.source,
        generation_rank=attempt.rank,
        metadata=attempt.metadata,
    )


def generate_maneuver_candidates(  # noqa: C901, PLR0912, PLR0913, PLR0915
    route: RouteGeometry,
    initial_state: RobotDynamicsState | Sequence[float] | Mapping[str, object],
    *,
    config: ManeuverPortfolioConfig | None = None,
    local_goal: Sequence[float] | None = None,
    projection_hint: RouteProjectionHint | None = None,
    crowd_density: float = 0.0,
    uncertainty_m: float = 0.0,
    adaptive_goal_config: AdaptiveLocalGoalConfig | None = None,
    static_verifier: StaticVerifier | Callable[..., object] | None = None,
    static_geometry: object | None = None,
    static_blocked: Callable[[np.ndarray], bool] | None = None,
) -> ManeuverGenerationResult:
    """Generate a bounded deterministic portfolio of maneuver candidates.

    Args:
        route: Canonical ordered route geometry from ``robot_sf.nav.global_route``.
        initial_state: Canonical ``RobotDynamicsState`` or finite five/six-value
            ``(x, y, heading, linear_speed, angular_speed[, steering_angle])``.
        config: Portfolio bounds and canonical actuator configuration.
        local_goal: Optional explicit route-relative goal.  When omitted, the
            canonical adaptive local-goal helper selects one.
        projection_hint: Optional route-branch continuity hint for the initial
            projection and adaptive local-goal selection.
        crowd_density: Context consumed only by adaptive local-goal selection.
        uncertainty_m: Context consumed only by adaptive local-goal selection.
        adaptive_goal_config: Optional canonical local-goal bounds.
        static_verifier: Callback returning ``True`` for statically feasible
            trajectories, optionally with minimum-clearance diagnostics.
        static_geometry: Optional ``MapDefinition`` or obstacle iterable adapted
            through canonical occupancy circle/segment checks.
        static_blocked: Optional canonical occupancy callback returning ``True``
            when a trajectory is blocked.  It is adapted to the verifier contract.

    Returns:
        :class:`ManeuverGenerationResult` containing retained candidates and a
        JSON-safe generation report.  The result is generation-only: no ranking or
        planner registration occurs.
    """

    started = time.perf_counter()
    cfg = config if config is not None else ManeuverPortfolioConfig()
    try:
        if not isinstance(route, RouteGeometry):
            raise TypeError("route must be a RouteGeometry")
        state = _finite_state(initial_state)
    except (TypeError, ValueError):
        # A malformed initial state cannot yield an executable stop.  Preserve a
        # structured report by returning an empty result rather than fabricating data.
        route_hash = getattr(route, "route_hash", "invalid")
        report = ManeuverGenerationReport(
            schema_version=MANEUVER_CANDIDATE_SCHEMA_VERSION,
            candidate_set_id="invalid",
            route_hash=route_hash,
            initial_robot_state=(),
            local_goal=None,
            requested_maneuvers=tuple(item.value for item in ManeuverId),
            per_maneuver_attempted_count=MappingProxyType({}),
            per_maneuver_valid_count=MappingProxyType({}),
            per_reason_rejection_count=MappingProxyType({"invalid_initial_state": 1}),
            ordered_candidate_ids=(),
            candidate_first_controls=MappingProxyType({}),
            stop_complete_within_horizon=MappingProxyType({}),
            static_min_clearance_m=MappingProxyType({}),
            generation_duration_ms=(time.perf_counter() - started) * 1000.0,
            truncated=False,
            truncation_rule=None,
            route_projection_status="invalid_initial_state",
        )
        return ManeuverGenerationResult((), report)

    projection = route.project((state.x, state.y), hint=projection_hint)
    projection_status = projection.status
    tangent: np.ndarray | None = None
    route_heading: float | None = None
    if projection.is_valid:
        try:
            tangent, route_heading = _route_tangent(route, projection)
        except ValueError:
            projection_status = "invalid_route"

    selected_goal: tuple[float, float] | None
    if local_goal is None:
        goal_result = route.adaptive_local_goal(
            (state.x, state.y),
            speed_m_s=state.linear_speed,
            crowd_density=float(crowd_density),
            uncertainty_m=float(uncertainty_m),
            config=adaptive_goal_config,
            hint=projection_hint,
        )
        selected_goal = (
            tuple(float(value) for value in goal_result.point) if goal_result.is_valid else None
        )
        if not goal_result.is_valid and projection_status == "ok":
            projection_status = str(goal_result.status)
    else:
        goal_array = np.asarray(local_goal, dtype=float).reshape(-1)
        if goal_array.size != 2 or not np.all(np.isfinite(goal_array)):
            selected_goal = None
            projection_status = "invalid_input"
        else:
            selected_goal = (float(goal_array[0]), float(goal_array[1]))

    static_checker: Callable[..., object] | None
    if static_verifier is not None:
        static_checker = static_verifier
    elif static_geometry is not None:
        static_checker = _obstacle_verifier(
            static_geometry,
            robot_radius_m=cfg.robot_radius_m,
            clearance_margin_m=cfg.static_clearance_margin_m,
        )
    elif static_blocked is not None:

        def _static_blocked_adapter(states: np.ndarray, **_: object) -> Mapping[str, object]:
            return {"feasible": not bool(static_blocked(states))}

        static_checker = _static_blocked_adapter
    else:
        static_checker = None

    dynamics = UnicycleDynamics(
        max_linear_speed=cfg.max_linear_speed_mps,
        max_angular_speed=cfg.max_angular_speed_radps,
    )
    attempts: list[_Attempt] = []
    attempted_counts: Counter[str] = Counter()
    valid_counts: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    static_clearances: dict[str, float | None] = {}

    def add_attempt(  # noqa: C901
        maneuver: ManeuverId,
        source: str,
        rank: int,
        target_speeds: Sequence[float],
        target_angular_speeds: Sequence[float],
        metadata: Mapping[str, JSONScalar],
        *,
        side_reason: str | None = None,
        side_sign: float | None = None,
    ) -> None:
        attempted_counts[maneuver.value] += 1
        try:
            controls = _build_controls(
                state,
                target_speeds=target_speeds,
                target_angular_speeds=target_angular_speeds,
                config=cfg,
                linear_deceleration_mps2=(
                    cfg.yield_deceleration_mps2
                    if maneuver is ManeuverId.YIELD_CREEP
                    else cfg.stop_deceleration_mps2
                ),
                braking_profile=maneuver is ManeuverId.CONTROLLED_STOP,
            )
            states = _rollout(state, controls, dt_s=cfg.dt_s, dynamics=dynamics)
        except ValueError as exc:
            reason = str(exc) if str(exc) in REJECTION_REASONS else "rollout_nonfinite"
            rejections[reason] += 1
            return
        route_progress = None
        if maneuver in (
            ManeuverId.ROUTE_FOLLOW,
            ManeuverId.PASS_LEFT,
            ManeuverId.PASS_RIGHT,
            ManeuverId.YIELD_CREEP,
        ):
            try:
                route_progress = route.candidate_progress(states[:, :2])
            except (TypeError, ValueError):
                route_progress = None
            if route_progress is None or not route_progress.is_valid:
                rejections["invalid_route_projection"] += 1
                return
        if side_sign is not None:
            if tangent is None:
                rejections["invalid_route_projection"] += 1
                return
            normal = np.asarray((-tangent[1], tangent[0]), dtype=float)
            relative_positions = states[:, :2] - states[0, :2]
            signed_lateral = side_sign * (relative_positions @ normal)
            peak_index = int(np.argmax(signed_lateral))
            peak_lateral = float(signed_lateral[peak_index])
            if (
                float(np.min(signed_lateral)) < -1.0e-8
                or peak_lateral <= 1.0e-6
                or peak_index >= len(signed_lateral) - 1
                or float(signed_lateral[-1]) >= peak_lateral - 1.0e-6
            ):
                rejections["pass_side_mismatch"] += 1
                return
        feasible, reason, min_clearance = _static_and_actuator_check(
            states, static_verifier=static_checker, config=cfg
        )
        if not feasible:
            if reason == "static_collision" and side_reason is not None:
                reason = side_reason
            rejections[reason or "static_collision"] += 1
            return
        attempt_metadata = dict(metadata)
        if maneuver is ManeuverId.CONTROLLED_STOP:
            attempt_metadata["stop_complete_within_horizon"] = bool(
                states[-1, 3] <= 1.0e-9 and abs(states[-1, 4]) <= 1.0e-9
            )
        if route_progress is not None:
            attempt_metadata["route_progress_status"] = str(route_progress.status)
            attempt_metadata["route_progress_m"] = (
                float(route_progress.signed_progress_m)
                if route_progress.signed_progress_m is not None
                else None
            )
        if min_clearance is not None:
            attempt_metadata["static_min_clearance_m"] = float(min_clearance)
        attempts.append(
            _Attempt(
                maneuver=maneuver,
                source=source,
                rank=rank,
                controls=controls,
                states=states,
                metadata=MappingProxyType(attempt_metadata),
            )
        )
        valid_counts[maneuver.value] += 1

    horizon = cfg.horizon_steps
    zero_angular = (0.0,) * horizon
    # Controlled stop is always attempted first and retained after global cap.
    stop_targets = (0.0,) * horizon
    # The angular stop target is a velocity target; unlike heading, it is simply
    # zero.  Keeping a named tuple prevents an accidental heading discontinuity
    # from becoming an angular command.
    stop_angular_targets = zero_angular
    stop_complete = state.linear_speed <= 1.0e-9 and abs(state.angular_speed) <= 1.0e-9
    predicted_speed = state.linear_speed
    for _ in range(horizon):
        predicted_speed = max(0.0, predicted_speed - cfg.stop_deceleration_mps2 * cfg.dt_s)
        if predicted_speed <= 1.0e-9 and abs(state.angular_speed) <= 1.0e-9:
            stop_complete = True
            break
    add_attempt(
        ManeuverId.CONTROLLED_STOP,
        "canonical_braking_profile",
        0,
        stop_targets,
        stop_angular_targets,
        _attempt_metadata(
            config=cfg,
            route=route,
            local_goal=selected_goal,
            stop_complete=stop_complete,
            stop_deceleration_mps2=float(cfg.stop_deceleration_mps2),
            braking_latency_s=float(cfg.actuator_limits.brake_latency_s),
            command_latency_s=float(cfg.actuator_limits.command_latency_s),
            latency_model="zero_latency_command_rollout",
        ),
    )

    if tangent is not None and route_heading is not None and selected_goal is not None:
        goal_heading = math.atan2(selected_goal[1] - state.y, selected_goal[0] - state.x)
        # Route-follow variants stay route-relative while one variant points at
        # the adaptive goal; the order is fixed by config tuple order.
        route_heading_targets = tuple(
            route_heading + offset for offset in cfg.route_heading_samples_rad
        )
        route_samples = (
            (offset, speed)
            for offset in cfg.route_heading_samples_rad
            for speed in cfg.route_speed_samples_mps
        )
        for rank, (heading_offset, speed) in enumerate(route_samples):
            if rank >= cfg.max_route_follow_candidates:
                break
            desired = route_heading + heading_offset
            if rank == 0:
                desired = goal_heading if math.isfinite(goal_heading) else route_heading_targets[0]
            add_attempt(
                ManeuverId.ROUTE_FOLLOW,
                "route_tangent_adaptive_goal",
                rank,
                (float(speed),) * horizon,
                _smooth_heading_targets(
                    initial_heading=state.heading,
                    desired_heading=desired,
                    horizon_steps=horizon,
                    dt_s=cfg.dt_s,
                ),
                _attempt_metadata(
                    config=cfg,
                    route=route,
                    local_goal=selected_goal,
                    stop_complete=False,
                    route_heading_rad=float(route_heading),
                    heading_offset_rad=float(heading_offset),
                    target_speed_mps=float(speed),
                ),
            )

        for maneuver, sign, reason in (
            (ManeuverId.PASS_LEFT, 1.0, "no_left_corridor"),
            (ManeuverId.PASS_RIGHT, -1.0, "no_right_corridor"),
        ):
            cap = (
                cfg.max_pass_left_candidates
                if maneuver is ManeuverId.PASS_LEFT
                else cfg.max_pass_right_candidates
            )
            pass_attempt_count = len(cfg.pass_lateral_offsets_m[:cap])
            if static_checker is None:
                attempted_counts[maneuver.value] += pass_attempt_count
                rejections["static_geometry_unavailable"] += pass_attempt_count
                continue
            if abs(wrap_angle_pi(state.heading - route_heading)) > _PASS_HEADING_TOLERANCE_RAD:
                attempted_counts[maneuver.value] += pass_attempt_count
                rejections["pass_heading_ineligible"] += pass_attempt_count
                continue
            for rank, lateral_offset in enumerate(cfg.pass_lateral_offsets_m[:cap]):
                offset = sign * float(lateral_offset)
                try:
                    pass_speeds, pass_angular_targets = _pass_reference_controls(
                        route,
                        projection=projection,
                        tangent=tangent,
                        lateral_offset_m=offset,
                        local_goal=selected_goal,
                        config=cfg,
                    )
                except ValueError:
                    rejections["invalid_route_projection"] += 1
                    continue
                add_attempt(
                    maneuver,
                    "route_frame_lateral_offset_rejoin",
                    rank,
                    pass_speeds,
                    pass_angular_targets,
                    _attempt_metadata(
                        config=cfg,
                        route=route,
                        local_goal=selected_goal,
                        stop_complete=False,
                        lateral_offset_m=float(offset),
                        pass_rejoin_fraction=float(cfg.pass_rejoin_fraction),
                    ),
                    side_reason=reason,
                    side_sign=sign,
                )

        if cfg.max_yield_candidates > 0:
            creep_heading = math.atan2(tangent[1], tangent[0])
            add_attempt(
                ManeuverId.YIELD_CREEP,
                "route_aligned_creep_profile",
                0,
                (cfg.creep_speed_mps,) * horizon,
                _smooth_heading_targets(
                    initial_heading=state.heading,
                    desired_heading=creep_heading,
                    horizon_steps=horizon,
                    dt_s=cfg.dt_s,
                ),
                _attempt_metadata(
                    config=cfg,
                    route=route,
                    local_goal=selected_goal,
                    stop_complete=False,
                    creep_speed_mps=float(cfg.creep_speed_mps),
                    yield_deceleration_mps2=float(cfg.yield_deceleration_mps2),
                ),
            )
    else:
        for maneuver in (
            ManeuverId.ROUTE_FOLLOW,
            ManeuverId.PASS_LEFT,
            ManeuverId.PASS_RIGHT,
            ManeuverId.YIELD_CREEP,
        ):
            rejections["invalid_route_projection"] += 1
            attempted_counts[maneuver.value] += 1

    # Retain the stop first, then deterministic family/rank order.  This makes
    # the safety-preserving cap explicit even when max_total_candidates == 1.
    family_order = (
        ManeuverId.CONTROLLED_STOP,
        ManeuverId.ROUTE_FOLLOW,
        ManeuverId.PASS_LEFT,
        ManeuverId.PASS_RIGHT,
        ManeuverId.YIELD_CREEP,
    )
    attempts.sort(key=lambda item: (family_order.index(item.maneuver), item.rank))
    truncated = len(attempts) > cfg.max_total_candidates
    if truncated:
        stop_attempts = [item for item in attempts if item.maneuver is ManeuverId.CONTROLLED_STOP]
        other_attempts = [
            item for item in attempts if item.maneuver is not ManeuverId.CONTROLLED_STOP
        ]
        retained_attempts = (
            stop_attempts[:1] + other_attempts[: max(cfg.max_total_candidates - 1, 0)]
        )
        retained_ids = {id(item) for item in retained_attempts}
        rejections["candidate_cap"] += len(attempts) - len(retained_attempts)
        attempts = [item for item in attempts if id(item) in retained_ids]

    candidates: list[ManeuverCandidate] = []
    for attempt in attempts:
        candidate_id = _candidate_id(cfg, attempt.maneuver, attempt.rank)
        try:
            candidate = _make_candidate(attempt, candidate_id=candidate_id, config=cfg)
        except (TypeError, ValueError):
            rejections["rollout_nonfinite"] += 1
            continue
        candidates.append(candidate)
        static_clearances[candidate_id] = (
            float(candidate.metadata["static_min_clearance_m"])
            if "static_min_clearance_m" in candidate.metadata
            else None
        )

    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    first_controls = {candidate.candidate_id: candidate.first_control for candidate in candidates}
    stop_flags = {
        candidate.candidate_id: bool(candidate.metadata.get("stop_complete_within_horizon", False))
        for candidate in candidates
    }
    report = ManeuverGenerationReport(
        schema_version=MANEUVER_CANDIDATE_SCHEMA_VERSION,
        candidate_set_id=_candidate_set_id(route, state, cfg, selected_goal, projection_hint),
        route_hash=route.route_hash,
        initial_robot_state=(
            float(state.x),
            float(state.y),
            float(state.heading),
            float(state.linear_speed),
            float(state.angular_speed),
        ),
        local_goal=selected_goal,
        requested_maneuvers=tuple(item.value for item in ManeuverId),
        per_maneuver_attempted_count=MappingProxyType(dict(attempted_counts)),
        per_maneuver_valid_count=MappingProxyType(
            {
                item.value: sum(candidate.maneuver is item for candidate in candidates)
                for item in ManeuverId
            }
        ),
        per_reason_rejection_count=MappingProxyType(dict(rejections)),
        ordered_candidate_ids=candidate_ids,
        candidate_first_controls=MappingProxyType(first_controls),
        stop_complete_within_horizon=MappingProxyType(stop_flags),
        static_min_clearance_m=MappingProxyType(static_clearances),
        generation_duration_ms=(time.perf_counter() - started) * 1000.0,
        truncated=truncated,
        truncation_rule=("retain_controlled_stop_then_family_rank_order" if truncated else None),
        route_projection_status=projection_status,
    )
    return ManeuverGenerationResult(tuple(candidates), report)


def generate_maneuver_portfolio(*args: Any, **kwargs: Any) -> ManeuverGenerationResult:
    """Compatibility alias for :func:`generate_maneuver_candidates`.

    Returns:
        Generated maneuver candidates and diagnostics.
    """

    return generate_maneuver_candidates(*args, **kwargs)


__all__ = [
    "MANEUVER_CANDIDATE_SCHEMA_VERSION",
    "Control",
    "JSONScalar",
    "ManeuverCandidate",
    "ManeuverGenerationReport",
    "ManeuverGenerationResult",
    "ManeuverId",
    "ManeuverPortfolioConfig",
    "StaticVerifier",
    "generate_maneuver_candidates",
    "generate_maneuver_portfolio",
]
