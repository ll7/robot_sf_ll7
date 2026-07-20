"""Experimental actuator-feasibility model for planner / trajectory validation (issue #6056).

A planned maneuver is only meaningful if it is physically executable under the robot's
actuator limits: acceleration / braking authority, yaw-rate and steering-rate bounds, and
the latencies between command and motion. This module is a pure, deterministic,
side-effect-free evaluator that separates two questions the existing trajectory verifier
and clearance diagnostics do not split explicitly:

- **geometrically clear** — does the geometry alone leave any room at all (the robot has
  not already reached the hazard)? This is a *geometry-only* verdict.
- **physically feasible** — can the robot actually execute the maneuver given its
  acceleration, deceleration, yaw/steering-rate, and latency limits?

The central value of this layer is the *distinction* between the two: a maneuver can be
geometrically clear (there is room) yet physically infeasible (the robot cannot brake,
turn, or change steering fast enough, or command/brake latency eats the available room).
The :func:`evaluate_actuator_feasibility` report returns an explicit
``actuator_feasible`` / ``geometry_only_clear`` / ``infeasible`` verdict and lists exactly
which actuator limit was violated.

Scope boundary: experimental diagnostic only, not a formal safety case, not conformalized,
not learned, and not wired into any planner control loop, release gate, or benchmark
scoring by default. All numeric limits are **provisional** defaults unless derived from a
measured AMMV / target platform — see ``docs/actuator_feasibility.md`` and
:data:`ACTUATOR_FEASIBILITY_CLAIM_BOUNDARY`. The implementation is pure so it can be unit
tested and called from offline analysis utilities.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

ACTUATOR_FEASIBILITY_SCHEMA = "actuator_feasibility.v1"
ACTUATOR_FEASIBILITY_CONFIG_KEY = "actuator_limits"
ACTUATOR_FEASIBILITY_CLAIM_BOUNDARY = (
    "experimental actuator-feasibility diagnostic; not a formal safety case; not "
    "conformalized; not learned; default planner behavior unchanged; numeric limits are "
    "PROVISIONAL defaults unless derived from a measured target platform (e.g. AMMV); "
    "geometrically-clear does not imply physically-feasible"
)

# Actuator-limit predicate identifiers. Each identifies one *actuator* constraint that the
# planned trajectory / maneuver may violate; ``violated_limits`` lists exactly these.
PRED_ACCEL_LIMIT = "accel_limit_exceeded"
PRED_DECEL_LIMIT = "decel_limit_exceeded"
PRED_YAW_RATE_LIMIT = "yaw_rate_limit_exceeded"
PRED_STEERING_RATE_LIMIT = "steering_rate_limit_exceeded"
PRED_BRAKE_DEADLINE = "fallback_brake_deadline_missed"

VERDICT_ACTUATOR_FEASIBLE = "actuator_feasible"
VERDICT_GEOMETRY_ONLY_CLEAR = "geometry_only_clear"
VERDICT_INFEASIBLE = "infeasible"

# A heading change is only meaningful when the robot is moving fast enough that the
# velocity direction is well-defined. Below this speed the yaw is ambiguous.
_MIN_HEADING_SPEED_MPS = 1.0e-3


@dataclass(frozen=True, slots=True)
class ActuatorLimitsConfig:
    """Provisional actuator limits for the experimental feasibility evaluator.

    All values are **provisional defaults** unless derived from a measured target
    platform (e.g. AMMV). They are conservative round numbers, not hardware claims.

    Attributes:
        max_accel_mps2: Maximum forward acceleration magnitude (longitudinal, m/s^2).
        max_decel_mps2: Maximum braking deceleration magnitude (longitudinal, m/s^2).
            Used both to bound commanded deceleration and to compute stopping distance.
        max_yaw_rate_radps: Maximum yaw rate magnitude (rad/s).
        max_steering_rate_radps: Maximum rate of change of yaw rate (rad/s^2), a proxy
            steering-rate / steering-discontinuity bound.
        command_latency_s: Reaction/command latency before a new command takes effect (s).
            Non-negative; adds ``v * command_latency_s`` to the stopping distance.
        brake_latency_s: Brake-engagement latency before deceleration begins (s).
            Non-negative; adds ``v * brake_latency_s`` to the stopping distance.
    """

    max_accel_mps2: float = 1.0
    max_decel_mps2: float = 1.5
    max_yaw_rate_radps: float = 1.0
    max_steering_rate_radps: float = 0.5
    command_latency_s: float = 0.15
    brake_latency_s: float = 0.2

    def __post_init__(self) -> None:
        """Validate limits so the evaluator cannot be silently misconfigured.

        Magnitude limits must be strictly positive (a zero/infinite limit would make the
        corresponding check either always fire or never fire). Latencies must be
        non-negative. All values must be finite.
        """
        for name in (
            "max_accel_mps2",
            "max_decel_mps2",
            "max_yaw_rate_radps",
            "max_steering_rate_radps",
        ):
            value = getattr(self, name)
            _require_finite(value, key=name)
            if value <= 0.0:
                raise ValueError(f"ActuatorLimitsConfig.{name} must be > 0; got {value}")
        for name in ("command_latency_s", "brake_latency_s"):
            value = getattr(self, name)
            _require_finite(value, key=name)
            if value < 0.0:
                raise ValueError(f"ActuatorLimitsConfig.{name} must be >= 0; got {value}")


@dataclass(frozen=True, slots=True)
class ActuatorFeasibilityReport:
    """Outcome of :func:`evaluate_actuator_feasibility`.

    Attributes:
        schema_version: Always :data:`ACTUATOR_FEASIBILITY_SCHEMA`.
        claim_boundary: Explicit experimental / provisional claim boundary string; always
            equal to :data:`ACTUATOR_FEASIBILITY_CLAIM_BOUNDARY`.
        geometrically_clear: ``True`` iff geometry alone leaves room — the available
            clearance to the hazard is non-negative (the robot has not already reached
            it). This is the *geometry-only* verdict.
        physically_feasible: ``True`` iff every actuator-limit check passes
            (acceleration, deceleration, yaw-rate, steering-rate, and the
            fallback-brake deadline).
        verdict: Aggregate verdict. ``actuator_feasible`` when geometrically clear AND
            physically feasible; ``geometry_only_clear`` when geometrically clear but
            physically infeasible (the distinguishing case); ``infeasible`` when not
            geometrically clear (already in / past contact).
        violated_limits: Tuple of actuator-limit predicate identifiers that fired, in
            evaluation order. Empty for a clean actuator-feasible result.
        available_clearance_m: Available geometric clearance to the hazard (m), as
            supplied to the evaluator. Negative means the robot is already in contact.
        max_speed_mps: Maximum robot speed observed over the trajectory (m/s), or
            ``None`` when the trajectory has no usable speed data.
        stopping_distance_m: Conservative stopping distance (m) from ``max_speed_mps``
            including command + brake latency, or ``None`` when no speed data is
            available. Braking authority uses ``max_decel_mps2``.
        observed_max_accel_mps2: Maximum commanded forward acceleration magnitude over
            the trajectory (m/s^2), or ``None`` when too few steps.
        observed_max_decel_mps2: Maximum commanded braking deceleration magnitude over
            the trajectory (m/s^2), or ``None`` when too few steps.
        observed_max_yaw_rate_radps: Maximum commanded yaw-rate magnitude (rad/s), or
            ``None`` when too few moving steps.
        observed_max_steering_rate_radps: Maximum commanded steering-rate magnitude
            (rad/s^2, proxy), or ``None`` when too few moving steps.
    """

    schema_version: str = ACTUATOR_FEASIBILITY_SCHEMA
    claim_boundary: str = field(default=ACTUATOR_FEASIBILITY_CLAIM_BOUNDARY)
    geometrically_clear: bool = False
    physically_feasible: bool = False
    verdict: Literal["actuator_feasible", "geometry_only_clear", "infeasible"] = VERDICT_INFEASIBLE
    violated_limits: tuple[str, ...] = ()
    available_clearance_m: float = 0.0
    max_speed_mps: float | None = None
    stopping_distance_m: float | None = None
    observed_max_accel_mps2: float | None = None
    observed_max_decel_mps2: float | None = None
    observed_max_yaw_rate_radps: float | None = None
    observed_max_steering_rate_radps: float | None = None


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _require_finite(value: float, *, key: str) -> float:
    """Raise ``ValueError`` if ``value`` is non-finite (NaN/inf).

    Non-finite limits would silently corrupt threshold comparisons and could cause an
    infeasible maneuver to be reported feasible. Reject them at the input boundary.

    Returns:
        The validated finite float value.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{key} must be finite; got {value}")
    return numeric


def _as_float2d(name: str, array: NDArray[np.floating] | object) -> NDArray[np.floating]:
    """Validate and coerce ``array`` to a finite float array of shape ``(T, 2)``.

    Returns:
        The validated finite float array of shape ``(T, 2)``.

    Raises:
        ValueError: If the array is not ``(T, 2)`` with ``T >= 1``, or holds non-finite
            values.
    """
    arr = np.asarray(array, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (T, 2); got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one timestep; got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values (no NaN or inf)")
    return arr


# ---------------------------------------------------------------------------
# Pure physics / kinematics helpers
# ---------------------------------------------------------------------------


def stopping_distance(speed_mps: float, config: ActuatorLimitsConfig) -> float:
    """Return the conservative stopping distance from ``speed_mps`` including latency.

    The robot continues at ``speed_mps`` for the combined command + brake latency before
    decelerating at ``max_decel_mps2``:

    ``d = v^2 / (2 * a) + v * (command_latency_s + brake_latency_s)``

    Negative speed is treated by its magnitude (stopping demand depends on |v|).

    Args:
        speed_mps: Current robot speed (m/s).
        config: Actuator limits supplying the braking authority and latencies.

    Returns:
        Stopping distance in metres (>= 0). ``0.0`` when ``speed_mps`` is zero.
    """
    speed = abs(_require_finite(speed_mps, key="speed_mps"))
    v_sq_term = (speed * speed) / (2.0 * config.max_decel_mps2)
    latency_term = speed * (config.command_latency_s + config.brake_latency_s)
    return float(v_sq_term + latency_term)


def brake_deadline_satisfied(
    speed_mps: float,
    available_clearance_m: float,
    config: ActuatorLimitsConfig,
) -> bool:
    """Return ``True`` iff the robot can stop before consuming ``available_clearance_m``.

    The fallback-brake deadline is satisfied when the latency-inclusive stopping distance
    (:func:`stopping_distance`) does not exceed the geometric clearance ahead. This is
    the actuator-level "can we actually stop in time" check that complements the
    geometry-only clearance verdict.

    Args:
        speed_mps: Current robot speed (m/s).
        available_clearance_m: Available geometric clearance to the hazard ahead (m).
        config: Actuator limits.

    Returns:
        ``True`` iff ``stopping_distance(speed, config) <= available_clearance_m``.
    """
    clearance = _require_finite(available_clearance_m, key="available_clearance_m")
    return stopping_distance(speed_mps, config) <= clearance


def _wrap_angle(angle: float) -> float:
    """Wrap an angle difference to ``[-pi, pi]``.

    Returns:
        The angle difference wrapped into ``[-pi, pi]``.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _observed_actuator_rates(
    robot_positions: NDArray[np.floating],
    robot_velocities: NDArray[np.floating],
    dt_s: float,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Return observed max accel / decel / yaw-rate / steering-rate / max-speed.

    Longitudinal accel/decel are derived from the change in scalar speed between
    consecutive timesteps (tangential along the motion direction). Yaw rate is the
    wrapped heading change between adjacent *moving* timesteps in the original
    trajectory; steering rate is the rate of change of yaw rate between consecutive
    valid transitions (a proxy bound on
    how fast the steering / yaw command can change). Any quantity that cannot be computed
    from too-short a trajectory is returned as ``None``.

    Returns:
        Tuple of (max_accel, max_decel, max_yaw_rate, max_steering_rate, max_speed).
    """
    speeds = np.linalg.norm(robot_velocities, axis=-1)
    max_speed: float | None = float(np.max(speeds)) if speeds.size else None

    max_accel: float | None = None
    max_decel: float | None = None
    if robot_velocities.shape[0] >= 2:
        # Tangential (longitudinal) acceleration: derivative of scalar speed.
        d_speed = np.diff(speeds)
        tangential = d_speed / dt_s
        accel_values = tangential[tangential > 0.0]
        decel_values = -tangential[tangential < 0.0]
        max_accel = float(np.max(accel_values)) if accel_values.size else 0.0
        max_decel = float(np.max(decel_values)) if decel_values.size else 0.0

    # Rates are only defined for adjacent moving samples in the original trajectory.
    # Filtering stopped samples first would use dt_s for a transition that actually
    # spans multiple timesteps and inflate the rates.
    max_yaw_rate: float | None = None
    max_steering_rate: float | None = None
    moving = speeds > _MIN_HEADING_SPEED_MPS
    if int(np.sum(moving)) >= 2:
        headings = np.arctan2(robot_velocities[:, 1], robot_velocities[:, 0])
        yaw_rates = np.full(robot_velocities.shape[0] - 1, np.nan)
        for i in range(yaw_rates.size):
            if moving[i] and moving[i + 1]:
                yaw_rates[i] = _wrap_angle(headings[i + 1] - headings[i]) / dt_s
        valid_yaw_rates = yaw_rates[~np.isnan(yaw_rates)]
        if valid_yaw_rates.size:
            max_yaw_rate = float(np.max(np.abs(valid_yaw_rates)))
        # A steering-rate delta is defined only for adjacent valid yaw-rate
        # intervals in the original trajectory. Do not diff the filtered values:
        # that would bridge a stopped sample and invent a steering discontinuity.
        steering_rates = []
        for i in range(yaw_rates.size - 1):
            if not np.isnan(yaw_rates[i]) and not np.isnan(yaw_rates[i + 1]):
                steering_rates.append(abs(yaw_rates[i + 1] - yaw_rates[i]) / dt_s)
        if steering_rates:
            max_steering_rate = float(max(steering_rates))

    return max_accel, max_decel, max_yaw_rate, max_steering_rate, max_speed


# ---------------------------------------------------------------------------
# Config schema loader
# ---------------------------------------------------------------------------


def load_actuator_limits(config: Mapping[str, Any]) -> ActuatorLimitsConfig:
    """Read and validate an ``actuator_limits`` block from a config mapping.

    This is the config-schema entry point for config-first workflows. The block may be
    supplied either as the mapping itself or nested under
    :data:`ACTUATOR_FEASIBILITY_CONFIG_KEY` (``actuator_limits``). An optional
    ``schema_version`` field, when present, must equal
    :data:`ACTUATOR_FEASIBILITY_SCHEMA`.

    Fails closed: a malformed block raises :class:`ValueError` rather than silently
    falling back to defaults, so a typo in a config cannot weaken the limits.

    Args:
        config: Mapping that either *is* the actuator-limits block or contains it under
            the ``actuator_limits`` key.

    Returns:
        Validated :class:`ActuatorLimitsConfig`.

    Raises:
        ValueError: If the config is not a mapping, the block is missing, the schema
            version is wrong, or any limit fails validation.
    """
    if not isinstance(config, Mapping):
        raise ValueError("actuator-feasibility config must be a mapping")
    block = config.get(ACTUATOR_FEASIBILITY_CONFIG_KEY, config)
    if not isinstance(block, Mapping):
        raise ValueError(f"{ACTUATOR_FEASIBILITY_CONFIG_KEY!r} block must be a mapping")
    schema_version = block.get("schema_version")
    if schema_version is not None and schema_version != ACTUATOR_FEASIBILITY_SCHEMA:
        raise ValueError(
            f"{ACTUATOR_FEASIBILITY_CONFIG_KEY}.schema_version must be "
            f"{ACTUATOR_FEASIBILITY_SCHEMA!r}; got {schema_version!r}"
        )
    try:
        return ActuatorLimitsConfig(
            max_accel_mps2=_require_finite(block.get("max_accel_mps2", 1.0), key="max_accel_mps2"),
            max_decel_mps2=_require_finite(block.get("max_decel_mps2", 1.5), key="max_decel_mps2"),
            max_yaw_rate_radps=_require_finite(
                block.get("max_yaw_rate_radps", 1.0), key="max_yaw_rate_radps"
            ),
            max_steering_rate_radps=_require_finite(
                block.get("max_steering_rate_radps", 0.5), key="max_steering_rate_radps"
            ),
            command_latency_s=_require_finite(
                block.get("command_latency_s", 0.15), key="command_latency_s"
            ),
            brake_latency_s=_require_finite(
                block.get("brake_latency_s", 0.2), key="brake_latency_s"
            ),
        )
    except KeyError as exc:
        raise ValueError(f"missing required actuator limit field: {exc}") from exc


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------


def _brake_deadline_violation(
    max_speed_mps: float | None,
    clearance_m: float,
    cfg: ActuatorLimitsConfig,
) -> tuple[list[str], float | None]:
    """Evaluate the fallback-brake deadline from the conservative worst-case speed.

    Returns:
        Tuple of (violated_predicates, stopping_distance_m). The stopping distance is
        ``None`` only when no speed data is available.
    """
    if max_speed_mps is None:
        return [], None
    stop_dist = stopping_distance(max_speed_mps, cfg)
    violated = (
        [] if brake_deadline_satisfied(max_speed_mps, clearance_m, cfg) else [PRED_BRAKE_DEADLINE]
    )
    return violated, stop_dist


def _actuator_rate_violations(
    observed_accel: float | None,
    observed_decel: float | None,
    observed_yaw: float | None,
    observed_steer: float | None,
    cfg: ActuatorLimitsConfig,
) -> list[str]:
    """Return the per-trajectory actuator-limit predicates that fired, in order."""
    violated: list[str] = []
    if observed_accel is not None and observed_accel > cfg.max_accel_mps2:
        violated.append(PRED_ACCEL_LIMIT)
    if observed_decel is not None and observed_decel > cfg.max_decel_mps2:
        violated.append(PRED_DECEL_LIMIT)
    if observed_yaw is not None and observed_yaw > cfg.max_yaw_rate_radps:
        violated.append(PRED_YAW_RATE_LIMIT)
    if observed_steer is not None and observed_steer > cfg.max_steering_rate_radps:
        violated.append(PRED_STEERING_RATE_LIMIT)
    return violated


def _resolve_verdict(
    geometrically_clear: bool,
    physically_feasible: bool,
) -> Literal["actuator_feasible", "geometry_only_clear", "infeasible"]:
    """Combine the geometry-only and physically-feasible flags into one verdict.

    Returns:
        ``infeasible`` when not geometrically clear; ``actuator_feasible`` when
        geometrically clear and physically feasible; ``geometry_only_clear`` when
        geometrically clear but physically infeasible.
    """
    if not geometrically_clear:
        return VERDICT_INFEASIBLE
    return VERDICT_ACTUATOR_FEASIBLE if physically_feasible else VERDICT_GEOMETRY_ONLY_CLEAR


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


def evaluate_actuator_feasibility(
    *,
    robot_positions: NDArray[np.floating] | object,
    robot_velocities: NDArray[np.floating] | object,
    dt_s: float,
    hazard_clearance_m: float,
    config: ActuatorLimitsConfig | None = None,
) -> ActuatorFeasibilityReport:
    """Evaluate actuator feasibility of a planned/executed robot trajectory.

    Computes the geometry-only clearance verdict, the per-trajectory actuator-limit
    checks (acceleration, deceleration, yaw-rate, steering-rate), and the
    fallback-brake deadline, then combines them into a verdict that distinguishes
    *geometrically clear* from *physically feasible*.

    Args:
        robot_positions: Robot positions over time, shape ``(T, 2)`` (metres).
        robot_velocities: Robot velocities over time, shape ``(T, 2)`` (m/s). Must match
            ``robot_positions``; required so accel / yaw / steering rates are computable.
        dt_s: Timestep duration in seconds. Must be positive.
        hazard_clearance_m: Available geometric clearance from the robot surface to the
            nearest hazard ahead (metres). This is the *geometry-only* clearance: a
            non-negative value means the robot has not yet reached the hazard; a negative
            value means the robot is already in contact and the verdict is ``infeasible``
            regardless of actuator limits.
        config: Actuator limits. Defaults to :class:`ActuatorLimitsConfig` (provisional).

    Returns:
        An :class:`ActuatorFeasibilityReport` with the verdict, the violated actuator
        limits, and the observed maxima used for the checks.

    Raises:
        ValueError: If array shapes are invalid, ``dt_s`` is non-positive, or any input
            holds non-finite values.
    """
    dt_s = _require_finite(dt_s, key="dt_s")
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be > 0; got {dt_s}")
    clearance = _require_finite(hazard_clearance_m, key="hazard_clearance_m")

    robot_pos = _as_float2d("robot_positions", robot_positions)
    robot_vel = _as_float2d("robot_velocities", robot_velocities)
    if robot_vel.shape != robot_pos.shape:
        raise ValueError(
            f"robot_velocities shape {robot_vel.shape} must match robot_positions "
            f"shape {robot_pos.shape}"
        )

    cfg = config if config is not None else ActuatorLimitsConfig()

    observed_accel, observed_decel, observed_yaw, observed_steer, max_speed = (
        _observed_actuator_rates(robot_pos, robot_vel, dt_s)
    )

    geometrically_clear = clearance >= 0.0
    brake_violations, stop_dist = _brake_deadline_violation(max_speed, clearance, cfg)
    rate_violations = _actuator_rate_violations(
        observed_accel, observed_decel, observed_yaw, observed_steer, cfg
    )
    violated = brake_violations + rate_violations
    physically_feasible = not violated
    verdict = _resolve_verdict(geometrically_clear, physically_feasible)

    return ActuatorFeasibilityReport(
        geometrically_clear=geometrically_clear,
        physically_feasible=physically_feasible,
        verdict=verdict,
        violated_limits=tuple(violated),
        available_clearance_m=clearance,
        max_speed_mps=max_speed,
        stopping_distance_m=stop_dist,
        observed_max_accel_mps2=observed_accel,
        observed_max_decel_mps2=observed_decel,
        observed_max_yaw_rate_radps=observed_yaw,
        observed_max_steering_rate_radps=observed_steer,
    )


def evaluate_encounter_actuator_feasibility(
    *,
    speed_mps: float,
    hazard_clearance_m: float,
    config: ActuatorLimitsConfig | None = None,
) -> ActuatorFeasibilityReport:
    """Evaluate actuator feasibility for a single encounter (no trajectory needed).

    Convenience entry point for the common case where only the current speed and the
    available geometric clearance are known — e.g. from a planner's instantaneous state.
    Only the fallback-brake deadline check applies (accel/yaw/steering rates require a
    trajectory and are reported as ``None``). This still produces the geometry-only vs
    actuator-feasible distinction via the braking/latency deadline.

    Args:
        speed_mps: Current robot speed (m/s).
        hazard_clearance_m: Available geometric clearance to the hazard ahead (metres).
        config: Actuator limits. Defaults to :class:`ActuatorLimitsConfig` (provisional).

    Returns:
        An :class:`ActuatorFeasibilityReport` with the braking-deadline verdict.

    Raises:
        ValueError: If inputs are non-finite.
    """
    speed = _require_finite(speed_mps, key="speed_mps")
    clearance = _require_finite(hazard_clearance_m, key="hazard_clearance_m")
    cfg = config if config is not None else ActuatorLimitsConfig()

    geometrically_clear = clearance >= 0.0
    violated, stop_dist = _brake_deadline_violation(abs(speed), clearance, cfg)
    physically_feasible = not violated
    verdict = _resolve_verdict(geometrically_clear, physically_feasible)

    return ActuatorFeasibilityReport(
        geometrically_clear=geometrically_clear,
        physically_feasible=physically_feasible,
        verdict=verdict,
        violated_limits=tuple(violated),
        available_clearance_m=clearance,
        max_speed_mps=float(abs(speed)),
        stopping_distance_m=stop_dist,
    )


__all__ = [
    "ACTUATOR_FEASIBILITY_CLAIM_BOUNDARY",
    "ACTUATOR_FEASIBILITY_CONFIG_KEY",
    "ACTUATOR_FEASIBILITY_SCHEMA",
    "PRED_ACCEL_LIMIT",
    "PRED_BRAKE_DEADLINE",
    "PRED_DECEL_LIMIT",
    "PRED_STEERING_RATE_LIMIT",
    "PRED_YAW_RATE_LIMIT",
    "VERDICT_ACTUATOR_FEASIBLE",
    "VERDICT_GEOMETRY_ONLY_CLEAR",
    "VERDICT_INFEASIBLE",
    "ActuatorFeasibilityReport",
    "ActuatorLimitsConfig",
    "brake_deadline_satisfied",
    "evaluate_actuator_feasibility",
    "evaluate_encounter_actuator_feasibility",
    "load_actuator_limits",
    "stopping_distance",
]
