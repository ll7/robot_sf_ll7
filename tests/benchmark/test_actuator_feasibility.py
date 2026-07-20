"""Tests for the experimental actuator-feasibility model (issue #6056).

Covers the four issue acceptance criteria:

1. A config schema represents acceleration, braking, yaw/steering rate, and latency.
2. At least one scenario report includes actuator-feasible versus geometry-only verdicts.
3. The report lists which actuator limit was violated.
4. (Provisional-values documentation is asserted via the claim-boundary constant.)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_sf.benchmark.actuator_feasibility import (
    ACTUATOR_FEASIBILITY_CLAIM_BOUNDARY,
    ACTUATOR_FEASIBILITY_CONFIG_KEY,
    ACTUATOR_FEASIBILITY_SCHEMA,
    PRED_ACCEL_LIMIT,
    PRED_BRAKE_DEADLINE,
    PRED_DECEL_LIMIT,
    PRED_STEERING_RATE_LIMIT,
    PRED_YAW_RATE_LIMIT,
    VERDICT_ACTUATOR_FEASIBLE,
    VERDICT_GEOMETRY_ONLY_CLEAR,
    VERDICT_INFEASIBLE,
    ActuatorFeasibilityReport,
    ActuatorLimitsConfig,
    brake_deadline_satisfied,
    evaluate_actuator_feasibility,
    evaluate_encounter_actuator_feasibility,
    load_actuator_limits,
    stopping_distance,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _straight_trajectory(
    *, n_steps: int = 6, speed: float = 0.5, dt_s: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, velocities) for a straight +x constant-speed path."""
    t = np.arange(n_steps, dtype=float) * dt_s
    positions = np.stack([t * speed, np.zeros(n_steps)], axis=1)
    velocities = np.tile([speed, 0.0], (n_steps, 1))
    return positions, velocities


# ---------------------------------------------------------------------------
# Acceptance criterion 1: config schema
# ---------------------------------------------------------------------------


def test_schema_and_claim_boundary_constants() -> None:
    """Schema version and claim boundary are stable, explicit, and provisional-flagged."""
    assert ACTUATOR_FEASIBILITY_SCHEMA == "actuator_feasibility.v1"
    assert "PROVISIONAL" in ACTUATOR_FEASIBILITY_CLAIM_BOUNDARY
    assert "not a formal safety case" in ACTUATOR_FEASIBILITY_CLAIM_BOUNDARY
    assert "default planner behavior unchanged" in ACTUATOR_FEASIBILITY_CLAIM_BOUNDARY


def test_config_defaults_represent_all_required_limits() -> None:
    """Defaults carry acceleration, braking, yaw rate, steering rate, and both latencies."""
    cfg = ActuatorLimitsConfig()
    # The six fields required by the issue candidate schema.
    assert cfg.max_accel_mps2 > 0.0
    assert cfg.max_decel_mps2 > 0.0
    assert cfg.max_yaw_rate_radps > 0.0
    assert cfg.max_steering_rate_radps > 0.0
    assert cfg.command_latency_s >= 0.0
    assert cfg.brake_latency_s >= 0.0


def test_config_validates_nonpositive_limits() -> None:
    """Non-positive magnitude limits raise ValueError (would weaken checks silently)."""
    with pytest.raises(ValueError, match="max_accel_mps2"):
        ActuatorLimitsConfig(max_accel_mps2=0.0)
    with pytest.raises(ValueError, match="max_decel_mps2"):
        ActuatorLimitsConfig(max_decel_mps2=-0.1)
    with pytest.raises(ValueError, match="max_yaw_rate_radps"):
        ActuatorLimitsConfig(max_yaw_rate_radps=0.0)
    with pytest.raises(ValueError, match="max_steering_rate_radps"):
        ActuatorLimitsConfig(max_steering_rate_radps=0.0)


def test_config_validates_negative_latency() -> None:
    """Negative latencies raise ValueError."""
    with pytest.raises(ValueError, match="command_latency_s"):
        ActuatorLimitsConfig(command_latency_s=-0.01)
    with pytest.raises(ValueError, match="brake_latency_s"):
        ActuatorLimitsConfig(brake_latency_s=-0.01)


def test_config_validates_nonfinite() -> None:
    """Non-finite limits raise ValueError (would corrupt threshold comparisons)."""
    with pytest.raises(ValueError, match="finite"):
        ActuatorLimitsConfig(max_accel_mps2=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        ActuatorLimitsConfig(max_decel_mps2=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        ActuatorLimitsConfig(command_latency_s=float("nan"))


def test_load_actuator_limits_reads_nested_block() -> None:
    """load_actuator_limits reads an actuator_limits block matching the issue schema."""
    config = {
        ACTUATOR_FEASIBILITY_CONFIG_KEY: {
            "schema_version": ACTUATOR_FEASIBILITY_SCHEMA,
            "max_accel_mps2": 1.0,
            "max_decel_mps2": 1.5,
            "max_yaw_rate_radps": 1.0,
            "max_steering_rate_radps": 0.5,
            "command_latency_s": 0.15,
            "brake_latency_s": 0.2,
        }
    }
    cfg = load_actuator_limits(config)
    assert cfg == ActuatorLimitsConfig()


def test_load_actuator_limits_accepts_flat_block_and_defaults() -> None:
    """The block can be supplied flat; omitted fields fall back to provisional defaults."""
    cfg = load_actuator_limits({"max_accel_mps2": 2.0})
    assert cfg.max_accel_mps2 == 2.0
    assert cfg.max_decel_mps2 == ActuatorLimitsConfig().max_decel_mps2


def test_load_actuator_limits_rejects_wrong_schema_version() -> None:
    """A mismatched schema_version fails closed rather than silently loading."""
    with pytest.raises(ValueError, match="schema_version"):
        load_actuator_limits(
            {
                ACTUATOR_FEASIBILITY_CONFIG_KEY: {
                    "schema_version": "actuator_feasibility.v0",
                    "max_accel_mps2": 1.0,
                }
            }
        )


def test_load_actuator_limits_rejects_non_mapping() -> None:
    """A non-mapping config fails closed with a clear ValueError."""
    with pytest.raises(ValueError, match="mapping"):
        load_actuator_limits("not a mapping")  # type: ignore[arg-type]


def test_load_actuator_limits_rejects_bad_limit() -> None:
    """Invalid limit values surface the underlying config validation error."""
    with pytest.raises(ValueError, match="max_decel_mps2"):
        load_actuator_limits({"max_decel_mps2": -1.0})


# ---------------------------------------------------------------------------
# Acceptance criteria 2 & 3: actuator-feasible vs geometry-only verdicts and
# which actuator limit was violated.
# ---------------------------------------------------------------------------


def test_feasible_straight_trajectory_is_actuator_feasible() -> None:
    """A slow straight path with ample clearance is geometrically clear AND feasible."""
    positions, velocities = _straight_trajectory(n_steps=6, speed=0.5)
    report = evaluate_actuator_feasibility(
        robot_positions=positions,
        robot_velocities=velocities,
        dt_s=0.1,
        hazard_clearance_m=5.0,
    )
    assert isinstance(report, ActuatorFeasibilityReport)
    assert report.verdict == VERDICT_ACTUATOR_FEASIBLE
    assert report.geometrically_clear is True
    assert report.physically_feasible is True
    assert report.violated_limits == ()
    assert report.claim_boundary == ACTUATOR_FEASIBILITY_CLAIM_BOUNDARY
    assert report.schema_version == ACTUATOR_FEASIBILITY_SCHEMA


def test_geometry_only_clear_when_brake_deadline_missed() -> None:
    """Geometric room exists but latency-inclusive braking cannot stop in time.

    This is the headline distinction: at 1.0 m/s with max_decel=1.5 the pure stopping
    distance is 1/(2*1.5) ~ 0.333 m; adding 0.35 s of latency gives ~0.333 + 0.35 = 0.683 m.
    With 0.5 m of clearance geometry says "room" (clearance >= 0) but the brake deadline
    is missed, so the verdict is geometry_only_clear and the brake predicate fires.
    """
    cfg = ActuatorLimitsConfig(max_decel_mps2=1.5, command_latency_s=0.15, brake_latency_s=0.2)
    positions, velocities = _straight_trajectory(n_steps=4, speed=1.0)
    report = evaluate_actuator_feasibility(
        robot_positions=positions,
        robot_velocities=velocities,
        dt_s=0.1,
        hazard_clearance_m=0.5,
        config=cfg,
    )
    assert report.geometrically_clear is True
    assert report.physically_feasible is False
    assert report.verdict == VERDICT_GEOMETRY_ONLY_CLEAR
    assert PRED_BRAKE_DEADLINE in report.violated_limits
    assert report.stopping_distance_m is not None
    assert report.stopping_distance_m > 0.5


def test_infeasible_when_already_in_contact() -> None:
    """Negative clearance (already in contact) is infeasible regardless of actuators."""
    positions, velocities = _straight_trajectory(n_steps=4, speed=0.3)
    report = evaluate_actuator_feasibility(
        robot_positions=positions,
        robot_velocities=velocities,
        dt_s=0.1,
        hazard_clearance_m=-0.1,
    )
    assert report.geometrically_clear is False
    assert report.verdict == VERDICT_INFEASIBLE


def test_accel_limit_violation_is_reported() -> None:
    """A commanded acceleration exceeding max_accel fires the accel-limit predicate."""
    cfg = ActuatorLimitsConfig(max_accel_mps2=0.5)
    # Speed rises 0.2 -> 1.0 over dt=0.1 => accel = 8.0 m/s^2 >> 0.5.
    velocities = np.array([[0.2, 0.0], [1.0, 0.0], [1.0, 0.0]])
    positions = np.cumsum(velocities * 0.1, axis=0)
    report = evaluate_actuator_feasibility(
        robot_positions=positions,
        robot_velocities=velocities,
        dt_s=0.1,
        hazard_clearance_m=10.0,
        config=cfg,
    )
    assert PRED_ACCEL_LIMIT in report.violated_limits
    assert report.observed_max_accel_mps2 is not None
    assert report.observed_max_accel_mps2 == pytest.approx(8.0, rel=1e-6)


def test_decel_limit_violation_is_reported() -> None:
    """A commanded deceleration exceeding max_decel fires the decel-limit predicate."""
    cfg = ActuatorLimitsConfig(max_decel_mps2=0.5)
    # Speed falls 1.0 -> 0.2 over dt=0.1 => decel = 8.0 m/s^2 >> 0.5.
    velocities = np.array([[1.0, 0.0], [0.2, 0.0], [0.2, 0.0]])
    positions = np.cumsum(velocities * 0.1, axis=0)
    report = evaluate_actuator_feasibility(
        robot_positions=positions,
        robot_velocities=velocities,
        dt_s=0.1,
        hazard_clearance_m=10.0,
        config=cfg,
    )
    assert PRED_DECEL_LIMIT in report.violated_limits
    assert report.observed_max_decel_mps2 == pytest.approx(8.0, rel=1e-6)


def test_yaw_rate_limit_violation_is_reported() -> None:
    """A commanded yaw rate exceeding max_yaw_rate fires the yaw-rate predicate.

    Heading flips from 0 to ~pi/2 between consecutive moving steps over dt=0.1 =>
    yaw rate ~ (pi/2)/0.1 ~ 15.7 rad/s, far above the 1.0 rad/s limit.
    """
    cfg = ActuatorLimitsConfig(max_yaw_rate_radps=1.0)
    v = 0.5
    headings = np.array([0.0, math.pi / 2.0, math.pi / 2.0])
    velocities = np.stack([v * np.cos(headings), v * np.sin(headings)], axis=1)
    positions = np.cumsum(velocities * 0.1, axis=0)
    report = evaluate_actuator_feasibility(
        robot_positions=positions,
        robot_velocities=velocities,
        dt_s=0.1,
        hazard_clearance_m=10.0,
        config=cfg,
    )
    assert PRED_YAW_RATE_LIMIT in report.violated_limits
    assert report.observed_max_yaw_rate_radps is not None
    assert report.observed_max_yaw_rate_radps > 1.0


def test_steering_rate_discontinuity_is_reported() -> None:
    """A sudden reversal of yaw rate (steering discontinuity) fires the steering-rate limit.

    Heading changes slowly then sharply reverses: the rate of change of yaw rate spikes,
    which is the steering-rate / steering-discontinuity proxy.
    """
    cfg = ActuatorLimitsConfig(max_steering_rate_radps=0.5)
    # Three distinct headings so yaw rate changes abruptly between the two intervals.
    headings = np.array([0.0, 0.3, 1.8])  # +0.3 rad then +1.5 rad over dt=0.1
    v = 0.5
    velocities = np.stack([v * np.cos(headings), v * np.sin(headings)], axis=1)
    positions = np.cumsum(velocities * 0.1, axis=0)
    report = evaluate_actuator_feasibility(
        robot_positions=positions,
        robot_velocities=velocities,
        dt_s=0.1,
        hazard_clearance_m=10.0,
        config=cfg,
    )
    assert PRED_STEERING_RATE_LIMIT in report.violated_limits
    assert report.observed_max_steering_rate_radps is not None
    assert report.observed_max_steering_rate_radps > 0.5


def test_latency_changes_brake_feasibility() -> None:
    """Adding latency widens stopping distance and can flip feasibility.

    With the same speed and clearance, zero-latency config is feasible but the default
    latency config is not — proving latency is part of the deadline, not just geometry.
    """
    positions, velocities = _straight_trajectory(n_steps=4, speed=1.0)
    no_latency = ActuatorLimitsConfig(command_latency_s=0.0, brake_latency_s=0.0)
    report_no_latency = evaluate_actuator_feasibility(
        robot_positions=positions,
        robot_velocities=velocities,
        dt_s=0.1,
        hazard_clearance_m=0.4,
        config=no_latency,
    )
    assert report_no_latency.physically_feasible is True
    assert report_no_latency.verdict == VERDICT_ACTUATOR_FEASIBLE

    report_with_latency = evaluate_actuator_feasibility(
        robot_positions=positions,
        robot_velocities=velocities,
        dt_s=0.1,
        hazard_clearance_m=0.4,
        config=ActuatorLimitsConfig(),  # default 0.15 + 0.2 s latency
    )
    assert report_with_latency.physically_feasible is False
    assert report_with_latency.verdict == VERDICT_GEOMETRY_ONLY_CLEAR
    assert PRED_BRAKE_DEADLINE in report_with_latency.violated_limits


def test_violated_limits_listed_in_evaluation_order() -> None:
    """Multiple violations are reported together; brake-deadline is evaluated first."""
    # High accel + missed brake deadline: both should appear, brake deadline first.
    cfg = ActuatorLimitsConfig(max_accel_mps2=0.1, max_decel_mps2=1.5)
    velocities = np.array([[0.2, 0.0], [2.0, 0.0], [2.0, 0.0]])
    positions = np.cumsum(velocities * 0.1, axis=0)
    report = evaluate_actuator_feasibility(
        robot_positions=positions,
        robot_velocities=velocities,
        dt_s=0.1,
        hazard_clearance_m=0.2,
        config=cfg,
    )
    assert report.verdict == VERDICT_GEOMETRY_ONLY_CLEAR
    assert PRED_BRAKE_DEADLINE in report.violated_limits
    assert PRED_ACCEL_LIMIT in report.violated_limits
    # Brake deadline is appended before the per-trajectory actuator checks.
    assert report.violated_limits.index(PRED_BRAKE_DEADLINE) < report.violated_limits.index(
        PRED_ACCEL_LIMIT
    )


# ---------------------------------------------------------------------------
# Encounter convenience entry point
# ---------------------------------------------------------------------------


def test_encounter_entry_point_distinguishes_geometry_only() -> None:
    """The encounter entry point returns the same verdict family without a trajectory."""
    cfg = ActuatorLimitsConfig()
    feasible = evaluate_encounter_actuator_feasibility(
        speed_mps=0.2, hazard_clearance_m=5.0, config=cfg
    )
    assert feasible.verdict == VERDICT_ACTUATOR_FEASIBLE
    assert feasible.violated_limits == ()

    geometry_only = evaluate_encounter_actuator_feasibility(
        speed_mps=1.0, hazard_clearance_m=0.3, config=cfg
    )
    assert geometry_only.verdict == VERDICT_GEOMETRY_ONLY_CLEAR
    assert PRED_BRAKE_DEADLINE in geometry_only.violated_limits


# ---------------------------------------------------------------------------
# Pure physics helpers
# ---------------------------------------------------------------------------


def test_stopping_distance_formula() -> None:
    """stopping_distance = v^2/(2a) + v*(command+brake latency)."""
    cfg = ActuatorLimitsConfig(max_decel_mps2=2.0, command_latency_s=0.1, brake_latency_s=0.1)
    # v=1.0: 1/(2*2) + 1.0*0.2 = 0.25 + 0.2 = 0.45 m
    assert stopping_distance(1.0, cfg) == pytest.approx(0.45, rel=1e-9)
    assert stopping_distance(0.0, cfg) == 0.0
    # Speed magnitude only.
    assert stopping_distance(-1.0, cfg) == pytest.approx(0.45, rel=1e-9)


def test_brake_deadline_satisfied_boundary() -> None:
    """brake_deadline_satisfied is True iff stopping distance <= clearance."""
    cfg = ActuatorLimitsConfig(max_decel_mps2=2.0, command_latency_s=0.0, brake_latency_s=0.0)
    # stopping distance at 2 m/s = 4/(2*2) = 1.0 m exactly.
    assert brake_deadline_satisfied(2.0, 1.0, cfg) is True
    assert brake_deadline_satisfied(2.0, 0.9, cfg) is False


# ---------------------------------------------------------------------------
# Input validation / fail-closed
# ---------------------------------------------------------------------------


def test_non_positive_dt_raises() -> None:
    """A non-positive dt_s raises ValueError."""
    positions, velocities = _straight_trajectory()
    with pytest.raises(ValueError, match="dt_s"):
        evaluate_actuator_feasibility(
            robot_positions=positions,
            robot_velocities=velocities,
            dt_s=0.0,
            hazard_clearance_m=1.0,
        )


def test_invalid_position_shape_raises() -> None:
    """A robot_positions array that is not (T, 2) raises ValueError."""
    with pytest.raises(ValueError, match="robot_positions"):
        evaluate_actuator_feasibility(
            robot_positions=np.zeros((4, 3)),
            robot_velocities=np.zeros((4, 2)),
            dt_s=0.1,
            hazard_clearance_m=1.0,
        )


def test_velocity_shape_mismatch_raises() -> None:
    """A robot_velocities shape mismatch raises ValueError."""
    with pytest.raises(ValueError, match="robot_velocities"):
        evaluate_actuator_feasibility(
            robot_positions=np.zeros((4, 2)),
            robot_velocities=np.zeros((3, 2)),
            dt_s=0.1,
            hazard_clearance_m=1.0,
        )


def test_non_finite_inputs_raise() -> None:
    """NaN/inf inputs are rejected at the boundary, never silently accepted."""
    positions, velocities = _straight_trajectory()
    bad_pos = positions.copy()
    bad_pos[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        evaluate_actuator_feasibility(
            robot_positions=bad_pos,
            robot_velocities=velocities,
            dt_s=0.1,
            hazard_clearance_m=1.0,
        )

    bad_vel = velocities.copy()
    bad_vel[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        evaluate_actuator_feasibility(
            robot_positions=positions,
            robot_velocities=bad_vel,
            dt_s=0.1,
            hazard_clearance_m=1.0,
        )


def test_single_timestep_skips_rate_checks() -> None:
    """A one-step trajectory has no accel/yaw/steering rates; only the brake check runs."""
    report = evaluate_actuator_feasibility(
        robot_positions=np.array([[0.0, 0.0]]),
        robot_velocities=np.array([[0.3, 0.0]]),
        dt_s=0.1,
        hazard_clearance_m=2.0,
    )
    assert report.verdict == VERDICT_ACTUATOR_FEASIBLE
    assert report.observed_max_accel_mps2 is None
    assert report.observed_max_yaw_rate_radps is None
    assert report.observed_max_steering_rate_radps is None
    assert report.max_speed_mps == pytest.approx(0.3)
    assert report.stopping_distance_m is not None


def test_non_consecutive_moving_timesteps_skip_rate_checks() -> None:
    """Stopped samples must not create synthetic yaw or steering transitions."""
    report = evaluate_actuator_feasibility(
        robot_positions=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.05]]),
        robot_velocities=np.array([[0.5, 0.0], [0.0, 0.0], [0.0, 0.5]]),
        dt_s=0.1,
        hazard_clearance_m=5.0,
        config=ActuatorLimitsConfig(max_accel_mps2=10.0, max_decel_mps2=10.0),
    )
    assert report.verdict == VERDICT_ACTUATOR_FEASIBLE
    assert report.observed_max_yaw_rate_radps is None
    assert report.observed_max_steering_rate_radps is None
