"""Tests for the experimental AMMV trajectory verifier (issue #4757) and execution deviation monitor (issue #6584)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_sf.benchmark.trajectory_verifier import (
    DECISION_ACCEPT,
    DECISION_FALLBACK_BRAKE,
    DECISION_WARN,
    EXECUTION_DEVIATION_CLAIM_BOUNDARY,
    EXECUTION_DEVIATION_REPORT_SCHEMA,
    EXECUTION_DEVIATION_SCHEMA,
    INTERVENTION_CONTINUE,
    INTERVENTION_FALLBACK_BRAKE,
    INTERVENTION_REPLAN,
    INTERVENTION_WARN,
    PRED_BRAKING_INFEASIBLE,
    PRED_CLEARANCE_HARD,
    PRED_CLEARANCE_WARN,
    PRED_RECOVERY_SMOOTHNESS,
    PRED_STALE_OR_MISSING_STATE,
    PRED_TTC_HARD,
    PRED_TTC_WARN,
    TRAJECTORY_VERIFIER_CLAIM_BOUNDARY,
    TRAJECTORY_VERIFIER_SCHEMA,
    ExecutionDeviationConfig,
    ExecutionDeviationDiagnosticCase,
    ExecutionDeviationDiagnosticReport,
    ExecutionDeviationResult,
    TrajectoryVerifierConfig,
    VerifierResult,
    monitor_execution_deviation,
    summarize_execution_deviation_diagnostics,
    verify_episode_trace_window,
    verify_trajectory,
)


def _straight_trajectory(
    *,
    n_steps: int = 10,
    robot_speed: float = 0.5,
    ped_offset: tuple[float, float] = (5.0, 0.0),
    ped_velocity: tuple[float, float] = (0.0, 0.0),
    dt_s: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (robot_pos, robot_vel, ped_pos, ped_vel) for a straight robot path.

    Pedestrian arrays have shape ``(n_steps, 1, 2)`` (one pedestrian over time).
    """
    t = np.arange(n_steps, dtype=float) * dt_s
    robot_pos = np.stack([t * robot_speed, np.zeros(n_steps)], axis=1)
    robot_vel = np.tile([robot_speed, 0.0], (n_steps, 1))
    ped_pos = np.tile(np.array([ped_offset], dtype=float), (n_steps, 1, 1))
    ped_vel = np.tile(np.array([ped_velocity], dtype=float), (n_steps, 1, 1))
    return robot_pos, robot_vel, ped_pos, ped_vel


def test_accept_straight_path_pedestrian_far() -> None:
    """A straight path with a far-away pedestrian and finite clearance/TTC is accepted."""
    robot_pos, robot_vel, ped_pos, ped_vel = _straight_trajectory(
        ped_offset=(5.0, 2.0), robot_speed=0.5
    )
    result = verify_trajectory(
        robot_positions=robot_pos,
        robot_velocities=robot_vel,
        pedestrian_positions=ped_pos,
        pedestrian_velocities=ped_vel,
        dt_s=0.1,
        robot_radius_m=0.3,
        pedestrian_radius_m=0.3,
    )
    assert result.decision == DECISION_ACCEPT
    assert result.violated_predicates == ()
    assert result.risk_score == 0.0
    assert result.min_distance_m is not None and result.min_distance_m > 1.0
    assert result.min_clearance_m is not None and result.min_clearance_m > 0.5
    assert result.min_ttc_s is None or result.min_ttc_s > 1.5
    assert result.braking_feasible is True
    assert result.claim_boundary == TRAJECTORY_VERIFIER_CLAIM_BOUNDARY


def test_warn_pedestrian_near_warning_clearance() -> None:
    """A pedestrian within the warning clearance band (but above hard) triggers warn."""
    n = 10
    robot_pos = np.tile([0.0, 0.0], (n, 1))
    robot_vel = np.tile([0.0, 0.0], (n, 1))
    cfg = TrajectoryVerifierConfig(min_clearance_m=0.1, warn_clearance_m=0.5)
    # clearance = 0.7 - 0.2 = 0.5 -> exactly at warn threshold boundary; accept or warn.
    ped_pos = np.array([[0.7, 0.0]])
    ped_vel = np.array([[0.0, 0.0]])
    result = verify_trajectory(
        robot_positions=robot_pos,
        robot_velocities=robot_vel,
        pedestrian_positions=ped_pos,
        pedestrian_velocities=ped_vel,
        dt_s=0.1,
        robot_radius_m=0.1,
        pedestrian_radius_m=0.1,
        config=cfg,
    )
    assert result.decision in {DECISION_WARN, DECISION_ACCEPT}
    # Now place clearance strictly inside the warn band (0.1, 0.5).
    ped_pos = np.array([[0.55, 0.0]])
    result = verify_trajectory(
        robot_positions=robot_pos,
        robot_velocities=robot_vel,
        pedestrian_positions=ped_pos,
        pedestrian_velocities=ped_vel,
        dt_s=0.1,
        robot_radius_m=0.1,
        pedestrian_radius_m=0.1,
        config=cfg,
    )
    assert result.decision == DECISION_WARN
    assert PRED_CLEARANCE_WARN in result.violated_predicates
    assert PRED_CLEARANCE_HARD not in result.violated_predicates
    assert 0.0 < result.risk_score <= 0.5


def test_warn_stale_prediction_age() -> None:
    """A prediction age above the stale threshold triggers a warn predicate."""
    robot_pos, robot_vel, ped_pos, ped_vel = _straight_trajectory(ped_offset=(5.0, 5.0))
    cfg = TrajectoryVerifierConfig(stale_prediction_max_age_s=0.2)
    result = verify_trajectory(
        robot_positions=robot_pos,
        robot_velocities=robot_vel,
        pedestrian_positions=ped_pos,
        pedestrian_velocities=ped_vel,
        dt_s=0.1,
        robot_radius_m=0.3,
        pedestrian_radius_m=0.3,
        config=cfg,
        prediction_age_s=0.5,
    )
    assert result.decision == DECISION_WARN
    assert PRED_STALE_OR_MISSING_STATE in result.violated_predicates
    assert result.risk_score >= 0.3


def test_fallback_clearance_below_hard_minimum() -> None:
    """A pedestrian within the hard clearance minimum triggers fallback_brake."""
    n = 5
    robot_pos = np.tile([0.0, 0.0], (n, 1))
    robot_vel = np.tile([0.0, 0.0], (n, 1))
    ped_pos = np.array([[0.4, 0.0]])  # clearance = 0.4 - 0.6 = -0.2 < min_clearance
    ped_vel = np.array([[0.0, 0.0]])
    result = verify_trajectory(
        robot_positions=robot_pos,
        robot_velocities=robot_vel,
        pedestrian_positions=ped_pos,
        pedestrian_velocities=ped_vel,
        dt_s=0.1,
        robot_radius_m=0.3,
        pedestrian_radius_m=0.3,
    )
    assert result.decision == DECISION_FALLBACK_BRAKE
    assert PRED_CLEARANCE_HARD in result.violated_predicates
    assert result.risk_score == 1.0
    assert result.min_clearance_m is not None and result.min_clearance_m < 0.25


def test_fallback_ttc_below_hard_minimum() -> None:
    """A pedestrian on a head-on collision course triggers the hard TTC predicate."""
    n = 20
    dt = 0.1
    t = np.arange(n, dtype=float) * dt
    # Robot at x=t*1.0 moving +x; ped at x=5.0-t*1.0 moving -x; closing at 2.0 m/s.
    robot_pos = np.stack([t * 1.0, np.zeros(n)], axis=1)
    robot_vel = np.tile([1.0, 0.0], (n, 1))
    ped_pos = np.stack([5.0 - t * 1.0, np.zeros(n)], axis=1)
    ped_vel = np.tile([-1.0, 0.0], (n, 1))
    cfg = TrajectoryVerifierConfig(min_ttc_s=1.0, warn_ttc_s=1.5, min_clearance_m=0.05)
    result = verify_trajectory(
        robot_positions=robot_pos,
        robot_velocities=robot_vel,
        pedestrian_positions=ped_pos,
        pedestrian_velocities=ped_vel,
        dt_s=dt,
        robot_radius_m=0.05,
        pedestrian_radius_m=0.05,
        config=cfg,
    )
    assert result.decision == DECISION_FALLBACK_BRAKE
    assert PRED_TTC_HARD in result.violated_predicates
    assert result.min_ttc_s is not None and result.min_ttc_s < 1.0


def test_fallback_braking_infeasible() -> None:
    """A pedestrian ahead within the stopping distance triggers braking-infeasible."""
    n = 5
    robot_pos = np.stack([np.linspace(0.0, 0.4, n), np.zeros(n)], axis=1)
    robot_vel = np.tile([2.0, 0.0], (n, 1))  # 2 m/s; d_stop = 4/(2*2.5) = 0.8 m
    ped_pos = np.array([[0.6, 0.0]])  # 0.2..0.6 m ahead, < 0.8 m stopping distance
    ped_vel = np.array([[0.0, 0.0]])
    cfg = TrajectoryVerifierConfig(
        min_clearance_m=0.05, warn_clearance_m=0.1, min_ttc_s=0.1, warn_ttc_s=0.2
    )
    result = verify_trajectory(
        robot_positions=robot_pos,
        robot_velocities=robot_vel,
        pedestrian_positions=ped_pos,
        pedestrian_velocities=ped_vel,
        dt_s=0.1,
        robot_radius_m=0.1,
        pedestrian_radius_m=0.1,
        config=cfg,
    )
    assert result.decision == DECISION_FALLBACK_BRAKE
    assert PRED_BRAKING_INFEASIBLE in result.violated_predicates
    assert result.braking_feasible is False


def test_missing_pedestrian_velocity_warns_not_fabricated() -> None:
    """Missing pedestrian velocities surface as a warn predicate, never a fabricated TTC."""
    robot_pos, robot_vel, ped_pos, _ = _straight_trajectory(ped_offset=(2.0, 0.0))
    result = verify_trajectory(
        robot_positions=robot_pos,
        robot_velocities=robot_vel,
        pedestrian_positions=ped_pos,
        pedestrian_velocities=None,
        dt_s=0.1,
        robot_radius_m=0.3,
        pedestrian_radius_m=0.3,
    )
    assert result.decision == DECISION_WARN
    assert PRED_STALE_OR_MISSING_STATE in result.violated_predicates
    assert PRED_TTC_HARD not in result.violated_predicates
    assert PRED_TTC_WARN not in result.violated_predicates
    assert result.min_ttc_s is None
    assert result.braking_feasible is None


def test_missing_robot_velocity_warns_and_skips_braking() -> None:
    """Missing robot velocity surfaces as warn and skips braking-feasibility evaluation."""
    _, _, ped_pos, ped_vel = _straight_trajectory(ped_offset=(3.0, 0.0))
    robot_pos = np.zeros((10, 2))
    result = verify_trajectory(
        robot_positions=robot_pos,
        robot_velocities=None,
        pedestrian_positions=ped_pos,
        pedestrian_velocities=ped_vel,
        dt_s=0.1,
        robot_radius_m=0.3,
        pedestrian_radius_m=0.3,
    )
    assert result.decision == DECISION_WARN
    assert PRED_STALE_OR_MISSING_STATE in result.violated_predicates
    assert result.braking_feasible is None
    assert result.min_ttc_s is None


def test_oscillatory_trajectory_triggers_smoothness_warn() -> None:
    """A trajectory with many large heading changes triggers the recovery-smoothness warn."""
    n = 16
    # Alternate heading by ~pi/2 each step above min speed; expect > 3 oscillations.
    angles = np.array([(0.0 if i % 2 == 0 else np.pi / 2) for i in range(n)])
    robot_vel = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 0.5
    robot_pos = np.cumsum(robot_vel * 0.1, axis=0)
    ped_pos = np.array([[10.0, 10.0]])
    ped_vel = np.array([[0.0, 0.0]])
    cfg = TrajectoryVerifierConfig(max_heading_oscillation_count=3)
    result = verify_trajectory(
        robot_positions=robot_pos,
        robot_velocities=robot_vel,
        pedestrian_positions=ped_pos,
        pedestrian_velocities=ped_vel,
        dt_s=0.1,
        robot_radius_m=0.2,
        pedestrian_radius_m=0.2,
        config=cfg,
    )
    assert result.decision == DECISION_WARN
    assert PRED_RECOVERY_SMOOTHNESS in result.violated_predicates
    assert result.risk_score > 0.0


def test_invalid_robot_positions_shape_raises() -> None:
    """A robot_positions array that is not (T, 2) raises a clear ValueError."""
    with pytest.raises(ValueError, match="robot_positions"):
        verify_trajectory(
            robot_positions=np.zeros((5, 3)),
            robot_velocities=None,
            pedestrian_positions=np.zeros((1, 2)),
            pedestrian_velocities=None,
            dt_s=0.1,
            robot_radius_m=0.3,
            pedestrian_radius_m=0.3,
        )


def test_invalid_pedestrian_positions_time_dim_raises() -> None:
    """A pedestrian_positions time dim mismatch raises a clear ValueError."""
    with pytest.raises(ValueError, match="pedestrian_positions"):
        verify_trajectory(
            robot_positions=np.zeros((5, 2)),
            robot_velocities=None,
            pedestrian_positions=np.zeros((3, 1, 2)),
            pedestrian_velocities=None,
            dt_s=0.1,
            robot_radius_m=0.3,
            pedestrian_radius_m=0.3,
        )


def test_invalid_robot_velocity_shape_raises() -> None:
    """A robot_velocities shape mismatch raises a clear ValueError."""
    with pytest.raises(ValueError, match="robot_velocities"):
        verify_trajectory(
            robot_positions=np.zeros((5, 2)),
            robot_velocities=np.zeros((4, 2)),
            pedestrian_positions=np.zeros((1, 2)),
            pedestrian_velocities=None,
            dt_s=0.1,
            robot_radius_m=0.3,
            pedestrian_radius_m=0.3,
        )


def test_invalid_pedestrian_velocity_shape_raises() -> None:
    """A pedestrian_velocities shape mismatch raises a clear ValueError."""
    with pytest.raises(ValueError, match="pedestrian_velocities"):
        verify_trajectory(
            robot_positions=np.zeros((5, 2)),
            robot_velocities=np.zeros((5, 2)),
            pedestrian_positions=np.zeros((5, 2)),
            pedestrian_velocities=np.zeros((4, 2)),
            dt_s=0.1,
            robot_radius_m=0.3,
            pedestrian_radius_m=0.3,
        )


def test_non_positive_dt_raises() -> None:
    """A non-positive dt_s raises ValueError."""
    robot_pos = np.zeros((3, 2))
    ped_pos = np.tile([2.0, 0.0], (3, 1))
    with pytest.raises(ValueError, match="dt_s"):
        verify_trajectory(
            robot_positions=robot_pos,
            robot_velocities=None,
            pedestrian_positions=ped_pos,
            pedestrian_velocities=None,
            dt_s=0.0,
            robot_radius_m=0.3,
            pedestrian_radius_m=0.3,
        )


def test_negative_radius_raises() -> None:
    """A negative radius raises ValueError."""
    robot_pos = np.zeros((3, 2))
    ped_pos = np.tile([2.0, 0.0], (3, 1))
    with pytest.raises(ValueError, match="robot_radius_m"):
        verify_trajectory(
            robot_positions=robot_pos,
            robot_velocities=None,
            pedestrian_positions=ped_pos,
            pedestrian_velocities=None,
            dt_s=0.1,
            robot_radius_m=-0.1,
            pedestrian_radius_m=0.3,
        )


def test_invalid_config_thresholds_raise() -> None:
    """Invalid TrajectoryVerifierConfig thresholds raise ValueError."""
    with pytest.raises(ValueError, match="warn_clearance_m"):
        TrajectoryVerifierConfig(min_clearance_m=0.5, warn_clearance_m=0.3)
    with pytest.raises(ValueError, match="warn_ttc_s"):
        TrajectoryVerifierConfig(min_ttc_s=2.0, warn_ttc_s=1.0)
    with pytest.raises(ValueError, match="max_brake_deceleration_mps2"):
        TrajectoryVerifierConfig(max_brake_deceleration_mps2=0.0)


def test_decision_precedence_fallback_over_warn() -> None:
    """A trajectory that fires both hard and soft predicates reports fallback_brake."""
    n = 5
    robot_pos = np.tile([0.0, 0.0], (n, 1))
    robot_vel = np.tile([0.0, 0.0], (n, 1))
    ped_pos = np.array([[0.3, 0.0]])  # clearance -0.3 < hard minimum
    ped_vel = np.array([[0.0, 0.0]])
    # Also stale to ensure warn would fire if hard did not.
    cfg = TrajectoryVerifierConfig(stale_prediction_max_age_s=0.1)
    result = verify_trajectory(
        robot_positions=robot_pos,
        robot_velocities=robot_vel,
        pedestrian_positions=ped_pos,
        pedestrian_velocities=ped_vel,
        dt_s=0.1,
        robot_radius_m=0.3,
        pedestrian_radius_m=0.3,
        config=cfg,
        prediction_age_s=0.5,
    )
    assert result.decision == DECISION_FALLBACK_BRAKE
    assert PRED_CLEARANCE_HARD in result.violated_predicates
    assert PRED_STALE_OR_MISSING_STATE in result.violated_predicates
    assert result.risk_score == 1.0


def test_static_pedestrian_positions_broadcast() -> None:
    """A static (N, 2) pedestrian array is broadcast across the robot time dim."""
    n = 10
    robot_pos = np.stack([np.linspace(0.0, 1.0, n), np.zeros(n)], axis=1)
    robot_vel = np.tile([0.1, 0.0], (n, 1))
    ped_pos_static = np.array([[5.0, 5.0]])  # shape (1, 2); broadcast to (n, 1, 2)
    ped_vel = np.zeros((n, 1, 2))
    result = verify_trajectory(
        robot_positions=robot_pos,
        robot_velocities=robot_vel,
        pedestrian_positions=ped_pos_static,
        pedestrian_velocities=ped_vel,
        dt_s=0.1,
        robot_radius_m=0.3,
        pedestrian_radius_m=0.3,
    )
    assert result.decision == DECISION_ACCEPT
    assert result.min_distance_m is not None and result.min_distance_m > 4.0


def test_verify_episode_trace_window_accepts_and_windows() -> None:
    """The opt-in trace-window helper slices and returns the same verifier contract."""
    robot_pos, robot_vel, ped_pos, ped_vel = _straight_trajectory(
        n_steps=20, ped_offset=(10.0, 0.0)
    )
    trace = {
        "robot_positions": robot_pos,
        "robot_velocities": robot_vel,
        "pedestrian_positions": ped_pos,
        "pedestrian_velocities": ped_vel,
        "dt_s": 0.1,
    }
    result = verify_episode_trace_window(trace, start=0, end=10)
    assert isinstance(result, VerifierResult)
    assert result.decision == DECISION_ACCEPT


def test_verify_episode_trace_window_missing_keys_raises() -> None:
    """The trace helper raises ValueError when required keys are missing."""
    with pytest.raises(ValueError, match="robot_positions"):
        verify_episode_trace_window({"pedestrian_positions": np.zeros((1, 2))})


def test_verify_episode_trace_window_empty_window_raises() -> None:
    """The trace helper raises ValueError for an empty window."""
    trace = {
        "robot_positions": np.zeros((5, 2)),
        "pedestrian_positions": np.zeros((5, 1, 2)),
        "dt_s": 0.1,
    }
    with pytest.raises(ValueError, match="window"):
        verify_episode_trace_window(trace, start=3, end=3)


def test_schema_and_claim_boundary_constants() -> None:
    """Schema and claim-boundary constants are stable and explicit."""
    assert TRAJECTORY_VERIFIER_SCHEMA == "trajectory_verifier.v1"
    assert "not a formal safety case" in TRAJECTORY_VERIFIER_CLAIM_BOUNDARY
    assert "not learned" in TRAJECTORY_VERIFIER_CLAIM_BOUNDARY
    assert "default planner behavior unchanged" in TRAJECTORY_VERIFIER_CLAIM_BOUNDARY


def test_braking_accounts_for_footprint_radii() -> None:
    """Braking is infeasible when the robot would stop inside the pedestrian footprint.

    Regression for the sum_radii gap: at 2 m/s ``d_stop = 0.8 m``. With the
    nearest along-heading center distance at 0.9 m and ``sum_radii = 0.2 m`` the
    available braking distance is ``0.9 - 0.2 = 0.7 m < 0.8 m`` -> infeasible.
    Ignoring the radii (comparing 0.9 m directly to 0.8 m) would wrongly call
    braking feasible even though the robot halts inside the pedestrian.
    """
    n = 5
    robot_pos = np.stack([np.linspace(0.0, 0.4, n), np.zeros(n)], axis=1)
    robot_vel = np.tile([2.0, 0.0], (n, 1))  # d_stop = 4 / (2 * 2.5) = 0.8 m
    ped_pos = np.array([[1.3, 0.0]])  # min along-distance = 1.3 - 0.4 = 0.9 m
    ped_vel = np.array([[0.0, 0.0]])
    # Loose clearance/TTC thresholds so only the braking predicate can fire here.
    cfg = TrajectoryVerifierConfig(
        min_clearance_m=0.05, warn_clearance_m=0.1, min_ttc_s=0.1, warn_ttc_s=0.2
    )
    result = verify_trajectory(
        robot_positions=robot_pos,
        robot_velocities=robot_vel,
        pedestrian_positions=ped_pos,
        pedestrian_velocities=ped_vel,
        dt_s=0.1,
        robot_radius_m=0.1,
        pedestrian_radius_m=0.1,  # sum_radii = 0.2 m
        config=cfg,
    )
    assert result.braking_feasible is False
    assert PRED_BRAKING_INFEASIBLE in result.violated_predicates


def test_heading_oscillation_counted_across_a_pause() -> None:
    """Heading reversals separated by a stopped timestep still count as oscillations.

    Regression for the adjacent-pair mask: previously a turn straddling a
    non-moving timestep was dropped because both bordering pairs contained a
    stopped step. Filtering to the moving subsequence first counts the reversal
    between consecutive *moving* timesteps regardless of intervening pauses.
    """
    velocities: list[list[float]] = []
    move_idx = 0
    for i in range(20):
        if i % 2 == 1:
            velocities.append([0.0, 0.0])  # pause between every moving step
        else:
            heading = 0.0 if move_idx % 2 == 0 else np.pi  # alternate east/west
            velocities.append([0.5 * float(np.cos(heading)), 0.5 * float(np.sin(heading))])
            move_idx += 1
    robot_vel = np.array(velocities)
    robot_pos = np.cumsum(robot_vel * 0.1, axis=0)
    ped_pos = np.array([[10.0, 10.0]])
    ped_vel = np.array([[0.0, 0.0]])
    cfg = TrajectoryVerifierConfig(max_heading_oscillation_count=3)
    result = verify_trajectory(
        robot_positions=robot_pos,
        robot_velocities=robot_vel,
        pedestrian_positions=ped_pos,
        pedestrian_velocities=ped_vel,
        dt_s=0.1,
        robot_radius_m=0.2,
        pedestrian_radius_m=0.2,
        config=cfg,
    )
    assert PRED_RECOVERY_SMOOTHNESS in result.violated_predicates


def test_non_finite_inputs_raise() -> None:
    """NaN/inf inputs are rejected at the boundary, never silently accepted.

    A ``nan`` in a position would defeat threshold comparisons (``nan < x`` is
    ``False``), so the verifier must fail closed with a clear ValueError.
    """
    robot_pos, robot_vel, ped_pos, ped_vel = _straight_trajectory(ped_offset=(5.0, 0.0))

    bad_robot = robot_pos.copy()
    bad_robot[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        verify_trajectory(
            robot_positions=bad_robot,
            robot_velocities=robot_vel,
            pedestrian_positions=ped_pos,
            pedestrian_velocities=ped_vel,
            dt_s=0.1,
            robot_radius_m=0.3,
            pedestrian_radius_m=0.3,
        )

    bad_ped = ped_pos.copy()
    bad_ped[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        verify_trajectory(
            robot_positions=robot_pos,
            robot_velocities=robot_vel,
            pedestrian_positions=bad_ped,
            pedestrian_velocities=ped_vel,
            dt_s=0.1,
            robot_radius_m=0.3,
            pedestrian_radius_m=0.3,
        )


# ---------------------------------------------------------------------------
# Execution-time deviation monitor tests (issue #6584)
# ---------------------------------------------------------------------------

# Calibration fixture: thresholds derived from this fixture only.
# Evaluation tests below use scenarios disjoint from calibration data.
_CALIBRATION_SOURCE = "test_calibration_fixture_v1"
_EVALUATION_SOURCE = "test_evaluation_fixture_v1"


def _deviation_config(**overrides: object) -> ExecutionDeviationConfig:
    """Return a config calibrated from the test calibration fixture."""
    defaults: dict[str, object] = {
        "warn_threshold": 0.5,
        "replan_threshold": 1.0,
        "fallback_brake_threshold": 2.0,
        "max_input_age_s": 0.5,
        "fail_closed_intervention": "warn",
        "calibration_source": _CALIBRATION_SOURCE,
        "evaluation_source": _EVALUATION_SOURCE,
    }
    defaults.update(overrides)
    return ExecutionDeviationConfig(**defaults)  # type: ignore[arg-type]


def _aligned_windows(
    n_steps: int = 10,
    dt_s: float = 0.1,
    robot_speed: float = 1.0,
    deviation: float = 0.0,
) -> dict[str, object]:
    """Return aligned predicted/observed windows with a constant position deviation.

    The predicted trajectory is a straight line at ``robot_speed`` along +x.
    The observed trajectory is offset by ``deviation`` in the y direction.
    """
    t = np.arange(n_steps, dtype=float) * dt_s
    pred_pos = np.stack([t * robot_speed, np.zeros(n_steps)], axis=1)
    obs_pos = pred_pos.copy()
    obs_pos[:, 1] += deviation
    pred_vel = np.tile([robot_speed, 0.0], (n_steps, 1))
    obs_vel = pred_vel.copy()
    return {
        "predicted_robot_positions": pred_pos,
        "observed_robot_positions": obs_pos,
        "predicted_robot_velocities": pred_vel,
        "observed_robot_velocities": obs_vel,
        "dt_s": dt_s,
    }


class TestExecutionDeviationConfig:
    """Validation and provenance tests for ExecutionDeviationConfig."""

    def test_valid_config(self) -> None:
        cfg = _deviation_config()
        assert cfg.schema_version == EXECUTION_DEVIATION_SCHEMA
        assert cfg.calibration_source == _CALIBRATION_SOURCE
        assert cfg.evaluation_source == _EVALUATION_SOURCE

    def test_empty_calibration_source_raises(self) -> None:
        with pytest.raises(ValueError, match="calibration_source"):
            ExecutionDeviationConfig(calibration_source="")

    def test_empty_evaluation_source_raises(self) -> None:
        with pytest.raises(ValueError, match="evaluation_source"):
            _deviation_config(evaluation_source="")

    def test_threshold_ordering_enforced(self) -> None:
        with pytest.raises(ValueError, match="replan_threshold"):
            _deviation_config(warn_threshold=1.0, replan_threshold=0.5)
        with pytest.raises(ValueError, match="fallback_brake_threshold"):
            _deviation_config(replan_threshold=2.0, fallback_brake_threshold=1.0)

    def test_negative_warn_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="warn_threshold"):
            _deviation_config(warn_threshold=-0.1)

    def test_non_positive_max_input_age_raises(self) -> None:
        with pytest.raises(ValueError, match="max_input_age_s"):
            _deviation_config(max_input_age_s=0.0)

    @pytest.mark.parametrize(
        "field_name",
        ("warn_threshold", "replan_threshold", "fallback_brake_threshold", "max_input_age_s"),
    )
    def test_non_finite_thresholds_raise(self, field_name: str) -> None:
        """Configuration rejects non-finite values that could bypass precedence."""
        with pytest.raises(ValueError, match=field_name):
            _deviation_config(**{field_name: math.nan})

    def test_invalid_fail_closed_intervention_raises(self) -> None:
        with pytest.raises(ValueError, match="fail_closed_intervention"):
            _deviation_config(fail_closed_intervention="continue")


class TestExecutionDeviationCleanExecution:
    """Clean execution: predicted matches observed, no deviation."""

    def test_identical_windows_continue(self) -> None:
        windows = _aligned_windows(deviation=0.0)
        result = monitor_execution_deviation(config=_deviation_config(), **windows)
        assert result.intervention == INTERVENTION_CONTINUE
        assert result.deviation_score == pytest.approx(0.0)
        assert result.fail_closed is False
        assert result.first_threshold_crossing_time_s is None
        assert result.claim_boundary == EXECUTION_DEVIATION_CLAIM_BOUNDARY
        assert result.schema_version == EXECUTION_DEVIATION_SCHEMA

    def test_small_deviation_below_warn_continues(self) -> None:
        windows = _aligned_windows(deviation=0.1)
        result = monitor_execution_deviation(config=_deviation_config(), **windows)
        assert result.intervention == INTERVENTION_CONTINUE
        assert result.deviation_score is not None
        assert result.deviation_score < 0.5

    def test_component_deviations_present(self) -> None:
        windows = _aligned_windows(deviation=0.0)
        result = monitor_execution_deviation(config=_deviation_config(), **windows)
        components = dict(result.component_deviations)
        assert "robot_position" in components
        assert "robot_velocity" in components


class TestExecutionDeviationPedestrianCourseChange:
    """Pedestrian course change: observed pedestrian deviates from prediction."""

    def test_pedestrian_course_change_detected(self) -> None:
        n = 10
        dt = 0.1
        t = np.arange(n, dtype=float) * dt
        pred_robot = np.stack([t * 1.0, np.zeros(n)], axis=1)
        obs_robot = pred_robot.copy()
        # Predicted ped walks straight; observed ped changes course at t=0.3s.
        pred_ped = np.tile([3.0, 2.0], (n, 1, 1)).astype(float)
        obs_ped = pred_ped.copy()
        obs_ped[3:, 0, 0] += 1.5  # course change: 1.5m x-deviation from step 3
        cfg = _deviation_config(warn_threshold=0.3)
        result = monitor_execution_deviation(
            predicted_robot_positions=pred_robot,
            observed_robot_positions=obs_robot,
            predicted_pedestrian_positions=pred_ped,
            observed_pedestrian_positions=obs_ped,
            dt_s=dt,
            config=cfg,
        )
        components = dict(result.component_deviations)
        assert "pedestrian_position" in components
        assert components["pedestrian_position"] > 0.3
        assert result.intervention in (INTERVENTION_WARN, INTERVENTION_REPLAN)
        assert result.first_threshold_crossing_time_s is not None
        assert result.first_threshold_crossing_time_s >= 0.3 - dt

    def test_pedestrian_deviation_contributes_to_score(self) -> None:
        n = 10
        pred_robot = np.zeros((n, 2))
        obs_robot = np.zeros((n, 2))
        pred_ped = np.zeros((n, 1, 2))
        obs_ped = np.zeros((n, 1, 2))
        obs_ped[:, 0, 0] = 3.0  # large pedestrian deviation (single ped)
        cfg = _deviation_config()
        result = monitor_execution_deviation(
            predicted_robot_positions=pred_robot,
            observed_robot_positions=obs_robot,
            predicted_pedestrian_positions=pred_ped,
            observed_pedestrian_positions=obs_ped,
            dt_s=0.1,
            config=cfg,
        )
        assert result.deviation_score is not None
        assert result.deviation_score >= 3.0
        assert result.intervention == INTERVENTION_FALLBACK_BRAKE


class TestExecutionDeviationActuatorDelay:
    """Actuator delay: observed robot lags behind predicted positions."""

    def test_actuator_delay_detected(self) -> None:
        n = 20
        dt = 0.1
        t = np.arange(n, dtype=float) * dt
        pred_pos = np.stack([t * 1.0, np.zeros(n)], axis=1)
        # Observed lags by 2 steps (0.2s delay): obs[t] = pred[t-2].
        obs_pos = np.zeros_like(pred_pos)
        obs_pos[2:] = pred_pos[:-2]
        obs_pos[:2] = pred_pos[0]
        pred_vel = np.tile([1.0, 0.0], (n, 1))
        obs_vel = np.zeros_like(pred_vel)
        obs_vel[2:] = pred_vel[:-2]
        cfg = _deviation_config(warn_threshold=0.1)
        result = monitor_execution_deviation(
            predicted_robot_positions=pred_pos,
            observed_robot_positions=obs_pos,
            predicted_robot_velocities=pred_vel,
            observed_robot_velocities=obs_vel,
            dt_s=dt,
            config=cfg,
        )
        assert result.intervention in (INTERVENTION_WARN, INTERVENTION_REPLAN)
        assert result.deviation_score is not None
        assert result.deviation_score > 0.1
        assert result.first_threshold_crossing_time_s is not None


class TestExecutionDeviationLocalizationBias:
    """Localization bias: constant offset between predicted and observed."""

    def test_constant_bias_detected(self) -> None:
        windows = _aligned_windows(deviation=0.8)
        cfg = _deviation_config(warn_threshold=0.5)
        result = monitor_execution_deviation(config=cfg, **windows)
        assert result.intervention == INTERVENTION_WARN
        assert result.deviation_score == pytest.approx(0.8, abs=0.01)

    def test_large_bias_triggers_replan(self) -> None:
        windows = _aligned_windows(deviation=1.5)
        cfg = _deviation_config()
        result = monitor_execution_deviation(config=cfg, **windows)
        assert result.intervention == INTERVENTION_REPLAN
        assert result.deviation_score == pytest.approx(1.5, abs=0.01)

    def test_very_large_bias_triggers_fallback_brake(self) -> None:
        windows = _aligned_windows(deviation=2.5)
        cfg = _deviation_config()
        result = monitor_execution_deviation(config=cfg, **windows)
        assert result.intervention == INTERVENTION_FALLBACK_BRAKE


class TestExecutionDeviationStaleInput:
    """Stale or missing inputs fail closed without fabricating a score."""

    def test_stale_input_fails_closed_warn(self) -> None:
        windows = _aligned_windows(deviation=0.0)
        cfg = _deviation_config(fail_closed_intervention="warn")
        result = monitor_execution_deviation(config=cfg, input_age_s=1.0, **windows)
        assert result.intervention == INTERVENTION_WARN
        assert result.deviation_score is None
        assert result.component_deviations == ()
        assert result.fail_closed is True
        assert result.input_age_s == 1.0

    def test_stale_input_fails_closed_fallback_brake(self) -> None:
        windows = _aligned_windows(deviation=0.0)
        cfg = _deviation_config(fail_closed_intervention="fallback_brake")
        result = monitor_execution_deviation(config=cfg, input_age_s=1.0, **windows)
        assert result.intervention == INTERVENTION_FALLBACK_BRAKE
        assert result.deviation_score is None
        assert result.fail_closed is True

    def test_missing_predicted_positions_fails_closed(self) -> None:
        cfg = _deviation_config()
        result = monitor_execution_deviation(
            predicted_robot_positions=None,
            observed_robot_positions=np.zeros((5, 2)),
            dt_s=0.1,
            config=cfg,
        )
        assert result.fail_closed is True
        assert result.deviation_score is None
        assert result.intervention == INTERVENTION_WARN

    def test_missing_observed_positions_fails_closed(self) -> None:
        cfg = _deviation_config()
        result = monitor_execution_deviation(
            predicted_robot_positions=np.zeros((5, 2)),
            observed_robot_positions=None,
            dt_s=0.1,
            config=cfg,
        )
        assert result.fail_closed is True
        assert result.deviation_score is None

    def test_input_age_at_boundary_not_stale(self) -> None:
        windows = _aligned_windows(deviation=0.0)
        cfg = _deviation_config(max_input_age_s=0.5)
        result = monitor_execution_deviation(config=cfg, input_age_s=0.5, **windows)
        assert result.fail_closed is False
        assert result.deviation_score is not None

    @pytest.mark.parametrize("input_age_s", (math.nan, -0.1))
    def test_invalid_input_age_fails_closed(self, input_age_s: float) -> None:
        """Invalid input age never yields a numeric deviation result."""
        result = monitor_execution_deviation(
            config=_deviation_config(), input_age_s=input_age_s, **_aligned_windows()
        )
        assert result.fail_closed is True
        assert result.deviation_score is None
        assert result.input_age_s is None


class TestExecutionDeviationMisalignedInputs:
    """Misaligned or non-finite inputs fail closed."""

    def test_shape_mismatch_fails_closed(self) -> None:
        cfg = _deviation_config()
        result = monitor_execution_deviation(
            predicted_robot_positions=np.zeros((5, 2)),
            observed_robot_positions=np.zeros((7, 2)),
            dt_s=0.1,
            config=cfg,
        )
        assert result.fail_closed is True
        assert result.deviation_score is None

    def test_wrong_ndim_fails_closed(self) -> None:
        cfg = _deviation_config()
        result = monitor_execution_deviation(
            predicted_robot_positions=np.zeros((5, 3)),
            observed_robot_positions=np.zeros((5, 3)),
            dt_s=0.1,
            config=cfg,
        )
        assert result.fail_closed is True

    def test_empty_array_fails_closed(self) -> None:
        cfg = _deviation_config()
        result = monitor_execution_deviation(
            predicted_robot_positions=np.zeros((0, 2)),
            observed_robot_positions=np.zeros((0, 2)),
            dt_s=0.1,
            config=cfg,
        )
        assert result.fail_closed is True

    def test_non_finite_predicted_fails_closed(self) -> None:
        pred = np.zeros((5, 2))
        pred[2, 0] = np.nan
        cfg = _deviation_config()
        result = monitor_execution_deviation(
            predicted_robot_positions=pred,
            observed_robot_positions=np.zeros((5, 2)),
            dt_s=0.1,
            config=cfg,
        )
        assert result.fail_closed is True
        assert result.deviation_score is None

    def test_non_finite_observed_fails_closed(self) -> None:
        obs = np.zeros((5, 2))
        obs[0, 1] = np.inf
        cfg = _deviation_config()
        result = monitor_execution_deviation(
            predicted_robot_positions=np.zeros((5, 2)),
            observed_robot_positions=obs,
            dt_s=0.1,
            config=cfg,
        )
        assert result.fail_closed is True

    def test_velocity_shape_mismatch_fails_closed(self) -> None:
        cfg = _deviation_config()
        result = monitor_execution_deviation(
            predicted_robot_positions=np.zeros((5, 2)),
            observed_robot_positions=np.zeros((5, 2)),
            predicted_robot_velocities=np.zeros((5, 2)),
            observed_robot_velocities=np.zeros((4, 2)),
            dt_s=0.1,
            config=cfg,
        )
        assert result.fail_closed is True

    def test_pedestrian_shape_mismatch_fails_closed(self) -> None:
        cfg = _deviation_config()
        result = monitor_execution_deviation(
            predicted_robot_positions=np.zeros((5, 2)),
            observed_robot_positions=np.zeros((5, 2)),
            predicted_pedestrian_positions=np.zeros((5, 2, 2)),
            observed_pedestrian_positions=np.zeros((5, 3, 2)),
            dt_s=0.1,
            config=cfg,
        )
        assert result.fail_closed is True

    def test_pedestrian_time_mismatch_fails_closed(self) -> None:
        cfg = _deviation_config()
        result = monitor_execution_deviation(
            predicted_robot_positions=np.zeros((5, 2)),
            observed_robot_positions=np.zeros((5, 2)),
            predicted_pedestrian_positions=np.zeros((3, 2, 2)),
            observed_pedestrian_positions=np.zeros((3, 2, 2)),
            dt_s=0.1,
            config=cfg,
        )
        assert result.fail_closed is True


class TestExecutionDeviationSplitOverlapRejection:
    """Calibration and evaluation fixtures must be disjoint."""

    def test_calibration_source_recorded(self) -> None:
        cfg = _deviation_config()
        assert cfg.calibration_source == _CALIBRATION_SOURCE
        assert cfg.calibration_source != ""
        assert cfg.evaluation_source == _EVALUATION_SOURCE
        assert cfg.evaluation_source != cfg.calibration_source

    def test_calibration_evaluation_overlap_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must differ"):
            _deviation_config(evaluation_source=_CALIBRATION_SOURCE)


class TestExecutionDeviationInterventionPrecedence:
    """Intervention labels follow explicit precedence ordering."""

    def test_precedence_continue_below_warn(self) -> None:
        windows = _aligned_windows(deviation=0.1)
        cfg = _deviation_config(warn_threshold=0.5)
        result = monitor_execution_deviation(config=cfg, **windows)
        assert result.intervention == INTERVENTION_CONTINUE

    def test_precedence_warn_above_warn_below_replan(self) -> None:
        windows = _aligned_windows(deviation=0.7)
        cfg = _deviation_config(warn_threshold=0.5, replan_threshold=1.0)
        result = monitor_execution_deviation(config=cfg, **windows)
        assert result.intervention == INTERVENTION_WARN

    def test_precedence_replan_above_replan_below_fallback(self) -> None:
        windows = _aligned_windows(deviation=1.5)
        cfg = _deviation_config(replan_threshold=1.0, fallback_brake_threshold=2.0)
        result = monitor_execution_deviation(config=cfg, **windows)
        assert result.intervention == INTERVENTION_REPLAN

    def test_precedence_fallback_brake_above_fallback_threshold(self) -> None:
        windows = _aligned_windows(deviation=2.5)
        cfg = _deviation_config(fallback_brake_threshold=2.0)
        result = monitor_execution_deviation(config=cfg, **windows)
        assert result.intervention == INTERVENTION_FALLBACK_BRAKE


class TestExecutionDeviationMiscellaneous:
    """Additional contract and edge-case tests."""

    def test_non_positive_dt_raises(self) -> None:
        """Non-positive or non-finite timesteps are rejected before scoring."""
        with pytest.raises(ValueError, match="dt_s"):
            monitor_execution_deviation(
                predicted_robot_positions=np.zeros((5, 2)),
                observed_robot_positions=np.zeros((5, 2)),
                dt_s=0.0,
                config=_deviation_config(),
            )
        with pytest.raises(ValueError, match="dt_s"):
            monitor_execution_deviation(
                predicted_robot_positions=np.zeros((5, 2)),
                observed_robot_positions=np.zeros((5, 2)),
                dt_s=math.nan,
                config=_deviation_config(),
            )

    def test_missing_config_rejected(self) -> None:
        """The monitor requires explicit calibration provenance before scoring."""
        with pytest.raises(ValueError, match="calibration_source"):
            monitor_execution_deviation(
                predicted_robot_positions=np.zeros((5, 2)),
                observed_robot_positions=np.zeros((5, 2)),
                dt_s=0.1,
            )

    def test_empty_pedestrian_window_fails_closed(self) -> None:
        """An empty optional pedestrian component cannot produce a NaN score."""
        result = monitor_execution_deviation(
            predicted_robot_positions=np.zeros((5, 2)),
            observed_robot_positions=np.zeros((5, 2)),
            predicted_pedestrian_positions=np.zeros((5, 0, 2)),
            observed_pedestrian_positions=np.zeros((5, 0, 2)),
            dt_s=0.1,
            config=_deviation_config(),
        )
        assert result.fail_closed is True
        assert result.deviation_score is None
        assert result.component_deviations == ()

    def test_result_is_frozen_dataclass(self) -> None:
        windows = _aligned_windows(deviation=0.0)
        result = monitor_execution_deviation(config=_deviation_config(), **windows)
        assert isinstance(result, ExecutionDeviationResult)
        with pytest.raises(AttributeError):
            result.intervention = "replan"  # type: ignore[misc]

    def test_schema_and_claim_boundary_constants(self) -> None:
        assert EXECUTION_DEVIATION_SCHEMA == "execution_deviation.v1"
        assert "offline" in EXECUTION_DEVIATION_CLAIM_BOUNDARY
        assert "not a control-loop intervention" in EXECUTION_DEVIATION_CLAIM_BOUNDARY
        assert "issue #4757" in EXECUTION_DEVIATION_CLAIM_BOUNDARY

    def test_no_score_fabricated_on_fail_closed(self) -> None:
        """Fail-closed results never contain a numeric score or denominators."""
        cfg = _deviation_config()
        result = monitor_execution_deviation(
            predicted_robot_positions=None,
            observed_robot_positions=None,
            dt_s=0.1,
            config=cfg,
        )
        assert result.deviation_score is None
        assert result.component_deviations == ()
        assert result.first_threshold_crossing_time_s is None

    def test_first_threshold_crossing_time_correct(self) -> None:
        """First crossing time corresponds to the first timestep above threshold."""
        n = 10
        dt = 0.1
        pred_pos = np.zeros((n, 2))
        obs_pos = np.zeros((n, 2))
        # Deviation appears only at step 5 onward.
        obs_pos[5:, 0] = 1.0
        cfg = _deviation_config(warn_threshold=0.5)
        result = monitor_execution_deviation(
            predicted_robot_positions=pred_pos,
            observed_robot_positions=obs_pos,
            dt_s=dt,
            config=cfg,
        )
        assert result.first_threshold_crossing_time_s == pytest.approx(5 * dt)

    def test_transient_threshold_crossing_has_matching_warn_intervention(self) -> None:
        """A one-step crossing cannot report detection while returning continue."""
        predicted = np.zeros((10, 2))
        observed = predicted.copy()
        observed[0, 0] = 1.0
        result = monitor_execution_deviation(
            predicted_robot_positions=predicted,
            observed_robot_positions=observed,
            dt_s=0.1,
            config=_deviation_config(warn_threshold=0.5),
        )
        assert result.deviation_score == pytest.approx(1.0)
        assert result.first_threshold_crossing_time_s == pytest.approx(0.0)
        assert result.intervention == INTERVENTION_WARN

    def test_separate_from_trajectory_verifier(self) -> None:
        """The deviation monitor does not alter verify_trajectory semantics."""
        robot_pos, robot_vel, ped_pos, ped_vel = _straight_trajectory(
            ped_offset=(5.0, 2.0), robot_speed=0.5
        )
        verifier_result = verify_trajectory(
            robot_positions=robot_pos,
            robot_velocities=robot_vel,
            pedestrian_positions=ped_pos,
            pedestrian_velocities=ped_vel,
            dt_s=0.1,
            robot_radius_m=0.3,
            pedestrian_radius_m=0.3,
        )
        assert verifier_result.decision == DECISION_ACCEPT
        deviation_result = monitor_execution_deviation(
            predicted_robot_positions=robot_pos,
            observed_robot_positions=robot_pos,
            dt_s=0.1,
            config=_deviation_config(),
        )
        assert deviation_result.intervention == INTERVENTION_CONTINUE
        assert verifier_result.claim_boundary != deviation_result.claim_boundary


class TestExecutionDeviationDiagnosticReport:
    """Diagnostic reporting is explicit about counts, denominators, and gaps."""

    def test_reports_available_detection_and_intervention_metrics(self) -> None:
        clean = monitor_execution_deviation(config=_deviation_config(), **_aligned_windows())
        detected = monitor_execution_deviation(
            config=_deviation_config(warn_threshold=0.5), **_aligned_windows(deviation=1.0)
        )
        report = summarize_execution_deviation_diagnostics(
            (
                ExecutionDeviationDiagnosticCase(clean, expected_deviation=False),
                ExecutionDeviationDiagnosticCase(
                    detected,
                    expected_deviation=True,
                    collision_or_near_miss=False,
                    repair_latency_s=0.25,
                ),
            )
        )
        assert isinstance(report, ExecutionDeviationDiagnosticReport)
        assert report.false_alarm_count == 0
        assert report.false_alarm_denominator == 1
        assert report.false_alarm_rate == pytest.approx(0.0)
        assert report.detection_count == 1
        assert report.detection_denominator == 1
        assert report.detection_recall == pytest.approx(1.0)
        assert report.detection_delay_s == pytest.approx(0.0)
        assert report.detection_delay_denominator == 1
        assert dict(report.intervention_counts) == {
            INTERVENTION_CONTINUE: 1,
            INTERVENTION_WARN: 1,
            INTERVENTION_REPLAN: 0,
            INTERVENTION_FALLBACK_BRAKE: 0,
        }
        assert report.intervention_denominator == 2
        assert report.intervention_rate == pytest.approx(0.5)
        assert report.repair_latency_status == "available"
        assert report.repair_latency_s == pytest.approx(0.25)
        assert report.residual_collision_near_miss_status == "available"
        assert report.residual_collision_near_miss_rate == pytest.approx(0.0)
        assert report.schema_version == EXECUTION_DEVIATION_REPORT_SCHEMA

    def test_marks_missing_collision_and_repair_evidence_unavailable(self) -> None:
        report = summarize_execution_deviation_diagnostics(
            (
                ExecutionDeviationDiagnosticCase(
                    monitor_execution_deviation(config=_deviation_config(), **_aligned_windows()),
                    expected_deviation=False,
                ),
            )
        )
        assert report.repair_latency_status == "unavailable"
        assert report.repair_latency_s is None
        assert report.repair_latency_denominator == 0
        assert report.residual_collision_near_miss_status == "unavailable"
        assert report.residual_collision_near_miss_rate is None
        assert report.residual_collision_near_miss_denominator == 0

    def test_missing_performance_denominators_are_not_fabricated(self) -> None:
        report = summarize_execution_deviation_diagnostics(())
        assert report.false_alarm_denominator == 0
        assert report.false_alarm_rate is None
        assert report.detection_denominator == 0
        assert report.detection_recall is None
        assert report.detection_delay_s is None
        assert report.intervention_denominator == 0
        assert report.intervention_rate is None

    def test_fail_closed_outcomes_are_counted_but_excluded_from_rates(self) -> None:
        clean = monitor_execution_deviation(config=_deviation_config(), **_aligned_windows())
        fail_closed = monitor_execution_deviation(
            predicted_robot_positions=None,
            observed_robot_positions=None,
            dt_s=0.1,
            config=_deviation_config(),
        )
        report = summarize_execution_deviation_diagnostics(
            (
                ExecutionDeviationDiagnosticCase(clean, expected_deviation=False),
                ExecutionDeviationDiagnosticCase(fail_closed, expected_deviation=True),
            )
        )
        assert dict(report.intervention_counts)[INTERVENTION_WARN] == 1
        assert report.fail_closed_count == 1
        assert report.intervention_denominator == 1
        assert report.intervention_rate == pytest.approx(0.0)
        assert report.detection_denominator == 0
        assert report.detection_recall is None
