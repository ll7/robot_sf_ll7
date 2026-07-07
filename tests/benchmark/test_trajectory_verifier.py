"""Tests for the experimental AMMV trajectory verifier (issue #4757)."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark.trajectory_verifier import (
    DECISION_ACCEPT,
    DECISION_FALLBACK_BRAKE,
    DECISION_WARN,
    PRED_BRAKING_INFEASIBLE,
    PRED_CLEARANCE_HARD,
    PRED_CLEARANCE_WARN,
    PRED_RECOVERY_SMOOTHNESS,
    PRED_STALE_OR_MISSING_STATE,
    PRED_TTC_HARD,
    PRED_TTC_WARN,
    TRAJECTORY_VERIFIER_CLAIM_BOUNDARY,
    TRAJECTORY_VERIFIER_SCHEMA,
    TrajectoryVerifierConfig,
    VerifierResult,
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
