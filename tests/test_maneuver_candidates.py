"""Focused contract tests for the generation-only maneuver portfolio."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from robot_sf.nav.global_route import RouteGeometry, RouteProjectionHint
from robot_sf.planner.maneuver_candidates import (
    ManeuverId,
    ManeuverPortfolioConfig,
    generate_maneuver_candidates,
)
from robot_sf.research.collision_risk import (
    RiskEstimatorConfig,
    estimate_action_conditioned_risk,
)
from robot_sf.robot.dynamics import RobotDynamicsState

DT_S = 0.1
HORIZON = 20


def _state(*, heading: float = 0.0, speed: float = 0.8, angular_speed: float = 0.0):
    """Return a finite canonical initial state."""

    return RobotDynamicsState(
        x=0.0,
        y=0.0,
        heading=heading,
        linear_speed=speed,
        angular_speed=angular_speed,
    )


def _route(angle: float = 0.0) -> RouteGeometry:
    """Return a straight route rotated by ``angle``."""

    direction = np.array([math.cos(angle), math.sin(angle)])
    return RouteGeometry((tuple(direction * 0.0), tuple(direction * 20.0)))


def _by_maneuver(result, maneuver: ManeuverId):
    """Return the first candidate for one maneuver family."""

    return next(candidate for candidate in result if candidate.maneuver is maneuver)


def test_empty_route_generates_all_families_with_one_shared_horizon() -> None:
    """An empty static scene exposes every required semantic family."""

    result = generate_maneuver_candidates(_route(), _state(), static_geometry=())

    assert {candidate.maneuver for candidate in result} == set(ManeuverId)
    assert all(candidate.states.shape == (HORIZON + 1, 5) for candidate in result)
    assert all(len(candidate.controls) == HORIZON for candidate in result)
    assert all(
        candidate.action.as_array(horizon_steps=HORIZON).shape == (HORIZON + 1, 2)
        for candidate in result
    )
    assert result.report.route_projection_status == "ok"


def test_candidates_are_deterministic_and_arrays_are_immutable() -> None:
    """Repeated calls preserve order and cannot be changed through returned arrays."""

    route = _route()
    first = generate_maneuver_candidates(route, _state(), static_geometry=())
    second = generate_maneuver_candidates(
        route,
        {"angular_speed": 0.0, "linear_speed": 0.8, "heading": 0.0, "y": 0.0, "x": 0.0},
        static_geometry=(),
    )

    assert first.report.ordered_candidate_ids == second.report.ordered_candidate_ids
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.states, right.states)
        assert np.array_equal(left.action.waypoints, right.action.waypoints)
        assert left.states.flags.writeable is False
        assert left.action.waypoints.flags.writeable is False
        with pytest.raises(ValueError):
            left.states[0, 0] = 99.0
        with pytest.raises(ValueError):
            left.action.waypoints[0, 0] = 99.0


def test_route_follow_makes_positive_arc_progress_and_risk_api_accepts_action() -> None:
    """The centerline candidate is a direct canonical risk-estimator input."""

    result = generate_maneuver_candidates(_route(), _state(), static_geometry=())
    candidate = _by_maneuver(result, ManeuverId.ROUTE_FOLLOW)
    progress = candidate.states[-1, 0] - candidate.states[0, 0]
    assert progress > 0.0

    risk = estimate_action_conditioned_risk(
        candidate.action,
        [],
        RiskEstimatorConfig(
            horizon_steps=HORIZON,
            dt_s=DT_S,
            n_samples=32,
            min_samples_for_estimate=1,
        ),
    )
    assert math.isfinite(risk.joint_contact_probability)


def test_pass_candidates_rejoin_and_mirror_in_route_frame() -> None:
    """Side trajectories mirror and reduce lateral error after their peak."""

    result = generate_maneuver_candidates(_route(), _state(), static_geometry=())
    left = _by_maneuver(result, ManeuverId.PASS_LEFT)
    right = _by_maneuver(result, ManeuverId.PASS_RIGHT)
    left_y = left.states[:, 1]
    right_y = right.states[:, 1]
    assert np.allclose(left_y, -right_y, atol=1.0e-10)
    peak_index = int(np.argmax(left_y))
    assert peak_index < len(left_y) - 1
    assert left_y[-1] < left_y[peak_index]


def test_rotated_route_defines_left_and_right_relative_to_tangent() -> None:
    """Rotation of the route rotates side displacement without changing signs."""

    angle = math.pi / 2.0
    result = generate_maneuver_candidates(_route(angle), _state(heading=angle), static_geometry=())
    left = _by_maneuver(result, ManeuverId.PASS_LEFT)
    right = _by_maneuver(result, ManeuverId.PASS_RIGHT)
    # For a +y route, left is -x and right is +x.
    assert left.states[:, 0].min() < -1.0e-5
    assert right.states[:, 0].max() > 1.0e-5


def test_blocked_left_rejects_only_left_and_preserves_stop() -> None:
    """Static topology rejection records the side reason and retains braking."""

    def no_left(states: np.ndarray, **_: object) -> bool:
        return bool(np.all(states[:, 1] <= 0.01))

    result = generate_maneuver_candidates(_route(), _state(), static_verifier=no_left)

    assert all(candidate.maneuver is not ManeuverId.PASS_LEFT for candidate in result)
    assert _by_maneuver(result, ManeuverId.PASS_RIGHT)
    assert _by_maneuver(result, ManeuverId.CONTROLLED_STOP)
    assert result.report.per_reason_rejection_count["no_left_corridor"] == 1


def test_caps_never_remove_controlled_stop() -> None:
    """Global truncation is deterministic and safety-preserving."""

    config = ManeuverPortfolioConfig(max_total_candidates=1)
    result = generate_maneuver_candidates(_route(), _state(), config=config, static_geometry=())

    assert len(result) == 1
    assert result[0].maneuver is ManeuverId.CONTROLLED_STOP
    assert result.report.truncated is True
    assert result.report.truncation_rule == "retain_controlled_stop_then_family_rank_order"


def test_short_horizon_reports_incomplete_stop() -> None:
    """A horizon shorter than braking time keeps a truthful stop diagnostic."""

    config = ManeuverPortfolioConfig(horizon_steps=2, max_total_candidates=1)
    result = generate_maneuver_candidates(_route(), _state(speed=1.0), config=config)
    stop = _by_maneuver(result, ManeuverId.CONTROLLED_STOP)

    assert stop.states[-1, 3] > 0.0
    assert stop.metadata["stop_complete_within_horizon"] is False


def test_stop_completion_requires_linear_and_angular_rest() -> None:
    """A short horizon with residual yaw rate cannot claim a complete stop."""

    config = ManeuverPortfolioConfig(horizon_steps=2, max_total_candidates=1)
    result = generate_maneuver_candidates(
        _route(), _state(speed=0.0, angular_speed=0.8), config=config
    )
    stop = _by_maneuver(result, ManeuverId.CONTROLLED_STOP)

    assert stop.states[-1, 3] == pytest.approx(0.0)
    assert abs(stop.states[-1, 4]) > 0.0
    assert stop.metadata["stop_complete_within_horizon"] is False


def test_unknown_static_topology_gates_pass_families() -> None:
    """Pass-side candidates require an explicit static-geometry source."""

    result = generate_maneuver_candidates(_route(), _state())

    assert all(
        candidate.maneuver not in (ManeuverId.PASS_LEFT, ManeuverId.PASS_RIGHT)
        for candidate in result
    )
    assert result.report.per_reason_rejection_count["static_geometry_unavailable"] == 2


def test_ineligible_pass_heading_is_reported() -> None:
    """A side maneuver is rejected when the initial heading is far from the route."""

    result = generate_maneuver_candidates(
        _route(), _state(heading=math.pi / 2.0), static_geometry=()
    )

    assert all(
        candidate.maneuver not in (ManeuverId.PASS_LEFT, ManeuverId.PASS_RIGHT)
        for candidate in result
    )
    assert result.report.per_reason_rejection_count["pass_heading_ineligible"] == 2


def test_pass_rejects_initial_yaw_rate_without_side_rejoin() -> None:
    """A residual yaw rate cannot be hidden behind a pass-side label."""

    result = generate_maneuver_candidates(_route(), _state(angular_speed=0.7), static_geometry=())

    assert all(
        candidate.maneuver not in (ManeuverId.PASS_LEFT, ManeuverId.PASS_RIGHT)
        for candidate in result
    )
    assert result.report.per_reason_rejection_count["pass_side_mismatch"] == 2


def test_stop_handles_nonzero_initial_angular_speed_and_jerk() -> None:
    """Stop preserves midpoint dynamics and both jerk bounds."""

    result = generate_maneuver_candidates(
        _route(), _state(speed=1.0, angular_speed=0.3), static_geometry=()
    )
    stop = _by_maneuver(result, ManeuverId.CONTROLLED_STOP)
    linear_accel = np.diff(stop.states[:, 3]) / DT_S
    angular_accel = np.diff(stop.states[:, 4]) / DT_S
    assert np.max(np.abs(np.diff(np.r_[0.0, linear_accel]) / DT_S)) <= 4.0 + 1.0e-7
    assert np.max(np.abs(np.diff(np.r_[0.0, angular_accel]) / DT_S)) <= 4.0 + 1.0e-7
    assert np.all(stop.states[stop.states[:, 3] <= 1.0e-12, 3] == 0.0)


def test_invalid_route_projection_does_not_fabricate_route_follow() -> None:
    """Ambiguous route projection fails closed while still attempting a stop."""

    route = RouteGeometry(((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0)))
    result = generate_maneuver_candidates(
        route,
        replace(_state(), x=1.0, y=1.0),
        local_goal=(1.0, 1.0),
    )

    assert all(candidate.maneuver is ManeuverId.CONTROLLED_STOP for candidate in result)
    assert result.report.route_projection_status == "ambiguous"
    assert result.report.per_reason_rejection_count["invalid_route_projection"] >= 1


def test_projection_hint_selects_a_route_branch_for_generation() -> None:
    """A caller hint preserves the intended branch at a self-intersection."""

    route = RouteGeometry(((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0)))
    result = generate_maneuver_candidates(
        route,
        RobotDynamicsState(
            x=1.0,
            y=1.0,
            heading=math.pi / 4.0,
            linear_speed=0.2,
            angular_speed=0.0,
        ),
        local_goal=(1.5, 1.5),
        projection_hint=RouteProjectionHint(
            previous_s_m=1.4,
            previous_segment_index=0,
            max_forward_jump_m=0.3,
            max_backtrack_m=0.3,
        ),
        static_geometry=(),
    )

    assert result.report.route_projection_status == "ok"


def test_projection_hint_rejects_a_discontinuous_initial_projection() -> None:
    """A stale branch hint must not admit route-dependent candidates."""

    result = generate_maneuver_candidates(
        _route(),
        replace(_state(), x=5.0),
        projection_hint=RouteProjectionHint(
            previous_s_m=0.0,
            previous_segment_index=0,
            max_forward_jump_m=1.0,
            max_backtrack_m=0.5,
        ),
    )

    assert result.report.route_projection_status == "discontinuous"
    assert all(candidate.maneuver is ManeuverId.CONTROLLED_STOP for candidate in result)


def test_nonfinite_state_and_config_fail_closed() -> None:
    """NaN inputs are rejected before a rollout can be represented."""

    result = generate_maneuver_candidates(_route(), (0.0, 0.0, math.nan, 0.0, 0.0))
    assert len(result) == 0
    assert result.report.per_reason_rejection_count["invalid_initial_state"] == 1
    with pytest.raises(ValueError):
        ManeuverPortfolioConfig(dt_s=math.nan)


def test_yield_creep_respects_speed_and_deceleration_limits() -> None:
    """Yield remains distinct from stop and obeys configured creep speed."""

    config = ManeuverPortfolioConfig(creep_speed_mps=0.25, yield_deceleration_mps2=0.4)
    result = generate_maneuver_candidates(_route(), _state(speed=1.0), config=config)
    yield_candidate = _by_maneuver(result, ManeuverId.YIELD_CREEP)

    assert yield_candidate.maneuver is ManeuverId.YIELD_CREEP
    assert yield_candidate.states[-1, 3] <= config.creep_speed_mps + 1.0e-8
    assert yield_candidate.metadata["stop_complete_within_horizon"] is False
