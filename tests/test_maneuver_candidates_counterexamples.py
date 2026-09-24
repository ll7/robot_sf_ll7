"""Adversarial counterexamples for the maneuver candidate portfolio.

These tests exercise physical, geometric, and ownership boundaries that are easy
to satisfy accidentally with a nominal straight-route smoke test.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from robot_sf.nav.global_route import RouteGeometry
from robot_sf.nav.obstacle import Obstacle
from robot_sf.planner.maneuver_candidates import (
    ManeuverId,
    ManeuverPortfolioConfig,
    generate_maneuver_candidates,
)
from robot_sf.robot.dynamics import RobotDynamicsState


def _straight_route() -> RouteGeometry:
    """Return a long, directed route with no turn-induced side effects."""

    return RouteGeometry(((0.0, 0.0), (10.0, 0.0)))


def _state(
    *,
    x: float = 0.0,
    speed: float = 0.8,
    heading: float = 0.0,
    angular_speed: float = 0.0,
) -> RobotDynamicsState:
    """Return a finite state on the straight route."""

    return RobotDynamicsState(
        x=x,
        y=0.0,
        heading=heading,
        linear_speed=speed,
        angular_speed=angular_speed,
    )


@pytest.mark.parametrize("initial_speed", (0.05, 0.2, 0.8, 1.0, 1.9))
def test_controlled_stop_is_finite_jerk_bounded_and_holds_zero(initial_speed: float) -> None:
    """Braking must ramp from zero acceleration, stop, and remain stopped."""

    config = ManeuverPortfolioConfig(horizon_steps=24)
    result = generate_maneuver_candidates(
        _straight_route(),
        _state(speed=initial_speed),
        config=config,
        local_goal=(5.0, 0.0),
        static_geometry=(),
    )
    stops = [
        candidate
        for candidate in result.candidates
        if candidate.maneuver is ManeuverId.CONTROLLED_STOP
    ]
    assert len(stops) == 1
    stop = stops[0]
    speeds = np.asarray(stop.states[:, 3], dtype=float)
    assert np.isfinite(stop.states).all()
    assert np.isfinite(np.asarray(stop.controls, dtype=float)).all()
    assert np.isfinite(np.diff(stop.states, axis=0)).all()
    assert np.isfinite(np.diff(stop.states, n=2, axis=0)).all()
    assert speeds[0] == pytest.approx(initial_speed)
    assert np.all(speeds >= -1.0e-12)
    assert np.all(np.diff(speeds) <= 1.0e-12)
    assert speeds[-1] == pytest.approx(0.0, abs=1.0e-12)
    first_zero = int(np.flatnonzero(speeds <= 1.0e-12)[0])
    assert np.all(speeds[first_zero:] <= 1.0e-12)

    acceleration = np.diff(speeds) / config.dt_s
    # The rollout contract documents zero acceleration immediately before the
    # first command; include that value when checking the first jerk sample.
    jerk = np.diff(np.concatenate(([0.0], acceleration))) / config.dt_s
    assert np.isfinite(acceleration).all()
    assert np.isfinite(jerk).all()
    assert np.max(np.abs(acceleration)) <= config.stop_deceleration_mps2 + 1.0e-8
    assert np.max(np.abs(jerk)) <= config.max_jerk_mps3 + 1.0e-7


@pytest.mark.parametrize("start_x", (0.0, 0.0425))
def test_thin_polygon_wall_rejects_crossing_and_initial_contact(start_x: float) -> None:
    """A swept thin wall and a state already inside it must reject every rollout."""

    wall = Polygon(((0.04, -1.0), (0.045, -1.0), (0.045, 1.0), (0.04, 1.0)))
    obstacle = Obstacle.from_geometry(wall)
    config = ManeuverPortfolioConfig(robot_radius_m=0.01, static_clearance_margin_m=0.0)
    result = generate_maneuver_candidates(
        _straight_route(),
        _state(x=start_x),
        config=config,
        local_goal=(5.0, 0.0),
        static_geometry=(obstacle,),
    )

    assert result.candidates == ()
    assert result.report.per_reason_rejection_count.get("static_collision", 0) >= 1


def test_empty_straight_route_emits_all_families_and_rejoins_after_lateral_peak() -> None:
    """An empty geometry tuple must preserve both pass sides and their rejoin phase."""

    result = generate_maneuver_candidates(
        _straight_route(),
        _state(),
        local_goal=(5.0, 0.0),
        static_geometry=(),
    )
    families = {candidate.maneuver for candidate in result.candidates}
    assert families == set(ManeuverId)

    for maneuver in (ManeuverId.PASS_LEFT, ManeuverId.PASS_RIGHT):
        candidate = next(item for item in result.candidates if item.maneuver is maneuver)
        lateral = np.abs(np.asarray(candidate.states[:, 1], dtype=float))
        peak_index = int(np.argmax(lateral))
        assert 0 < peak_index < len(lateral) - 1
        assert np.all(np.diff(lateral[peak_index:]) <= 1.0e-9)
        assert lateral[-1] < lateral[peak_index] - 1.0e-6


def test_missing_geometry_does_not_emit_pass_candidates_but_empty_geometry_does() -> None:
    """Pass maneuvers require an explicit static-geometry context."""

    without_geometry = generate_maneuver_candidates(
        _straight_route(),
        _state(),
        local_goal=(5.0, 0.0),
    )
    assert all(
        candidate.maneuver not in (ManeuverId.PASS_LEFT, ManeuverId.PASS_RIGHT)
        for candidate in without_geometry
    )

    with_empty_geometry = generate_maneuver_candidates(
        _straight_route(),
        _state(),
        local_goal=(5.0, 0.0),
        static_geometry=(),
    )
    assert {candidate.maneuver for candidate in with_empty_geometry} == set(ManeuverId)


def test_zero_yield_cap_emits_no_yield_candidate() -> None:
    """A zero family cap must suppress yield-creep generation."""

    result = generate_maneuver_candidates(
        _straight_route(),
        _state(),
        config=ManeuverPortfolioConfig(max_yield_candidates=0),
        local_goal=(5.0, 0.0),
        static_geometry=(),
    )

    assert all(candidate.maneuver is not ManeuverId.YIELD_CREEP for candidate in result)


def test_zero_linear_speed_with_residual_angular_speed_reports_incomplete_stop() -> None:
    """A stop is incomplete while nonzero angular motion remains in a short horizon."""

    result = generate_maneuver_candidates(
        _straight_route(),
        _state(speed=0.0, angular_speed=0.8),
        config=ManeuverPortfolioConfig(horizon_steps=2, max_total_candidates=1),
        local_goal=(5.0, 0.0),
        static_geometry=(),
    )
    stop = next(
        candidate for candidate in result if candidate.maneuver is ManeuverId.CONTROLLED_STOP
    )

    assert stop.states[-1, 3] == pytest.approx(0.0, abs=1.0e-12)
    assert abs(stop.states[-1, 4]) > 1.0e-6
    assert stop.metadata["stop_complete_within_horizon"] is False


@pytest.mark.parametrize(
    ("heading", "angular_speed"),
    ((np.pi / 2.0, 0.0), (0.0, 0.3), (0.0, 0.7)),
)
def test_misaligned_heading_does_not_flip_pass_right_side_label(
    heading: float, angular_speed: float
) -> None:
    """A right-pass candidate must stay right and rejoin after its excursion."""

    result = generate_maneuver_candidates(
        _straight_route(),
        _state(heading=heading, angular_speed=angular_speed),
        local_goal=(5.0, 0.0),
        static_geometry=(),
    )
    right_candidates = [
        candidate for candidate in result if candidate.maneuver is ManeuverId.PASS_RIGHT
    ]
    if not right_candidates:
        assert result.report.per_maneuver_attempted_count.get("pass_right", 0) >= 1
        return

    for candidate in right_candidates:
        lateral = np.asarray(candidate.states[:, 1], dtype=float)
        assert np.max(lateral) <= 1.0e-8
        peak_index = int(np.argmin(lateral))
        assert lateral[peak_index] < -1.0e-6
        assert 0 < peak_index < len(lateral) - 1
        post_peak = np.abs(lateral[peak_index:])
        assert np.all(np.diff(post_peak) <= 1.0e-9)
        assert post_peak[-1] < post_peak[0] - 1.0e-6


def test_exposed_trajectory_arrays_cannot_become_writeable() -> None:
    """Read-only trajectory ownership must survive a setflags write escalation."""

    result = generate_maneuver_candidates(
        _straight_route(),
        _state(),
        local_goal=(5.0, 0.0),
        static_geometry=(),
    )
    candidate = result.candidates[0]
    for array in (candidate.states, candidate.action.waypoints):
        before = array.copy()
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)
        np.testing.assert_array_equal(array, before)


def test_nan_portfolio_config_is_rejected() -> None:
    """Configuration validation must fail closed on non-finite numeric bounds."""

    with pytest.raises(ValueError, match="finite"):
        ManeuverPortfolioConfig(max_jerk_mps3=float("nan"))
