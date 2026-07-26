"""Focused unit tests for the bounded residual-control reactive adversary (#4360).

These tests cover the capability-only slice: residual-control bounds (speed,
acceleration, jerk, heading, route deviation), 0.5 s macro-action cadence,
walkable-space projection, inter-agent separation, opt-in gating (off by
default), nominal-Social-Force base-law preservation (perturb not replace), and
fail-closed behavior on malformed / non-finite input.

This is a capability-only slice: it makes no benchmark, planner-ranking, safety,
or paper-facing claim and defines no stress-case metric.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from robot_sf.ped_npc.residual_adversary import (
    DEFAULT_MACRO_ACTION_DT_S,
    BoundedResidualAdversary,
    ResidualAdversaryConfig,
    ResidualAdversaryObservation,
    ScriptedPullResidualAdversaryPolicy,
    bound_heading_change,
    bound_route_deviation,
    bound_speed,
    build_default_residual_adversary,
    clamp_magnitude,
    enforce_inter_agent_separation,
    project_residual_displacement_walkable,
    rate_limit_jerk,
    residual_displacement_from_accel,
)

ROBOT_POSE = ((0.0, 0.0), 0.0)


# ---------------------------------------------------------------------------
# Config and opt-in gating
# ---------------------------------------------------------------------------


def test_config_defaults_are_opt_in_and_off() -> None:
    """The adversary must be off by default so defaults are unchanged."""
    config = ResidualAdversaryConfig()
    assert config.is_active is False
    assert config.macro_action_dt_s == pytest.approx(DEFAULT_MACRO_ACTION_DT_S)


def test_config_rejects_non_positive_or_non_finite_bounds() -> None:
    """Malformed bounds must fail closed at construction."""
    with pytest.raises(ValueError):
        ResidualAdversaryConfig(is_active=True, max_residual_accel_mps2=0.0)
    with pytest.raises(ValueError):
        ResidualAdversaryConfig(is_active=True, max_jerk_mps3=-1.0)
    with pytest.raises(ValueError):
        ResidualAdversaryConfig(is_active=True, macro_action_dt_s=float("nan"))
    with pytest.raises(ValueError):
        ResidualAdversaryConfig(is_active=True, min_separation_m=0.0)


def test_build_returns_none_when_inactive() -> None:
    """When ``is_active`` is False the factory must return None (no state allocated)."""
    assert build_default_residual_adversary(ResidualAdversaryConfig(), 0.1, 3) is None


def test_resolve_target_mask_supports_all_minus_one_and_list() -> None:
    """``-1`` targets everyone; a list targets only valid indices."""
    config_all = ResidualAdversaryConfig(is_active=True, target_ped_idx=-1)
    np.testing.assert_array_equal(config_all.resolve_target_mask(3), [True, True, True])
    config_list = ResidualAdversaryConfig(is_active=True, target_ped_idx=[0, 2])
    np.testing.assert_array_equal(config_list.resolve_target_mask(3), [True, False, True])
    config_oob = ResidualAdversaryConfig(is_active=True, target_ped_idx=[99])
    np.testing.assert_array_equal(config_oob.resolve_target_mask(2), [False, False])


# ---------------------------------------------------------------------------
# Bound helpers
# ---------------------------------------------------------------------------


def test_clamp_magnitude_preserves_direction_and_caps_norm() -> None:
    """Magnitude clamp keeps direction and bounds the norm; small rows unchanged."""
    residual = np.array([[3.0, 4.0], [0.1, 0.0]], dtype=float)
    clamped = clamp_magnitude(residual, 2.5)
    assert np.linalg.norm(clamped[0]) == pytest.approx(2.5)
    assert clamped[0, 0] > 0 and clamped[0, 1] > 0  # direction preserved
    np.testing.assert_allclose(clamped[1], [0.1, 0.0])


def test_rate_limit_jerk_caps_step_toward_proposal() -> None:
    """Jerk rate-limiting moves the residual by at most max_jerk * dt per step."""
    previous = np.zeros((1, 2), dtype=float)
    proposed = np.array([[10.0, 0.0]], dtype=float)
    dt = 0.1
    max_jerk = 7.5
    out = rate_limit_jerk(proposed, previous, dt, max_jerk)
    # Step magnitude must equal the cap, direction toward the proposal.
    assert np.linalg.norm(out[0]) == pytest.approx(max_jerk * dt)
    assert out[0, 0] > 0


def test_bound_speed_caps_resulting_speed() -> None:
    """The residual may not push a pedestrian beyond its max_speed."""
    velocity = np.array([[1.0, 0.0]], dtype=float)
    max_speeds = np.array([1.2], dtype=float)
    residual = np.array([[5.0, 0.0]], dtype=float)  # would spike speed far above cap
    dt = 0.1
    out = bound_speed(residual, velocity, max_speeds, dt, max_speed_delta_mps=5.0)
    resulting = velocity + out * dt
    assert np.linalg.norm(resulting[0]) <= max_speeds[0] + 1e-6


def test_bound_speed_caps_speed_delta_component() -> None:
    """The forward speed-increasing component is capped by max_speed_delta."""
    velocity = np.array([[0.0, 0.0]], dtype=float)
    max_speeds = np.array([10.0], dtype=float)
    residual = np.array([[5.0, 0.0]], dtype=float)
    dt = 0.1
    delta = 0.3
    out = bound_speed(residual, velocity, max_speeds, dt, max_speed_delta_mps=delta)
    forward_added = float(out[0, 0]) * dt
    assert forward_added <= delta + 1e-9


def test_bound_heading_change_caps_perpendicular_component() -> None:
    """A pure turning residual is capped so the angular change is bounded."""
    velocity = np.array([[1.0, 0.0]], dtype=float)
    # Large perpendicular residual that would rotate velocity sharply.
    residual = np.array([[0.0, 100.0]], dtype=float)
    allowance = 0.05
    out = bound_heading_change(residual, velocity, allowance)
    # The perpendicular component magnitude must be <= speed * allowance.
    assert abs(out[0, 1]) <= 1.0 * allowance + 1e-9


def test_bound_heading_change_leaves_stationary_rows_unchanged() -> None:
    """A pedestrian with near-zero velocity has no defined heading; residual passes through."""
    velocity = np.array([[0.0, 0.0]], dtype=float)
    residual = np.array([[1.0, 1.0]], dtype=float)
    out = bound_heading_change(residual, velocity, 0.1)
    np.testing.assert_allclose(out, residual)


def test_bound_route_deviation_scales_residual_to_corridor() -> None:
    """A residual that would leave the corridor is scaled back toward zero."""
    polyline = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=float)
    positions = np.array([[5.0, 0.0]], dtype=float)
    # Residual pointing straight away from the route (in +y).
    residual = np.array([[0.0, 5.0]], dtype=float)
    target_indices = np.array([0], dtype=int)
    out = bound_route_deviation(
        residual, positions, 0.1, [polyline], target_indices, max_route_deviation_m=1.0
    )
    displacement = residual_displacement_from_accel(out, 0.1)
    candidate = positions + displacement
    assert abs(candidate[0, 1]) <= 1.0 + 1e-6


def test_bound_route_deviation_zeroes_outward_when_already_outside() -> None:
    """When already outside the corridor, the outward residual is zeroed."""
    polyline = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=float)
    positions = np.array([[5.0, 5.0]], dtype=float)  # already 5 m off route
    residual = np.array([[0.0, 5.0]], dtype=float)
    out = bound_route_deviation(
        residual, positions, 0.1, [polyline], np.array([0]), max_route_deviation_m=1.0
    )
    np.testing.assert_allclose(out[0], [0.0, 0.0], atol=1e-6)


def test_project_residual_displacement_walkable_pushes_out_of_obstacle() -> None:
    """A residual driving into a wall is redirected to the obstacle margin."""
    positions = np.array([[1.0, 1.0]], dtype=float)
    # Displacement would place the ped at x=0.05, inside a wall at x=0.
    displacement = np.array([[-0.95, 0.0]], dtype=float)
    obstacle = np.array([[[0.0, -5.0], [0.0, 5.0]]], dtype=float)  # vertical wall at x=0
    corrected = project_residual_displacement_walkable(
        positions, displacement, obstacle, None, radius=0.4, margin_m=0.1
    )
    candidate = positions + corrected
    # The candidate must sit at least radius + margin from the wall.
    assert candidate[0, 0] >= 0.4 + 0.1 - 1e-6


def test_project_residual_displacement_walkable_clamps_to_bounds() -> None:
    """A residual leaving the map is clamped inside the bounds with clearance."""
    positions = np.array([[0.2, 0.2]], dtype=float)
    displacement = np.array([[-5.0, -5.0]], dtype=float)
    bounds = ((0.0, 10.0), (0.0, 10.0))
    corrected = project_residual_displacement_walkable(
        positions, displacement, None, bounds, radius=0.4, margin_m=0.1
    )
    candidate = positions + corrected
    assert candidate[0, 0] >= 0.4 + 0.1 - 1e-6
    assert candidate[0, 1] >= 0.4 + 0.1 - 1e-6


def test_enforce_inter_agent_separation_prevents_overlap() -> None:
    """Targeted displacements are scaled so agents keep the minimum separation.

    The adversary must not *cause* a separation violation: when the initial gap
    already satisfies the minimum, a residual closing the gap below it is scaled
    back so the gap is preserved at the boundary.
    """
    positions = np.array([[0.0, 0.0], [1.5, 0.0]], dtype=float)
    # Move agent 0 toward agent 1, which would close the 1.5 m gap below 1.0 m.
    displacement = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float)
    target_mask = np.array([True, False])
    corrected = enforce_inter_agent_separation(displacement, positions, target_mask, 1.0)
    candidate = positions + corrected
    gap = float(np.linalg.norm(candidate[0] - candidate[1]))
    assert gap >= 1.0 - 1e-6


def test_enforce_inter_agent_separation_leaves_non_targeted_unchanged() -> None:
    """Non-targeted rows must never be modified by the separation projection."""
    positions = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=float)
    displacement = np.array([[0.0, 0.0], [0.3, 0.4]], dtype=float)
    target_mask = np.array([False, False])
    corrected = enforce_inter_agent_separation(displacement, positions, target_mask, 1.0)
    np.testing.assert_allclose(corrected, displacement)


def test_enforce_inter_agent_separation_preserves_all_target_pairs() -> None:
    """All targeted candidates are projected together, without stale-row overlap."""
    positions = np.array([[0.0, 0.0], [1.2, 0.0]], dtype=float)
    displacement = np.array([[0.5, 0.0], [-0.5, 0.0]], dtype=float)
    corrected = enforce_inter_agent_separation(displacement, positions, np.array([True, True]), 0.6)
    candidate = positions + corrected
    assert np.linalg.norm(candidate[0] - candidate[1]) >= 0.6 - 1e-9


def test_enforce_inter_agent_separation_checks_lower_index_non_target() -> None:
    """A targeted row also preserves separation from earlier non-targeted rows."""
    positions = np.array([[0.0, 0.0], [1.2, 0.0]], dtype=float)
    displacement = np.array([[0.0, 0.0], [-1.0, 0.0]], dtype=float)
    corrected = enforce_inter_agent_separation(
        displacement, positions, np.array([False, True]), 0.6
    )
    candidate = positions + corrected
    assert np.linalg.norm(candidate[0] - candidate[1]) >= 0.6 - 1e-9


# ---------------------------------------------------------------------------
# Fail-closed behavior on malformed / non-finite input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bound_fn,kwargs",
    [
        (clamp_magnitude, {"max_magnitude": 1.0}),
        (rate_limit_jerk, {"previous": np.zeros((1, 2)), "dt_s": 0.1, "max_jerk_mps3": 1.0}),
        (
            bound_speed,
            {
                "velocities": np.zeros((1, 2)),
                "max_speeds": np.array([1.0]),
                "dt_s": 0.1,
                "max_speed_delta_mps": 1.0,
            },
        ),
        (bound_heading_change, {"velocities": np.zeros((1, 2)), "per_step_allowance_rad": 0.1}),
    ],
)
def test_bound_helpers_fail_closed_on_non_finite_input(bound_fn, kwargs) -> None:
    """Every bound helper must raise on non-finite residual input."""
    bad = np.array([[float("nan"), 0.0]], dtype=float)
    with pytest.raises(ValueError):
        bound_fn(bad, **kwargs)


def test_controller_fail_closed_on_non_finite_state() -> None:
    """The controller must raise when pedestrian state is non-finite."""
    config = ResidualAdversaryConfig(is_active=True)
    adversary = BoundedResidualAdversary(
        config=config,
        policy=ScriptedPullResidualAdversaryPolicy(),
        dt_s=0.1,
        num_peds=1,
    )
    bad_positions = np.array([[float("inf"), 0.0]], dtype=float)
    with pytest.raises(ValueError):
        adversary.step_residual(
            bad_positions,
            np.zeros((1, 2)),
            np.array([1.0]),
            ROBOT_POSE,
        )


def test_controller_fail_closed_on_non_finite_policy_output() -> None:
    """A policy emitting non-finite residual must be rejected by the controller."""

    @dataclass
    class _BadPolicy:
        def propose_residual(self, observation: ResidualAdversaryObservation) -> np.ndarray:
            return np.array([[float("nan"), 0.0]], dtype=float)

    config = ResidualAdversaryConfig(is_active=True)
    adversary = BoundedResidualAdversary(config=config, policy=_BadPolicy(), dt_s=0.1, num_peds=1)
    with pytest.raises(ValueError):
        adversary.step_residual(
            np.array([[1.0, 1.0]]), np.zeros((1, 2)), np.array([1.0]), ROBOT_POSE
        )


# ---------------------------------------------------------------------------
# Cadence
# ---------------------------------------------------------------------------


def test_macro_action_cadence_holds_proposal_between_boundaries() -> None:
    """A fresh proposal is requested every round(macro_dt/dt) steps and held otherwise."""
    call_counter = {"count": 0}

    @dataclass
    class _CountingPolicy:
        def propose_residual(self, observation: ResidualAdversaryObservation) -> np.ndarray:
            call_counter["count"] += 1
            # Constant proposal so the held value is observable after rate-limiting settles.
            return np.array([[1.0, 0.0]], dtype=float)

    dt = 0.1
    config = ResidualAdversaryConfig(is_active=True, macro_action_dt_s=0.5, max_jerk_mps3=1e9)
    adversary = BoundedResidualAdversary(
        config=config, policy=_CountingPolicy(), dt_s=dt, num_peds=1
    )
    assert adversary.macro_action_steps == 5
    positions = np.array([[2.0, 0.0]])
    velocities = np.array([[0.0, 0.0]])
    max_speeds = np.array([1.5])
    for _ in range(5):
        adversary.step_residual(positions, velocities, max_speeds, ROBOT_POSE)
    # 5 steps with a 5-step cadence -> exactly one macro-action proposal requested.
    assert call_counter["count"] == 1
    adversary.step_residual(positions, velocities, max_speeds, ROBOT_POSE)
    assert call_counter["count"] == 2  # step 6 is the next boundary


def test_default_macro_cadence_is_half_second() -> None:
    """The pre-registered macro-action cadence is 0.5 s."""
    assert DEFAULT_MACRO_ACTION_DT_S == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Controller integration: bounds enforcement and perturb-not-replace
# ---------------------------------------------------------------------------


def test_non_targeted_pedestrians_receive_zero_residual() -> None:
    """Only targeted pedestrians are perturbed; others keep a zero residual."""
    config = ResidualAdversaryConfig(is_active=True, target_ped_idx=[0])
    adversary = BoundedResidualAdversary(
        config=config,
        policy=ScriptedPullResidualAdversaryPolicy(),
        dt_s=0.1,
        num_peds=2,
    )
    positions = np.array([[2.0, 0.0], [2.0, 2.0]])
    velocities = np.zeros((2, 2))
    max_speeds = np.array([1.5, 1.5])
    residual = adversary.step_residual(positions, velocities, max_speeds, ROBOT_POSE)
    assert np.linalg.norm(residual[1]) == pytest.approx(0.0)
    assert np.linalg.norm(residual[0]) > 0.0


def test_residual_acceleration_stays_within_magnitude_bound() -> None:
    """The applied residual must never exceed the acceleration magnitude bound."""
    max_accel = 0.8
    config = ResidualAdversaryConfig(
        is_active=True, max_residual_accel_mps2=max_accel, max_jerk_mps3=1e9
    )
    adversary = BoundedResidualAdversary(
        config=config,
        policy=ScriptedPullResidualAdversaryPolicy(max_pull_accel_mps2=50.0),
        dt_s=0.1,
        num_peds=1,
    )
    positions = np.array([[2.0, 0.0]])
    velocities = np.array([[0.0, 0.0]])
    max_speeds = np.array([1.5])
    for _ in range(12):
        residual = adversary.step_residual(positions, velocities, max_speeds, ROBOT_POSE)
        assert np.linalg.norm(residual[0]) <= max_accel + 1e-9


def test_residual_jerk_is_bounded_across_steps() -> None:
    """The change in applied residual between consecutive steps respects the jerk bound."""
    max_jerk = 5.0
    dt = 0.1
    config = ResidualAdversaryConfig(
        is_active=True, max_jerk_mps3=max_jerk, max_residual_accel_mps2=2.0
    )
    adversary = BoundedResidualAdversary(
        config=config,
        policy=ScriptedPullResidualAdversaryPolicy(max_pull_accel_mps2=2.0),
        dt_s=dt,
        num_peds=1,
    )
    positions = np.array([[2.0, 0.0]])
    velocities = np.array([[0.0, 0.0]])
    max_speeds = np.array([1.5])
    prev = adversary.step_residual(positions, velocities, max_speeds, ROBOT_POSE)
    for _ in range(8):
        nxt = adversary.step_residual(positions, velocities, max_speeds, ROBOT_POSE)
        change = np.linalg.norm(nxt - prev) / dt
        assert change <= max_jerk + 1e-6
        prev = nxt


def test_residual_jerk_remains_bounded_after_route_projection() -> None:
    """Route projection cannot bypass the final per-step jerk limit."""
    max_jerk = 1.0
    dt = 0.1

    @dataclass
    class _OutwardPolicy:
        def propose_residual(self, observation: ResidualAdversaryObservation) -> np.ndarray:
            return np.array([[0.0, 5.0]])

    config = ResidualAdversaryConfig(
        is_active=True,
        macro_action_dt_s=dt,
        max_residual_accel_mps2=10.0,
        max_jerk_mps3=max_jerk,
        max_route_deviation_m=0.05,
    )
    adversary = BoundedResidualAdversary(
        config=config,
        policy=_OutwardPolicy(),
        dt_s=dt,
        num_peds=1,
        route_polylines=[np.array([[0.0, 0.0], [2.0, 0.0]])],
    )
    positions = np.array([[0.0, 0.05]])
    velocities = np.zeros((1, 2))
    max_speeds = np.array([10.0])
    adversary._last_residual = np.array([[0.0, 0.5]])
    previous = adversary.last_residual
    current = adversary.step_residual(positions, velocities, max_speeds, ROBOT_POSE)
    assert np.linalg.norm(current - previous) / dt <= max_jerk + 1e-9


def test_residual_does_not_exceed_max_speed() -> None:
    """After applying the residual, the resulting speed stays within max_speed."""
    config = ResidualAdversaryConfig(
        is_active=True,
        max_residual_accel_mps2=2.0,
        max_jerk_mps3=1e9,
        max_speed_delta_mps=2.0,
    )
    adversary = BoundedResidualAdversary(
        config=config,
        policy=ScriptedPullResidualAdversaryPolicy(max_pull_accel_mps2=2.0),
        dt_s=0.1,
        num_peds=1,
    )
    velocity = np.array([[1.4, 0.0]])  # close to the 1.5 cap
    positions = np.array([[2.0, 0.0]])
    max_speeds = np.array([1.5])
    residual = adversary.step_residual(positions, velocity, max_speeds, ROBOT_POSE)
    resulting = velocity + residual * 0.1
    assert np.linalg.norm(resulting[0]) <= 1.5 + 1e-6


def test_zero_proposal_preserves_nominal_forces() -> None:
    """With a zero-residual proposal the controller adds nothing (base law preserved)."""

    @dataclass
    class _ZeroPolicy:
        def propose_residual(self, observation: ResidualAdversaryObservation) -> np.ndarray:
            return np.zeros((1, 2))

    config = ResidualAdversaryConfig(is_active=True)
    adversary = BoundedResidualAdversary(config=config, policy=_ZeroPolicy(), dt_s=0.1, num_peds=1)
    positions = np.array([[2.0, 0.0]])
    velocities = np.array([[0.0, 0.0]])
    max_speeds = np.array([1.5])
    residual = adversary.step_residual(positions, velocities, max_speeds, ROBOT_POSE)
    np.testing.assert_allclose(residual, np.zeros((1, 2)))


def test_reset_clears_held_state() -> None:
    """``reset`` restarts the macro-action cadence and clears the held residual."""
    config = ResidualAdversaryConfig(is_active=True)
    adversary = BoundedResidualAdversary(
        config=config,
        policy=ScriptedPullResidualAdversaryPolicy(),
        dt_s=0.1,
        num_peds=1,
    )
    positions = np.array([[2.0, 0.0]])
    velocities = np.array([[0.0, 0.0]])
    max_speeds = np.array([1.5])
    adversary.step_residual(positions, velocities, max_speeds, ROBOT_POSE)
    assert adversary.step_index > 0
    adversary.reset()
    assert adversary.step_index == 0
    assert adversary.macro_action_index == 0


def test_empty_crowd_returns_empty_residual() -> None:
    """An adversary sized for zero pedestrians returns an empty residual cleanly."""
    config = ResidualAdversaryConfig(is_active=True)
    adversary = BoundedResidualAdversary(
        config=config,
        policy=ScriptedPullResidualAdversaryPolicy(),
        dt_s=0.1,
        num_peds=0,
    )
    residual = adversary.step_residual(
        np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,)), ROBOT_POSE
    )
    assert residual.shape == (0, 2)


def test_scripted_pull_proposal_points_toward_robot_ahead_point() -> None:
    """The scripted policy pulls targeted peds toward a point ahead of the robot."""
    policy = ScriptedPullResidualAdversaryPolicy(max_pull_accel_mps2=1.0, pull_offset_m=1.0)
    observation = ResidualAdversaryObservation(
        positions=np.array([[0.0, 1.0]]),
        velocities=np.zeros((1, 2)),
        max_speeds=np.array([1.5]),
        target_ped_mask=np.array([True]),
        robot_pose=((0.0, 0.0), 0.0),  # heading +x -> attraction point (1, 0)
        sim_time_s=0.0,
        step_index=0,
        macro_action_index=0,
    )
    proposal = policy.propose_residual(observation)
    # Direction from (0,1) toward (1,0) is (1,-1)/sqrt(2), scaled to magnitude 1.
    expected = np.array([[1.0, -1.0]]) / math.sqrt(2.0)
    np.testing.assert_allclose(proposal, expected, atol=1e-9)
