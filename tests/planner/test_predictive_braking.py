"""Behavioural counterexamples for the opt-in stopping bound (#10111)."""

import numpy as np
import pytest

from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleCandidate,
    HybridRuleLocalPlannerAdapter,
    build_hybrid_rule_local_planner_config,
)


def planner(**overrides):
    """Enable the successor through the production mapping builder."""
    return HybridRuleLocalPlannerAdapter(
        build_hybrid_rule_local_planner_config(
            {
                "planner_variant": "hybrid_rule_v4_clearance_braking",
                "v4_predictive_braking_enabled": True,
                "v4_prediction_speed_error": 0.2,
                **overrides,
            }
        )
    )


def state(position=(1.7, 0.0), velocity=(1.0, 0.0), speed=0.0, dt=0.1):
    """A metre-radius robot and 0.4 m pedestrian, using development fixtures."""
    return {
        "robot_pos": np.zeros(2),
        "heading": 0.0,
        "current_speed": speed,
        "ped_pos": np.array([position]),
        "ped_vel": np.array([velocity]),
        "robot_radius": 1.0,
        "ped_radius": 0.4,
        "dt": dt,
        "observation": {},
    }


def test_observed_retreat_is_not_limited_by_present_position():
    """A safe receding pedestrian no longer clamps every direction to 0.15 m/s."""
    p = planner()
    s = state(speed=0.5)
    assert p._v4_human_speed_cap(s) == 2.0
    assert (
        p._v4_braking_rejection(
            candidate=HybridRuleCandidate(0.6, 0.0, "dynamic_window"),
            state=s,
            collision_radius=1.5,
        )
        is None
    )


def test_crossing_between_samples_rejected():
    """Both endpoints are clear, but the pedestrian crosses the stopping segment."""
    p = planner(v4_prediction_speed_error=0.0)
    s = state(position=(0.05, 2.0), velocity=(0.0, -2.0), dt=2.0)
    result = p._v4_braking_rejection(
        candidate=HybridRuleCandidate(0.1, 0.0, "dynamic_window"),
        state=s,
        collision_radius=1.5,
    )
    assert result is not None, "endpoint-only braking misses crossing contact"
    assert result["reason"] == "braking_infeasible"


def test_prediction_error_tube_rejects_nominally_safe_retreat():
    """Nominal prediction clears; allowed inward velocity error cannot be ignored."""
    p = planner(v4_prediction_speed_error=2.0)
    s = state(speed=0.5)
    result = p._v4_braking_rejection(
        candidate=HybridRuleCandidate(0.6, 0.0, "dynamic_window"),
        state=s,
        collision_radius=1.5,
    )
    assert result is not None, "unbounded confidence in pedestrian retreat"
    assert result["reason"] == "braking_infeasible"


def test_stationary_pedestrian_still_requires_full_stop():
    """No credit is invented for a stationary pedestrian or truncated horizon."""
    p = planner()
    s = state(position=(2.0, 0.0), velocity=(0.0, 0.0), speed=1.0)
    result = p._v4_braking_rejection(
        candidate=HybridRuleCandidate(1.0, 0.0, "dynamic_window"),
        state=s,
        collision_radius=1.5,
    )
    assert result is not None


def test_disabled_switch_preserves_legacy_cap():
    """An unchanged 0.0.8 mapping retains its present-position speed band."""
    p = planner(v4_predictive_braking_enabled=False)
    assert p._v4_human_speed_cap(state()) == pytest.approx(0.15)


@pytest.mark.parametrize(
    "overrides",
    [
        {"v4_braking_check_enabled": False},
        {"planner_variant": "hybrid_rule_v3_teb_like_rollout"},
        {"v4_prediction_speed_error": -0.1},
        {"v4_prediction_speed_error": float("nan")},
        {"v4_prediction_speed_error": float("inf")},
    ],
)
def test_predictive_bound_cannot_be_disabled_or_use_invalid_uncertainty(overrides):
    """A malformed opt-in cannot remove the only remaining speed safety gate."""
    with pytest.raises(ValueError):
        planner(**overrides)


def test_accepted_stops_keep_separation_under_bounded_pedestrian_error():
    """Check accepted straight stops against independent differential-drive odometry."""
    from robot_sf.robot.differential_drive import (
        DifferentialDriveMotion,
        DifferentialDriveSettings,
        DifferentialDriveState,
    )

    p = planner(v4_reaction_time=0.3)
    motion = DifferentialDriveMotion(DifferentialDriveSettings())
    accepted = 0
    for gap in (0.3, 0.6, 1.0, 2.0):
        for speed in (0.0, 0.3, 0.8, 1.5):
            for ped_speed in (-0.5, 0.0, 0.6, 1.2):
                s = state(position=(1.4 + gap, 0), velocity=(ped_speed, 0), speed=speed)
                command = HybridRuleCandidate(min(speed + 0.1, 2), 0, "dynamic_window")
                if p._v4_braking_rejection(candidate=command, state=s, collision_radius=1.5):
                    continue
                accepted += 1
                drive = DifferentialDriveState(
                    velocity=(speed, 0),
                    wheel_speeds=(
                        speed / motion.config.wheel_radius,
                        speed / motion.config.wheel_radius,
                    ),
                )
                elapsed = 0.0
                for i in range(30):
                    old_x = drive.pose[0][0]
                    target = command.linear if i < 3 else 0.0
                    motion.move(drive, ((target - drive.velocity[0]) / 0.1, 0), 0.1)
                    # A worst allowed inward drift (0.2 m/s) remains in the tube.
                    for fraction in np.linspace(0, 1, 21):
                        t = elapsed + fraction * 0.1
                        x = old_x + fraction * (drive.pose[0][0] - old_x)
                        assert abs(1.4 + gap + (ped_speed - 0.2) * t - x) > 1.5
                    elapsed += 0.1
                    if i >= 3 and drive.velocity[0] == 0:
                        break
    assert accepted > 10, "the separation oracle must exercise accepted movement"
