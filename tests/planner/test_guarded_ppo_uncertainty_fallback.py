"""Tests for guarded PPO uncertainty-triggered fallback decisions."""

from __future__ import annotations

import pytest

from robot_sf.planner.guarded_ppo import GuardedPPOAdapter, build_guarded_ppo_config
from tests.planner.test_guarded_ppo import _FallbackAdapter, _obs


def _uncertainty_guard(
    *,
    mode: str = "stop",
    fallback_adapter: _FallbackAdapter | None = None,
    extra: dict[str, object] | None = None,
) -> GuardedPPOAdapter:
    cfg: dict[str, object] = {
        "guard_near_field_distance": 2.5,
        "guard_hard_ped_clearance": 0.1,
        "guard_first_step_ped_clearance": 0.1,
        "guard_min_ttc": 0.1,
        "uncertainty_fallback_enabled": True,
        "uncertainty_base_radius_m": 0.1,
        "uncertainty_conformal_radius_m": 0.5,
        "uncertainty_buffer_intrusion_threshold": 0.0,
        "uncertainty_collision_probability_threshold": 1.0,
        "uncertainty_fallback_mode": mode,
        "uncertainty_slow_down_speed_m_s": 0.2,
    }
    if extra:
        cfg.update(extra)
    return GuardedPPOAdapter(
        config=build_guarded_ppo_config(cfg),
        fallback_adapter=fallback_adapter or _FallbackAdapter((0.0, 0.0)),
    )


def test_uncertainty_fallback_trigger_stop_on_buffer_intrusion() -> None:
    """Conformal-buffer intrusion overrides an otherwise guard-safe PPO command."""
    guard = _uncertainty_guard()

    decision = guard.choose_command_decision(
        _obs(ped_positions=[(1.0, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.4, 0.0),
    )

    assert decision.decision_label == "uncertainty_fallback_stop"
    assert decision.filtered_action == (0.0, 0.0)
    assert decision.override_applied
    assert decision.uncertainty_metadata["triggered"] is True
    assert "uncertainty_buffer_intrusion" in decision.uncertainty_metadata["trigger_conditions"]
    assert (
        decision.calibration_metadata["claim_boundary"] == "diagnostic_proxy_not_safety_guarantee"
    )


def test_uncertainty_fallback_trigger_slow_down_preserves_turning() -> None:
    """Slow-down mode clamps linear speed while preserving angular velocity."""
    guard = _uncertainty_guard(mode="slow_down")

    decision = guard.choose_command_decision(
        _obs(ped_positions=[(1.0, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.6, 0.7),
    )

    assert decision.decision_label == "uncertainty_fallback_slow_down"
    assert decision.filtered_action == (0.2, 0.7)
    assert decision.override_applied


def test_uncertainty_fallback_trigger_uses_configured_fallback_adapter() -> None:
    """Configured fallback mode delegates to the existing fallback adapter."""
    fallback = _FallbackAdapter((0.05, -0.3))
    guard = _uncertainty_guard(mode="fallback", fallback_adapter=fallback)

    decision = guard.choose_command_decision(
        _obs(ped_positions=[(1.0, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.4, 0.0),
    )

    assert decision.decision_label == "uncertainty_fallback_configured"
    assert decision.filtered_action == (0.05, -0.3)
    assert fallback.plan_calls == 1


def test_uncertainty_fallback_trigger_low_ttc_records_reason() -> None:
    """Low time-to-collision threshold records its uncertainty trigger reason."""
    guard = _uncertainty_guard(
        extra={
            "uncertainty_conformal_radius_m": 0.0,
            "uncertainty_min_ttc_threshold_s": 2.0,
        }
    )

    decision = guard.choose_command_decision(
        _obs(ped_positions=[(0.9, 0.0)], ped_velocities=[(-0.8, 0.0)]),
        (0.0, 0.0),
    )

    assert decision.decision_label == "uncertainty_fallback_stop"
    assert "low_ttc" in decision.uncertainty_metadata["trigger_conditions"]


def test_uncertainty_fallback_disabled_preserves_existing_guarded_ppo_behavior() -> None:
    """Default-off uncertainty fallback preserves guard-safe PPO pass-through."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(
            {
                "guard_near_field_distance": 2.5,
                "guard_hard_ped_clearance": 0.1,
                "guard_first_step_ped_clearance": 0.1,
                "guard_min_ttc": 0.1,
            }
        ),
        fallback_adapter=_FallbackAdapter((0.0, 0.0)),
    )

    decision = guard.choose_command_decision(
        _obs(ped_positions=[(1.0, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.4, 0.0),
    )

    assert decision.decision_label == "ppo_safe"
    assert decision.filtered_action == (0.4, 0.0)
    assert not decision.override_applied


@pytest.mark.parametrize(
    "field,value",
    [
        ("uncertainty_buffer_intrusion_threshold", -0.01),
        ("uncertainty_collision_probability_threshold", 1.01),
        ("uncertainty_base_radius_m", -0.1),
        ("uncertainty_base_radius_m", 0.0),
        ("uncertainty_conformal_radius_m", -0.1),
        ("uncertainty_slow_down_speed_m_s", -0.1),
        ("uncertainty_min_ttc_threshold_s", -0.1),
    ],
)
def test_uncertainty_fallback_config_validation(field: str, value: object) -> None:
    """Uncertainty fallback config validates ranges fail-closed."""
    with pytest.raises(ValueError):
        build_guarded_ppo_config({"uncertainty_fallback_enabled": True, field: value})


def test_uncertainty_fallback_config_rejects_unknown_mode() -> None:
    """Unknown uncertainty fallback modes are rejected."""
    with pytest.raises(ValueError):
        build_guarded_ppo_config(
            {"uncertainty_fallback_enabled": True, "uncertainty_fallback_mode": "swerve"}
        )
