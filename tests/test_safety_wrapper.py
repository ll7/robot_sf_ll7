"""Tests for the planner-agnostic safety wrapper (issue #3501)."""

from __future__ import annotations

import pytest

from robot_sf.robot.safety_wrapper import (
    INTERVENTION_DISABLED,
    INTERVENTION_HARD_STOP,
    INTERVENTION_NONE,
    INTERVENTION_SPEED_CAP,
    SAFETY_WRAPPER_SCHEMA,
    SafetyContext,
    SafetyWrapperConfig,
    apply_safety_wrapper,
)

_ENABLED = SafetyWrapperConfig(enabled=True)
_SAFE = SafetyContext(min_pedestrian_distance_m=10.0, min_clearance_m=5.0, min_ttc_s=None)


def test_disabled_by_default_is_pass_through() -> None:
    """With the default (disabled) config the action must pass through unchanged."""
    result = apply_safety_wrapper(1.5, 0.3, _SAFE)

    assert result["intervention"] == INTERVENTION_DISABLED
    assert result["enabled"] is False
    assert result["corrected_linear_velocity"] == 1.5
    assert result["corrected_angular_velocity"] == 0.3


def test_safe_context_when_enabled_is_pass_through() -> None:
    """An enabled wrapper must not intervene when the context is safe."""
    result = apply_safety_wrapper(1.5, 0.3, _SAFE, _ENABLED)

    assert result["intervention"] == INTERVENTION_NONE
    assert result["intervened"] is False
    assert result["corrected_linear_velocity"] == 1.5


def test_speed_cap_near_pedestrians() -> None:
    """Within the caution radius an over-cap speed must be clamped to the ceiling."""
    context = SafetyContext(min_pedestrian_distance_m=1.0, min_clearance_m=5.0)
    result = apply_safety_wrapper(2.0, 0.1, context, _ENABLED)

    assert result["intervention"] == INTERVENTION_SPEED_CAP
    assert result["corrected_linear_velocity"] == _ENABLED.capped_speed_m_s
    assert result["corrected_angular_velocity"] == 0.1  # turning preserved


def test_no_speed_cap_when_already_slow_near_pedestrians() -> None:
    """A speed already below the cap must not be modified."""
    context = SafetyContext(min_pedestrian_distance_m=1.0, min_clearance_m=5.0)
    result = apply_safety_wrapper(0.4, 0.0, context, _ENABLED)

    assert result["intervention"] == INTERVENTION_NONE
    assert result["corrected_linear_velocity"] == 0.4


def test_hard_stop_on_low_ttc() -> None:
    """A critical time-to-collision must veto a hard stop (zero forward speed)."""
    context = SafetyContext(min_pedestrian_distance_m=1.0, min_clearance_m=5.0, min_ttc_s=0.5)
    result = apply_safety_wrapper(2.0, 0.2, context, _ENABLED)

    assert result["intervention"] == INTERVENTION_HARD_STOP
    assert result["corrected_linear_velocity"] == 0.0
    assert result["corrected_angular_velocity"] == 0.2  # may still turn to yield


def test_hard_stop_on_low_clearance() -> None:
    """Critically low clearance must veto a hard stop."""
    context = SafetyContext(min_pedestrian_distance_m=5.0, min_clearance_m=0.2)
    result = apply_safety_wrapper(1.0, 0.0, context, _ENABLED)

    assert result["intervention"] == INTERVENTION_HARD_STOP
    assert result["corrected_linear_velocity"] == 0.0


def test_veto_takes_precedence_over_speed_cap() -> None:
    """When both apply, the hard-stop veto must override the speed cap."""
    context = SafetyContext(min_pedestrian_distance_m=1.0, min_clearance_m=0.1, min_ttc_s=0.5)
    result = apply_safety_wrapper(2.0, 0.0, context, _ENABLED)

    assert result["intervention"] == INTERVENTION_HARD_STOP
    assert result["corrected_linear_velocity"] == 0.0


def test_record_is_schema_tagged_and_reports_original() -> None:
    """The record must be versioned and preserve the original commanded action."""
    result = apply_safety_wrapper(2.0, 0.5, _SAFE, _ENABLED)

    assert result["schema_version"] == SAFETY_WRAPPER_SCHEMA
    assert result["evidence_kind"] == "diagnostic_proxy"
    assert result["original_linear_velocity"] == 2.0
    assert result["original_angular_velocity"] == 0.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pedestrian_caution_radius_m": 0.0},
        {"capped_speed_m_s": -1.0},
        {"ttc_veto_threshold_s": 0.0},
        {"clearance_veto_m": -0.1},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, float]) -> None:
    """Non-physical thresholds must fail closed at construction."""
    with pytest.raises(ValueError):
        SafetyWrapperConfig(enabled=True, **kwargs)
