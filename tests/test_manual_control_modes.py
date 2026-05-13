"""Tests for manual-control mode identifiers."""

import pytest

from robot_sf.manual_control.modes import (
    ManualControlMode,
    ManualViewMode,
    ensure_supported_mvp_mode,
)


def test_ensure_supported_mvp_mode_accepts_keyboard_hold_fixed_map():
    """Current MVP mode pair should be accepted."""
    ensure_supported_mvp_mode(
        control_mode=ManualControlMode.KEYBOARD_HOLD,
        view_mode=ManualViewMode.FIXED_MAP,
    )


def test_ensure_supported_mvp_mode_rejects_unimplemented_control_mode():
    """Stretch control modes should fail closed until implemented."""
    with pytest.raises(NotImplementedError, match="mouse_target"):
        ensure_supported_mvp_mode(
            control_mode=ManualControlMode.MOUSE_TARGET,
            view_mode=ManualViewMode.FIXED_MAP,
        )


def test_ensure_supported_mvp_mode_rejects_unimplemented_view_mode():
    """Stretch view modes should fail closed until implemented."""
    with pytest.raises(NotImplementedError, match="ego_up"):
        ensure_supported_mvp_mode(
            control_mode=ManualControlMode.KEYBOARD_HOLD,
            view_mode=ManualViewMode.EGO_UP,
        )


def test_ensure_supported_mvp_mode_accepts_supported_string_inputs():
    """Public mode guards should accept the serialized MVP string values."""
    ensure_supported_mvp_mode(control_mode="keyboard_hold", view_mode="fixed_map")


def test_ensure_supported_mvp_mode_rejects_unknown_strings() -> None:
    """Unknown serialized mode values should still fail closed."""
    with pytest.raises(NotImplementedError, match="joystick"):
        ensure_supported_mvp_mode(control_mode="joystick", view_mode="fixed_map")
