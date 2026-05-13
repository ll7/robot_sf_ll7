"""Tests for manual-control mode identifiers."""

import pytest

from robot_sf.manual_control.modes import (
    CONTROL_MODE_REGISTRY,
    ManualControlMode,
    ManualViewMode,
    ensure_supported_manual_mode,
    ensure_supported_mvp_mode,
    view_mode_spec,
)


def test_ensure_supported_mvp_mode_accepts_keyboard_hold_fixed_map():
    """Current MVP mode pair should be accepted."""
    ensure_supported_mvp_mode(
        control_mode=ManualControlMode.KEYBOARD_HOLD,
        view_mode=ManualViewMode.FIXED_MAP,
    )


def test_ensure_supported_mvp_mode_rejects_unimplemented_control_mode():
    """Action-space mismatches should fail closed with the missing mapper named."""
    with pytest.raises(NotImplementedError, match="holonomic"):
        ensure_supported_manual_mode(
            control_mode=ManualControlMode.MOUSE_TARGET,
            view_mode=ManualViewMode.FIXED_MAP,
            robot_action_space="holonomic",
        )


def test_ensure_supported_mvp_mode_rejects_unimplemented_view_mode():
    """Stretch view modes should fail closed until implemented."""
    with pytest.raises(NotImplementedError, match="ego_up"):
        ensure_supported_mvp_mode(
            control_mode=ManualControlMode.KEYBOARD_HOLD,
            view_mode=ManualViewMode.EGO_UP,
        )


def test_post_mvp_control_modes_are_registered_with_versions_and_labels():
    """New steering modes should be registry-visible and artifact-filterable."""
    cruise = CONTROL_MODE_REGISTRY[ManualControlMode.KEYBOARD_CRUISE]
    mouse = CONTROL_MODE_REGISTRY[ManualControlMode.MOUSE_TARGET]

    assert cruise.input_mapping_version == "keyboard_cruise_diff_drive_v1"
    assert mouse.input_mapping_version == "mouse_target_diff_drive_v1"
    assert "persistent target velocity" in cruise.overlay_label
    assert "steering intent" in mouse.overlay_label


def test_ego_up_view_fails_closed_with_documented_blocker():
    """Ego-up should be explicit but blocked until the renderer exposes camera hooks."""
    spec = view_mode_spec(ManualViewMode.EGO_UP)

    assert spec.implemented is False
    assert "renderer camera transform hook" in str(spec.blocker)
    with pytest.raises(NotImplementedError, match="renderer camera transform hook"):
        ensure_supported_manual_mode(
            control_mode=ManualControlMode.KEYBOARD_CRUISE,
            view_mode=ManualViewMode.EGO_UP,
        )


def test_ensure_supported_mvp_mode_accepts_supported_string_inputs():
    """Public mode guards should accept the serialized MVP string values."""
    ensure_supported_mvp_mode(control_mode="keyboard_hold", view_mode="fixed_map")


def test_ensure_supported_mvp_mode_rejects_unknown_strings() -> None:
    """Unknown serialized mode values should still fail closed."""
    with pytest.raises(NotImplementedError, match="joystick"):
        ensure_supported_mvp_mode(control_mode="joystick", view_mode="fixed_map")
