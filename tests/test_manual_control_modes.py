"""Tests for manual-control mode identifiers."""

import pytest

from robot_sf.manual_control.modes import (
    CONTROL_MODE_REGISTRY,
    ManualControlMode,
    ManualViewMode,
    configure_manual_view_renderer,
    control_mode_for_input_mapping_version,
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


def test_ensure_supported_mvp_mode_rejects_robot_static_until_camera_hook_exists():
    """Stretch view modes should fail closed until implemented."""
    with pytest.raises(NotImplementedError, match="robot_static"):
        ensure_supported_mvp_mode(
            control_mode=ManualControlMode.KEYBOARD_HOLD,
            view_mode=ManualViewMode.ROBOT_STATIC,
        )


def test_post_mvp_control_modes_are_registered_with_versions_and_labels():
    """New steering modes should be registry-visible and artifact-filterable."""
    cruise = CONTROL_MODE_REGISTRY[ManualControlMode.KEYBOARD_CRUISE]
    mouse = CONTROL_MODE_REGISTRY[ManualControlMode.MOUSE_TARGET]

    assert cruise.input_mapping_version == "keyboard_cruise_diff_drive_v1"
    assert mouse.input_mapping_version == "mouse_target_diff_drive_v1"
    assert "persistent target velocity" in cruise.overlay_label
    assert "steering intent" in mouse.overlay_label


def test_input_mapping_versions_resolve_back_to_control_modes() -> None:
    """Recorded mapping versions should round-trip back to their owning control mode."""
    assert (
        control_mode_for_input_mapping_version("keyboard_cruise_diff_drive_v1")
        == ManualControlMode.KEYBOARD_CRUISE
    )
    with pytest.raises(ValueError, match="unknown manual-control input mapping version"):
        control_mode_for_input_mapping_version("joystick_arcade_v1")


def test_ego_up_view_is_supported_by_mode_registry():
    """Ego-up should be selectable now that the renderer exposes camera hooks."""
    spec = view_mode_spec(ManualViewMode.EGO_UP)

    assert spec.implemented is True
    assert spec.blocker is None
    ensure_supported_manual_mode(
        control_mode=ManualControlMode.KEYBOARD_CRUISE,
        view_mode=ManualViewMode.EGO_UP,
    )


def test_ego_up_renderer_adapter_requires_camera_transform_hook():
    """Ego-up should still fail closed when a renderer cannot apply the camera transform."""
    with pytest.raises(NotImplementedError, match="camera transform hook"):
        configure_manual_view_renderer(object(), ManualViewMode.EGO_UP)


def test_fixed_map_renderer_adapter_allows_renderers_without_camera_hook():
    """Fixed-map remains compatible with renderers that have no manual camera hook."""
    configure_manual_view_renderer(object(), ManualViewMode.FIXED_MAP)


def test_ego_up_renderer_adapter_configures_supported_renderer():
    """Supported renderers receive the selected manual view mode."""

    class _Renderer:
        configured_mode = None

        def set_manual_view_mode(self, mode):
            self.configured_mode = mode

    renderer = _Renderer()
    configure_manual_view_renderer(renderer, ManualViewMode.EGO_UP)

    assert renderer.configured_mode == ManualViewMode.EGO_UP


def test_ensure_supported_mvp_mode_accepts_supported_string_inputs():
    """Public mode guards should accept the serialized MVP string values."""
    ensure_supported_mvp_mode(control_mode="keyboard_hold", view_mode="fixed_map")


def test_ensure_supported_mvp_mode_rejects_unknown_strings() -> None:
    """Unknown serialized mode values should still fail closed."""
    with pytest.raises(NotImplementedError, match="joystick"):
        ensure_supported_mvp_mode(control_mode="joystick", view_mode="fixed_map")
