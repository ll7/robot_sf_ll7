"""Stable mode identifiers for manual-control runners and manifests."""

from __future__ import annotations

from enum import StrEnum


class ManualControlMode(StrEnum):
    """Supported manual-control input mapping modes."""

    KEYBOARD_HOLD = "keyboard_hold"
    KEYBOARD_CRUISE = "keyboard_cruise"
    MOUSE_TARGET = "mouse_target"


class ManualViewMode(StrEnum):
    """Supported manual-control camera/view modes."""

    FIXED_MAP = "fixed_map"
    EGO_UP = "ego_up"
    ROBOT_STATIC = "robot_static"


SUPPORTED_MVP_CONTROL_MODES = (ManualControlMode.KEYBOARD_HOLD,)
"""Control modes implemented in the current MVP foundation."""

SUPPORTED_MVP_VIEW_MODES = (ManualViewMode.FIXED_MAP,)
"""View modes implemented in the current MVP foundation."""


def ensure_supported_mvp_mode(
    *,
    control_mode: ManualControlMode | str,
    view_mode: ManualViewMode | str,
) -> None:
    """Fail closed when a requested manual-control mode is not implemented yet."""
    try:
        normalized_control_mode = (
            control_mode
            if isinstance(control_mode, ManualControlMode)
            else ManualControlMode(control_mode)
        )
    except ValueError as exc:
        raise NotImplementedError(
            f"manual control mode is not implemented: {control_mode}"
        ) from exc

    try:
        normalized_view_mode = (
            view_mode if isinstance(view_mode, ManualViewMode) else ManualViewMode(view_mode)
        )
    except ValueError as exc:
        raise NotImplementedError(f"manual view mode is not implemented: {view_mode}") from exc

    if normalized_control_mode not in SUPPORTED_MVP_CONTROL_MODES:
        raise NotImplementedError(
            f"manual control mode is not implemented: {normalized_control_mode.value}"
        )
    if normalized_view_mode not in SUPPORTED_MVP_VIEW_MODES:
        raise NotImplementedError(
            f"manual view mode is not implemented: {normalized_view_mode.value}"
        )
