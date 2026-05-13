"""Stable mode identifiers for manual-control runners and manifests."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ManualControlModeSpec:
    """Registry metadata for one manual-control input mode."""

    mode: ManualControlMode
    input_mapping_version: str
    overlay_label: str
    robot_action_space: str = "differential_drive"
    implemented: bool = True


@dataclass(frozen=True)
class ManualViewModeSpec:
    """Registry metadata for one manual-control view mode."""

    mode: ManualViewMode
    overlay_label: str
    implemented: bool = True
    blocker: str | None = None


CONTROL_MODE_REGISTRY: dict[ManualControlMode, ManualControlModeSpec] = {
    ManualControlMode.KEYBOARD_HOLD: ManualControlModeSpec(
        mode=ManualControlMode.KEYBOARD_HOLD,
        input_mapping_version="manual_keyboard_diff_drive_hold_v1",
        overlay_label="Keyboard hold: WASD/arrows command velocity while held; Space brakes",
    ),
    ManualControlMode.KEYBOARD_CRUISE: ManualControlModeSpec(
        mode=ManualControlMode.KEYBOARD_CRUISE,
        input_mapping_version="keyboard_cruise_diff_drive_v1",
        overlay_label="Keyboard cruise: WASD/arrows adjust persistent target velocity; Space stops",
    ),
    ManualControlMode.MOUSE_TARGET: ManualControlModeSpec(
        mode=ManualControlMode.MOUSE_TARGET,
        input_mapping_version="mouse_target_diff_drive_v1",
        overlay_label="Mouse target: cursor/click sets local steering intent for differential drive",
    ),
}
"""Versioned manual-control input mode registry."""

VIEW_MODE_REGISTRY: dict[ManualViewMode, ManualViewModeSpec] = {
    ManualViewMode.FIXED_MAP: ManualViewModeSpec(
        mode=ManualViewMode.FIXED_MAP,
        overlay_label="Fixed map view: world-oriented static camera",
    ),
    ManualViewMode.EGO_UP: ManualViewModeSpec(
        mode=ManualViewMode.EGO_UP,
        overlay_label="Ego-up view: robot-centered camera with robot facing up",
        implemented=False,
        blocker=(
            "ego_up_view_v1 requires an interactive renderer camera transform hook; "
            "the current manual-control foundation exposes pure mode metadata only"
        ),
    ),
    ManualViewMode.ROBOT_STATIC: ManualViewModeSpec(
        mode=ManualViewMode.ROBOT_STATIC,
        overlay_label="Robot-static view: robot-centered camera without ego-up rotation",
        implemented=False,
        blocker="robot_static view has no renderer camera transform hook in this foundation",
    ),
}
"""Versioned manual-control view mode registry."""


def parse_manual_control_mode(value: str | ManualControlMode) -> ManualControlMode:
    """Parse a manual-control mode identifier and fail closed for unknown values.

    Returns
    -------
    ManualControlMode
        Parsed control mode.
    """
    if isinstance(value, ManualControlMode):
        return value
    try:
        return ManualControlMode(str(value))
    except ValueError as exc:
        supported = ", ".join(mode.value for mode in ManualControlMode)
        raise ValueError(f"unknown manual control mode {value!r}; supported: {supported}") from exc


def parse_manual_view_mode(value: str | ManualViewMode) -> ManualViewMode:
    """Parse a manual-view mode identifier and fail closed for unknown values.

    Returns
    -------
    ManualViewMode
        Parsed view mode.
    """
    if isinstance(value, ManualViewMode):
        return value
    try:
        return ManualViewMode(str(value))
    except ValueError as exc:
        supported = ", ".join(mode.value for mode in ManualViewMode)
        raise ValueError(f"unknown manual view mode {value!r}; supported: {supported}") from exc


def control_mode_spec(mode: str | ManualControlMode) -> ManualControlModeSpec:
    """Return registry metadata for a control mode."""
    parsed = parse_manual_control_mode(mode)
    return CONTROL_MODE_REGISTRY[parsed]


def control_mode_for_input_mapping_version(input_mapping_version: str) -> ManualControlMode:
    """Return the control mode that owns a versioned input mapping identifier."""
    normalized = str(input_mapping_version)
    for spec in CONTROL_MODE_REGISTRY.values():
        if spec.input_mapping_version == normalized:
            return spec.mode
    raise ValueError(
        "unknown manual-control input mapping version "
        f"{input_mapping_version!r}; expected one of: "
        + ", ".join(spec.input_mapping_version for spec in CONTROL_MODE_REGISTRY.values())
    )


def view_mode_spec(mode: str | ManualViewMode) -> ManualViewModeSpec:
    """Return registry metadata for a view mode."""
    parsed = parse_manual_view_mode(mode)
    return VIEW_MODE_REGISTRY[parsed]


def ensure_supported_manual_mode(
    *,
    control_mode: str | ManualControlMode,
    view_mode: str | ManualViewMode,
    robot_action_space: str = "differential_drive",
) -> None:
    """Fail closed when a requested manual-control mode combination is unsupported."""
    control_spec = control_mode_spec(control_mode)
    view_spec = view_mode_spec(view_mode)
    if not control_spec.implemented:
        raise NotImplementedError(
            f"manual control mode is not implemented: {control_spec.mode.value}"
        )
    if control_spec.robot_action_space != robot_action_space:
        raise NotImplementedError(
            "manual control mapper is not available for "
            f"{robot_action_space}; mode {control_spec.mode.value} supports "
            f"{control_spec.robot_action_space}"
        )
    if not view_spec.implemented:
        blocker = f"; blocker: {view_spec.blocker}" if view_spec.blocker else ""
        raise NotImplementedError(
            f"manual view mode is not implemented: {view_spec.mode.value}{blocker}"
        )


def ensure_supported_mvp_mode(
    *,
    control_mode: ManualControlMode | str,
    view_mode: ManualViewMode | str,
) -> None:
    """Fail closed when a requested manual-control mode is not implemented yet."""
    try:
        ensure_supported_manual_mode(control_mode=control_mode, view_mode=view_mode)
    except ValueError as exc:
        message = str(exc)
        if "manual view mode" in message:
            raise NotImplementedError(f"manual view mode is not implemented: {view_mode}") from exc
        raise NotImplementedError(
            f"manual control mode is not implemented: {control_mode}"
        ) from exc
