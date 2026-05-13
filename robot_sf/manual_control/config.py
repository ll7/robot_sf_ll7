"""Configuration surface for manual-control mode selection."""

from __future__ import annotations

from dataclasses import dataclass

from robot_sf.manual_control.modes import (
    ManualControlMode,
    ManualViewMode,
    control_mode_spec,
    ensure_supported_manual_mode,
    parse_manual_control_mode,
    parse_manual_view_mode,
    view_mode_spec,
)


@dataclass(frozen=True)
class ManualControlRuntimeConfig:
    """Typed runtime configuration for manual-control mode selection."""

    control_mode: ManualControlMode = ManualControlMode.KEYBOARD_HOLD
    view_mode: ManualViewMode = ManualViewMode.FIXED_MAP
    robot_action_space: str = "differential_drive"

    @classmethod
    def from_strings(
        cls,
        *,
        control_mode: str = ManualControlMode.KEYBOARD_HOLD.value,
        view_mode: str = ManualViewMode.FIXED_MAP.value,
        robot_action_space: str = "differential_drive",
    ) -> ManualControlRuntimeConfig:
        """Build and validate a runtime config from CLI/config strings.

        Returns
        -------
        ManualControlRuntimeConfig
            Validated manual-control runtime configuration.
        """
        config = cls(
            control_mode=parse_manual_control_mode(control_mode),
            view_mode=parse_manual_view_mode(view_mode),
            robot_action_space=str(robot_action_space),
        )
        config.ensure_supported()
        return config

    def ensure_supported(self) -> None:
        """Fail closed if this mode combination is not executable."""
        ensure_supported_manual_mode(
            control_mode=self.control_mode,
            view_mode=self.view_mode,
            robot_action_space=self.robot_action_space,
        )

    @property
    def input_mapping_version(self) -> str:
        """Return the versioned input mapper associated with this control mode."""
        return control_mode_spec(self.control_mode).input_mapping_version

    def overlay_metadata(self) -> dict[str, str]:
        """Return UI overlay labels for the active control and view modes."""
        control_spec = control_mode_spec(self.control_mode)
        view_spec = view_mode_spec(self.view_mode)
        return {
            "control_mode": control_spec.mode.value,
            "view_mode": view_spec.mode.value,
            "input_mapping_version": control_spec.input_mapping_version,
            "control_overlay_label": control_spec.overlay_label,
            "view_overlay_label": view_spec.overlay_label,
        }

    def to_json_dict(self) -> dict[str, str]:
        """Return a compact JSON-compatible config dictionary."""
        return {
            "control_mode": self.control_mode.value,
            "view_mode": self.view_mode.value,
            "robot_action_space": self.robot_action_space,
            "input_mapping_version": self.input_mapping_version,
            **self.overlay_metadata(),
        }
