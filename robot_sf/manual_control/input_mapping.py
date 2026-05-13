"""Pure input mappers for manual-control sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.manual_control.modes import (
    ManualControlMode,
    control_mode_spec,
    parse_manual_control_mode,
)
from robot_sf.robot.differential_drive import DifferentialDriveSettings

if TYPE_CHECKING:
    from collections.abc import Iterable

    from robot_sf.common.types import DifferentialDriveAction
    from robot_sf.robot.bicycle_drive import BicycleDriveSettings
    from robot_sf.robot.holonomic_drive import HolonomicDriveSettings

FORWARD_KEYS = frozenset({"w", "up"})
BACKWARD_KEYS = frozenset({"s", "down"})
LEFT_KEYS = frozenset({"a", "left"})
RIGHT_KEYS = frozenset({"d", "right"})
BRAKE_KEYS = frozenset({"space", "brake", "stop"})


def _clip_velocity(
    *,
    linear: float,
    angular: float,
    settings: DifferentialDriveSettings,
) -> tuple[float, float]:
    """Clip a differential-drive target velocity to robot limits.

    Returns
    -------
    tuple[float, float]
        Clipped ``(linear, angular)`` target velocity.
    """
    min_linear = settings.min_linear_speed if settings.allow_backwards else 0.0
    return (
        float(np.clip(linear, min_linear, settings.max_linear_speed)),
        float(np.clip(angular, -settings.max_angular_speed, settings.max_angular_speed)),
    )


@dataclass(frozen=True)
class ManualKeyState:
    """Normalized keyboard state for one manual-control step."""

    pressed: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_keys(cls, keys: Iterable[str]) -> ManualKeyState:
        """Normalize raw key labels into lowercase symbolic key names.

        Returns
        -------
        ManualKeyState
            Normalized immutable key state.
        """
        return cls(frozenset(str(key).strip().lower() for key in keys if str(key).strip()))

    def any_pressed(self, candidates: frozenset[str]) -> bool:
        """Return whether any candidate key is currently pressed.

        Returns
        -------
        bool
            True when any candidate is present in the current key set.
        """
        return bool(self.pressed & candidates)


@dataclass(frozen=True)
class DifferentialDriveKeyboardMapper:
    """Map keyboard hold controls into differential-drive velocity deltas.

    The repository's differential-drive action is a ``(delta_linear, delta_angular)``
    command applied to the current robot velocity. This mapper therefore treats
    keyboard input as a target velocity and emits the delta needed to move from the
    current velocity to that target for the next environment step.
    """

    settings: DifferentialDriveSettings = field(default_factory=DifferentialDriveSettings)
    input_mapping_version: str = "manual_keyboard_diff_drive_hold_v1"

    def map_action(
        self,
        keys: ManualKeyState | Iterable[str],
        *,
        current_velocity: tuple[float, float] = (0.0, 0.0),
    ) -> DifferentialDriveAction:
        """Return the action delta for the current key state.

        Args:
            keys: Normalized key state or raw key labels.
            current_velocity: Current ``(linear, angular)`` robot velocity.

        Returns:
            Differential-drive action ``(delta_linear, delta_angular)``.
        """
        key_state = keys if isinstance(keys, ManualKeyState) else ManualKeyState.from_keys(keys)
        target_linear, target_angular = self._target_velocity(key_state)
        current_linear, current_angular = current_velocity
        return (
            float(target_linear - current_linear),
            float(target_angular - current_angular),
        )

    def metadata(self) -> dict[str, object]:
        """Return manifest metadata for recorded manual-control sessions.

        Returns
        -------
        dict[str, object]
            JSON-compatible input-mapping metadata.
        """
        spec = control_mode_spec(ManualControlMode.KEYBOARD_HOLD)
        return {
            "input_mapping_version": self.input_mapping_version,
            "control_mode": spec.mode.value,
            "robot_action_space": spec.robot_action_space,
            "overlay_label": spec.overlay_label,
            "forward_keys": sorted(FORWARD_KEYS),
            "backward_keys": sorted(BACKWARD_KEYS),
            "left_keys": sorted(LEFT_KEYS),
            "right_keys": sorted(RIGHT_KEYS),
            "brake_keys": sorted(BRAKE_KEYS),
        }

    def _target_velocity(self, keys: ManualKeyState) -> tuple[float, float]:
        """Compute target velocity from held keys.

        Returns
        -------
        tuple[float, float]
            Target ``(linear, angular)`` velocity.
        """
        if keys.any_pressed(BRAKE_KEYS):
            return 0.0, 0.0

        target_linear = self._target_linear_velocity(keys)
        target_angular = self._target_angular_velocity(keys)
        return target_linear, target_angular

    def _target_linear_velocity(self, keys: ManualKeyState) -> float:
        """Compute target linear velocity from forward/backward keys.

        Returns
        -------
        float
            Target linear velocity in meters per second.
        """
        forward = keys.any_pressed(FORWARD_KEYS)
        backward = keys.any_pressed(BACKWARD_KEYS)
        if forward == backward:
            return 0.0
        if forward:
            return float(self.settings.max_linear_speed)
        if self.settings.allow_backwards:
            return float(self.settings.min_linear_speed)
        return 0.0

    def _target_angular_velocity(self, keys: ManualKeyState) -> float:
        """Compute target angular velocity from left/right keys.

        Returns
        -------
        float
            Target angular velocity in radians per second.
        """
        left = keys.any_pressed(LEFT_KEYS)
        right = keys.any_pressed(RIGHT_KEYS)
        if left == right:
            return 0.0
        angular = self.settings.max_angular_speed
        return float(np.clip(angular if left else -angular, -angular, angular))


@dataclass(frozen=True)
class DifferentialDriveCruiseKeyboardMapper:
    """Map keyboard cruise controls into differential-drive velocity deltas.

    Cruise mode is deliberately stateless inside the mapper. The runner owns the
    persistent target velocity and passes it back as ``current_target_velocity`` on
    each step, which keeps recording/replay deterministic and easy to test.
    """

    settings: DifferentialDriveSettings = field(default_factory=DifferentialDriveSettings)
    linear_step: float = 0.25
    angular_step: float = 0.25
    input_mapping_version: str = "keyboard_cruise_diff_drive_v1"

    def next_target_velocity(
        self,
        keys: ManualKeyState | Iterable[str],
        *,
        current_target_velocity: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[float, float]:
        """Return the next persistent target velocity after applying key input."""
        key_state = keys if isinstance(keys, ManualKeyState) else ManualKeyState.from_keys(keys)
        if key_state.any_pressed(BRAKE_KEYS):
            return 0.0, 0.0

        target_linear, target_angular = current_target_velocity
        if key_state.any_pressed(FORWARD_KEYS):
            target_linear += self.linear_step
        if key_state.any_pressed(BACKWARD_KEYS):
            target_linear -= self.linear_step
        if key_state.any_pressed(LEFT_KEYS):
            target_angular += self.angular_step
        if key_state.any_pressed(RIGHT_KEYS):
            target_angular -= self.angular_step
        return _clip_velocity(
            linear=target_linear,
            angular=target_angular,
            settings=self.settings,
        )

    def map_action(
        self,
        keys: ManualKeyState | Iterable[str],
        *,
        current_velocity: tuple[float, float] = (0.0, 0.0),
        current_target_velocity: tuple[float, float] = (0.0, 0.0),
    ) -> DifferentialDriveAction:
        """Return the action delta after updating the cruise target velocity."""
        target_linear, target_angular = self.next_target_velocity(
            keys,
            current_target_velocity=current_target_velocity,
        )
        current_linear, current_angular = current_velocity
        return (
            float(target_linear - current_linear),
            float(target_angular - current_angular),
        )

    def metadata(self) -> dict[str, object]:
        """Return manifest metadata for recorded cruise-control sessions."""
        spec = control_mode_spec(ManualControlMode.KEYBOARD_CRUISE)
        return {
            "input_mapping_version": self.input_mapping_version,
            "control_mode": spec.mode.value,
            "robot_action_space": spec.robot_action_space,
            "overlay_label": spec.overlay_label,
            "linear_step": self.linear_step,
            "angular_step": self.angular_step,
            "forward_keys": sorted(FORWARD_KEYS),
            "backward_keys": sorted(BACKWARD_KEYS),
            "left_keys": sorted(LEFT_KEYS),
            "right_keys": sorted(RIGHT_KEYS),
            "brake_keys": sorted(BRAKE_KEYS),
        }


@dataclass(frozen=True)
class ManualMouseTarget:
    """Local mouse steering target for differential-drive manual control."""

    x: float
    y: float
    speed: float | None = None

    @classmethod
    def from_xy(cls, target: tuple[float, float]) -> ManualMouseTarget:
        """Build a local mouse target from an ``(x, y)`` tuple.

        Returns
        -------
        ManualMouseTarget
            Local mouse target.
        """
        return cls(x=float(target[0]), y=float(target[1]))


@dataclass(frozen=True)
class DifferentialDriveMouseTargetMapper:
    """Map a local mouse target into differential-drive velocity deltas."""

    settings: DifferentialDriveSettings = field(default_factory=DifferentialDriveSettings)
    linear_gain: float = 0.8
    angular_gain: float = 1.5
    stop_radius: float = 0.05
    input_mapping_version: str = "mouse_target_diff_drive_v1"

    def target_velocity(
        self,
        target: ManualMouseTarget | tuple[float, float] | None,
    ) -> tuple[float, float]:
        """Return the target velocity implied by a local mouse target."""
        if target is None:
            raise ValueError("mouse_target_diff_drive_v1 requires a local mouse target")
        mouse_target = (
            target if isinstance(target, ManualMouseTarget) else ManualMouseTarget.from_xy(target)
        )
        distance = float(np.hypot(mouse_target.x, mouse_target.y))
        if distance <= self.stop_radius:
            return 0.0, 0.0

        desired_heading = float(np.arctan2(mouse_target.y, mouse_target.x))
        requested_speed = (
            float(mouse_target.speed)
            if mouse_target.speed is not None
            else distance * self.linear_gain
        )
        return _clip_velocity(
            linear=requested_speed,
            angular=desired_heading * self.angular_gain,
            settings=self.settings,
        )

    def map_action(
        self,
        target: ManualMouseTarget | tuple[float, float] | None,
        *,
        current_velocity: tuple[float, float] = (0.0, 0.0),
    ) -> DifferentialDriveAction:
        """Return the action delta for a local mouse steering target."""
        target_linear, target_angular = self.target_velocity(target)
        current_linear, current_angular = current_velocity
        return (
            float(target_linear - current_linear),
            float(target_angular - current_angular),
        )

    def metadata(self) -> dict[str, object]:
        """Return manifest metadata for recorded mouse-target sessions."""
        spec = control_mode_spec(ManualControlMode.MOUSE_TARGET)
        return {
            "input_mapping_version": self.input_mapping_version,
            "control_mode": spec.mode.value,
            "robot_action_space": spec.robot_action_space,
            "overlay_label": spec.overlay_label,
            "linear_gain": self.linear_gain,
            "angular_gain": self.angular_gain,
            "stop_radius": self.stop_radius,
        }


def mapper_for_robot_config(
    robot_config: DifferentialDriveSettings | BicycleDriveSettings | HolonomicDriveSettings,
) -> DifferentialDriveKeyboardMapper:
    """Return the supported manual-control mapper for a robot configuration.

    The MVP supports differential-drive keyboard hold controls only. Other
    action spaces fail closed until their explicit mapper versions are added.

    Returns
    -------
    DifferentialDriveKeyboardMapper
        Keyboard-hold mapper for differential-drive robots.
    """
    if isinstance(robot_config, DifferentialDriveSettings):
        return DifferentialDriveKeyboardMapper(robot_config)
    raise NotImplementedError(
        "Manual control mapper is not available for "
        f"{type(robot_config).__name__}; supported action space: differential_drive"
    )


def mapper_for_manual_mode(
    robot_config: DifferentialDriveSettings | BicycleDriveSettings | HolonomicDriveSettings,
    control_mode: str | ManualControlMode,
) -> (
    DifferentialDriveKeyboardMapper
    | DifferentialDriveCruiseKeyboardMapper
    | DifferentialDriveMouseTargetMapper
):
    """Return the supported manual-control mapper for a robot config and mode."""
    mode = parse_manual_control_mode(control_mode)
    if not isinstance(robot_config, DifferentialDriveSettings):
        raise NotImplementedError(
            "Manual control mapper is not available for "
            f"{type(robot_config).__name__}; mode {mode.value} supports differential_drive"
        )
    if mode == ManualControlMode.KEYBOARD_HOLD:
        return DifferentialDriveKeyboardMapper(robot_config)
    if mode == ManualControlMode.KEYBOARD_CRUISE:
        return DifferentialDriveCruiseKeyboardMapper(robot_config)
    if mode == ManualControlMode.MOUSE_TARGET:
        return DifferentialDriveMouseTargetMapper(robot_config)
    raise NotImplementedError(f"manual control mode is not implemented: {mode.value}")
