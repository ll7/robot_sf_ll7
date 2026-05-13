"""Pure input mappers for manual-control sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.manual_control.modes import ManualControlMode
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
        return {
            "input_mapping_version": self.input_mapping_version,
            "control_mode": ManualControlMode.KEYBOARD_HOLD.value,
            "robot_action_space": "differential_drive",
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
