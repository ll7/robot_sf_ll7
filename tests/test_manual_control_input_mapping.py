"""Tests for pure manual-control input mapping."""

from robot_sf.manual_control.input_mapping import (
    DifferentialDriveCruiseKeyboardMapper,
    DifferentialDriveKeyboardMapper,
    DifferentialDriveMouseTargetMapper,
    ManualKeyState,
    ManualMouseTarget,
    mapper_for_manual_mode,
    mapper_for_robot_config,
)
from robot_sf.manual_control.modes import ManualControlMode
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings


def test_manual_key_state_normalizes_labels():
    """Raw key labels should normalize to lowercase symbolic names."""
    state = ManualKeyState.from_keys([" W ", "LEFT", ""])

    assert state.pressed == frozenset({"w", "left"})


def test_differential_mapper_emits_delta_to_forward_left_target():
    """Hold-to-command mapping should emit velocity deltas toward the target."""
    mapper = DifferentialDriveKeyboardMapper(
        DifferentialDriveSettings(max_linear_speed=2.0, max_angular_speed=1.0)
    )

    action = mapper.map_action(["w", "a"], current_velocity=(0.5, 0.25))

    assert action == (1.5, 0.75)


def test_differential_mapper_releases_to_stop():
    """No movement keys should command a delta back to zero velocity."""
    mapper = DifferentialDriveKeyboardMapper()

    action = mapper.map_action([], current_velocity=(1.0, -0.5))

    assert action == (-1.0, 0.5)


def test_differential_mapper_blocks_reverse_unless_enabled():
    """Backward key should brake when reverse driving is disabled."""
    mapper = DifferentialDriveKeyboardMapper(DifferentialDriveSettings(allow_backwards=False))

    action = mapper.map_action(["s"], current_velocity=(0.5, 0.0))

    assert action == (-0.5, 0.0)


def test_differential_mapper_allows_reverse_when_configured():
    """Backward key should target negative max speed when reverse is enabled."""
    mapper = DifferentialDriveKeyboardMapper(
        DifferentialDriveSettings(max_linear_speed=2.0, allow_backwards=True)
    )

    action = mapper.map_action(["down"], current_velocity=(0.5, 0.0))

    assert action == (-2.5, 0.0)


def test_differential_mapper_metadata_versions_recordings():
    """Mapper metadata should identify the control mode for manifests."""
    metadata = DifferentialDriveKeyboardMapper().metadata()

    assert metadata["input_mapping_version"] == "manual_keyboard_diff_drive_hold_v1"
    assert metadata["control_mode"] == "keyboard_hold"
    assert metadata["robot_action_space"] == "differential_drive"


def test_mapper_for_robot_config_selects_differential_drive_mapper():
    """Manual mapper selection should support the MVP differential-drive path."""
    mapper = mapper_for_robot_config(DifferentialDriveSettings())

    assert isinstance(mapper, DifferentialDriveKeyboardMapper)


def test_mapper_for_robot_config_fails_closed_for_unsupported_action_space():
    """Unsupported action spaces should fail closed with the missing mapper named."""
    try:
        mapper_for_robot_config(BicycleDriveSettings())
    except NotImplementedError as exc:
        assert "BicycleDriveSettings" in str(exc)
        assert "differential_drive" in str(exc)
    else:  # pragma: no cover - defensive assertion style for clearer failure
        raise AssertionError("expected NotImplementedError")


def test_keyboard_cruise_mapper_increments_persistent_target_velocity():
    """Cruise mode should adjust a persistent target instead of requiring held velocity."""
    mapper = DifferentialDriveCruiseKeyboardMapper(
        DifferentialDriveSettings(max_linear_speed=1.0, max_angular_speed=0.5),
        linear_step=0.4,
        angular_step=0.25,
    )

    target = mapper.next_target_velocity(["w", "a"], current_target_velocity=(0.8, 0.4))
    action = mapper.map_action(
        ["w", "a"],
        current_velocity=(0.5, 0.1),
        current_target_velocity=(0.8, 0.4),
    )

    assert target == (1.0, 0.5)
    assert action == (0.5, 0.4)


def test_keyboard_cruise_mapper_brake_resets_target_velocity():
    """Cruise brake should reset persistent target velocity to zero."""
    mapper = DifferentialDriveCruiseKeyboardMapper()

    assert mapper.next_target_velocity(["space"], current_target_velocity=(1.0, -1.0)) == (0.0, 0.0)
    assert mapper.map_action(["space"], current_velocity=(0.25, -0.5)) == (-0.25, 0.5)


def test_mouse_target_mapper_points_at_local_target():
    """Mouse-target mode should convert local intent into bounded velocity deltas."""
    mapper = DifferentialDriveMouseTargetMapper(
        DifferentialDriveSettings(max_linear_speed=2.0, max_angular_speed=1.0),
        linear_gain=1.0,
        angular_gain=1.0,
    )

    action = mapper.map_action(ManualMouseTarget(x=0.0, y=2.0), current_velocity=(0.5, 0.25))

    assert action == (1.5, 0.75)


def test_mouse_target_mapper_requires_target():
    """Mouse-target mode should fail closed when no cursor/click target is available."""
    mapper = DifferentialDriveMouseTargetMapper()

    try:
        mapper.map_action(None)
    except ValueError as exc:
        assert "requires a local mouse target" in str(exc)
    else:  # pragma: no cover - defensive assertion style for clearer failure
        raise AssertionError("expected ValueError")


def test_mapper_for_manual_mode_selects_registered_differential_mappers():
    """Mapper factory should select each implemented differential-drive mode."""
    settings = DifferentialDriveSettings()

    assert isinstance(
        mapper_for_manual_mode(settings, ManualControlMode.KEYBOARD_HOLD),
        DifferentialDriveKeyboardMapper,
    )
    assert isinstance(
        mapper_for_manual_mode(settings, ManualControlMode.KEYBOARD_CRUISE),
        DifferentialDriveCruiseKeyboardMapper,
    )
    assert isinstance(
        mapper_for_manual_mode(settings, ManualControlMode.MOUSE_TARGET),
        DifferentialDriveMouseTargetMapper,
    )


def test_mapper_for_manual_mode_fails_closed_for_unsupported_action_space():
    """Unsupported action spaces should name the missing mode/action-space combination."""
    try:
        mapper_for_manual_mode(BicycleDriveSettings(), ManualControlMode.MOUSE_TARGET)
    except NotImplementedError as exc:
        assert "BicycleDriveSettings" in str(exc)
        assert "mouse_target" in str(exc)
        assert "differential_drive" in str(exc)
    else:  # pragma: no cover - defensive assertion style for clearer failure
        raise AssertionError("expected NotImplementedError")
