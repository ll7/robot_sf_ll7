"""Tests for first-class acceleration/braking actuation limits and the
stopping-distance envelope in run metadata (issue #4976).

These are focused unit tests only -- they exercise the drive models' deceleration
caps directly and the JSON-safe envelope helper, without running any campaign.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from robot_sf.benchmark.run_config_provenance import metric_affecting_run_config
from robot_sf.robot.actuation_envelope import (
    ACTUATION_ENVELOPE_INTERPRETATION,
    ACTUATION_ENVELOPE_SCHEMA,
    actuation_envelope_from_drive_config,
    stopping_distance,
)
from robot_sf.robot.bicycle_drive import (
    BicycleDriveRobot,
    BicycleDriveSettings,
    BicycleDriveState,
    BicycleMotion,
)
from robot_sf.robot.differential_drive import (
    DifferentialDriveMotion,
    DifferentialDriveRobot,
    DifferentialDriveSettings,
    DifferentialDriveState,
)

# ---------------------------------------------------------------------------
# Stopping-distance envelope math
# ---------------------------------------------------------------------------


def test_stopping_distance_formula() -> None:
    """The envelope is the kinematic ``v^2 / (2a)`` braking distance."""
    # v=2 m/s, a=1 m/s^2 -> d = 4 / 2 = 2 m
    assert stopping_distance(2.0, 1.0) == pytest.approx(2.0)
    # v=3 m/s, a=3 m/s^2 -> d = 9 / 6 = 1.5 m
    assert stopping_distance(3.0, 3.0) == pytest.approx(1.5)
    # At rest, no distance needed.
    assert stopping_distance(0.0, 1.0) == pytest.approx(0.0)


def test_stopping_distance_rejects_invalid_inputs() -> None:
    """Non-positive deceleration and negative speed fail closed instead of dividing badly."""
    with pytest.raises(ValueError, match="deceleration"):
        stopping_distance(2.0, 0.0)
    with pytest.raises(ValueError, match="deceleration"):
        stopping_distance(2.0, -1.0)
    with pytest.raises(ValueError, match="speed"):
        stopping_distance(-1.0, 1.0)


def test_stronger_braking_shrinks_stopping_distance() -> None:
    """Doubling braking authority roughly halves the worst-case stopping distance."""
    weak = stopping_distance(3.0, 1.0)
    strong = stopping_distance(3.0, 2.0)
    assert strong == pytest.approx(weak / 2.0)


# ---------------------------------------------------------------------------
# Backward compatibility: symmetric default (braking == acceleration)
# ---------------------------------------------------------------------------


def test_bicycle_default_braking_equals_acceleration() -> None:
    """Default bicycle braking authority equals forward acceleration (legacy)."""
    settings = BicycleDriveSettings(max_accel=2.0)
    assert settings.max_decel == 2.0


def test_diff_drive_default_braking_equals_acceleration() -> None:
    """Default differential braking authority equals forward acceleration (legacy)."""
    settings = DifferentialDriveSettings(max_linear_accel=0.4)
    assert settings.max_linear_decel == 0.4


def test_diff_drive_default_action_space_unchanged() -> None:
    """Without explicit braking, the action-space low bound stays symmetric."""
    robot = DifferentialDriveRobot(
        DifferentialDriveSettings(
            max_linear_speed=3.0,
            max_angular_speed=2.0,
            max_linear_accel=0.4,
            max_angular_accel=0.3,
        )
    )
    # Legacy symmetric bound: low == -high.
    assert tuple(robot.action_space.low) == pytest.approx((-0.4, -0.3))


# ---------------------------------------------------------------------------
# Commanded stops respect the deceleration cap (issue #4976 acceptance)
# ---------------------------------------------------------------------------


def test_bicycle_commanded_stop_respects_deceleration_cap() -> None:
    """A bicycle hard-stop command cannot decelerate harder than ``max_decel``."""
    # Asymmetric authority: forward accel 1.0, braking decel 0.5 (weaker brakes).
    motion = BicycleMotion(BicycleDriveSettings(max_velocity=5.0, max_accel=1.0, max_decel=0.5))
    state = BicycleDriveState(((0.0, 0.0), 0.0), 2.0)  # cruising at 2 m/s

    # Command an extreme brake (-1000 m/s^2). The deceleration cap must clamp it.
    motion.move(state, (-1000.0, 0.0), 1.0)

    # Only 0.5 m/s of speed can be bled in one second.
    assert state.velocity == pytest.approx(1.5)


def test_bicycle_braking_can_exceed_forward_acceleration() -> None:
    """Asymmetric braking authority lets the model stop harder than it accelerates."""
    # Strong brakes (5.0) but weak forward accel (1.0).
    motion = BicycleMotion(BicycleDriveSettings(max_velocity=5.0, max_accel=1.0, max_decel=5.0))

    # Forward command of 1000 m/s^2 is clamped to the weak forward cap (1.0).
    forward_state = BicycleDriveState(((0.0, 0.0), 0.0), 0.0)
    motion.move(forward_state, (1000.0, 0.0), 1.0)
    assert forward_state.velocity == pytest.approx(1.0)

    # Brake command of -1000 m/s^2 from 5 m/s is clamped to the strong brake cap (5.0),
    # stopping in a single second.
    brake_state = BicycleDriveState(((0.0, 0.0), 0.0), 5.0)
    motion.move(brake_state, (-1000.0, 0.0), 1.0)
    assert brake_state.velocity == pytest.approx(0.0)


def test_diff_drive_commanded_stop_respects_deceleration_cap() -> None:
    """A differential-drive hard-stop command cannot brake harder than ``max_linear_decel``."""
    motion = DifferentialDriveMotion(
        DifferentialDriveSettings(
            max_linear_speed=10.0,
            max_linear_accel=1.0,
            max_linear_decel=0.5,  # weaker brakes than forward thrust
        )
    )
    state = DifferentialDriveState(((0.0, 0.0), 0.0), (2.0, 0.0), (0.0, 0.0), (0.0, 0.0))

    # Command an extreme linear brake (-1000 m/s^2); it must clamp to the brake cap.
    motion.move(state, (-1000.0, 0.0), 1.0)

    # Only 0.5 m/s of speed can be bled in one second.
    assert state.velocity[0] == pytest.approx(1.5)


def test_diff_drive_braking_can_exceed_forward_acceleration() -> None:
    """Asymmetric braking authority lets the diff drive stop harder than it accelerates."""
    motion = DifferentialDriveMotion(
        DifferentialDriveSettings(
            max_linear_speed=10.0,
            max_linear_accel=1.0,
            max_linear_decel=5.0,
        )
    )
    # Forward command clamped to the weak forward cap (1.0).
    forward_state = DifferentialDriveState(((0.0, 0.0), 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    motion.move(forward_state, (1000.0, 0.0), 1.0)
    assert forward_state.velocity[0] == pytest.approx(1.0)

    # Brake command clamped to the strong brake cap (5.0): stops from 5 m/s in 1s.
    brake_state = DifferentialDriveState(((0.0, 0.0), 0.0), (5.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    motion.move(brake_state, (-1000.0, 0.0), 1.0)
    assert brake_state.velocity[0] == pytest.approx(0.0)


def test_diff_drive_action_space_reflects_asymmetric_braking() -> None:
    """The action-space low bound tracks braking authority when it is explicit."""
    robot = DifferentialDriveRobot(
        DifferentialDriveSettings(
            max_linear_speed=3.0,
            max_angular_speed=2.0,
            max_linear_accel=0.4,
            max_linear_decel=2.0,
            max_angular_accel=0.3,
        )
    )
    # Linear low reflects the 2.0 brake cap; angular low stays symmetric.
    assert tuple(robot.action_space.low) == pytest.approx((-2.0, -0.3))
    assert tuple(robot.action_space.high) == pytest.approx((0.4, 0.3))


def test_bicycle_action_space_reflects_asymmetric_braking() -> None:
    """The bicycle action-space low bound tracks braking authority when explicit."""
    robot = BicycleDriveRobot(
        BicycleDriveSettings(max_velocity=3.0, max_accel=1.0, max_decel=2.5, max_steer=0.5)
    )
    assert tuple(robot.action_space.low) == pytest.approx((-2.5, -0.5))
    assert tuple(robot.action_space.high) == pytest.approx((1.0, 0.5))


def test_diff_drive_rejects_non_positive_braking_authority() -> None:
    """An explicit non-positive braking authority fails closed at construction."""
    with pytest.raises(ValueError, match="max_linear_decel"):
        DifferentialDriveSettings(max_linear_decel=0.0)


# ---------------------------------------------------------------------------
# Stopping-distance envelope surfaces in run metadata (issue #4976 acceptance)
# ---------------------------------------------------------------------------


def test_envelope_omitted_for_velocity_level_drive_model() -> None:
    """A drive model with no first-class deceleration cap omits the envelope."""
    # Holonomic-style config exposes no braking-authority field.
    config = SimpleNamespace(robot_config=SimpleNamespace(max_speed=2.0))
    assert actuation_envelope_from_drive_config(config.robot_config) is None


def test_envelope_omitted_when_config_has_no_robot_config() -> None:
    """A config without a robot_config yields no envelope block."""
    block = metric_affecting_run_config(SimpleNamespace())
    assert "actuation_envelope" not in block


def test_diff_drive_envelope_surfaces_in_run_metadata() -> None:
    """The differential-drive envelope appears in the metric-affecting run config."""
    config = SimpleNamespace(
        robot_config=DifferentialDriveSettings(
            max_linear_speed=3.0, max_linear_accel=1.0, max_linear_decel=3.0
        )
    )
    block = metric_affecting_run_config(config)

    envelope = block["actuation_envelope"]
    assert envelope["schema"] == ACTUATION_ENVELOPE_SCHEMA
    assert envelope["drive_model"] == "differential_drive"
    assert envelope["max_forward_accel_m_s2"] == pytest.approx(1.0)
    assert envelope["max_braking_decel_m_s2"] == pytest.approx(3.0)
    assert envelope["braking_distinct_from_accel"] is True
    assert envelope["peak_forward_speed_m_s"] == pytest.approx(3.0)
    # v=3, a=3 -> d = 9 / 6 = 1.5 m
    assert envelope["stopping_distance_envelope_m"] == pytest.approx(1.5)
    assert envelope["interpretation"] == ACTUATION_ENVELOPE_INTERPRETATION


def test_bicycle_envelope_surfaces_in_run_metadata() -> None:
    """The bicycle-drive envelope appears in the metric-affecting run config."""
    config = SimpleNamespace(
        robot_config=BicycleDriveSettings(max_velocity=3.0, max_accel=1.0, max_decel=2.0)
    )
    block = metric_affecting_run_config(config)

    envelope = block["actuation_envelope"]
    assert envelope["drive_model"] == "bicycle_drive"
    assert envelope["max_forward_accel_m_s2"] == pytest.approx(1.0)
    assert envelope["max_braking_decel_m_s2"] == pytest.approx(2.0)
    # v=3, a=2 -> d = 9 / 4 = 2.25 m
    assert envelope["stopping_distance_envelope_m"] == pytest.approx(2.25)


def test_default_envelope_is_symmetric_and_not_distinct() -> None:
    """Default braking (== acceleration) is recorded as not distinct from accel."""
    config = SimpleNamespace(
        robot_config=DifferentialDriveSettings(max_linear_speed=2.0, max_linear_accel=0.5)
    )
    block = metric_affecting_run_config(config)
    envelope = block["actuation_envelope"]
    assert envelope["braking_distinct_from_accel"] is False
    assert envelope["max_braking_decel_m_s2"] == pytest.approx(0.5)
    # v=2, a=0.5 -> d = 4 / 1 = 4 m
    assert envelope["stopping_distance_envelope_m"] == pytest.approx(4.0)


def test_envelope_block_is_json_serializable() -> None:
    """The envelope block round-trips through JSON without custom encoders."""
    config = SimpleNamespace(
        robot_config=BicycleDriveSettings(max_velocity=3.0, max_accel=1.0, max_decel=2.0)
    )
    block = metric_affecting_run_config(config)
    # Must not raise; values must be plain JSON types.
    round_tripped = json.loads(json.dumps(block))
    assert round_tripped == block
    assert isinstance(round_tripped["actuation_envelope"]["stopping_distance_envelope_m"], float)


def test_weaker_braking_larger_envelope_changes_metric_meaning() -> None:
    """Weaker braking produces a larger stopping envelope -- the metric-affecting point."""
    strong_brakes = metric_affecting_run_config(
        SimpleNamespace(
            robot_config=DifferentialDriveSettings(
                max_linear_speed=3.0, max_linear_accel=1.0, max_linear_decel=3.0
            )
        )
    )
    weak_brakes = metric_affecting_run_config(
        SimpleNamespace(
            robot_config=DifferentialDriveSettings(
                max_linear_speed=3.0, max_linear_accel=1.0, max_linear_decel=1.0
            )
        )
    )
    strong_d = strong_brakes["actuation_envelope"]["stopping_distance_envelope_m"]
    weak_d = weak_brakes["actuation_envelope"]["stopping_distance_envelope_m"]
    # Weak brakes need more room to stop -> larger envelope.
    assert weak_d > strong_d
    assert math.isclose(weak_d / strong_d, 3.0)
