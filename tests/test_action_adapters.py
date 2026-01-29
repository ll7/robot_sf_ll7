"""Tests for action-space adapters."""

import numpy as np

from robot_sf.robot.action_adapters import DiffDriveAdapterConfig, holonomic_to_diff_drive_action


def test_holonomic_to_diff_drive_forward():
    """Forward holonomic motion yields forward diff-drive action."""
    pose = ((0.0, 0.0), 0.0)
    action = holonomic_to_diff_drive_action(
        np.array([1.0, 0.0]),
        pose,
        max_linear_speed=1.0,
        max_angular_speed=1.0,
    )
    assert action[0] > 0.0
    assert abs(action[1]) < 1e-3


def test_holonomic_to_diff_drive_turns_left():
    """Leftward holonomic motion yields positive angular command."""
    pose = ((0.0, 0.0), 0.0)
    action = holonomic_to_diff_drive_action(
        np.array([0.0, 1.0]),
        pose,
        max_linear_speed=1.0,
        max_angular_speed=1.0,
    )
    assert action[0] >= 0.0
    assert action[1] > 0.0


def test_holonomic_to_diff_drive_clamps_speed():
    """Holonomic speeds clamp to the configured linear max."""
    pose = ((0.0, 0.0), 0.0)
    action = holonomic_to_diff_drive_action(
        np.array([10.0, 0.0]),
        pose,
        max_linear_speed=0.5,
        max_angular_speed=1.0,
    )
    assert action[0] <= 0.5 + 1e-6


def test_holonomic_to_diff_drive_allows_backwards():
    """Allow backwards motion when heading is opposite to desired velocity."""
    pose = ((0.0, 0.0), 0.0)
    action_no_back = holonomic_to_diff_drive_action(
        np.array([-1.0, 0.0]),
        pose,
        max_linear_speed=1.0,
        max_angular_speed=1.0,
        config=DiffDriveAdapterConfig(allow_backwards=False),
    )
    assert action_no_back[0] >= 0.0

    action_backwards = holonomic_to_diff_drive_action(
        np.array([-1.0, 0.0]),
        pose,
        max_linear_speed=1.0,
        max_angular_speed=1.0,
        config=DiffDriveAdapterConfig(allow_backwards=True),
    )
    assert action_backwards[0] < 0.0
