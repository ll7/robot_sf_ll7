"""Tests for dynamic pedestrian occlusion semantics."""

import numpy as np
import pytest

from robot_sf.sensor.socnav_observation import dynamic_pedestrian_occlusion_mask


def test_dynamic_occlusion_hides_far_pedestrian_behind_near_body():
    """A nearer pedestrian disk should hide a farther center-line target."""
    ped_positions = np.asarray([[2.0, 0.0], [4.0, 0.0]], dtype=np.float32)

    visible = dynamic_pedestrian_occlusion_mask(
        ped_positions,
        robot_pos=np.asarray([0.0, 0.0], dtype=np.float32),
        pedestrian_radius=0.4,
    )

    assert visible.tolist() == [True, False]


def test_dynamic_occlusion_keeps_adjacent_pedestrian_visible():
    """A pedestrian outside the nearer body disk should remain visible."""
    ped_positions = np.asarray([[2.0, 0.0], [4.0, 1.0]], dtype=np.float32)

    visible = dynamic_pedestrian_occlusion_mask(
        ped_positions,
        robot_pos=np.asarray([0.0, 0.0], dtype=np.float32),
        pedestrian_radius=0.4,
    )

    assert visible.tolist() == [True, True]


def test_dynamic_occlusion_respects_existing_visibility_mask():
    """Already-hidden pedestrians should not become visible or act as blockers."""
    ped_positions = np.asarray([[2.0, 0.0], [4.0, 0.0]], dtype=np.float32)

    visible = dynamic_pedestrian_occlusion_mask(
        ped_positions,
        robot_pos=np.asarray([0.0, 0.0], dtype=np.float32),
        pedestrian_radius=0.4,
        base_visible=np.asarray([False, True]),
    )

    assert visible.tolist() == [False, True]


def test_dynamic_occlusion_rejects_negative_radius():
    """Invalid body radius should fail closed."""
    with pytest.raises(ValueError, match="pedestrian_radius"):
        dynamic_pedestrian_occlusion_mask(
            np.asarray([[2.0, 0.0]], dtype=np.float32),
            robot_pos=np.asarray([0.0, 0.0], dtype=np.float32),
            pedestrian_radius=-0.1,
        )


def test_dynamic_occlusion_rejects_malformed_base_visibility_mask():
    """Malformed base visibility masks should fail with a clear contract error."""
    with pytest.raises(ValueError, match="base_visible"):
        dynamic_pedestrian_occlusion_mask(
            np.asarray([[2.0, 0.0], [4.0, 0.0]], dtype=np.float32),
            robot_pos=np.asarray([0.0, 0.0], dtype=np.float32),
            pedestrian_radius=0.4,
            base_visible=np.asarray([[True, False]]),
        )
