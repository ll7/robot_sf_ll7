"""Contract tests for observation-frame normalization."""

from math import pi

import numpy as np
import pytest

from robot_sf.sensor.pedestrian_tracking import (
    RobotPoseGlobal,
    transform_covariance_from_global_xy,
    transform_covariance_to_global_xy,
    transform_heading_from_global_xy,
    transform_heading_to_global_xy,
    transform_history_from_global_xy,
    transform_history_to_global_xy,
    transform_position_from_global_xy,
    transform_position_to_global_xy,
    transform_velocity_from_global_xy,
    transform_velocity_to_global_xy,
)


def test_point_velocity_covariance_and_heading_round_trip() -> None:
    """Global normalization and its inverse preserve finite planar geometry."""
    pose = RobotPoseGlobal(position_global_xy=np.array([10.0, -2.0]), heading_rad=pi / 2.0)
    local_points = np.array([[1.0, 2.0], [-2.0, 0.5]])
    local_velocities = np.array([[1.0, 0.0], [0.5, -1.0]])
    local_covariance = np.array([[[2.0, 0.2], [0.2, 1.0]], [[1.0, 0.0], [0.0, 3.0]]])

    global_points = transform_position_to_global_xy(local_points, "robot_ego_xy", pose)
    global_velocities = transform_velocity_to_global_xy(local_velocities, "robot_ego_xy", pose)
    global_covariance = transform_covariance_to_global_xy(local_covariance, "robot_ego_xy", pose)

    np.testing.assert_allclose(
        transform_position_from_global_xy(global_points, "robot_ego_xy", pose), local_points
    )
    np.testing.assert_allclose(
        transform_velocity_from_global_xy(global_velocities, "robot_ego_xy", pose),
        local_velocities,
    )
    np.testing.assert_allclose(
        transform_covariance_from_global_xy(global_covariance, "robot_ego_xy", pose),
        local_covariance,
        atol=1e-10,
    )
    assert transform_heading_to_global_xy(0.0, "robot_ego_xy", pose) == pytest.approx(pi / 2.0)
    assert transform_heading_from_global_xy(pi / 2.0, "robot_ego_xy", pose) == pytest.approx(0.0)
    assert np.all(np.linalg.eigvalsh(global_covariance) >= -1e-9)


def test_history_uses_same_step_pose_and_preserves_oldest_to_newest_order() -> None:
    """Historical rows use their own robot transform rather than the latest heading."""
    poses = (
        RobotPoseGlobal(np.array([0.0, 0.0]), 0.0),
        RobotPoseGlobal(np.array([10.0, 0.0]), pi / 2.0),
    )
    local_history = np.array([[[1.0, 0.0]], [[1.0, 0.0]]])
    global_history = transform_history_to_global_xy(
        local_history, "robot_ego_xy", poses, value_kind="position"
    )

    np.testing.assert_allclose(global_history[:, 0], [[1.0, 0.0], [10.0, 1.0]])
    np.testing.assert_allclose(
        transform_history_from_global_xy(
            global_history, "robot_ego_xy", poses, value_kind="position"
        ),
        local_history,
        atol=1e-10,
    )


def test_unknown_or_nonfinite_frame_transform_fails_closed() -> None:
    """Unknown frames and non-finite transforms cannot enter the actor contract."""
    pose = RobotPoseGlobal(np.zeros(2), 0.0)
    with pytest.raises(ValueError, match="coordinate_frame"):
        transform_position_to_global_xy(np.zeros((1, 2)), "camera", pose)
    with pytest.raises(ValueError, match="positions"):
        transform_position_to_global_xy(np.array([[np.nan, 0.0]]), "global_xy", pose)
    with pytest.raises(ValueError, match="heading_rad"):
        RobotPoseGlobal(np.zeros(2), np.inf)


def test_global_frame_translation_is_applied_only_to_positions() -> None:
    """Global-frame vectors do not acquire the robot translation."""
    pose = RobotPoseGlobal(np.array([20.0, 30.0]), 0.7)
    point = transform_position_to_global_xy(np.array([[1.0, 2.0]]), "global_xy", pose)
    velocity = transform_velocity_to_global_xy(np.array([[1.0, 2.0]]), "global_xy", pose)
    np.testing.assert_allclose(point, [[1.0, 2.0]])
    np.testing.assert_allclose(velocity, [[1.0, 2.0]])


@pytest.mark.parametrize(
    "transform",
    [transform_history_to_global_xy, transform_history_from_global_xy],
)
def test_empty_history_still_validates_shape_and_frame(transform) -> None:
    """Empty history batches retain the same fail-closed shape and frame checks as non-empty ones."""
    with pytest.raises(ValueError, match="final shape dimension 2"):
        transform(np.empty((0, 3)), "global_xy", ())
    with pytest.raises(ValueError, match="coordinate_frame"):
        transform(np.empty((0, 2)), "camera", ())

    result = transform(np.empty((0, 2)), "global_xy", ())
    assert result.shape == (0, 2)


def test_empty_covariance_history_rejects_malformed_shape() -> None:
    """An empty covariance history cannot bypass its final matrix-shape contract."""
    with pytest.raises(ValueError, match=r"final shape \(2, 2\)"):
        transform_history_to_global_xy(
            np.empty((0, 2, 3)),
            "global_xy",
            (),
            value_kind="covariance",
        )


def test_covariance_roundoff_is_projected_to_positive_semidefinite() -> None:
    """Tiny numerical negative eigenvalues are clamped without accepting an indefinite input."""
    covariance = np.array([[[1.0, 1.0], [1.0, 1.0 - 5e-8]]])
    normalized = transform_covariance_to_global_xy(
        covariance, "global_xy", RobotPoseGlobal(np.zeros(2), 0.0)
    )

    assert np.min(np.linalg.eigvalsh(normalized)) >= -1e-12
