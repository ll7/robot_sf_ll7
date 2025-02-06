import numpy as np
import pytest
from robot_sf.render.lidar_visual import render_lidar


def test_render_lidar_basic():
    # Basic test with simple values
    robot_pos = [0, 0]
    distances = np.array([1, 2, 3])
    directions = np.array([0, np.pi / 2, np.pi])

    # Compute expected endpoints manually:
    expected = np.array(
        [
            [[0, 0], [0 + np.cos(0) * 1, 0 + np.sin(0) * 1]],
            [[0, 0], [0 + np.cos(np.pi / 2) * 2, 0 + np.sin(np.pi / 2) * 2]],
            [[0, 0], [0 + np.cos(np.pi) * 3, 0 + np.sin(np.pi) * 3]],
        ]
    )

    result = render_lidar(robot_pos, distances, directions)
    assert result.shape == (3, 2, 2)
    np.testing.assert_allclose(result, expected)


def test_render_lidar_zero_distances():
    # When distances are all zeros, the endpoints should equal the robot_pos.
    robot_pos = [5, -3]
    distances = np.array([0, 0, 0])
    directions = np.array([0, 1, 2])

    expected = np.array([[[5, -3], [5, -3]], [[5, -3], [5, -3]], [[5, -3], [5, -3]]])

    result = render_lidar(robot_pos, distances, directions)
    np.testing.assert_allclose(result, expected)


def test_render_lidar_empty_input():
    # When lists are empty, expect an empty array.
    robot_pos = [1, 1]
    distances = np.array([])
    directions = np.array([])

    result = render_lidar(robot_pos, distances, directions)
    # Expected output is an empty array with shape (0,)
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_render_lidar_negative_values():
    # Test with negative distances and directions values.
    robot_pos = [0, 0]
    distances = np.array([-1, -2])
    directions = np.array([-np.pi / 4, -np.pi / 2])

    expected = np.array(
        [
            [[0, 0], [0 + np.cos(-np.pi / 4) * (-1), 0 + np.sin(-np.pi / 4) * (-1)]],
            [[0, 0], [0 + np.cos(-np.pi / 2) * (-2), 0 + np.sin(-np.pi / 2) * (-2)]],
        ]
    )

    result = render_lidar(robot_pos, distances, directions)
    np.testing.assert_allclose(result, expected)


def test_render_lidar_nonzero_robot_pos():
    # Test with a non-zero robot position.
    robot_pos = [10, 20]
    distances = np.array([3])
    directions = np.array([np.pi / 3])

    expected_end = [
        robot_pos[0] + np.cos(np.pi / 3) * 3,
        robot_pos[1] + np.sin(np.pi / 3) * 3,
    ]
    expected = np.array([[[10, 20], expected_end]])

    result = render_lidar(robot_pos, distances, directions)
    np.testing.assert_allclose(result, expected)


def test_render_lidar_output_dtype():
    # Ensure the returned value is a numpy array with correct dtype.
    robot_pos = [0, 0]
    distances = np.array([1.5, 2.5])
    directions = np.array([0.0, np.pi / 2])

    result = render_lidar(robot_pos, distances, directions)
    assert isinstance(result, np.ndarray)
    # You can test for the output dtype if necessary, assuming float dtype.
    assert result.dtype in [np.float32, np.float64]


# Additional tests using parametrization can be added if different combinations are needed.
@pytest.mark.parametrize(
    "robot_pos, distances, directions",
    [
        ([0, 0], np.array([1]), np.array([0])),
        ([5, 5], np.array([2, 3]), np.array([np.pi / 4, np.pi / 2])),
    ],
)
def test_render_lidar_parametrized(robot_pos, distances, directions):
    # Compute expected endpoints for each parameter set
    expected_list = []
    for d, theta in zip(distances, directions):
        expected_list.append(
            [
                [robot_pos[0], robot_pos[1]],
                [robot_pos[0] + np.cos(theta) * d, robot_pos[1] + np.sin(theta) * d],
            ]
        )
    expected = np.array(expected_list)

    result = render_lidar(robot_pos, distances, directions)
    np.testing.assert_allclose(result, expected)
