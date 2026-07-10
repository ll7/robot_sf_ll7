"""Contracts for opt-in cross-environment LiDAR obstacle batching."""

import numpy as np
import pytest

from robot_sf.sensor import range_sensor


def _synthetic_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out_ranges = np.array(
        [
            [10.0, 10.0, 10.0, 10.0],
            [8.0, 8.0, 8.0, 8.0],
            [6.0, 6.0, 6.0, 6.0],
        ],
        dtype=np.float64,
    )
    scanner_positions = np.array([[0.0, 0.0], [10.0, -2.0], [1.0, 1.0]])
    obstacles = np.zeros((3, 3, 4), dtype=np.float64)
    obstacles[0, 0] = [2.0, -1.0, 2.0, 1.0]
    obstacles[0, 1] = [-3.0, -1.0, -3.0, 1.0]
    obstacles[1, 0] = [9.0, 1.0, 11.0, 1.0]
    obstacles[1, 1] = [12.0, -3.0, 12.0, -1.0]
    obstacles[1, 2] = [100.0, 100.0, 101.0, 101.0]
    obstacle_counts = np.array([2, 3, 0], dtype=np.int64)
    ray_angles = np.tile(
        np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]),
        (3, 1),
    )
    return out_ranges, scanner_positions, obstacles, obstacle_counts, ray_angles


def test_multi_environment_batch_is_bit_identical_to_scalar_kernel() -> None:
    """Every batch row matches an independent call to the established kernel exactly."""
    actual, scanner_positions, obstacles, obstacle_counts, ray_angles = _synthetic_batch()
    expected = actual.copy()
    for env_idx in range(expected.shape[0]):
        range_sensor.raycast_obstacles(
            expected[env_idx],
            scanner_positions[env_idx],
            obstacles[env_idx, : obstacle_counts[env_idx]],
            ray_angles[env_idx],
        )

    range_sensor.raycast_obstacles_batch(
        actual,
        scanner_positions,
        obstacles,
        obstacle_counts,
        ray_angles,
    )

    assert np.array_equal(actual, expected)


def test_single_environment_batch_uses_bit_identical_scalar_fallback(monkeypatch) -> None:
    """A one-row request bypasses the batch kernel and retains scalar output bits."""
    out_ranges, scanner_positions, obstacles, obstacle_counts, ray_angles = _synthetic_batch()
    actual = out_ranges[:1].copy()
    expected = actual.copy()
    range_sensor.raycast_obstacles(
        expected[0],
        scanner_positions[0],
        obstacles[0, : obstacle_counts[0]],
        ray_angles[0],
    )

    def _unexpected_batch_dispatch(*_args) -> None:
        raise AssertionError("single-environment fallback dispatched the batch kernel")

    monkeypatch.setattr(range_sensor, "_raycast_obstacles_batch_kernel", _unexpected_batch_dispatch)
    range_sensor.raycast_obstacles_batch(
        actual,
        scanner_positions[:1],
        obstacles[:1],
        obstacle_counts[:1],
        ray_angles[:1],
    )

    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    ("field", "replacement", "error", "message"),
    [
        ("out_ranges", np.zeros((0, 4)), ValueError, "out_ranges"),
        ("out_ranges", np.zeros((3, 4), dtype=np.int64), TypeError, "out_ranges"),
        ("scanner_positions", np.zeros((2, 2)), ValueError, "scanner_positions"),
        ("obstacles", np.zeros((3, 3, 3)), ValueError, "obstacles"),
        ("obstacle_counts", np.zeros(3, dtype=np.float64), TypeError, "obstacle_counts"),
        ("obstacle_counts", np.array([2, 4, 0]), ValueError, "obstacle_counts"),
        ("obstacle_counts", np.array([2, 3, -1]), ValueError, "obstacle_counts"),
        ("ray_angles", np.zeros((3, 3)), ValueError, "ray_angles"),
    ],
)
def test_batch_contract_fails_closed_for_invalid_inputs(
    field: str,
    replacement: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    """Malformed padded batches fail before entering the compiled kernel."""
    out_ranges, scanner_positions, obstacles, obstacle_counts, ray_angles = _synthetic_batch()
    inputs = {
        "out_ranges": out_ranges,
        "scanner_positions": scanner_positions,
        "obstacles": obstacles,
        "obstacle_counts": obstacle_counts,
        "ray_angles": ray_angles,
    }
    inputs[field] = replacement

    with pytest.raises(error, match=message):
        range_sensor.raycast_obstacles_batch(**inputs)


def test_batch_contract_rejects_read_only_output() -> None:
    """The mutating adapter rejects an output array that cannot be updated."""
    out_ranges, scanner_positions, obstacles, obstacle_counts, ray_angles = _synthetic_batch()
    out_ranges.flags.writeable = False

    with pytest.raises(TypeError, match="writable"):
        range_sensor.raycast_obstacles_batch(
            out_ranges,
            scanner_positions,
            obstacles,
            obstacle_counts,
            ray_angles,
        )
