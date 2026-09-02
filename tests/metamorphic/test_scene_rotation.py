"""Rotation metamorphism for the crowd-only environment."""

from __future__ import annotations

import numpy as np

from tests.metamorphic.support import BASE_MAP, assert_trace_equal, rotated_map, run_episode


def _inverse_rotate_positions(values: np.ndarray) -> np.ndarray:
    """Map rotated points back to the base frame around the square center."""
    center = np.asarray((10.0, 10.0), dtype=values.dtype)
    centered = values - center
    return np.stack((centered[..., 1], -centered[..., 0]), axis=-1) + center


def _inverse_rotate_vectors(values: np.ndarray) -> np.ndarray:
    """Map rotated vectors back to the base frame."""
    return np.stack((values[..., 1], -values[..., 0]), axis=-1)


def test_scene_rotation_preserves_dynamics_in_rotated_frame() -> None:
    """A 90-degree scene rotation rotates all position-like and vector outputs."""
    base = run_episode(BASE_MAP)
    rotated = run_episode(rotated_map())

    assert_trace_equal(
        base,
        rotated,
        transforms={
            "positions": _inverse_rotate_positions,
            "goals": _inverse_rotate_positions,
            "velocities": _inverse_rotate_vectors,
            "forces": _inverse_rotate_vectors,
        },
    )
