"""Unit tests for free-space point sampling within bounds."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from robot_sf.nav.free_space_sampling import sample_free_points_in_bounds


def test_reproducibility() -> None:
    """Identically seeded Generators produce identical outputs, distinct seeds differ."""
    bounds = (0.0, 10.0, 0.0, 10.0)
    num_samples = 10

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    rng3 = np.random.default_rng(999)

    pts1 = sample_free_points_in_bounds(bounds, num_samples, rng=rng1)
    pts2 = sample_free_points_in_bounds(bounds, num_samples, rng=rng2)
    pts3 = sample_free_points_in_bounds(bounds, num_samples, rng=rng3)

    assert isinstance(pts1, list)
    assert len(pts1) == num_samples
    assert all(isinstance(pt, tuple) and len(pt) == 2 for pt in pts1)

    assert pts1 == pts2
    assert pts1 != pts3


def test_backward_compatibility_omitted_and_none_rng() -> None:
    """Omitting rng or explicitly passing None defaults to random sampling."""
    bounds = (0.0, 10.0, 0.0, 10.0)
    num_samples = 5

    pts_omitted = sample_free_points_in_bounds(bounds, num_samples)
    pts_none = sample_free_points_in_bounds(bounds, num_samples, rng=None)

    assert isinstance(pts_omitted, list)
    assert len(pts_omitted) == num_samples
    assert isinstance(pts_none, list)
    assert len(pts_none) == num_samples


def test_obstacle_rejection() -> None:
    """Sampled points avoid obstacle polygon interiors."""
    bounds = (0.0, 10.0, 0.0, 10.0)
    obstacle = Polygon([(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)])
    num_samples = 20
    rng = np.random.default_rng(123)

    pts = sample_free_points_in_bounds(
        bounds,
        num_samples,
        obstacle_polygons=[obstacle],
        rng=rng,
    )

    assert len(pts) == num_samples
    for x, y in pts:
        assert 0.0 <= x <= 10.0
        assert 0.0 <= y <= 10.0
        # Verify point is outside the obstacle interior
        assert not obstacle.contains(Point(x, y))


def test_runtime_error_attempt_exhaustion() -> None:
    """RuntimeError is raised when max attempt budget is exhausted without finding enough points."""
    bounds = (0.0, 10.0, 0.0, 10.0)
    # Obstacle covering the entire bounding box
    full_obstacle = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    rng = np.random.default_rng(42)

    with pytest.raises(RuntimeError, match="Failed to sample"):
        sample_free_points_in_bounds(
            bounds,
            num_samples=5,
            obstacle_polygons=[full_obstacle],
            max_attempts_per_point=10,
            rng=rng,
        )
