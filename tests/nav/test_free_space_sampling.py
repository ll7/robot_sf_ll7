"""Tests for seedable RNG in free-space point sampling."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from robot_sf.nav.free_space_sampling import sample_free_points_in_bounds

BOUNDS = (0.0, 10.0, 0.0, 10.0)


def test_reproducibility_with_seeded_rng() -> None:
    rng = np.random.default_rng(42)
    result_a = sample_free_points_in_bounds(BOUNDS, 5, rng=rng)
    rng = np.random.default_rng(42)
    result_b = sample_free_points_in_bounds(BOUNDS, 5, rng=rng)
    assert result_a == result_b


def test_different_seeds_produce_different_outputs() -> None:
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(99)
    result_a = sample_free_points_in_bounds(BOUNDS, 5, rng=rng_a)
    result_b = sample_free_points_in_bounds(BOUNDS, 5, rng=rng_b)
    assert result_a != result_b


def test_returns_correct_number_of_points() -> None:
    rng = np.random.default_rng(0)
    result = sample_free_points_in_bounds(BOUNDS, 10, rng=rng)
    assert len(result) == 10


def test_points_within_bounds() -> None:
    rng = np.random.default_rng(1)
    x_min, x_max, y_min, y_max = BOUNDS
    result = sample_free_points_in_bounds(BOUNDS, 20, rng=rng)
    for x, y in result:
        assert x_min <= x <= x_max
        assert y_min <= y <= y_max


def test_points_reject_obstacles() -> None:
    obstacle = Polygon([(2.0, 2.0), (2.0, 8.0), (8.0, 8.0), (8.0, 2.0)])
    rng = np.random.default_rng(123)
    result = sample_free_points_in_bounds(BOUNDS, 10, obstacle_polygons=[obstacle], rng=rng)
    for x, y in result:
        assert not obstacle.contains(Point(x, y))


def test_runtime_error_on_budget_exhaustion() -> None:
    full_obstacle = Polygon([(-1.0, -1.0), (-1.0, 11.0), (11.0, 11.0), (11.0, -1.0)])
    rng = np.random.default_rng(999)
    with pytest.raises(RuntimeError, match="Failed to sample"):
        sample_free_points_in_bounds(
            BOUNDS, 5, obstacle_polygons=[full_obstacle], max_attempts_per_point=2, rng=rng
        )


def test_no_obstacles_returns_points() -> None:
    rng = np.random.default_rng(42)
    result = sample_free_points_in_bounds(BOUNDS, 3, rng=rng)
    assert len(result) == 3


def test_backward_compatible_without_rng() -> None:
    result = sample_free_points_in_bounds(BOUNDS, 3)
    assert len(result) == 3
    for x, y in result:
        assert 0.0 <= x <= 10.0
        assert 0.0 <= y <= 10.0


def test_omitted_rng_preserves_global_numpy_seed_reproducibility() -> None:
    np.random.seed(42)
    result_a = sample_free_points_in_bounds(BOUNDS, 5)
    np.random.seed(42)
    result_b = sample_free_points_in_bounds(BOUNDS, 5)
    assert result_a == result_b


def test_backward_compatible_with_explicit_none_rng() -> None:
    result = sample_free_points_in_bounds(BOUNDS, 3, rng=None)
    assert len(result) == 3


def test_zero_samples_returns_empty_list() -> None:
    rng = np.random.default_rng(0)
    result = sample_free_points_in_bounds(BOUNDS, 0, rng=rng)
    assert result == []
