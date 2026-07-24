"""Tests for the Douglas-Peucker path smoothing algorithm."""

from __future__ import annotations

import math

from robot_sf.planner.path_smoother import douglas_peucker


def test_short_path_passthrough_empty() -> None:
    path: list[tuple[float, float]] = []
    result = douglas_peucker(path, epsilon=1.0)
    assert result == []


def test_short_path_passthrough_single_point() -> None:
    path = [(3.0, 4.0)]
    result = douglas_peucker(path, epsilon=1.0)
    assert result == [(3.0, 4.0)]


def test_short_path_passthrough_two_points() -> None:
    path = [(0.0, 0.0), (10.0, 10.0)]
    result = douglas_peucker(path, epsilon=1.0)
    assert result == [(0.0, 0.0), (10.0, 10.0)]


def test_non_positive_epsilon_zero() -> None:
    path = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
    result = douglas_peucker(path, epsilon=0.0)
    assert result == [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]


def test_non_positive_epsilon_negative() -> None:
    path = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
    result = douglas_peucker(path, epsilon=-1.0)
    assert result == [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]


def test_endpoint_preservation_first_and_last() -> None:
    path = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0), (4.0, 0.0)]
    result = douglas_peucker(path, epsilon=100.0)
    assert result[0] == (0.0, 0.0)
    assert result[-1] == (4.0, 0.0)


def test_collinear_point_reduction() -> None:
    path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
    result = douglas_peucker(path, epsilon=0.1)
    assert result == [(0.0, 0.0), (4.0, 0.0)]


def test_collinear_with_bulge_under_epsilon() -> None:
    path = [(0.0, 0.0), (2.0, 0.01), (4.0, 0.0)]
    result = douglas_peucker(path, epsilon=0.1)
    assert result == [(0.0, 0.0), (4.0, 0.0)]


def test_corner_point_retention() -> None:
    path = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
    result = douglas_peucker(path, epsilon=0.1)
    assert result == [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]


def test_multiple_corners_retained() -> None:
    path = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0), (15.0, 5.0)]
    result = douglas_peucker(path, epsilon=0.1)
    assert len(result) == 4
    assert result == [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0), (15.0, 5.0)]


def test_degenerate_zero_length_segment() -> None:
    path = [(0.0, 0.0), (0.0, 0.0), (5.0, 0.0)]
    result = douglas_peucker(path, epsilon=0.1)
    assert len(result) == 2
    assert result[0] == (0.0, 0.0)
    assert result[-1] == (5.0, 0.0)


def test_degenerate_multi_repeated_point() -> None:
    path = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (5.0, 0.0)]
    result = douglas_peucker(path, epsilon=0.1)
    assert len(result) >= 2


def test_point_ordering() -> None:
    path = [(0.0, 0.0), (2.0, 1.0), (4.0, 0.0), (6.0, 1.0), (8.0, 0.0)]
    result = douglas_peucker(path, epsilon=0.5)
    for i in range(len(result) - 1):
        x_i, _ = result[i]
        x_j, _ = result[i + 1]
        assert x_i <= x_j


def test_tuple_return_types() -> None:
    path = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
    result = douglas_peucker(path, epsilon=0.1)
    for pt in result:
        assert isinstance(pt, tuple)
        assert len(pt) == 2
        x, y = pt
        assert isinstance(x, float)
        assert isinstance(y, float)


def test_large_epsilon_returns_only_endpoints() -> None:
    path = [(0.0, 0.0), (1.0, 10.0), (2.0, -10.0), (3.0, 10.0), (4.0, 0.0)]
    result = douglas_peucker(path, epsilon=1e9)
    assert result == [(0.0, 0.0), (4.0, 0.0)]


def test_tiny_epsilon_returns_all_points() -> None:
    path = [(0.0, 0.0), (0.5, 0.01), (1.0, 0.0)]
    result = douglas_peucker(path, epsilon=1e-9)
    assert len(result) == 3


def test_float_input_types() -> None:
    result = douglas_peucker([(0, 0), (5, 5), (10, 0)], epsilon=0.1)
    for pt in result:
        x, y = pt
        assert isinstance(x, float)
        assert isinstance(y, float)


def test_collinear_diagonal_reduction() -> None:
    path = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = douglas_peucker(path, epsilon=0.1)
    assert result == [(0.0, 0.0), (3.0, 3.0)]


def test_perpendicular_offset_barely_under_epsilon() -> None:
    d = math.sqrt(0.005)  # sqrt(0.005) ≈ 0.0707 < 0.1
    path = [(0.0, 0.0), (1.0, d), (2.0, 0.0)]
    result = douglas_peucker(path, epsilon=0.1)
    assert len(result) == 2


def test_perpendicular_offset_barely_over_epsilon() -> None:
    d = math.sqrt(0.02)  # sqrt(0.02) ≈ 0.1414 > 0.1
    path = [(0.0, 0.0), (1.0, d), (2.0, 0.0)]
    result = douglas_peucker(path, epsilon=0.1)
    assert len(result) == 3
