"""
Tests for shared geometry utilities to keep distance calculations consistent across modules.
"""

import pytest

from robot_sf.common.geometry import euclid_dist


def test_euclid_dist_zero_length() -> None:
    """euclid_dist returns zero for identical points so overlap checks do not add phantom gaps."""
    point = (1.2, -3.4)
    assert euclid_dist(point, point) == 0.0


@pytest.mark.parametrize(
    ("vec_1", "vec_2", "expected"),
    [
        ((0.0, 0.0), (3.0, 4.0), 5.0),
        ((-1.5, 2.0), (2.5, -2.0), 5.656854249),
        ((1e6, -1e6), (-1e6, 1e6), 2_828_427.12474619),
    ],
)
def test_euclid_dist_expected_values(
    vec_1: tuple[float, float],
    vec_2: tuple[float, float],
    expected: float,
) -> None:
    """euclid_dist matches the Pythagorean expectation, keeping physics and sensing math aligned."""
    assert euclid_dist(vec_1, vec_2) == pytest.approx(expected)


def test_euclid_dist_is_symmetric() -> None:
    """Distance is symmetric to avoid order-dependent collision or force computations."""
    vec_a = (1.0, -2.0)
    vec_b = (-3.0, 4.5)
    assert euclid_dist(vec_a, vec_b) == pytest.approx(euclid_dist(vec_b, vec_a))
