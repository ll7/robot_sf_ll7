"""Tests for shared math helpers."""

from __future__ import annotations

from math import atan2, cos, pi, sin

import pytest

from robot_sf.common.math_utils import (
    normalize_angle_atan2,
    wrap_angle_pi,
    wrap_angle_pi_closed,
)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (pi, -pi),
        (3.0 * pi, -pi),
        (-3.0 * pi, -pi),
        (4.0 * pi, 0.0),
    ],
)
def test_wrap_angle_pi_matches_modulo_semantics(angle: float, expected: float) -> None:
    """Modulo wrapper keeps the historical half-open interval behavior."""
    assert wrap_angle_pi(angle) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (pi, pi),
        (3.0 * pi, pi),
        (-pi, -pi),
        (-3.0 * pi, -pi),
        (4.0 * pi, 0.0),
    ],
)
def test_wrap_angle_pi_closed_preserves_positive_pi(angle: float, expected: float) -> None:
    """Closed wrapper keeps positive odd multiples of pi positive."""
    assert wrap_angle_pi_closed(angle) == pytest.approx(expected)


@pytest.mark.parametrize("angle", [-7.2, -pi, -0.1, 0.0, 0.1, pi, 7.2])
def test_normalize_angle_atan2_matches_drive_model_formula(angle: float) -> None:
    """Drive model helper preserves the previous atan2(sin, cos) formula."""
    expected = atan2(sin(angle), cos(angle))
    assert normalize_angle_atan2(angle) == pytest.approx(expected)
