"""Tests for shared math helpers."""

from __future__ import annotations

from math import atan2, cos, isnan, pi, sin

import pytest

from robot_sf.common.math_utils import (
    clip_scalar,
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


@pytest.mark.parametrize(
    ("value", "lower", "upper", "expected"),
    [
        (-2.0, -1.0, 1.0, -1.0),
        (0.25, -1.0, 1.0, 0.25),
        (2.0, -1.0, 1.0, 1.0),
    ],
)
def test_clip_scalar_clips_bounds(
    value: float, lower: float, upper: float, expected: float
) -> None:
    """Scalar clipping matches inclusive bound semantics without NumPy dispatch."""
    assert clip_scalar(value, lower, upper) == expected


def test_clip_scalar_preserves_nan() -> None:
    """Scalar clipping keeps np.clip scalar NaN behavior."""
    assert isnan(clip_scalar(float("nan"), -1.0, 1.0))
