"""Tests for robot_sf.benchmark.constants — central benchmark constants."""

from __future__ import annotations

from robot_sf.benchmark.constants import (
    COLLISION_DIST,
    COMFORT_FORCE_THRESHOLD,
    EPISODE_SCHEMA_VERSION,
    NEAR_MISS_DIST,
)


def test_collision_dist_is_positive_float() -> None:
    """COLLISION_DIST must be a positive float used as center-distance threshold."""
    assert isinstance(COLLISION_DIST, float)
    assert COLLISION_DIST > 0.0


def test_near_miss_dist_exceeds_collision_dist() -> None:
    """NEAR_MISS_DIST must be strictly greater than COLLISION_DIST."""
    assert isinstance(NEAR_MISS_DIST, float)
    assert NEAR_MISS_DIST > COLLISION_DIST


def test_comfort_force_threshold_is_positive() -> None:
    """COMFORT_FORCE_THRESHOLD must be a positive float."""
    assert isinstance(COMFORT_FORCE_THRESHOLD, float)
    assert COMFORT_FORCE_THRESHOLD > 0.0


def test_episode_schema_version_is_frozen_v1() -> None:
    """EPISODE_SCHEMA_VERSION must be the frozen 'v1' string."""
    assert EPISODE_SCHEMA_VERSION == "v1"
    assert isinstance(EPISODE_SCHEMA_VERSION, str)


def test_constants_are_finite() -> None:
    """All numeric constants must be finite (not NaN or inf)."""
    import math

    for value in (COLLISION_DIST, NEAR_MISS_DIST, COMFORT_FORCE_THRESHOLD):
        assert math.isfinite(value)


def test_all_exports_present() -> None:
    """The module __all__ must list exactly the four public constants."""
    from robot_sf.benchmark import constants

    assert set(constants.__all__) == {
        "COLLISION_DIST",
        "COMFORT_FORCE_THRESHOLD",
        "EPISODE_SCHEMA_VERSION",
        "NEAR_MISS_DIST",
    }
