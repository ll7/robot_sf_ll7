"""Freeze benchmark collision and near-miss metric semantics."""

from __future__ import annotations

from robot_sf.benchmark.thresholds import (
    default_threshold_profile,
    legacy_missing_threshold_profile,
)


def test_default_threshold_profile_freezes_clearance_as_safety_metric() -> None:
    """Benchmark-facing pedestrian safety metrics use surface clearance."""
    profile = default_threshold_profile()

    assert profile["pedestrian_safety_metric"] == "surface_clearance_m"
    assert profile["pedestrian_collision_definition"] == "min_clearance_m < 0"
    assert profile["near_miss_definition"] == "0 <= min_clearance_m < near_miss_distance_m"
    assert profile["center_distance_pedestrian_role"] == "geometric_diagnostic_only"
    assert profile["center_distance_collision_definition"] == (
        "center_distance_m < collision_distance_m"
    )


def test_legacy_missing_threshold_profile_names_center_distance_as_diagnostic() -> None:
    """Legacy center-distance bands remain explicit diagnostics, not safety metrics."""
    profile = legacy_missing_threshold_profile()

    assert profile["pedestrian_safety_metric"] == "center_distance_m"
    assert profile["center_distance_pedestrian_role"] == "legacy_geometric_diagnostic"
    assert profile["pedestrian_collision_definition"] == (
        "center_distance_m < collision_distance_m"
    )
