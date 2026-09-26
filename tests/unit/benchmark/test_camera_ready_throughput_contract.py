"""Unit checks for camera-ready campaign throughput provenance metadata."""

from __future__ import annotations

from robot_sf.benchmark.camera_ready.campaign import _run_meta_throughput_definition


def test_run_meta_throughput_definition_describes_serialized_episode_rows() -> None:
    """Keep throughput scope, fields, and units directly covered by core CI."""
    assert _run_meta_throughput_definition() == {
        "scope": "campaign_all_planner_arms",
        "numerator_field": "total_episodes",
        "numerator_unit": "episode_rows",
        "numerator_semantics": "serialized_episode_rows",
        "denominator_field": "runtime_sec",
        "denominator_unit": "seconds",
        "denominator_semantics": "campaign_elapsed_through_outcome_snapshot",
        "rate_field": "episodes_per_second",
        "rate_unit": "episode_rows/second",
    }
