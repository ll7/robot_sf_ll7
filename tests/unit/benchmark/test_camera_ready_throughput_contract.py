"""Unit checks for camera-ready campaign throughput provenance metadata."""

from __future__ import annotations

from robot_sf.benchmark.camera_ready._reporting import _write_arm_rollup
from robot_sf.benchmark.camera_ready._run_state import _arm_episode_counts
from robot_sf.benchmark.camera_ready.campaign import (
    _campaign_episode_counts,
    _run_meta_throughput_definition,
)


def test_run_meta_throughput_definition_describes_invocation_rows_and_retained_rows() -> None:
    """Keep throughput scope, fields, and units directly covered by core CI."""
    assert _run_meta_throughput_definition() == {
        "scope": "campaign_all_planner_arms",
        "numerator_field": "episodes_written_this_invocation",
        "numerator_unit": "episode_rows",
        "numerator_semantics": "episode_rows_newly_written_during_this_campaign_invocation",
        "retained_count_field": "total_episodes",
        "retained_count_semantics": "complete_serialized_episode_rows_retained_across_resume",
        "denominator_field": "runtime_sec",
        "denominator_unit": "seconds",
        "denominator_semantics": "campaign_elapsed_through_outcome_snapshot",
        "rate_field": "episodes_per_second",
        "rate_unit": "episode_rows/second",
    }


def test_campaign_episode_counts_preserve_fresh_full_campaign_shape() -> None:
    """Fresh full-matrix numerator remains 20,160 while resume separates counts."""
    fresh_runs = [
        {
            "summary": {
                "episodes_total": 1_440,
                "episodes_written_this_invocation": 1_440,
            }
        }
        for _ in range(14)
    ]
    assert _campaign_episode_counts(fresh_runs) == (20_160, 20_160)

    resumed_cached_run = {
        "summary": {
            "written": 1_440,
            "episodes_total": 1_440,
            "episodes_written_this_invocation": 0,
        }
    }
    assert _campaign_episode_counts([resumed_cached_run]) == (1_440, 0)


def test_arm_episode_counts_preserve_legacy_and_invocation_shapes() -> None:
    """Legacy and invocation-specific summaries keep distinct count semantics."""
    assert _arm_episode_counts({"written": 7}) == (7, 7)
    assert _arm_episode_counts({"episodes_total": 9}) == (9, 9)
    assert _arm_episode_counts({"episodes_written_this_invocation": 2}) == (2, 2)
    assert _arm_episode_counts({}) == (0, 0)


def test_arm_rollup_report_renders_error_columns_with_retained_rows() -> None:
    """An errored arm uses the expanded table header and keeps both row counts."""
    lines: list[str] = []
    _write_arm_rollup(
        lines,
        [
            {
                "planner_key": "goal",
                "kinematics": "diff",
                "status": "partial-failure",
                "episodes_written": 2,
                "episodes_total": 5,
                "episodes_failed": 1,
                "first_error": "worker failed",
                "distinct_error_count": 1,
            }
        ],
    )

    assert "written this invocation | retained rows" in lines[3]
    assert lines[4] == "|---|---|---|---:|---:|---:|---|---:|"
    assert lines[5] == "| goal | diff | partial-failure | 2 | 5 | 1 | worker failed | 1 |"
