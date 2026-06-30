"""Tests for policy-search safety metric semantics."""

from __future__ import annotations

from scripts.validation.policy_search_common import (
    classify_failure_mode,
    summarize_policy_search_records,
)


def _row(metrics: dict[str, float]) -> dict:
    """Build a minimal policy-search episode row."""
    return {
        "scenario_id": "classic_crossing",
        "termination_reason": "max_steps",
        "metrics": metrics,
    }


def test_policy_search_near_miss_rate_uses_clearance_safety_metric() -> None:
    """Center-distance fallback is reported separately as a diagnostic."""
    records = [
        _row({"min_clearance": 0.10, "min_distance": 1.50}),
        _row({"min_clearance": -0.10, "min_distance": 0.30}),
    ]

    summary = summarize_policy_search_records(records)

    assert summary["near_miss_semantics"] == "surface_clearance_safety_metric"
    assert summary["near_miss_rate"] == 0.5
    assert summary["center_distance_near_miss_diagnostic_rate"] == 0.5


def test_policy_search_failure_mode_names_center_distance_diagnostic() -> None:
    """Legacy center-distance bands do not reuse the safety near-miss label."""
    assert (
        classify_failure_mode(_row({"min_clearance": -0.10, "min_distance": 0.30}))
        == "center_distance_near_miss_diagnostic"
    )
    assert classify_failure_mode(_row({"min_clearance": 0.10, "min_distance": 1.50})) == (
        "near_miss_intrusive"
    )
