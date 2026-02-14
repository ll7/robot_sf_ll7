"""Tests for threshold-profile handling in benchmark aggregation reports."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.aggregate import compute_aggregates
from robot_sf.benchmark.errors import AggregationMetadataError
from robot_sf.benchmark.thresholds import build_metric_parameters


def _record(ep_id: str, algo: str, metric_params: dict) -> dict:
    """Build a minimal aggregate-ready record with explicit threshold metadata."""
    return {
        "episode_id": ep_id,
        "scenario_id": f"scenario-{ep_id}",
        "scenario_params": {"algo": algo},
        "metrics": {"success_rate": 1.0},
        "metric_parameters": metric_params,
    }


def test_compute_aggregates_includes_threshold_profile_meta() -> None:
    """Aggregation metadata should include threshold profile and signature."""
    recs = [
        _record("1", "sf", build_metric_parameters()),
        _record("2", "sf", build_metric_parameters()),
    ]
    summary = compute_aggregates(recs, group_by="scenario_params.algo")
    meta = summary["_meta"]["metric_parameters"]
    assert meta["threshold_profile"]["profile_id"] == "social_nav_thresholds_v1"
    assert isinstance(meta["threshold_signature"], str)
    assert meta["explicit_profile_records"] == 2


def test_compute_aggregates_raises_for_inconsistent_threshold_profiles() -> None:
    """Aggregation should fail when records contain mixed threshold profiles."""
    recs = [
        _record("1", "sf", build_metric_parameters()),
        _record("2", "sf", build_metric_parameters(profile={"near_miss_distance_m": 0.6})),
    ]
    with pytest.raises(AggregationMetadataError):
        compute_aggregates(recs, group_by="scenario_params.algo")
