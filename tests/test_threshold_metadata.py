"""Tests for benchmark metric-threshold profile metadata and consistency checks."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.errors import AggregationMetadataError
from robot_sf.benchmark.thresholds import (
    build_metric_parameters,
    ensure_metric_parameters,
    validate_threshold_parameter_consistency,
)


def test_ensure_metric_parameters_adds_profile_when_missing() -> None:
    """Episode records should get canonical metric-parameter metadata when absent."""
    record = {"episode_id": "ep-1", "metrics": {"success": True}}
    ensure_metric_parameters(record)
    params = record.get("metric_parameters")
    assert isinstance(params, dict)
    assert "threshold_profile" in params
    assert "threshold_signature" in params


def test_consistency_validation_accepts_missing_profiles_with_default_inference() -> None:
    """Aggregation consistency check should infer defaults for legacy records."""
    records = [
        {"episode_id": "ep-1", "metrics": {"success": True}},
        {"episode_id": "ep-2", "metrics": {"success": False}},
    ]
    meta = validate_threshold_parameter_consistency(records)
    assert meta["missing_profile_records"] == 2
    assert meta["explicit_profile_records"] == 0
    assert isinstance(meta["threshold_signature"], str)


def test_consistency_validation_rejects_mixed_threshold_profiles() -> None:
    """Records with different threshold profiles should fail consistency validation."""
    base_params = build_metric_parameters()
    changed_params = build_metric_parameters(
        profile={
            "collision_distance_m": 0.3,
        },
    )
    records = [
        {"episode_id": "ep-1", "metric_parameters": base_params, "metrics": {"success": True}},
        {"episode_id": "ep-2", "metric_parameters": changed_params, "metrics": {"success": True}},
    ]
    with pytest.raises(AggregationMetadataError):
        validate_threshold_parameter_consistency(records)
