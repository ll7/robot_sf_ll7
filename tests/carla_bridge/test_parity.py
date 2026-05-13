"""Tests for CARLA oracle replay parity comparison."""

import pytest

from robot_sf.carla_bridge.parity import compare_oracle_replay_metrics


def test_compare_oracle_replay_metrics_reports_numeric_deltas_and_bool_match():
    """Nominal fixture comparison should emit metric deltas without CARLA imports."""
    report = compare_oracle_replay_metrics(
        {"metrics": {"success": True, "min_distance_m": 1.0}},
        {"metrics": {"success": True, "min_distance_m": 0.8}},
        metric_names=("success", "min_distance_m"),
    )

    assert report["comparison_schema"] == "carla_oracle_replay_parity_v1"
    assert report["status"] == "comparable"
    assert report["metrics"][0]["status"] == "match"
    assert report["metrics"][1]["delta"] == pytest.approx(-0.2)


def test_compare_oracle_replay_metrics_marks_missing_fields_unavailable():
    """Missing CARLA/Robot-SF fields should be explicit, not synthetic."""
    report = compare_oracle_replay_metrics(
        {"metrics": {"success": True}},
        {"metrics": {}},
        metric_names=("success", "snqi"),
    )

    assert report["status"] == "unavailable"
    assert report["metrics"][0]["status"] == "unavailable"
    assert report["metrics"][0]["reason"] == "missing CARLA metric"
    assert report["metrics"][1]["reason"] == "missing Robot-SF metric"


def test_compare_oracle_replay_metrics_rejects_degraded_carla_mode():
    """Fallback/degraded CARLA replay modes must not count as parity evidence."""
    report = compare_oracle_replay_metrics(
        {"metrics": {"success": True}},
        {"mode": "fallback", "metrics": {"success": True}},
        metric_names=("success",),
    )

    assert report["status"] == "unavailable"
    assert "fallback" in report["reason"]
    assert report["metrics"][0]["status"] == "unavailable"
