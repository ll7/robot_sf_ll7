"""Tests for CARLA oracle replay parity comparison."""

import math

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


def test_compare_oracle_replay_metrics_rejects_degraded_status_even_with_native_mode():
    """A degraded CARLA status should fail closed even when mode still says native."""
    report = compare_oracle_replay_metrics(
        {"metrics": {"success": True}},
        {"mode": "native", "status": "failed", "metrics": {"success": True}},
        metric_names=("success",),
    )

    assert report["status"] == "unavailable"
    assert "failed" in report["reason"]
    assert report["metrics"][0]["reason"] == report["reason"]


def test_compare_oracle_replay_metrics_marks_non_finite_numbers_unavailable():
    """NaN and infinity should stay unavailable instead of becoming JSON deltas."""
    report = compare_oracle_replay_metrics(
        {"metrics": {"snqi": math.nan}},
        {"metrics": {"snqi": math.inf}},
        metric_names=("snqi",),
    )

    assert report["status"] == "unavailable"
    assert report["metrics"][0]["status"] == "unavailable"
    assert report["metrics"][0]["reason"] == "metric values are not numeric or boolean"
