"""Unit tests for metric aggregation module."""

import numpy as np
import pytest

from robot_sf.research.aggregation import (
    aggregate_metrics,
    bootstrap_ci,
    compute_completeness_score,
)


def test_aggregate_metrics_basic():
    """Test basic metric aggregation with minimal data."""
    metric_records = [
        {
            "seed": 42,
            "policy_type": "baseline",
            "success_rate": 0.7,
            "timesteps_to_convergence": 500000,
        },
        {
            "seed": 123,
            "policy_type": "baseline",
            "success_rate": 0.75,
            "timesteps_to_convergence": 480000,
        },
        {
            "seed": 42,
            "policy_type": "pretrained",
            "success_rate": 0.85,
            "timesteps_to_convergence": 280000,
        },
        {
            "seed": 123,
            "policy_type": "pretrained",
            "success_rate": 0.88,
            "timesteps_to_convergence": 270000,
        },
    ]

    result = aggregate_metrics(metric_records, group_by="policy_type", ci_samples=100, seed=42)

    # Check that we have results for both conditions
    conditions = {r["condition"] for r in result}
    assert "baseline" in conditions
    assert "pretrained" in conditions

    # Check metrics are present
    metrics = {r["metric_name"] for r in result}
    assert "success_rate" in metrics
    assert "timesteps_to_convergence" in metrics

    # Check baseline success_rate aggregation
    baseline_success = next(
        r for r in result if r["condition"] == "baseline" and r["metric_name"] == "success_rate"
    )
    assert baseline_success["mean"] == pytest.approx(0.725, abs=1e-6)
    assert baseline_success["sample_size"] == 2
    assert baseline_success["ci_low"] is not None
    assert baseline_success["ci_high"] is not None


def test_aggregate_metrics_single_value():
    """Test aggregation with single value per condition (no CI)."""
    metric_records = [
        {"seed": 42, "policy_type": "baseline", "success_rate": 0.7},
    ]

    result = aggregate_metrics(metric_records, group_by="policy_type", ci_samples=100, seed=42)

    baseline_success = next(
        r for r in result if r["condition"] == "baseline" and r["metric_name"] == "success_rate"
    )
    assert baseline_success["mean"] == 0.7
    assert baseline_success["sample_size"] == 1
    assert baseline_success["ci_low"] is None  # No CI for single value
    assert baseline_success["ci_high"] is None


def test_bootstrap_ci_basic():
    """Test bootstrap CI computation."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci_low, ci_high = bootstrap_ci(values, ci_samples=1000, ci_confidence=0.95, seed=42)

    assert ci_low is not None
    assert ci_high is not None
    assert ci_low < np.mean(values)
    assert ci_high > np.mean(values)
    assert ci_low < ci_high


def test_bootstrap_ci_insufficient_data():
    """Test bootstrap CI with insufficient data."""
    values = [1.0]
    ci_low, ci_high = bootstrap_ci(values, ci_samples=1000, seed=42)

    assert ci_low is None
    assert ci_high is None


def test_aggregate_metrics_empty():
    """Test aggregation with empty records."""
    result = aggregate_metrics([], group_by="policy_type")
    assert result == []


def test_aggregate_metrics_missing_values():
    """Test aggregation with missing/None values."""
    metric_records = [
        {"seed": 42, "policy_type": "baseline", "success_rate": 0.7, "collision_rate": None},
        {"seed": 123, "policy_type": "baseline", "success_rate": 0.75, "collision_rate": 0.1},
    ]

    result = aggregate_metrics(metric_records, group_by="policy_type")

    # success_rate should have 2 samples
    baseline_success = next(
        r for r in result if r["condition"] == "baseline" and r["metric_name"] == "success_rate"
    )
    assert baseline_success["sample_size"] == 2

    # collision_rate should have 1 sample (None dropped)
    baseline_collision = next(
        r for r in result if r["condition"] == "baseline" and r["metric_name"] == "collision_rate"
    )
    assert baseline_collision["sample_size"] == 1
    assert baseline_collision["mean"] == 0.1


def test_completeness_score():
    """Completeness scoring tracks missing and failed seeds."""

    completeness = compute_completeness_score(
        expected_seeds=[1, 2, 3], completed_seeds=[1, 2], failed_seeds=[3]
    )

    assert completeness["score"] == pytest.approx(66.7, rel=1e-2)
    assert completeness["missing_seeds"] == ["3"]
    assert completeness["failed_seeds"] == ["3"]
    assert completeness["status"] == "PARTIAL"
