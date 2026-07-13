"""Unit tests for metric aggregation module."""

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

    # Assert non-metric/metadata columns are NOT aggregated as metrics
    assert "seed" not in metrics
    assert "policy_type" not in metrics
    assert "variant_id" not in metrics

    # Check baseline success_rate aggregation
    baseline_success = next(
        r for r in result if r["condition"] == "baseline" and r["metric_name"] == "success_rate"
    )
    assert baseline_success["mean"] == pytest.approx(0.725, abs=1e-6)
    assert baseline_success["median"] == pytest.approx(0.725, abs=1e-6)
    assert baseline_success["p95"] == pytest.approx(0.7475, abs=1e-6)
    assert baseline_success["std"] == pytest.approx(0.035355, abs=1e-5)
    assert baseline_success["sample_size"] == 2
    assert baseline_success["ci_low"] == pytest.approx(0.7, abs=1e-6)
    assert baseline_success["ci_high"] == pytest.approx(0.75, abs=1e-6)


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
    assert baseline_success["median"] == 0.7
    assert baseline_success["p95"] == 0.7
    assert baseline_success["std"] == 0.0
    assert baseline_success["sample_size"] == 1
    assert baseline_success["ci_low"] is None  # No CI for single value
    assert baseline_success["ci_high"] is None


def test_bootstrap_ci_basic():
    """Test bootstrap CI computation."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci_low, ci_high = bootstrap_ci(values, ci_samples=1000, ci_confidence=0.95, seed=42)

    assert ci_low == pytest.approx(1.8, abs=1e-6)
    assert ci_high == pytest.approx(4.2, abs=1e-6)


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

    assert completeness["score"] == 66.7
    assert completeness["missing_seeds"] == []
    assert completeness["failed_seeds"] == ["3"]
    assert completeness["status"] == "PARTIAL"
    assert completeness["expected"] == 3
    assert completeness["completed"] == 2


def test_aggregate_metrics_missing_groupby_field():
    """Verify that aggregate_metrics returns [] if group_by field is missing from df."""
    metric_records = [{"seed": 42, "success_rate": 0.7}]
    result = aggregate_metrics(metric_records, group_by="policy_type")
    assert result == []


def test_aggregate_metrics_seed_reproducibility():
    """Verify that seed parameter ensures reproducible CIs, and different seeds differ."""
    metric_records = [
        {"seed": i, "policy_type": "baseline", "success_rate": float(i) / 50.0} for i in range(50)
    ]
    res1 = aggregate_metrics(metric_records, seed=42)
    res2 = aggregate_metrics(metric_records, seed=42)
    res3 = aggregate_metrics(metric_records, seed=123)

    ci1 = (res1[0]["ci_low"], res1[0]["ci_high"])
    ci2 = (res2[0]["ci_low"], res2[0]["ci_high"])
    ci3 = (res3[0]["ci_low"], res3[0]["ci_high"])

    assert ci1 == ci2
    assert ci1 != ci3
    assert res1[0]["ci_low"] == pytest.approx(0.40916, abs=1e-5)
    assert res1[0]["ci_high"] == pytest.approx(0.56323, abs=1e-5)


def test_aggregate_metrics_entirely_none_metric():
    """Verify that a metric column that is entirely None is skipped, without breaking the loop."""
    metric_records = [
        {"seed": 1, "policy_type": "baseline", "success_rate": 0.7, "collision_rate": None},
        {"seed": 2, "policy_type": "baseline", "success_rate": 0.8, "collision_rate": None},
    ]
    result = aggregate_metrics(metric_records)
    metrics = {r["metric_name"] for r in result}
    assert "success_rate" in metrics
    assert "collision_rate" not in metrics


def test_bootstrap_ci_exactly_two_values():
    """Verify bootstrap_ci computes a valid CI when values has exactly 2 elements."""
    values = [1.0, 2.0]
    ci_low, ci_high = bootstrap_ci(values, seed=42)
    assert ci_low is not None
    assert ci_high is not None
    assert ci_low <= ci_high


def test_bootstrap_ci_defaults():
    """Verify bootstrap_ci works correctly with default arguments (no confidence or samples specified)."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci_low, ci_high = bootstrap_ci(values, seed=42)
    assert ci_low is not None
    assert ci_high is not None
    assert ci_low < ci_high


def test_completeness_score_empty_expected():
    """Verify completeness score handles empty expected_seeds by returning score 0.0."""
    completeness = compute_completeness_score(expected_seeds=[], completed_seeds=[])
    assert completeness["score"] == 0.0
    assert completeness["status"] == "PASS"


def test_completeness_score_sorting():
    """Verify completeness score correctly sorts missing and failed seeds."""
    completeness = compute_completeness_score(
        expected_seeds=[10, 2, "abc", 1, "1a"],
        completed_seeds=[1],
        failed_seeds=[10, 2, "abc", "1a"],
    )
    # missing: empty list as all others are in failed or completed
    assert completeness["missing_seeds"] == []
    # failed seeds should sort numeric first: 10, 2, 1a, abc
    assert completeness["failed_seeds"] == ["10", "2", "1a", "abc"]


def test_completeness_score_fail():
    """Verify completeness score status is FAIL when there are no completed seeds."""
    completeness = compute_completeness_score(expected_seeds=[1, 2], completed_seeds=[])
    assert completeness["status"] == "FAIL"
    assert completeness["score"] == 0.0
    assert completeness["expected"] == 2
    assert completeness["completed"] == 0
