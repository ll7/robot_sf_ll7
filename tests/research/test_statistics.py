"""Unit tests for statistical analysis module."""

import pytest

from robot_sf.research.statistics import cohen_d, evaluate_hypothesis, paired_t_test


def test_paired_t_test_basic():
    """Test basic paired t-test."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.5, 2.5, 3.5, 4.5, 5.5]

    result = paired_t_test(x, y)

    assert "t_stat" in result
    assert "p_value" in result
    assert "n" in result
    assert result["n"] == 5
    assert result["t_stat"] is not None
    assert result["p_value"] is not None
    assert 0 <= result["p_value"] <= 1


def test_paired_t_test_insufficient_data():
    """Test paired t-test with insufficient data."""
    x = [1.0]
    y = [1.5]

    result = paired_t_test(x, y)

    assert result["t_stat"] is None
    assert result["p_value"] is None
    assert result["n"] == 1


def test_paired_t_test_mismatched_lengths():
    """Test paired t-test with mismatched lengths."""
    x = [1.0, 2.0, 3.0]
    y = [1.5, 2.5]

    result = paired_t_test(x, y)

    assert result["t_stat"] is None
    assert result["p_value"] is None
    assert result["n"] == 2


def test_cohen_d_basic():
    """Test Cohen's d effect size calculation."""
    # Use data with varying differences to get non-zero std
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.5, 3.0, 3.5, 5.0, 6.5]  # Differences: [-1.5, -1.0, -0.5, -1.0, -1.5]

    effect_size = cohen_d(x, y)

    assert effect_size is not None
    assert isinstance(effect_size, float)
    # With varying differences, we should get a valid effect size
    assert abs(effect_size) > 0


def test_cohen_d_no_difference():
    """Test Cohen's d when there is no difference."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 2.0, 3.0, 4.0, 5.0]

    effect_size = cohen_d(x, y)

    # Effect size should be 0 or None (std is 0)
    assert effect_size is None or effect_size == pytest.approx(0.0, abs=1e-6)


def test_cohen_d_insufficient_data():
    """Test Cohen's d with insufficient data."""
    x = [1.0]
    y = [2.0]

    effect_size = cohen_d(x, y)

    assert effect_size is None


def test_evaluate_hypothesis_pass():
    """Test hypothesis evaluation that passes."""
    baseline = [500000, 480000, 520000]  # Mean ~500k
    pretrained = [280000, 270000, 290000]  # Mean ~280k, ~44% reduction

    result = evaluate_hypothesis(baseline, pretrained, threshold=40.0)

    assert result["decision"] == "PASS"
    assert result["measured_value"] is not None
    assert result["measured_value"] >= 40.0
    assert result["threshold_value"] == 40.0
    assert "note" in result


def test_evaluate_hypothesis_fail_insufficient_improvement():
    """Test hypothesis evaluation that fails due to insufficient improvement."""
    baseline = [500000, 480000, 520000]
    pretrained = [400000, 390000, 410000]  # Only ~20% reduction

    result = evaluate_hypothesis(baseline, pretrained, threshold=40.0)

    assert result["decision"] == "FAIL"
    assert result["measured_value"] is not None
    assert result["measured_value"] < 40.0
    assert "threshold" in result["note"].lower()


def test_evaluate_hypothesis_fail_degradation():
    """Test hypothesis evaluation that fails due to performance degradation."""
    baseline = [500000, 480000, 520000]
    pretrained = [600000, 590000, 610000]  # Worse than baseline

    result = evaluate_hypothesis(baseline, pretrained, threshold=40.0)

    assert result["decision"] == "FAIL"
    assert result["measured_value"] is not None
    assert result["measured_value"] < 0
    assert "degrad" in result["note"].lower()


def test_evaluate_hypothesis_incomplete():
    """Test hypothesis evaluation with incomplete data."""
    baseline = []
    pretrained = [280000]

    result = evaluate_hypothesis(baseline, pretrained, threshold=40.0)

    assert result["decision"] == "INCOMPLETE"
    assert "note" in result


def test_evaluate_hypothesis_zero_baseline():
    """Test hypothesis evaluation with zero baseline."""
    baseline = [0.0, 0.0]
    pretrained = [100.0, 100.0]

    result = evaluate_hypothesis(baseline, pretrained, threshold=40.0)

    assert result["decision"] == "INCOMPLETE"
    assert "zero" in result["note"].lower() or "mean" in result["note"].lower()
