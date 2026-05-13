"""Tests for manual-control baseline comparison helpers."""

import pytest

from robot_sf.manual_control.baseline import BaselineMetric, MetricDirection, PolicyBaseline


def test_baseline_compare_higher_is_better_metric():
    """Candidate should beat higher-is-better metric only above tolerance."""
    baseline = PolicyBaseline(
        policy_id="best-policy",
        source="model/registry.yaml",
        primary_metric="success_rate",
        metrics={
            "success_rate": BaselineMetric(
                name="success_rate",
                value=0.7,
                direction=MetricDirection.HIGHER_IS_BETTER,
                tolerance=0.01,
            )
        },
    )

    comparison = baseline.compare({"success_rate": 0.72})

    assert comparison.beat_baseline is True
    assert comparison.metric_results == {"success_rate": True}


def test_baseline_compare_lower_is_better_metric():
    """Candidate should beat lower-is-better metric only below tolerance."""
    baseline = PolicyBaseline(
        policy_id="best-policy",
        source="model/registry.yaml",
        primary_metric="collision_rate",
        metrics={
            "collision_rate": BaselineMetric(
                name="collision_rate",
                value=0.2,
                direction=MetricDirection.LOWER_IS_BETTER,
                tolerance=0.01,
            )
        },
    )

    comparison = baseline.compare({"collision_rate": 0.18})

    assert comparison.beat_baseline is True
    assert comparison.metric_results == {"collision_rate": True}


def test_baseline_compare_requires_primary_candidate_metric():
    """Missing primary candidate metric should fail closed."""
    baseline = PolicyBaseline(
        policy_id="best-policy",
        source="model/registry.yaml",
        primary_metric="snqi",
        metrics={
            "snqi": BaselineMetric(
                name="snqi",
                value=0.5,
                direction=MetricDirection.HIGHER_IS_BETTER,
            )
        },
    )

    with pytest.raises(KeyError, match="candidate metric"):
        baseline.compare({})


def test_baseline_manifest_serialization_is_json_compatible():
    """Baseline manifests should preserve source and metric direction."""
    baseline = PolicyBaseline(
        policy_id="best-policy",
        source="model/registry.yaml",
        primary_metric="snqi",
        metrics={
            "snqi": BaselineMetric(
                name="snqi",
                value=0.5,
                direction=MetricDirection.HIGHER_IS_BETTER,
            )
        },
        metadata={"commit": "abc123"},
    )

    manifest = baseline.to_manifest_dict()

    assert manifest["policy_id"] == "best-policy"
    assert manifest["metrics"]["snqi"]["direction"] == "higher_is_better"
    assert manifest["metadata"] == {"commit": "abc123"}
