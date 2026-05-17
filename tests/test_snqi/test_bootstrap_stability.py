"""Tests for SNQI bootstrap ranking-stability calculation."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark.snqi.bootstrap import bootstrap_stability


def _episodes() -> list[dict]:
    """Return synthetic episodes with a stable SNQI ordering."""
    return [
        {"algo": "alpha", "metrics": {"snqi": 0.95}},
        {"algo": "alpha", "metrics": {"snqi": 0.92}},
        {"algo": "alpha", "metrics": {"snqi": 0.91}},
        {"algo": "beta", "metrics": {"snqi": 0.62}},
        {"algo": "beta", "metrics": {"snqi": 0.59}},
        {"algo": "beta", "metrics": {"snqi": 0.58}},
        {"algo": "gamma", "metrics": {"snqi": 0.25}},
        {"algo": "gamma", "metrics": {"snqi": 0.22}},
        {"algo": "gamma", "metrics": {"snqi": 0.20}},
    ]


def test_bootstrap_stability_computes_deterministic_non_placeholder_result() -> None:
    """Bootstrap stability should return evidence-grade values for valid SNQI episodes."""
    left = bootstrap_stability(
        _episodes(),
        {"snqi": 1.0},
        rng=np.random.default_rng(123),
        samples=20,
    )
    right = bootstrap_stability(
        _episodes(),
        {"snqi": 1.0},
        rng=np.random.default_rng(123),
        samples=20,
    )

    assert left == right
    assert left["status"] == "ok"
    assert left["stability"] == pytest.approx(1.0)
    assert left["samples"] == 20
    assert left["method"] == "bootstrap_spearman"
    assert left["details"]["baseline_ordering"] == ["alpha", "beta", "gamma"]
    assert left["details"]["groups"] == {
        "alpha": {"episodes": 3, "mean_snqi": pytest.approx(0.9266666667)},
        "beta": {"episodes": 3, "mean_snqi": pytest.approx(0.5966666667)},
        "gamma": {"episodes": 3, "mean_snqi": pytest.approx(0.2233333333)},
    }


def test_bootstrap_stability_rejects_missing_snqi_metric() -> None:
    """Missing SNQI values should fail closed instead of producing placeholder evidence."""
    with pytest.raises(ValueError, match="metrics.snqi"):
        bootstrap_stability(
            [{"algo": "alpha", "metrics": {"success": 1.0}}],
            {"snqi": 1.0},
            rng=np.random.default_rng(123),
        )


def test_bootstrap_stability_requires_multiple_groups() -> None:
    """Ranking stability is undefined for a single planner group."""
    with pytest.raises(ValueError, match="at least two"):
        bootstrap_stability(
            [
                {"algo": "alpha", "metrics": {"snqi": 0.9}},
                {"algo": "alpha", "metrics": {"snqi": 0.8}},
            ],
            {"snqi": 1.0},
            rng=np.random.default_rng(123),
        )
