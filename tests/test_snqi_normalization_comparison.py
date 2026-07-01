"""Tests for normalization comparison correlations in recompute script.

Validates that when comparing normalization strategies:
- Base strategy 'median_p95' is present
- All correlation values are finite and within [-1, 1]
- Alternate strategies exist (given synthetic data)
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from robot_sf.benchmark.snqi.compute import compute_snqi
from scripts.recompute_snqi_weights import SNQIWeightRecomputer

_METRICS = {
    "success": 1.0,
    "time_to_goal_norm": 3.0,
    "collisions": 9.0,
    "near_misses": 9.0,
    "comfort_exposure": 2.0,
    "force_exceed_events": 9.0,
    "jerk_mean": 9.0,
}
_BASELINE_STATS_V1 = {
    "time_to_goal_norm": {"med": 0.0, "p95": 1.0},
    "collisions": {"med": 0.0, "p95": 1.0},
    "near_misses": {"med": 0.0, "p95": 1.0},
    "comfort_exposure": {"med": 0.0, "p95": 1.0},
    "force_exceed_events": {"med": 0.0, "p95": 1.0},
    "jerk_mean": {"med": 0.0, "p95": 1.0},
}
_WEIGHTS = {
    "w_success": 1.0,
    "w_time": 1.0,
    "w_collisions": 1.0,
    "w_near": 1.0,
    "w_comfort": 1.0,
    "w_force_exceed": 1.0,
    "w_jerk": 1.0,
}


def _episodes():
    """TODO docstring. Document this function."""
    rng = np.random.default_rng(5)
    episodes = []
    for i in range(15):
        episodes.append(
            {
                "scenario_id": f"scn_{i}",
                "metrics": {
                    "success": float(rng.integers(0, 2)),
                    "time_to_goal_norm": float(rng.uniform(0.2, 1.1)),
                    "collisions": float(rng.integers(0, 5)),
                    "near_misses": float(rng.integers(0, 7)),
                    "comfort_exposure": float(rng.uniform(0.0, 0.6)),
                    "force_exceed_events": float(rng.integers(0, 6)),
                    "jerk_mean": float(rng.uniform(0.0, 1.2)),
                },
            },
        )
    return episodes


def _baseline(episodes):
    """TODO docstring. Document this function.

    Args:
        episodes: TODO docstring.
    """
    metrics = {k: [] for k in ["collisions", "near_misses", "force_exceed_events", "jerk_mean"]}
    for e in episodes:
        for k in metrics:
            metrics[k].append(e["metrics"][k])
    stats = {}
    for k, vals in metrics.items():
        arr = np.array(vals, dtype=float)
        stats[k] = {"med": float(np.median(arr)), "p95": float(np.percentile(arr, 95))}
    return stats


def test_normalization_comparison_correlations_range():
    """TODO docstring. Document this function."""
    eps = _episodes()
    base = _baseline(eps)
    recomputer = SNQIWeightRecomputer(eps, base)
    weights = recomputer.default_weights()
    comparison = recomputer.compare_normalization_strategies(weights)

    assert "median_p95" in comparison, "Base normalization strategy missing"
    # Should have at least one alternate strategy
    assert any(k != "median_p95" for k in comparison)

    for name, data in comparison.items():
        assert "correlation_with_base" in data
        corr = data["correlation_with_base"]
        assert np.isfinite(corr), f"Correlation not finite for {name}"
        assert -1.0 <= corr <= 1.0, f"Correlation out of range for {name}: {corr}"


def test_compute_snqi_default_preserves_v0_score() -> None:
    """Default score version preserves legacy raw time/comfort behavior."""
    assert compute_snqi(_METRICS, _WEIGHTS, _BASELINE_STATS_V1) == pytest.approx(-8.0)


def test_compute_snqi_v1_uses_bounded_comparable_penalty_terms() -> None:
    """SNQI-v1 clamps all penalty terms onto the same baseline-relative basis."""
    assert compute_snqi(
        _METRICS,
        _WEIGHTS,
        _BASELINE_STATS_V1,
        score_version="SNQI-v1",
    ) == pytest.approx(-5.0)


def test_snqi_v1_missing_baseline_stats_fail_closed() -> None:
    """SNQI-v1 must not silently zero missing normalized terms."""
    missing_time = copy.deepcopy(_BASELINE_STATS_V1)
    missing_time.pop("time_to_goal_norm")

    with pytest.raises(ValueError, match="missing med/p95"):
        compute_snqi(_METRICS, _WEIGHTS, missing_time, score_version="SNQI-v1")


def test_snqi_v1_invalid_baseline_spread_fails_closed() -> None:
    """SNQI-v1 requires positive median/p95 spread for each penalty metric."""
    invalid_time = copy.deepcopy(_BASELINE_STATS_V1)
    invalid_time["time_to_goal_norm"] = {"med": 1.0, "p95": 1.0}

    with pytest.raises(ValueError, match="non-positive spread"):
        compute_snqi(_METRICS, _WEIGHTS, invalid_time, score_version="SNQI-v1")


def test_compute_snqi_unknown_score_version_fails_closed() -> None:
    """Unknown score versions should fail closed instead of falling back."""
    with pytest.raises(ValueError, match="unknown SNQI score version"):
        compute_snqi(_METRICS, _WEIGHTS, _BASELINE_STATS_V1, score_version="SNQI-v2")
