"""Tests for normalization comparison correlations in recompute script.

Validates that when comparing normalization strategies:
- Base strategy 'median_p95' is present
- All correlation values are finite and within [-1, 1]
- Alternate strategies exist (given synthetic data)
"""

from __future__ import annotations

import numpy as np

from scripts.recompute_snqi_weights import SNQIWeightRecomputer


def _episodes():
    """Episodes.

    Returns:
        Any: Auto-generated placeholder description.
    """
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
    """Baseline.

    Args:
        episodes: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
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
    """Test normalization comparison correlations range.

    Returns:
        Any: Auto-generated placeholder description.
    """
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
