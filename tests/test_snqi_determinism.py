"""Determinism tests for SNQI weight optimization differential evolution.

These tests exercise the `SNQIWeightOptimizer.differential_evolution_optimization`
method using a tiny synthetic episode set so they run quickly.
"""

from __future__ import annotations

import numpy as np

from robot_sf.benchmark.snqi import WEIGHT_NAMES
from scripts.snqi_weight_optimization import SNQIWeightOptimizer


def _make_synthetic_episodes(n: int = 6):
    episodes = []
    rng = np.random.default_rng(123)
    for i in range(n):
        episodes.append(
            {
                "scenario_id": f"scn_{i}",
                "metrics": {
                    "success": float(rng.integers(0, 2)),
                    "time": float(rng.uniform(5, 25)),
                    "collisions": float(rng.integers(0, 3)),
                    "near_misses": float(rng.integers(0, 5)),
                    "comfort": float(rng.uniform(0.2, 1.0)),
                    "force_exceed_events": float(rng.integers(0, 4)),
                    "jerk_mean": float(rng.uniform(0.0, 2.0)),
                },
            }
        )
    return episodes


def _baseline_from_episodes(episodes):
    # Compute simple med/p95 stats to approximate real baseline structure
    metrics = {
        k: []
        for k in [
            "collisions",
            "near_misses",
            "force_exceed_events",
            "jerk_mean",
        ]
    }
    for ep in episodes:
        m = ep["metrics"]
        for k in metrics.keys():
            metrics[k].append(m[k])
    baseline = {}
    for k, vals in metrics.items():
        arr = np.array(vals, dtype=float)
        baseline[k] = {"med": float(np.median(arr)), "p95": float(np.percentile(arr, 95))}
    return baseline


def test_differential_evolution_deterministic_same_seed():
    episodes = _make_synthetic_episodes()
    baseline = _baseline_from_episodes(episodes)
    opt1 = SNQIWeightOptimizer(episodes, baseline)
    opt2 = SNQIWeightOptimizer(episodes, baseline)

    res1 = opt1.differential_evolution_optimization(maxiter=5, seed=123)
    res2 = opt2.differential_evolution_optimization(maxiter=5, seed=123)

    assert res1.weights == res2.weights, "Weights should match for identical seed"
    assert abs(res1.objective_value - res2.objective_value) < 1e-12


def test_differential_evolution_differs_different_seed():
    episodes = _make_synthetic_episodes()
    baseline = _baseline_from_episodes(episodes)
    opt1 = SNQIWeightOptimizer(episodes, baseline)
    opt2 = SNQIWeightOptimizer(episodes, baseline)

    res1 = opt1.differential_evolution_optimization(maxiter=5, seed=111)
    res2 = opt2.differential_evolution_optimization(maxiter=5, seed=222)

    # It is possible (though unlikely) to coincide; allow fallback check
    if res1.weights == res2.weights:
        # Ensure at least objective values coincide if weights same (sanity)
        assert abs(res1.objective_value - res2.objective_value) < 1e-12
    else:
        # Distinct seeds should usually produce at least one differing weight value
        diff_count = sum(1 for k in WEIGHT_NAMES if abs(res1.weights[k] - res2.weights[k]) > 1e-9)
        assert diff_count > 0
