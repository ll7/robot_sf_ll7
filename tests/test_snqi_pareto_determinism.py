"""Determinism test for Pareto sampling in SNQI weight recomputation.

Ensures that with a fixed NumPy seed the `pareto` strategy yields identical
selected weights and alternative set across runs. Uses a moderate sample size
reduction to keep runtime low.
"""

from __future__ import annotations

import numpy as np

from scripts.recompute_snqi_weights import SNQIWeightRecomputer


def _episodes(n: int = 10):
    """TODO docstring. Document this function.

    Args:
        n: TODO docstring.
    """
    rng = np.random.default_rng(7)
    eps = []
    for i in range(n):
        eps.append(
            {
                "scenario_id": f"scn_{i}",
                "metrics": {
                    "success": float(rng.integers(0, 2)),
                    "time_to_goal_norm": float(rng.uniform(0.2, 1.0)),
                    "collisions": float(rng.integers(0, 5)),
                    "near_misses": float(rng.integers(0, 6)),
                    "comfort_exposure": float(rng.uniform(0.0, 0.5)),
                    "force_exceed_events": float(rng.integers(0, 5)),
                    "jerk_mean": float(rng.uniform(0.0, 1.0)),
                },
            },
        )
    return eps


def _baseline(episodes):
    """TODO docstring. Document this function.

    Args:
        episodes: TODO docstring.
    """
    metrics = {k: [] for k in ["collisions", "near_misses", "force_exceed_events", "jerk_mean"]}
    for ep in episodes:
        for k in metrics:
            metrics[k].append(ep["metrics"][k])
    stats = {}
    for k, vals in metrics.items():
        arr = np.array(vals, dtype=float)
        stats[k] = {"med": float(np.median(arr)), "p95": float(np.percentile(arr, 95))}
    return stats


def test_pareto_sampling_deterministic():
    """TODO docstring. Document this function."""
    eps = _episodes()
    baseline = _baseline(eps)

    # First run
    np.random.seed(999)
    recomputer1 = SNQIWeightRecomputer(eps, baseline)
    res1 = recomputer1.recompute_with_strategy("pareto")

    # Second run (reset seed)
    np.random.seed(999)
    recomputer2 = SNQIWeightRecomputer(eps, baseline)
    res2 = recomputer2.recompute_with_strategy("pareto")

    assert res1["weights"] == res2["weights"], "Pareto selected weights differ under same seed"

    # Compare top alternatives (order & contents)
    alts1 = res1.get("pareto_alternatives", [])
    alts2 = res2.get("pareto_alternatives", [])
    assert len(alts1) == len(alts2)
    for a1, a2 in zip(alts1, alts2, strict=False):
        assert a1["weights"] == a2["weights"]
        # Floating metrics: allow tiny tolerance
        for k in ["discriminative_power", "stability", "mean_score"]:
            assert abs(a1[k] - a2[k]) < 1e-12
