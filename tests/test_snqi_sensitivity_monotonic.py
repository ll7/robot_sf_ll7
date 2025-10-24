"""Test that increasing a penalty weight does not increase mean SNQI score on synthetic data.

We construct synthetic episodes where increasing collision penalty weight should
not raise (and typically lowers) average score. This provides a weak monotonicity
regression guard for objective consistency.
"""

from __future__ import annotations

import itertools

import numpy as np

from robot_sf.benchmark.snqi.compute import compute_snqi


def _synthetic_eps(num: int = 12):
    rng = np.random.default_rng(123)
    episodes = []
    for i in range(num):
        episodes.append(
            {
                "scenario_id": f"e{i}",
                "metrics": {
                    "success": float(rng.integers(0, 2)),
                    "time_to_goal_norm": float(rng.uniform(0.2, 1.0)),
                    "collisions": float(rng.integers(0, 4)),
                    "near_misses": float(rng.integers(0, 5)),
                    "comfort_exposure": float(rng.uniform(0.0, 0.5)),
                    "force_exceed_events": float(rng.integers(0, 3)),
                    "jerk_mean": float(rng.uniform(0.0, 0.8)),
                },
            },
        )
    return episodes


def _baseline(episodes):
    metrics = {k: [] for k in ["collisions", "near_misses", "force_exceed_events", "jerk_mean"]}
    for ep in episodes:
        m = ep["metrics"]
        for k in metrics:
            metrics[k].append(m[k])
    stats = {}
    for k, vals in metrics.items():
        arr = np.array(vals, dtype=float)
        stats[k] = {"med": float(np.median(arr)), "p95": float(np.percentile(arr, 95))}
    return stats


def test_collision_weight_monotonicity():
    eps = _synthetic_eps()
    base = _baseline(eps)

    # Evaluate mean SNQI across increasing collision weight
    weights_template = {
        "w_success": 2.0,
        "w_time": 1.0,
        "w_collisions": 1.0,  # will vary
        "w_near": 1.0,
        "w_comfort": 1.0,
        "w_force_exceed": 1.0,
        "w_jerk": 1.0,
    }

    means = []
    for coll_w in [0.5, 1.0, 1.5, 2.0, 2.5]:
        weights = dict(weights_template)
        weights["w_collisions"] = coll_w
        scores = [compute_snqi(ep["metrics"], weights, base) for ep in eps]
        means.append(float(np.mean(scores)))

    # Monotonic non-increasing (allow tiny numerical noise tolerance)
    for earlier, later in itertools.pairwise(means):
        assert later <= earlier + 1e-9, (
            f"Mean score increased with higher collision weight: {means}"
        )
