"""Tests for SNQI weight recomputation strategies.

Validates that each exposed strategy from `SNQIWeightRecomputer` produces:
- A weights dict containing all canonical weight names
- Weight values within expected numeric bounds (0 <= w <= 3.0)
- Statistics block with required overall keys
- Pareto strategy includes non‑empty pareto_alternatives list (<=5) with expected fields

These tests use a small synthetic episode set for speed. The intent is structural
and range validation rather than asserting specific numeric outcomes.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark.snqi import WEIGHT_NAMES  # type: ignore
from scripts.recompute_snqi_weights import SNQIWeightRecomputer


@pytest.fixture(scope="module", name="synthetic_episodes")
def _synthetic_episodes_fixture():
    """Synthetic episodes fixture.

    Returns:
        Any: Auto-generated placeholder description.
    """
    rng = np.random.default_rng(42)
    episodes = []
    for i in range(12):
        episodes.append(
            {
                "scenario_id": f"scn_{i}",
                # mimic structure used in scripts (metrics nested)
                "metrics": {
                    "success": float(rng.integers(0, 2)),
                    "time_to_goal_norm": float(rng.uniform(0.2, 1.2)),
                    "collisions": float(rng.integers(0, 4)),
                    "near_misses": float(rng.integers(0, 8)),
                    "comfort_exposure": float(rng.uniform(0.0, 0.6)),
                    "force_exceed_events": float(rng.integers(0, 6)),
                    "jerk_mean": float(rng.uniform(0.0, 1.5)),
                },
            },
        )
    return episodes


@pytest.fixture(scope="module", name="baseline_stats")
def _baseline_stats_fixture(synthetic_episodes):
    """Baseline stats fixture.

    Args:
        synthetic_episodes: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    # Build baseline median/p95 stats matching compute_snqi expectations
    metrics = {k: [] for k in ["collisions", "near_misses", "force_exceed_events", "jerk_mean"]}
    for ep in synthetic_episodes:
        m = ep["metrics"]
        for k in metrics:  # collect
            metrics[k].append(m[k])
    stats = {}
    for k, vals in metrics.items():
        arr = np.array(vals, dtype=float)
        stats[k] = {"med": float(np.median(arr)), "p95": float(np.percentile(arr, 95))}
    return stats


@pytest.mark.parametrize(
    "strategy",
    ["default", "balanced", "safety_focused", "efficiency_focused", "pareto"],
)
def test_recompute_strategies_structure(strategy, synthetic_episodes, baseline_stats):
    """Test recompute strategies structure.

    Args:
        strategy: Auto-generated placeholder description.
        synthetic_episodes: Auto-generated placeholder description.
        baseline_stats: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    # Seed NumPy RNG for deterministic pareto sampling
    np.random.seed(123)
    recomputer = SNQIWeightRecomputer(synthetic_episodes, baseline_stats)
    result = recomputer.recompute_with_strategy(strategy)

    # Base keys
    assert "weights" in result, f"Missing weights for strategy {strategy}"
    assert "statistics" in result, f"Missing statistics for strategy {strategy}"
    assert "normalization_strategy" in result

    # Weights coverage & range
    weights = result["weights"]
    missing = set(WEIGHT_NAMES) - set(weights.keys())
    assert not missing, f"Strategy {strategy} missing weight keys: {missing}"
    for name, value in weights.items():
        assert isinstance(value, int | float), f"Weight {name} not numeric"
        assert 0.0 <= float(value) <= 3.5, f"Weight {name} out of expected range: {value}"

    # Statistics structure (overall block)
    stats = result["statistics"]
    assert "overall" in stats, "Missing overall stats"
    for key in ["mean", "std", "min", "max", "range"]:
        assert key in stats["overall"], f"Missing overall stat key {key}"
        assert isinstance(stats["overall"][key], float), f"Overall stat {key} not float"

    # Pareto specific assertions
    if strategy == "pareto":
        assert "pareto_alternatives" in result, "Pareto strategy missing pareto_alternatives"
        alts = result["pareto_alternatives"]
        assert 0 < len(alts) <= 5
        for alt in alts:
            for k in ["weights", "discriminative_power", "stability", "mean_score"]:
                assert k in alt, f"Alternative missing {k}"
            # Basic sanity on weight alt values
            for wname, wval in alt["weights"].items():
                assert wname in WEIGHT_NAMES
                assert 0.0 <= float(wval) <= 3.5
