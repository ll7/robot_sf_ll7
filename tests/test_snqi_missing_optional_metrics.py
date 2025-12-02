"""Tests handling of episodes missing optional metrics for SNQI tools.

Ensures that recomputation and optimization logic do not fail when some
metrics (collisions, near_misses, force_exceed_events, jerk_mean, comfort_exposure)
are absent in episode records. Also verifies weights computation returns finite
values.
"""

from __future__ import annotations

import numpy as np

from scripts.recompute_snqi_weights import SNQIWeightRecomputer, load_episodes_data
from scripts.snqi_weight_optimization import SNQIWeightOptimizer


def _episodes_partial():
    """Episodes partial.

    Returns:
        Any: Auto-generated placeholder description.
    """
    # Some episodes with missing metrics fields
    return [
        {"scenario_id": "a", "metrics": {"success": 1.0, "time_to_goal_norm": 0.4}},
        {
            "scenario_id": "b",
            "metrics": {"success": 0.0, "time_to_goal_norm": 0.9, "collisions": 1},
        },
        {
            "scenario_id": "c",
            "metrics": {"success": 1.0, "time_to_goal_norm": 0.6, "near_misses": 2},
        },
    ]


def _baseline_stub():
    """Baseline stub.

    Returns:
        Any: Auto-generated placeholder description.
    """
    # Minimal baseline: supply stats only for metrics that may appear
    return {
        "collisions": {"med": 0.0, "p95": 2.0},
        "near_misses": {"med": 0.0, "p95": 5.0},
        "force_exceed_events": {"med": 0.0, "p95": 1.0},
        "jerk_mean": {"med": 0.0, "p95": 0.5},
    }


def test_recompute_with_missing_optional_metrics():
    """Test recompute with missing optional metrics.

    Returns:
        Any: Auto-generated placeholder description.
    """
    eps = _episodes_partial()
    recomputer = SNQIWeightRecomputer(eps, _baseline_stub())
    res = recomputer.recompute_with_strategy("default")
    assert "weights" in res
    assert all(np.isfinite(list(res["weights"].values())))
    # Statistics should have overall block
    assert "overall" in res["statistics"]


def test_optimization_with_missing_optional_metrics():
    """Test optimization with missing optional metrics.

    Returns:
        Any: Auto-generated placeholder description.
    """
    eps = _episodes_partial()
    optimizer = SNQIWeightOptimizer(eps, _baseline_stub())
    # Differential evolution might be overkill; use small grid for speed
    grid_res = optimizer.grid_search_optimization(grid_resolution=2)
    assert grid_res.weights
    assert all(np.isfinite(list(grid_res.weights.values())))


def test_load_episodes_skips_malformed(tmp_path):
    """Test load episodes skips malformed.

    Args:
        tmp_path: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    # Create JSONL with some bad lines
    good1 = {"scenario_id": "x", "metrics": {"success": 1.0, "time_to_goal_norm": 0.5}}
    good2 = {"scenario_id": "y", "metrics": {"success": 0.0, "time_to_goal_norm": 0.8}}
    content = "\n".join(
        [
            "{",  # malformed
            "not json",  # malformed
            "" + str(good1).replace("'", '"'),
            "" + str(good2).replace("'", '"'),
            "",  # empty line
        ],
    )
    p = tmp_path / "episodes.jsonl"
    p.write_text(content, encoding="utf-8")
    loaded = load_episodes_data(p)
    # Should only load the two valid objects
    assert len(loaded) == 2
