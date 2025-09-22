"""SNQI weighting & normalization behavior tests.

Focus:
  - Missing weights default to 1.0
  - Missing baseline stats cause optimistic (0.0) normalized penalties
  - Provided baseline stats normalize values into [0,1]

We construct a synthetic metric_values dict and vary baseline coverage.
"""

from __future__ import annotations

import math

from robot_sf.benchmark.metrics import snqi

METRICS = {
    "success": 1.0,
    "time_to_goal_norm": 0.2,
    "collisions": 3.0,
    "near_misses": 5.0,
    "comfort_exposure": 0.1,
    "force_exceed_events": 4.0,
    "jerk_mean": 2.0,
    "curvature_mean": 0.5,
}


def test_snqi_missing_baselines_optimistic():
    """Only collisions have baseline stats; others default to normalized 0."""
    baseline = {
        "collisions": {"med": 1.0, "p95": 5.0},  # value 3 -> (3-1)/(5-1)=0.5
    }
    weights = {"w_success": 2.0, "w_time": 1.0}  # other weights default to 1
    score = snqi(METRICS, weights, baseline_stats=baseline)
    # Expected: 2*1 - 1*0.2 - 1*0.5 - 1*0.0 - 1*0.1 - rest zeros = 1.2
    assert math.isclose(score, 1.2, rel_tol=1e-9), score


def test_snqi_full_baselines_all_penalties_count():
    """Providing baselines for all penalized metrics lowers the score appropriately."""
    baseline = {
        "collisions": {"med": 1.0, "p95": 5.0},  # norm 0.5
        "near_misses": {"med": 2.0, "p95": 10.0},  # (5-2)/(10-2)=0.375
        "force_exceed_events": {"med": 1.0, "p95": 9.0},  # (4-1)/8=0.375
        "jerk_mean": {"med": 0.5, "p95": 2.5},  # (2.0-0.5)/2=0.75
        "curvature_mean": {"med": 0.2, "p95": 0.6},  # (0.5-0.2)/0.4=0.75
    }
    weights = {"w_success": 2.0, "w_time": 1.0}
    score = snqi(METRICS, weights, baseline_stats=baseline)
    # Penalty sum: time 0.2 + coll 0.5 + near 0.375 + comfort 0.1 + force 0.375 + jerk 0.75 + curvature 0.75 = 3.05
    # Score = 2 - 3.05 = -1.05
    assert math.isclose(score, -1.05, rel_tol=1e-9), score
