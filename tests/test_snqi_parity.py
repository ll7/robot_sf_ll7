"""Parity tests for SNQI canonical implementation.

These tests ensure the shared `compute_snqi` in `robot_sf.benchmark.snqi` produces
identical results to a legacy inline formula representative of pre-refactor
scripts. This guards against accidental drift when modifying normalization or
weight application logic.
"""

from __future__ import annotations

from math import isclose

import pytest

from robot_sf.benchmark.snqi import compute_snqi


# Legacy inline implementation copied (logic only) to assert parity.
# NOTE: Keep this in sync only for parity testing; do NOT import elsewhere.
def _legacy_compute_snqi(metrics, weight_map, baseline_map):  # type: ignore[missing-type-doc]
    """Legacy compute snqi.

    Args:
        metrics: Auto-generated placeholder description.
        weight_map: Auto-generated placeholder description.
        baseline_map: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """

    def _normalize(name: str, value: float):
        """Normalize.

        Args:
            name: Auto-generated placeholder description.
            value: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        if name not in baseline_map:
            return 0.0
        med = baseline_map[name].get("med", 0.0)
        p95 = baseline_map[name].get("p95", med)
        eps = 1e-6
        denom = (p95 - med) if (p95 - med) > eps else 1.0
        norm = (value - med) / denom
        norm = max(norm, 0.0)
        norm = min(norm, 1.0)
        return norm

    success = metrics.get("success", 0.0)
    if isinstance(success, bool):
        success = 1.0 if success else 0.0

    time_norm = metrics.get("time_to_goal_norm", 1.0)
    coll = _normalize("collisions", metrics.get("collisions", 0.0))
    near = _normalize("near_misses", metrics.get("near_misses", 0.0))
    comfort = metrics.get("comfort_exposure", 0.0)
    force_ex = _normalize("force_exceed_events", metrics.get("force_exceed_events", 0.0))
    jerk_n = _normalize("jerk_mean", metrics.get("jerk_mean", 0.0))

    score = (
        weight_map.get("w_success", 1.0) * success
        - weight_map.get("w_time", 1.0) * time_norm
        - weight_map.get("w_collisions", 1.0) * coll
        - weight_map.get("w_near", 1.0) * near
        - weight_map.get("w_comfort", 1.0) * comfort
        - weight_map.get("w_force_exceed", 1.0) * force_ex
        - weight_map.get("w_jerk", 1.0) * jerk_n
    )
    return float(score)


@pytest.fixture
def baseline_stats():  # Representative synthetic stats
    """Baseline stats.

    Returns:
        Any: Auto-generated placeholder description.
    """
    return {
        "collisions": {"med": 0.0, "p95": 3.0},
        "near_misses": {"med": 1.0, "p95": 6.0},
        "force_exceed_events": {"med": 0.0, "p95": 10.0},
        "jerk_mean": {"med": 0.05, "p95": 0.55},
    }


@pytest.fixture
def weights():
    """Weights.

    Returns:
        Any: Auto-generated placeholder description.
    """
    return {
        "w_success": 2.0,
        "w_time": 1.2,
        "w_collisions": 1.5,
        "w_near": 0.8,
        "w_comfort": 0.5,
        "w_force_exceed": 0.7,
        "w_jerk": 0.9,
    }


@pytest.mark.parametrize(
    "metrics",
    [
        {  # success fast run
            "success": True,
            "time_to_goal_norm": 0.3,
            "collisions": 0,
            "near_misses": 2,
            "comfort_exposure": 0.1,
            "force_exceed_events": 1,
            "jerk_mean": 0.2,
        },
        {  # failure with many collisions
            "success": False,
            "time_to_goal_norm": 1.1,
            "collisions": 5,
            "near_misses": 7,  # exceeds p95 for clamping
            "comfort_exposure": 0.4,
            "force_exceed_events": 12,  # clamp
            "jerk_mean": 0.9,  # clamp
        },
        {  # mid quality
            "success": 1.0,
            "time_to_goal_norm": 0.7,
            "collisions": 1,
            "near_misses": 3,
            "comfort_exposure": 0.2,
            "force_exceed_events": 3,
            "jerk_mean": 0.3,
        },
        {  # edge: missing optional metrics
            "success": True,
            "time_to_goal_norm": 0.5,
            # collisions omitted
            # near_misses omitted
            # force_exceed_events omitted
            # jerk_mean omitted
            "comfort_exposure": 0.05,
        },
    ],
)
def test_compute_snqi_parity(metrics, weights, baseline_stats):
    """Canonical compute_snqi must match legacy implementation for varied scenarios."""
    legacy = _legacy_compute_snqi(metrics, weights, baseline_stats)
    canonical = compute_snqi(metrics, weights, baseline_stats)
    assert isclose(legacy, canonical, rel_tol=1e-9, abs_tol=1e-12)


def test_monotonic_success_improves_score(weights, baseline_stats):
    """Increasing success (0->1) should not reduce SNQI score when other metrics fixed."""
    metrics_fail = {"success": False, "time_to_goal_norm": 0.5, "comfort_exposure": 0.1}
    metrics_success = {"success": True, "time_to_goal_norm": 0.5, "comfort_exposure": 0.1}
    score_fail = compute_snqi(metrics_fail, weights, baseline_stats)
    score_success = compute_snqi(metrics_success, weights, baseline_stats)
    assert score_success >= score_fail


def test_clamping_effect(weights, baseline_stats):
    """Values beyond p95 should not increase penalty beyond normalized 1.0 for negative terms."""
    metrics = {
        "success": True,
        "time_to_goal_norm": 0.5,
        "collisions": 100,  # huge
        "near_misses": 100,  # huge
        "force_exceed_events": 500,
        "jerk_mean": 50.0,
        "comfort_exposure": 0.2,
    }
    score_extreme = compute_snqi(metrics, weights, baseline_stats)

    # Same metrics but trimmed to just at/near p95 threshold values
    metrics_p95 = {
        "success": True,
        "time_to_goal_norm": 0.5,
        "collisions": 3,
        "near_misses": 6,
        "force_exceed_events": 10,
        "jerk_mean": 0.55,
        "comfort_exposure": 0.2,
    }
    score_p95 = compute_snqi(metrics_p95, weights, baseline_stats)
    # Differences should be very small (clamped) — allow small numerical tolerance
    assert isclose(score_extreme, score_p95, rel_tol=1e-9, abs_tol=1e-9)
