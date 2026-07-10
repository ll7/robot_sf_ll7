"""SNQI zero-weight NaN contract (issue #5132).

`compute_all_metrics()` emits ``comfort_exposure = NaN`` when pedestrian force
data is absent. In the SNQI scalarization the term ``w_comfort * comfort_exposure``
collapsed to ``NaN`` via the IEEE 754 rule ``0.0 * NaN = NaN``, silently poisoning
the whole score even when ``w_comfort = 0.0`` (the explicit "exclude this term"
setting).

These tests pin the contract: a zero-weighted term must never propagate ``NaN``,
while a non-zero weight still multiplies through so genuine ``NaN`` signals are
not masked.
"""

from __future__ import annotations

import math
from math import isclose

import pytest

from robot_sf.benchmark.snqi.compute import (
    _weighted_term,
    compute_snqi_v0,
    compute_snqi_v1,
)
from robot_sf.benchmark.snqi.types import SNQIWeights

NaN = float("nan")


# A baseline that provides coverage for every metric the two score versions read.
_BASELINE = {
    "collisions": {"med": 0.0, "p95": 3.0},
    "near_misses": {"med": 1.0, "p95": 6.0},
    "force_exceed_events": {"med": 0.0, "p95": 10.0},
    "jerk_mean": {"med": 0.05, "p95": 0.55},
    "comfort_exposure": {"med": 0.0, "p95": 0.5},
    "time_to_goal_norm": {"med": 0.3, "p95": 1.2},
}


def _zero_comfort_force_weights() -> dict[str, float]:
    """Canonical weights but with comfort/force terms excluded (issue scenario)."""
    return {
        "w_success": 1.0,
        "w_time": 0.8,
        "w_collisions": 2.0,
        "w_near": 1.0,
        "w_comfort": 0.0,
        "w_force_exceed": 0.0,
        "w_jerk": 0.3,
    }


# ---------------------------------------------------------------------------
# Unit: helper
# ---------------------------------------------------------------------------


def test_weighted_term_zero_weight_short_circuits_nan():
    """Zero weight returns 0.0 even when the value is NaN."""
    assert _weighted_term(0.0, NaN) == 0.0
    assert _weighted_term(0.0, 1.0) == 0.0
    assert _weighted_term(0.0, float("inf")) == 0.0


def test_weighted_term_nonzero_weight_propagates_nan():
    """A non-zero weight still multiplies through (NaN signal is not masked)."""
    assert math.isnan(_weighted_term(1.5, NaN))
    assert isclose(_weighted_term(2.0, 3.0), 6.0)


# ---------------------------------------------------------------------------
# Regression: the exact issue #5132 scenario (v0)
# ---------------------------------------------------------------------------


def test_v0_zero_weight_nan_comfort_force_yields_finite_score():
    """Reproduces issue #5132: NaN comfort/force under zero weights must be finite."""
    metrics = {
        "success": True,
        "time_to_goal_norm": 0.5,
        "collisions": 0,
        "near_misses": 2,
        # force data legitimately absent -> NaN metrics, exactly as emitted by
        # compute_all_metrics() on an episode without pedestrian force data.
        "comfort_exposure": NaN,
        "force_exceed_events": NaN,
        "jerk_mean": 0.2,
    }
    weights = _zero_comfort_force_weights()
    score = compute_snqi_v0(metrics, weights, _BASELINE)
    assert math.isfinite(score), f"SNQI score must be finite, got {score}"


def test_v0_zero_weight_nan_metric_matches_force_free_metrics():
    """A zero-weighted NaN term must equal the same score computed with 0.0 values.

    This mirrors the maintainer workaround (build a force-free metrics dict) and
    asserts the two paths now agree.
    """
    weights = _zero_comfort_force_weights()
    metrics_nan = {
        "success": 1.0,
        "time_to_goal_norm": 0.6,
        "collisions": 1,
        "near_misses": 3,
        "comfort_exposure": NaN,
        "force_exceed_events": NaN,
        "jerk_mean": 0.3,
    }
    metrics_zero = {
        "success": 1.0,
        "time_to_goal_norm": 0.6,
        "collisions": 1,
        "near_misses": 3,
        "comfort_exposure": 0.0,
        "force_exceed_events": 0.0,
        "jerk_mean": 0.3,
    }
    assert isclose(
        compute_snqi_v0(metrics_nan, weights, _BASELINE),
        compute_snqi_v0(metrics_zero, weights, _BASELINE),
        rel_tol=0.0,
        abs_tol=0.0,
    )


@pytest.mark.parametrize(
    "nan_key",
    [
        "comfort_exposure",
        "time_to_goal_norm",
        "collisions",
        "near_misses",
        "force_exceed_events",
        "jerk_mean",
    ],
)
def test_v0_each_penalty_term_zero_weight_is_nan_safe(nan_key):
    """Every penalty term must be NaN-safe when its own weight is zero."""
    metrics = {
        "success": True,
        "time_to_goal_norm": 0.5,
        "collisions": 0,
        "near_misses": 0,
        "comfort_exposure": 0.1,
        "force_exceed_events": 0,
        "jerk_mean": 0.1,
    }
    metrics[nan_key] = NaN
    weights = {
        "w_success": 1.0,
        "w_time": 0.8,
        "w_collisions": 2.0,
        "w_near": 1.0,
        "w_comfort": 0.5,
        "w_force_exceed": 1.5,
        "w_jerk": 0.3,
    }
    weight_for = {
        "comfort_exposure": "w_comfort",
        "time_to_goal_norm": "w_time",
        "collisions": "w_collisions",
        "near_misses": "w_near",
        "force_exceed_events": "w_force_exceed",
        "jerk_mean": "w_jerk",
    }
    weights[weight_for[nan_key]] = 0.0
    score = compute_snqi_v0(metrics, weights, _BASELINE)
    assert math.isfinite(score), f"{nan_key} (weight 0.0) leaked NaN -> {score}"


def test_v0_zero_weight_success_term_is_nan_safe():
    """The success term with zero weight must not propagate a NaN success either."""
    metrics = {"success": NaN, "time_to_goal_norm": 0.5}
    weights = {
        "w_success": 0.0,
        "w_time": 0.8,
        "w_collisions": 0.0,
        "w_near": 0.0,
        "w_comfort": 0.0,
        "w_force_exceed": 0.0,
        "w_jerk": 0.0,
    }
    score = compute_snqi_v0(metrics, weights, _BASELINE)
    assert math.isfinite(score)


# ---------------------------------------------------------------------------
# Regression-free behavior: nonzero weight still propagates NaN (no masking)
# ---------------------------------------------------------------------------


def test_v0_nonzero_weight_propagates_nan():
    """A NaN metric under a non-zero weight must still collapse the score to NaN.

    Guards against over-broadly masking NaN (which would hide real signals).
    """
    metrics = {
        "success": True,
        "time_to_goal_norm": 0.5,
        "comfort_exposure": NaN,
    }
    weights = {
        "w_success": 1.0,
        "w_time": 0.8,
        "w_collisions": 0.0,
        "w_near": 0.0,
        "w_comfort": 0.5,  # non-zero -> NaN must propagate
        "w_force_exceed": 0.0,
        "w_jerk": 0.0,
    }
    assert math.isnan(compute_snqi_v0(metrics, weights, _BASELINE))


# ---------------------------------------------------------------------------
# v1 parity of the zero-weight contract
# ---------------------------------------------------------------------------


def test_v1_zero_weight_nan_term_yields_finite_score():
    """The opt-in v1 scorer shares the zero-weight NaN contract."""
    metrics = {
        "success": True,
        "time_to_goal_norm": 0.5,
        "collisions": 1,
        "near_misses": 3,
        "comfort_exposure": NaN,
        "force_exceed_events": 0,
        "jerk_mean": 0.2,
    }
    weights = _zero_comfort_force_weights()
    score = compute_snqi_v1(metrics, weights, _BASELINE)
    assert math.isfinite(score), f"SNQI-v1 score must be finite, got {score}"


def test_v1_zero_weight_nan_matches_force_free():
    """v1 zero-weight NaN term matches the force-free (0.0) variant."""
    weights = _zero_comfort_force_weights()
    common = {
        "success": 1.0,
        "time_to_goal_norm": 0.6,
        "collisions": 1,
        "near_misses": 3,
        "jerk_mean": 0.3,
    }
    metrics_nan = {**common, "comfort_exposure": NaN, "force_exceed_events": NaN}
    metrics_zero = {**common, "comfort_exposure": 0.0, "force_exceed_events": 0.0}
    assert isclose(
        compute_snqi_v1(metrics_nan, weights, _BASELINE),
        compute_snqi_v1(metrics_zero, weights, _BASELINE),
        rel_tol=0.0,
        abs_tol=0.0,
    )


# ---------------------------------------------------------------------------
# Concrete value: zero weight changes nothing vs plain finite arithmetic
# ---------------------------------------------------------------------------


def test_v0_zero_weight_finite_value_unchanged():
    """For finite values, zero-weight short-circuit equals the old 0*w==0.0 result."""
    metrics = {
        "success": True,
        "time_to_goal_norm": 0.5,
        "collisions": 1,
        "near_misses": 2,
        "comfort_exposure": 0.1,
        "force_exceed_events": 1,
        "jerk_mean": 0.2,
    }
    full_weights = {
        "w_success": 1.0,
        "w_time": 0.8,
        "w_collisions": 2.0,
        "w_near": 1.0,
        "w_comfort": 0.5,
        "w_force_exceed": 1.5,
        "w_jerk": 0.3,
    }
    expected = (
        1.0 * 1.0
        - 0.8 * 0.5
        - 2.0 * (1.0 / 3.0)
        - 1.0 * (1.0 / 5.0)
        - 0.5 * 0.1
        - 1.5 * (1.0 / 10.0)
        - 0.3 * ((0.2 - 0.05) / (0.55 - 0.05))
    )
    assert isclose(compute_snqi_v0(metrics, full_weights, _BASELINE), expected, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# SNQIWeights provenance type still imports/constructs (light smoke)
# ---------------------------------------------------------------------------


def test_snqi_weights_type_smoke():
    """Cheap import/construct smoke for the provenance type touched by neighbors."""
    w = SNQIWeights(
        weights_version="1.0",
        created_at="2026-07-10T00:00:00",
        git_sha="abcd1234",
        baseline_stats_path="computed",
        baseline_stats_hash="0" * 16,
        normalization_strategy="median_p95_clamp",
        bootstrap_params={"method": "canonical"},
        components=[
            "w_success",
            "w_time",
            "w_collisions",
            "w_near",
            "w_comfort",
            "w_force_exceed",
            "w_jerk",
        ],
        weights={"w_success": 1.0},
    )
    assert w.weights["w_success"] == 1.0
