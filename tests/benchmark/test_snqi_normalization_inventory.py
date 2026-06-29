"""Tests for the SNQI per-term normalization inventory (issue #3699).

These tests prove the diagnostic surfaces the mixed raw/baseline-normalized
condition *fail-closed* and that adding the inventory does not change any
``compute_snqi`` output. The key anti-drift guarantee is
``test_inventory_reconstructs_compute_snqi``: the SNQI score rebuilt from the
inventory's term table must equal ``compute_snqi`` exactly, so the inventory's
scaling labels cannot silently diverge from the scoring code they describe.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.snqi.compute import compute_snqi
from robot_sf.benchmark.snqi.normalization_inventory import (
    SCALING_BASELINE_NORMALIZED,
    SCALING_RAW,
    SNQI_TERM_SCALING,
    build_snqi_normalization_inventory,
    format_normalization_report,
    scaled_term_value,
)

# Baseline stats covering exactly the four count-type penalty terms that
# ``compute_snqi`` routes through ``normalize_metric``.
_BASELINE_STATS = {
    "collisions": {"med": 1.0, "p95": 5.0},
    "near_misses": {"med": 2.0, "p95": 8.0},
    "force_exceed_events": {"med": 0.0, "p95": 4.0},
    "jerk_mean": {"med": 0.1, "p95": 1.0},
}

_WEIGHTS = {
    "w_success": 1.0,
    "w_time": 0.8,
    "w_collisions": 2.0,
    "w_near": 1.0,
    "w_comfort": 0.5,
    "w_force_exceed": 1.5,
    "w_jerk": 0.3,
}

_METRICS = {
    "success": 1.0,
    "time_to_goal_norm": 0.7,
    "collisions": 3.0,
    "near_misses": 4.0,
    "comfort_exposure": 0.25,
    "force_exceed_events": 2.0,
    "jerk_mean": 0.5,
}


def test_inventory_reconstructs_compute_snqi():
    """Rebuilding the score from the inventory equals ``compute_snqi`` exactly.

    This anchors the inventory's scaling labels to the real scoring code: if a
    term's ``scaling``/``sign``/``metric_key``/``default`` ever drifts from
    ``compute_snqi``, this reconstruction diverges and the test fails.
    """
    reconstructed = 0.0
    for term in SNQI_TERM_SCALING:
        value = scaled_term_value(term, _METRICS, _BASELINE_STATS)
        reconstructed += term.sign * _WEIGHTS[term.weight_name] * value

    expected = compute_snqi(_METRICS, _WEIGHTS, _BASELINE_STATS)
    assert reconstructed == pytest.approx(expected)


def test_reconstruction_holds_with_missing_metrics():
    """Per-term defaults match ``compute_snqi`` when metrics are absent."""
    sparse = {"success": 1.0}  # every other term falls back to its default
    reconstructed = sum(
        term.sign * _WEIGHTS[term.weight_name] * scaled_term_value(term, sparse, _BASELINE_STATS)
        for term in SNQI_TERM_SCALING
    )
    expected = compute_snqi(sparse, _WEIGHTS, _BASELINE_STATS)
    assert reconstructed == pytest.approx(expected)


def test_mixed_scale_is_surfaced():
    """The inventory flags that penalty terms span raw and normalized scales."""
    inv = build_snqi_normalization_inventory(_BASELINE_STATS)
    assert inv.mixed_scale is True
    assert inv.is_consistent is False
    raw_penalty = {t.term for t in inv.raw_penalty_terms}
    assert raw_penalty == {"time", "comfort"}
    normalized_penalty = {t.term for t in inv.normalized_penalty_terms}
    assert normalized_penalty == {"collisions", "near", "force_exceed", "jerk"}


def test_unbounded_terms_are_the_raw_penalty_terms():
    """``time`` and ``comfort`` are reported as the unbounded penalty terms."""
    inv = build_snqi_normalization_inventory(_BASELINE_STATS)
    assert {t.term for t in inv.unbounded_terms} == {"time", "comfort"}


def test_raw_terms_are_unbounded_in_behavior():
    """A raw term's scaled value grows without bound; normalized stays in [0, 1].

    This is the behavioral counterpart to the static ``bounded`` labels: it
    confirms the labels describe real ``compute_snqi`` behavior.
    """
    time_term = next(t for t in SNQI_TERM_SCALING if t.term == "time")
    coll_term = next(t for t in SNQI_TERM_SCALING if t.term == "collisions")

    huge = {"time_to_goal_norm": 1000.0, "collisions": 1000.0}
    assert scaled_term_value(time_term, huge, _BASELINE_STATS) == pytest.approx(1000.0)
    # Baseline-normalized term saturates at the clamp upper bound.
    assert scaled_term_value(coll_term, huge, _BASELINE_STATS) == pytest.approx(1.0)


def test_missing_baseline_coverage_is_surfaced_fail_closed():
    """Normalized terms with no median/p95 are reported, not silently passed."""
    partial = {"collisions": {"med": 1.0, "p95": 5.0}}  # near/force/jerk missing
    inv = build_snqi_normalization_inventory(partial)
    missing = {t.metric_key for t in inv.missing_baseline_coverage}
    assert missing == {"near_misses", "force_exceed_events", "jerk_mean"}


def test_no_baseline_stats_reports_all_normalized_uncovered():
    """Without baseline stats, every normalized term is reported as uncovered."""
    inv = build_snqi_normalization_inventory(None)
    missing = {t.metric_key for t in inv.missing_baseline_coverage}
    assert missing == {"near_misses", "force_exceed_events", "jerk_mean", "collisions"}


def test_inventory_does_not_change_compute_snqi_outputs():
    """Building the inventory must not mutate metrics/weights/stats or scores."""
    before = compute_snqi(_METRICS, _WEIGHTS, _BASELINE_STATS)
    _ = build_snqi_normalization_inventory(_BASELINE_STATS)
    after = compute_snqi(_METRICS, _WEIGHTS, _BASELINE_STATS)
    assert before == after


def test_term_table_covers_every_weight():
    """Every SNQI weight has exactly one term; signs match reward/penalty roles."""
    weight_names = [t.weight_name for t in SNQI_TERM_SCALING]
    assert sorted(weight_names) == sorted(_WEIGHTS)
    success = next(t for t in SNQI_TERM_SCALING if t.term == "success")
    assert success.sign == 1
    assert all(t.sign == -1 for t in SNQI_TERM_SCALING if t.term != "success")
    # Scaling values are restricted to the two known regimes.
    assert all(t.scaling in {SCALING_RAW, SCALING_BASELINE_NORMALIZED} for t in SNQI_TERM_SCALING)


def test_to_dict_is_json_serializable_and_self_consistent():
    """The payload round-trips through JSON and reports the mixed-scale flag."""
    import json

    inv = build_snqi_normalization_inventory(_BASELINE_STATS)
    payload = inv.to_dict()
    encoded = json.loads(json.dumps(payload))
    assert encoded["mixed_scale"] is True
    assert set(encoded["raw_penalty_terms"]) == {"time", "comfort"}
    assert encoded["is_consistent"] is False
    status_by_term = {term["term"]: term["normalization_status"] for term in encoded["terms"]}
    assert status_by_term["success"] == "raw_bounded"
    assert status_by_term["time"] == "raw_unbounded"
    assert status_by_term["comfort"] == "raw_unbounded"
    assert status_by_term["collisions"] == "baseline_normalized_bounded"


def test_format_report_is_human_readable():
    """The text report names the issue and every term."""
    inv = build_snqi_normalization_inventory(_BASELINE_STATS)
    text = format_normalization_report(inv)
    assert "issue #3699" in text
    for term in SNQI_TERM_SCALING:
        assert term.term in text
