"""Tests for SNQI campaign diagnostics and contract helpers."""

from __future__ import annotations

from robot_sf.benchmark.snqi.campaign_contract import (
    SnqiContractThresholds,
    calibrate_weights,
    compute_baseline_stats_from_episodes,
    evaluate_snqi_contract,
    sanitize_baseline_stats,
)


def _sample_rows() -> list[dict[str, object]]:
    return [
        {
            "planner_key": "goal",
            "kinematics": "differential_drive",
            "success_mean": 1.0,
            "collisions_mean": 0.0,
            "near_misses_mean": 0.0,
            "comfort_exposure_mean": 0.1,
        },
        {
            "planner_key": "social_force",
            "kinematics": "differential_drive",
            "success_mean": 0.2,
            "collisions_mean": 1.0,
            "near_misses_mean": 2.0,
            "comfort_exposure_mean": 0.6,
        },
    ]


def _sample_episodes() -> list[dict[str, object]]:
    return [
        {
            "planner_key": "goal",
            "kinematics": "differential_drive",
            "metrics": {
                "success": 1.0,
                "time_to_goal_norm": 0.2,
                "collisions": 0.0,
                "near_misses": 0.0,
                "comfort_exposure": 0.1,
                "force_exceed_events": 0.0,
                "jerk_mean": 0.1,
            },
        },
        {
            "planner_key": "social_force",
            "kinematics": "differential_drive",
            "metrics": {
                "success": 0.0,
                "time_to_goal_norm": 0.9,
                "collisions": 1.0,
                "near_misses": 2.0,
                "comfort_exposure": 0.7,
                "force_exceed_events": 2.0,
                "jerk_mean": 0.8,
            },
        },
    ]


def _baseline() -> dict[str, dict[str, float]]:
    return {
        "time_to_goal_norm": {"med": 0.1, "p95": 1.0},
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 0.0, "p95": 1.0},
        "force_exceed_events": {"med": 0.0, "p95": 2.0},
        "jerk_mean": {"med": 0.1, "p95": 0.9},
    }


def test_sanitize_baseline_stats_adjusts_degenerate_entries() -> None:
    """Degenerate med==p95 baseline entries should be widened deterministically."""
    baseline, warnings = sanitize_baseline_stats(
        {
            "collisions": {"med": 0.0, "p95": 0.0},
            "near_misses": {"med": 1.0, "p95": 1.0},
        }
    )
    assert baseline["collisions"]["p95"] > baseline["collisions"]["med"]
    assert baseline["near_misses"]["p95"] > baseline["near_misses"]["med"]
    assert warnings


def test_evaluate_snqi_contract_returns_pass_for_well_aligned_data() -> None:
    """Contract should pass when ranking alignment and separation are both healthy."""
    evaluation = evaluate_snqi_contract(
        _sample_rows(),
        _sample_episodes(),
        weights={
            "w_success": 0.2,
            "w_time": 0.2,
            "w_collisions": 0.2,
            "w_near": 0.1,
            "w_comfort": 0.1,
            "w_force_exceed": 0.1,
            "w_jerk": 0.1,
        },
        baseline=_baseline(),
        thresholds=SnqiContractThresholds(
            rank_alignment_warn=0.5,
            rank_alignment_fail=0.3,
            outcome_separation_warn=0.05,
            outcome_separation_fail=0.0,
        ),
    )
    assert evaluation.status == "pass"
    assert evaluation.rank_alignment_spearman > 0.5
    assert evaluation.outcome_separation > 0.05


def test_calibrate_weights_is_deterministic_for_fixed_seed() -> None:
    """Calibration should return identical recommendation for identical seed/trials."""
    first = calibrate_weights(
        _sample_rows(),
        _sample_episodes(),
        baseline=_baseline(),
        seed=123,
        trials=500,
    )
    second = calibrate_weights(
        _sample_rows(),
        _sample_episodes(),
        baseline=_baseline(),
        seed=123,
        trials=500,
    )
    assert first["weights"] == second["weights"]
    assert first["metrics"] == second["metrics"]


def test_compute_baseline_stats_from_episodes_propagates_adjustment_warnings() -> None:
    """Episode-derived baseline builder should return degeneracy adjustment warnings."""
    baseline, warnings = compute_baseline_stats_from_episodes(
        [
            {
                "metrics": {
                    "time_to_goal_norm": 0.0,
                    "collisions": 0.0,
                    "near_misses": 0.0,
                    "force_exceed_events": 0.0,
                    "jerk_mean": 0.0,
                }
            }
        ]
    )
    assert baseline["collisions"]["p95"] > baseline["collisions"]["med"]
    assert warnings
