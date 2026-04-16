"""Tests for SNQI calibration robustness helpers."""

from __future__ import annotations

from robot_sf.benchmark.snqi.calibration import (
    analyze_snqi_calibration,
    derive_planner_rows_from_episodes,
    normalization_anchor_variants,
    weight_variants,
)


def _weights() -> dict[str, float]:
    return {
        "w_success": 0.20,
        "w_time": 0.10,
        "w_collisions": 0.15,
        "w_near": 0.25,
        "w_comfort": 0.15,
        "w_force_exceed": 0.10,
        "w_jerk": 0.05,
    }


def _baseline() -> dict[str, dict[str, float]]:
    return {
        "time_to_goal_norm": {"med": 0.5, "p95": 1.0},
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 0.0, "p95": 4.0},
        "force_exceed_events": {"med": 0.0, "p95": 6.0},
        "jerk_mean": {"med": 0.05, "p95": 0.5},
    }


def _episodes() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(4):
        rows.append(
            {
                "planner_key": "safe",
                "kinematics": "differential_drive",
                "metrics": {
                    "success": 1.0,
                    "time_to_goal_norm": 0.55 + idx * 0.02,
                    "collisions": 0.0,
                    "near_misses": float(idx),
                    "comfort_exposure": 0.05 + idx * 0.01,
                    "force_exceed_events": float(idx),
                    "jerk_mean": 0.06 + idx * 0.01,
                },
            }
        )
        rows.append(
            {
                "planner_key": "risky",
                "kinematics": "differential_drive",
                "metrics": {
                    "success": 0.0 if idx % 2 else 1.0,
                    "time_to_goal_norm": 0.85 + idx * 0.03,
                    "collisions": 1.0 if idx == 3 else 0.0,
                    "near_misses": 3.0 + idx,
                    "comfort_exposure": 0.20 + idx * 0.02,
                    "force_exceed_events": 4.0 + idx,
                    "jerk_mean": 0.25 + idx * 0.03,
                },
            }
        )
    return rows


def test_weight_variants_preserve_simplex_sum() -> None:
    """Local perturbation variants should preserve the v3 total weight mass."""
    variants = weight_variants(_weights(), epsilon=0.2)
    expected_total = sum(_weights().values())
    assert "local_w_success_up" in variants
    assert "component_subset_no_jerk" in variants
    for weights in variants.values():
        assert abs(sum(weights.values()) - expected_total) < 1e-12


def test_normalization_anchor_variants_include_dataset_anchors() -> None:
    """Anchor variants should derive deterministic dataset quantile baselines."""
    variants = normalization_anchor_variants(_episodes(), _baseline())
    assert set(variants) == {
        "v3_fixed",
        "dataset_median_p95",
        "dataset_median_p90",
        "dataset_median_max",
        "dataset_p25_p75",
    }
    assert variants["dataset_median_p95"]["near_misses"]["p95"] > 0.0
    assert (
        variants["dataset_p25_p75"]["time_to_goal_norm"]["p95"]
        > variants["dataset_p25_p75"]["time_to_goal_norm"]["med"]
    )


def test_analyze_snqi_calibration_reports_recommendation_and_variants() -> None:
    """Calibration analysis should compare weight and anchor variants against v3."""
    episodes = _episodes()
    payload = analyze_snqi_calibration(
        episodes,
        weights=_weights(),
        baseline=_baseline(),
        planner_rows=derive_planner_rows_from_episodes(episodes),
        epsilon=0.15,
    )

    assert payload["schema_version"] == "snqi-calibration-analysis.v1"
    assert payload["episodes"] == len(episodes)
    assert payload["recommendation"]["decision"] in {
        "keep_v3_fixed",
        "demote_snqi_further",
        "propose_candidate_v4_contract",
    }
    variants = {row["variant"]: row for row in payload["variants"]}
    assert variants["v3_fixed"]["variant_type"] == "baseline"
    assert variants["dataset_median_p90"]["variant_type"] == "anchor"
    assert variants["local_w_near_up"]["variant_type"] == "weight"
    assert payload["sensitivity_summary"]["local_weight_min_planner_rank_correlation"] >= -1.0
