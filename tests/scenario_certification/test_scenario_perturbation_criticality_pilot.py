"""Tests for scenario perturbation criticality pilot aggregation."""

from __future__ import annotations

import pytest

from scripts.validation.run_scenario_perturbation_criticality_pilot import (
    build_pair_table,
    classify_episode_status,
    summarize_pairs,
)


def test_classify_episode_status_separates_fallback_and_invalid_rows() -> None:
    """Fallback/degraded/invalid records should not count as completed evidence."""
    assert classify_episode_status(None) == "missing"
    assert classify_episode_status({"scenario_exclusion": {"status": "invalid"}}) == "invalid"
    assert (
        classify_episode_status({"algorithm_metadata": {"planner": {"execution_mode": "fallback"}}})
        == "fallback"
    )
    assert (
        classify_episode_status({"algorithm_metadata": {"planner": {"execution_mode": "degraded"}}})
        == "degraded"
    )
    assert classify_episode_status({"termination_reason": "error"}) == "failed"
    assert classify_episode_status({"termination_reason": "success"}) == "completed"


def test_build_pair_table_computes_completed_pair_deltas() -> None:
    """No-op and route-offset rows should pair by source scenario, planner, and seed."""
    metadata = {
        "demo_noop": {
            "source_scenario_id": "demo",
            "variant_id": "demo_noop",
            "family": "noop",
        },
        "demo_offset": {
            "source_scenario_id": "demo",
            "variant_id": "demo_offset",
            "family": "robot_route_offset",
        },
    }
    records = {
        "goal": [
            {
                "scenario_id": "demo_noop",
                "seed": 111,
                "termination_reason": "success",
                "metrics": {"min_distance": 2.0},
            },
            {
                "scenario_id": "demo_offset",
                "seed": 111,
                "termination_reason": "max_steps",
                "metrics": {"min_distance": 1.5},
            },
        ]
    }

    pairs = build_pair_table(records, metadata)

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["pair_status"] == "completed"
    assert pair["success_delta"] == pytest.approx(-1.0)
    assert pair["timeout_delta"] == pytest.approx(1.0)
    assert pair["collision_delta"] == pytest.approx(0.0)
    assert pair["min_distance_delta"] == pytest.approx(-0.5)
    assert summarize_pairs(pairs)["mean_deltas_completed_pairs"]["success_delta"] == pytest.approx(
        -1.0
    )


def test_build_pair_table_excludes_fallback_rows_from_completed_deltas() -> None:
    """Fallback perturbed rows should be visible but excluded from completed-pair means."""
    metadata = {
        "demo_noop": {
            "source_scenario_id": "demo",
            "variant_id": "demo_noop",
            "family": "noop",
        },
        "demo_offset": {
            "source_scenario_id": "demo",
            "variant_id": "demo_offset",
            "family": "robot_route_offset",
        },
    }
    records = {
        "orca": [
            {"scenario_id": "demo_noop", "seed": 111, "termination_reason": "success"},
            {
                "scenario_id": "demo_offset",
                "seed": 111,
                "termination_reason": "success",
                "algorithm_metadata": {"mode": "fallback"},
            },
        ]
    }

    pairs = build_pair_table(records, metadata)
    summary = summarize_pairs(pairs)

    assert pairs[0]["pair_status"] == "excluded"
    assert pairs[0]["perturbed_status"] == "fallback"
    assert summary["status_counts"] == {"excluded": 1}
    assert summary["mean_deltas_completed_pairs"] == {}
