"""Tests for benchmark-facing safety shield decision contracts."""

from __future__ import annotations

import pytest

from robot_sf.planner.safety_shield import (
    ShieldDecision,
    shield_metrics_from_stats,
    update_shield_stats,
)


def test_shield_decision_metadata_separates_proposed_filtered_and_prediction_fields() -> None:
    """ShieldDecision should serialize action, constraint, fallback, and prediction details."""
    decision = ShieldDecision(
        proposed_action=(0.8, 0.0),
        filtered_action=(0.0, 1.0),
        decision_label="fallback_safe",
        intervention_reason="ppo_action_violated_short_horizon_constraints",
        violated_constraints=("pedestrian_clearance", "time_to_collision"),
        prediction_source="short_horizon_rollout",
        prediction_horizon_steps=6,
        prediction_dt=0.2,
        uncertainty_metadata={"mode": "deterministic"},
        calibration_metadata={"status": "not_calibrated"},
        fallback_controller_state={"policy": "RiskDWAPlannerAdapter"},
        selected_evaluation={"safe": True, "min_ped_clear": 0.9},
        proposed_evaluation={"safe": False, "min_ped_clear": 0.2},
    )

    metadata = decision.to_metadata()

    assert metadata["schema_version"] == "shield-decision.v1"
    assert metadata["proposed_action"] == [0.8, 0.0]
    assert metadata["filtered_action"] == [0.0, 1.0]
    assert metadata["intervened"] is True
    assert metadata["override_applied"] is True
    assert metadata["hard_constraint_violation"] is False
    assert metadata["violated_constraints"] == ["pedestrian_clearance", "time_to_collision"]
    assert metadata["prediction"]["source"] == "short_horizon_rollout"
    assert metadata["prediction"]["calibration"]["status"] == "not_calibrated"


def test_update_shield_stats_and_metrics_track_interventions_and_violations() -> None:
    """Stats should expose benchmark rates without conflating them with reward/success."""
    stats: dict[str, object] = {}
    update_shield_stats(
        stats,
        ShieldDecision(
            proposed_action=(0.4, 0.0),
            filtered_action=(0.4, 0.0),
            decision_label="ppo_safe",
            intervention_reason="proposed_action_satisfies_shield",
        ),
    )
    update_shield_stats(
        stats,
        ShieldDecision(
            proposed_action=(0.8, 0.0),
            filtered_action=(0.0, 0.0),
            decision_label="stop_best_effort",
            intervention_reason="no_safe_command_available",
            violated_constraints=("pedestrian_clearance",),
            selected_evaluation={"safe": False},
            proposed_evaluation={"safe": False},
        ),
    )

    assert stats["decision_count"] == 2
    assert stats["pass_through_count"] == 1
    assert stats["intervention_count"] == 1
    assert stats["override_count"] == 1
    assert stats["hard_constraint_violation_count"] == 1
    assert stats["decision_counts"]["ppo_safe"] == 1
    assert stats["decision_counts"]["stop_best_effort"] == 1

    metrics = shield_metrics_from_stats(stats)
    assert metrics["shield_decision_count"] == 2
    assert metrics["shield_intervention_rate"] == pytest.approx(0.5)
    assert metrics["shield_override_rate"] == pytest.approx(0.5)
    assert metrics["shield_hard_constraint_violation_rate"] == pytest.approx(0.5)
