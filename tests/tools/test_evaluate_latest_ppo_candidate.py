"""Tests for latest-W&B PPO evaluation helper."""

from __future__ import annotations

from scripts.tools import evaluate_latest_ppo_candidate as latest_eval


def test_promotion_summary_flags_gate_and_weakest_scenarios() -> None:
    """Promotion summary should derive gate fields from policy-analysis output."""
    payload = {
        "summary": {
            "episodes": 10,
            "success_rate": 0.85,
            "collision_rate": 0.05,
            "termination_reason_counts": {"success": 8, "collision": 1, "max_steps": 1},
            "metric_means": {"path_efficiency": 0.7},
        },
        "aggregates": {
            "s1": {"success_rate": {"mean": 1.0}, "collision_rate": {"mean": 0.0}},
            "s2": {"success_rate": {"mean": 0.2}, "collision_rate": {"mean": 0.4}},
        },
        "problem_episodes": [],
    }
    summary = latest_eval._promotion_summary(payload)
    assert summary["gate_pass"] is True
    assert summary["episodes"] == 10
    assert summary["weakest_scenarios"][0]["scenario_id"] == "s2"


def test_promotion_summary_rejects_problematic_candidate() -> None:
    """Weak success/collision rates should fail the promotion gate even if episodes are present."""
    payload = {
        "summary": {
            "episodes": 5,
            "success_rate": 0.6,
            "collision_rate": 0.2,
            "termination_reason_counts": {"success": 3, "collision": 1, "max_steps": 1},
            "metric_means": {},
        },
        "aggregates": {},
        "problem_episodes": [{"scenario_id": "bad", "seed": 1}],
    }
    summary = latest_eval._promotion_summary(payload)
    assert summary["gate_pass"] is False
    assert summary["problem_episode_count"] == 1
