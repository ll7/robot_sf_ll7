"""Tests for latest-W&B PPO evaluation helper."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import yaml

from scripts.tools import evaluate_latest_ppo_candidate as latest_eval

if TYPE_CHECKING:
    from pathlib import Path


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


def test_build_benchmark_algo_config_threads_predictive_settings(tmp_path: Path) -> None:
    """Temporary benchmark PPO config should inherit predictive settings from training config."""
    training_config = tmp_path / "train.yaml"
    training_config.write_text(
        yaml.safe_dump(
            {
                "env_overrides": {
                    "predictive_foresight_enabled": True,
                    "predictive_foresight_model_id": "predictive_proxy_selected_v2_full",
                    "predictive_foresight_horizon_steps": 8,
                    "predictive_foresight_max_agents": 16,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "algo.yaml"
    latest_eval._build_benchmark_algo_config(
        training_config_path=training_config,
        checkpoint_path=tmp_path / "model.zip",
        output_path=output_path,
    )
    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload["model_path"].endswith("model.zip")
    assert payload["obs_mode"] == "dict"
    assert payload["predictive_foresight_enabled"] is True
    assert payload["predictive_foresight_model_id"] == "predictive_proxy_selected_v2_full"


def test_benchmark_summary_flags_contradictions_and_gate() -> None:
    """Benchmark gate should reject contradictory collision-plus-success records."""
    records = [
        {
            "scenario_id": "s1",
            "seed": 1,
            "termination_reason": "success",
            "metrics": {"success_rate": 1.0, "collision_rate": 0.0},
        },
        {
            "scenario_id": "s2",
            "seed": 2,
            "termination_reason": "collision",
            "metrics": {"success_rate": 1.0, "collision_rate": 1.0},
        },
    ]
    summary = latest_eval._benchmark_summary(records)
    assert summary["gate_pass"] is False
    assert summary["problem_episode_count"] == 2
    assert summary["weakest_scenarios"][0]["scenario_id"] == "s2"


def test_registry_entry_from_candidate_uses_selection_metadata(tmp_path: Path) -> None:
    """Promotion registry entries should carry W&B provenance and training config path."""
    entry = latest_eval._registry_entry_from_candidate(
        model_id="ppo_candidate",
        display_name="PPO Candidate",
        selection={
            "run_id": "abc123",
            "run_path": "ll7/robot_sf/abc123",
            "run_name": "ppo_candidate_run",
        },
        checkpoint_path=tmp_path / "model.zip",
        training_config=tmp_path / "train.yaml",
        decision={
            "policy_gate": {"success_rate": 0.9, "collision_rate": 0.05},
            "benchmark_gate": {"success_rate": 0.85, "collision_rate": 0.08},
        },
    )
    assert entry["model_id"] == "ppo_candidate"
    assert entry["wandb_run_path"] == "ll7/robot_sf/abc123"
    assert entry["wandb_entity"] == "ll7"
    assert entry["wandb_project"] == "robot_sf"
    assert "Promoted via scripts/tools/evaluate_latest_ppo_candidate.py." in entry["notes"][0]


def test_write_promotion_report_includes_benchmark_and_decision(tmp_path: Path) -> None:
    """Promotion report payload should include both gates and decision outcome."""
    report_json, report_md = latest_eval._write_promotion_report(
        output_root=tmp_path,
        selection={
            "run_id": "abc123",
            "run_name": "ppo_candidate_run",
            "run_path": "ll7/robot_sf/abc123",
            "state": "finished",
            "created_at": "2026-03-12T10:00:00Z",
            "downloaded_model": "output/model.zip",
        },
        policy_result={
            "summary_json": "policy/summary.json",
            "report_md": "policy/report.md",
            "video_root": "videos",
        },
        benchmark_result={
            "episodes_jsonl": "benchmark/episodes.jsonl",
            "summary_json": "benchmark/summary.json",
            "algo_config": "benchmark/ppo_candidate_algo.yaml",
        },
        decision={
            "policy_gate": {
                "success_rate": 0.9,
                "collision_rate": 0.05,
                "gate_pass": True,
            },
            "benchmark_gate": {
                "success_rate": 0.85,
                "collision_rate": 0.08,
                "problem_episode_count": 0,
                "gate_pass": True,
                "weakest_scenarios": [
                    {"scenario_id": "s1", "success_rate": 0.5, "collision_rate": 0.2}
                ],
            },
            "promote": True,
            "rationale": "both policy-analysis and benchmark gates passed",
            "registry_updated": True,
        },
    )
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["decision"]["promote"] is True
    assert "Benchmark Gate" in report_md.read_text(encoding="utf-8")


def test_decision_exit_code_maps_gate_result() -> None:
    """Successful promotion decisions should exit zero; failed gates should exit two."""
    assert latest_eval._decision_exit_code({"promote": True}) == 0
    assert latest_eval._decision_exit_code({"promote": False}) == 2
