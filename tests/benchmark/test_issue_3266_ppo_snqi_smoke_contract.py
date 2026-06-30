"""Regression tests for issue #3266 PPO/SNQI blocker-resolution evidence."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/benchmarks/issue_3266_scenario_horizon_ppo_snqi_smoke.yaml"
SUMMARY_PATH = ROOT / "docs/context/evidence/issue_3266_ppo_snqi_smoke_2026-06-23/summary.json"


def test_issue_3266_smoke_config_stays_small_and_non_paper_facing() -> None:
    """The issue #3266 rerun remains a smallest-slice blocker-resolution smoke."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["paper_facing"] is False
    assert config["scenario_candidates"] == ["francis2023_blind_corner"]
    assert config["seed_policy"] == {
        "mode": "fixed-list",
        "seeds": [219],
        "seed_sets_path": "configs/benchmarks/seed_sets_v1.yaml",
    }
    assert [planner["key"] for planner in config["planners"]] == ["goal", "ppo"]

    ppo = next(planner for planner in config["planners"] if planner["key"] == "ppo")
    assert ppo["planner_group"] == "experimental"
    assert ppo["algo"] == "ppo"
    assert ppo["benchmark_profile"] == "experimental"
    assert ppo["adapter_impact_eval"] is True

    ppo_config_path = ROOT / ppo["algo_config"]
    ppo_config = yaml.safe_load(ppo_config_path.read_text(encoding="utf-8"))
    assert ppo_config["obs_mode"] == "dict"
    assert ppo_config["fallback_to_goal"] is False

    snqi_contract = config["snqi_contract"]
    assert snqi_contract["enabled"] is True
    assert snqi_contract["enforcement"] == "warn"


def test_issue_3266_summary_is_valid_smoke_not_results_evidence() -> None:
    """The tracked summary must not overstate the blocker-resolution smoke result."""
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    assert summary["issue"] == 3266
    assert summary["source_config"] == str(CONFIG_PATH.relative_to(ROOT))
    assert summary["evidence_status"] == "valid_blocker_resolution_smoke"
    assert "not paper-facing Results evidence" in summary["claim_boundary"]

    campaign = summary["campaign"]
    assert campaign["status"] == "benchmark_success"
    assert campaign["benchmark_success"] is True
    assert campaign["total_runs"] == 2
    assert campaign["successful_runs"] == 2
    assert campaign["unexpected_failed_runs"] == 0
    assert campaign["row_status_summary"]["unexpected_failed_rows"] == 0
    assert campaign["row_status_summary"]["fallback_or_degraded_rows"] == 0

    ppo = next(row for row in summary["planner_rows"] if row["planner_key"] == "ppo")
    assert ppo["execution_mode"] == "native"
    assert ppo["readiness_status"] == "native"
    assert ppo["benchmark_success"] is True
    assert ppo["learned_policy_contract_status"] == "pass"

    snqi_contract = summary["snqi_contract"]
    assert snqi_contract["contract_status"] == "pass"
    assert (
        snqi_contract["positioning_recommendation"] == "downgrade_to_appendix_or_implementation_aid"
    )
    assert (
        "full paper-facing scenario-horizon Results claim"
        in summary["interpretation"]["not_resolved"]
    )
