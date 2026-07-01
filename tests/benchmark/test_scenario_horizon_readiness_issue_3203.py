"""Regression tests for issue #3203 scenario-horizon re-export configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.scenario_horizon_readiness import (
    DIAGNOSTIC_ONLY,
    classify_scenario_horizon_readiness,
)
from robot_sf.common.artifact_paths import get_repository_root

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/benchmarks/issue_3203_scenario_horizons_h500_reexport_valid.yaml"
LEGACY_CONFIG_PATH = (
    ROOT / "configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml"
)


def test_issue_3203_config_preserves_original_table_inputs() -> None:
    """The #3203 rerun preserves the original scenario-horizon inputs."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    legacy = yaml.safe_load(LEGACY_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["paper_facing"] is False
    assert config["scenario_matrix"] == legacy["scenario_matrix"]
    assert config["scenario_horizons"] == legacy["scenario_horizons"]
    assert config["comparability_mapping"] == legacy["comparability_mapping"]
    assert config["route_clearance_certifications"] == legacy["route_clearance_certifications"]
    assert config["seed_policy"]["mode"] == "seed-set"
    assert config["seed_policy"]["seed_set"] == "eval"
    assert config["kinematics_matrix"] == ["differential_drive"]
    assert config["export_publication_bundle"] is False


def test_issue_3203_config_keeps_ppo_repaired_and_in_scope() -> None:
    """PPO remains in the matrix on the repaired native dict-observation path."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    planner = next(item for item in config["planners"] if item["key"] == "ppo")

    assert planner["algo"] == "ppo"
    assert planner["algo_config"].endswith(
        "ppo_issue_791_eval_aligned_large_capacity_portable.yaml"
    )
    assert planner["adapter_impact_eval"] is True
    assert planner["workers"] == 1


def test_issue_3203_config_makes_snqi_contract_explicit() -> None:
    """SNQI diagnostics are required to decide whether the rerun is promotable."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    snqi = config["snqi_contract"]

    assert snqi["enabled"] is True
    assert snqi["enforcement"] == "warn"
    assert snqi["rank_alignment_fail_threshold"] == pytest.approx(0.3)
    assert snqi["outcome_separation_fail_threshold"] == pytest.approx(0.0)


def test_issue_3203_reexport_summary_if_present_stays_diagnostic_when_snqi_fails() -> None:
    """The 2026-07-01 rerun is not Results evidence while SNQI fails."""
    artifact = (
        get_repository_root()
        / "docs/context/evidence/issue_3203_scenario_horizon_reexport_2026-07-01"
        / "reports/campaign_summary.json"
    )
    if not artifact.exists():
        pytest.skip("issue #3203 2026-07-01 diagnostic rerun artifact not present")

    readiness = classify_scenario_horizon_readiness(artifact)
    payload = readiness.to_payload()
    summary = json.loads(artifact.read_text(encoding="utf-8"))
    ppo_row = next(row for row in summary["planner_rows"] if row["planner_key"] == "ppo")

    assert readiness.status == DIAGNOSTIC_ONLY, json.dumps(payload, indent=2)
    assert readiness.ppo_status == "ok"
    assert payload["snqi_contract_status"] == "fail"
    assert ppo_row["execution_mode"] == "native"
    assert ppo_row["learned_policy_contract_status"] == "pass"
    assert summary["campaign"]["snqi_contract_status"] == "fail"
    assert any("SNQI contract status" in blocker for blocker in readiness.blockers)
