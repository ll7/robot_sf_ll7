"""Regression tests for issue #3203 scenario-horizon re-export configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.scenario_horizon_readiness import (
    BLOCKED,
    DIAGNOSTIC_ONLY,
    TABLE_REEXPORT_READY,
    classify_scenario_horizon_readiness,
    classify_scenario_horizon_table_reexport_readiness,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/benchmarks/issue_3203_scenario_horizons_h500_reexport_valid.yaml"
LEGACY_CONFIG_PATH = (
    ROOT / "configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml"
)
NARROW_PACKET_PATH = (
    ROOT
    / "docs/context/evidence/issue_3203_scenario_horizon_table_reexport_closure_2026-07-04"
    / "readiness_packet.json"
)
SUMMARY_PATH = (
    ROOT
    / "docs/context/evidence/issue_3203_scenario_horizon_reexport_2026-07-01"
    / "reports/campaign_summary.json"
)


def _packet_for(path: Path) -> dict[str, object]:
    """Return narrow packet data retargeted to a temporary artifact."""
    packet = json.loads(NARROW_PACKET_PATH.read_text(encoding="utf-8"))
    packet["source_artifact"] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return packet


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
    artifact = SUMMARY_PATH
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


def test_issue_3203_narrow_packet_accepts_table_reexport_only() -> None:
    """A predeclared packet can close table-readiness without promoting SNQI."""
    artifact = SUMMARY_PATH

    readiness = classify_scenario_horizon_table_reexport_readiness(artifact, NARROW_PACKET_PATH)
    payload = readiness.to_payload()

    assert readiness.status == TABLE_REEXPORT_READY, json.dumps(payload, indent=2)
    assert readiness.claim_boundary == "scenario_horizon_table_reexport_only"
    assert readiness.readiness_packet == str(NARROW_PACKET_PATH)
    assert readiness.snqi_contract_status == "fail"
    assert readiness.blockers == []
    assert any("SNQI remains excluded" in note for note in readiness.notes)


def test_issue_3203_narrow_packet_requires_predeclared_snqi_exclusion(
    tmp_path: Path,
) -> None:
    """Missing SNQI exclusion keeps the narrow packet diagnostic-only."""
    artifact = SUMMARY_PATH
    packet = json.loads(NARROW_PACKET_PATH.read_text(encoding="utf-8"))
    packet.pop("snqi_exclusion")
    packet_path = tmp_path / "readiness_packet_without_snqi_exclusion.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    readiness = classify_scenario_horizon_table_reexport_readiness(artifact, packet_path)

    assert readiness.status == DIAGNOSTIC_ONLY
    assert any("snqi_exclusion" in blocker for blocker in readiness.blockers)


def test_issue_3203_narrow_packet_missing_file_blocks() -> None:
    """A packet path must exist before a narrow readiness verdict can pass."""
    readiness = classify_scenario_horizon_table_reexport_readiness(
        SUMMARY_PATH,
        NARROW_PACKET_PATH.with_name("missing_packet.json"),
    )

    assert readiness.status == BLOCKED
    assert any("Readiness packet not found" in blocker for blocker in readiness.blockers)


def test_issue_3203_narrow_packet_rejects_stale_or_broad_packet(tmp_path: Path) -> None:
    """Packet metadata must remain narrow, matched, and checksummed."""
    packet = json.loads(NARROW_PACKET_PATH.read_text(encoding="utf-8"))
    packet["claim_boundary"] = "benchmark_results"
    packet["paper_facing"] = True
    packet["benchmark_results_claim"] = True
    packet["snqi_validity_claim"] = True
    packet["source_artifact"] = {"path": "wrong.json", "sha256": "0" * 64}
    packet_path = tmp_path / "broad_packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    readiness = classify_scenario_horizon_table_reexport_readiness(SUMMARY_PATH, packet_path)

    assert readiness.status == DIAGNOSTIC_ONLY
    assert any("claim_boundary" in blocker for blocker in readiness.blockers)
    assert any("paper_facing=false" in blocker for blocker in readiness.blockers)
    assert any("benchmark_results_claim=false" in blocker for blocker in readiness.blockers)
    assert any("snqi_validity_claim=false" in blocker for blocker in readiness.blockers)
    assert any("source_artifact.path" in blocker for blocker in readiness.blockers)
    assert any("sha256" in blocker for blocker in readiness.blockers)


def test_issue_3203_narrow_packet_rejects_malformed_summary(tmp_path: Path) -> None:
    """Malformed campaign summary structures stay diagnostic-only."""
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    summary["snqi_contract_status"] = "fail"
    summary["campaign"] = []
    summary_path = tmp_path / "campaign_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    packet = _packet_for(summary_path)
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    readiness = classify_scenario_horizon_table_reexport_readiness(summary_path, packet_path)

    assert readiness.status == DIAGNOSTIC_ONLY
    assert any("campaign must be object" in blocker for blocker in readiness.blockers)


def test_issue_3203_narrow_packet_rejects_required_observation_drift(
    tmp_path: Path,
) -> None:
    """Campaign and PPO observation drift fail the narrow packet."""
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    summary["campaign"]["exit_code"] = 1
    ppo_row = next(row for row in summary["planner_rows"] if row["planner_key"] == "ppo")
    ppo_row["status"] = "failed"
    ppo_row["execution_mode"] = "adapter"
    ppo_row["learned_policy_contract_status"] = "fail"
    summary_path = tmp_path / "campaign_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    packet = _packet_for(summary_path)
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    readiness = classify_scenario_horizon_table_reexport_readiness(summary_path, packet_path)

    assert readiness.status == DIAGNOSTIC_ONLY
    assert any("campaign.exit_code=1" in blocker for blocker in readiness.blockers)
    assert any("PPO row status" in blocker for blocker in readiness.blockers)
    assert any("PPO execution_mode" in blocker for blocker in readiness.blockers)
    assert any("learned_policy_contract_status" in blocker for blocker in readiness.blockers)
    assert any("not table-ready" in blocker for blocker in readiness.blockers)
