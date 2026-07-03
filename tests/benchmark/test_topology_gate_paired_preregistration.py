"""Tests for issue #3465 topology-gate paired preregistration."""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.topology_gate_paired_preregistration import (
    EXPECTED_ARMS,
    build_topology_gate_readiness_packet,
    check_topology_gate_arms,
    load_topology_gate_paired_config,
    topology_gate_config_hash,
    validate_topology_gate_paired_config,
    write_readiness_packet,
)

CONFIG_PATH = Path("configs/benchmarks/issue_3465_topology_gate_paired.yaml")


def _config() -> dict[str, object]:
    return load_topology_gate_paired_config(CONFIG_PATH)


def test_preregistered_config_builds_fail_closed_readiness_packet() -> None:
    """The pre-run config is reproducible but blocked until #3463 completion or waiver."""

    packet = build_topology_gate_readiness_packet(_config())

    assert packet["benchmark_evidence"] is False
    assert packet["status"] == "blocked_corrective_issue"
    assert packet["blockers"] == ["blocked_corrective_issue"]
    assert packet["arm_check"]["complete"] is True
    assert packet["row_check"]["complete"] is True
    assert packet["row_count"] == 2
    assert {row["topology_gate_arm"] for row in packet["rows"]} == set(EXPECTED_ARMS)
    assert {row["planner"] for row in packet["rows"]} == {"topology_guided_hybrid_rule_v0"}
    assert {row["scenario_id"] for row in packet["rows"]} == {
        "classic_realworld_double_bottleneck_high"
    }
    assert {row["seed"] for row in packet["rows"]} == {111}
    assert {row["horizon"] for row in packet["rows"]} == {160}


def test_arms_differ_only_by_topology_gate_flag() -> None:
    """Enabled and disabled arms may differ only by near_parity_diversity_gate_enabled."""

    config = _config()
    report = check_topology_gate_arms(config["arms"])  # type: ignore[index]

    assert report["complete"] is True
    assert report["allowed_difference"] == "near_parity_diversity_gate_enabled"

    drifted = copy.deepcopy(config)
    drifted["arms"][1]["policy_config"]["static_clearance_weight"] = 0.7  # type: ignore[index]
    drift_report = check_topology_gate_arms(drifted["arms"])  # type: ignore[index]

    assert drift_report["complete"] is False
    assert drift_report["field_mismatches"] == [
        {
            "allowed_difference": "near_parity_diversity_gate_enabled",
            "unexpected_fields": ["static_clearance_weight"],
        }
    ]


def test_topology_gate_config_hash_is_pinned() -> None:
    """Arm policy-config changes must update the pinned readiness hash."""

    config = _config()
    assert config["readiness"]["topology_gate_config_hash"] == topology_gate_config_hash(
        config["arms"]  # type: ignore[arg-type,index]
    )

    drifted = copy.deepcopy(config)
    drifted["readiness"]["topology_gate_config_hash"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="topology_gate_config_hash"):
        validate_topology_gate_paired_config(drifted)


def test_ineligible_rows_fail_closed() -> None:
    """Fallback/degraded or other ineligible row declarations block the future run."""

    config = _config()
    config["readiness"]["known_ineligible_rows"] = [  # type: ignore[index]
        {"planner": "topology_guided_hybrid_rule_v0", "reason": "fallback"}
    ]
    packet = build_topology_gate_readiness_packet(config)

    assert packet["status"] == "blocked_corrective_issue"
    assert packet["blockers"] == ["blocked_corrective_issue", "blocked_ineligible_rows"]


def test_config_rejects_transient_queue_routing_state() -> None:
    """Tracked config must not encode target hosts or packet-lineage routing state."""

    config = _config()
    config["target_host"] = "imech039"

    with pytest.raises(ValueError, match="transient queue-routing state"):
        validate_topology_gate_paired_config(config)


def test_write_readiness_packet_outputs_deterministic_json(tmp_path: Path) -> None:
    """The readiness packet writer emits a deterministic JSON file."""

    packet = build_topology_gate_readiness_packet(_config())
    out_path = write_readiness_packet(packet, tmp_path)

    assert out_path.name == "issue_3465_topology_gate_paired_readiness.json"
    assert out_path.read_text(encoding="utf-8").endswith("\n")
    assert '"status": "blocked_corrective_issue"' in out_path.read_text(encoding="utf-8")


def test_cli_accepts_output_directory_outside_repo(tmp_path: Path) -> None:
    """The CLI writes the pre-run readiness packet without running a benchmark."""

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/build_issue3465_topology_gate_paired_readiness.py",
            "--config",
            str(CONFIG_PATH),
            "--out",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=blocked_corrective_issue" in completed.stdout
    assert "row_check.complete=True" in completed.stdout
    assert (tmp_path / "issue_3465_topology_gate_paired_readiness.json").is_file()
