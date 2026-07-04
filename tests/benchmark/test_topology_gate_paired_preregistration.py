"""Tests for issue #3465 topology-gate paired preregistration."""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.topology_gate_paired_preregistration import (
    CORRECTIVE_REQUIRED_CAPABILITIES,
    EXPECTED_ARMS,
    build_topology_gate_readiness_packet,
    check_topology_gate_arms,
    load_topology_gate_paired_config,
    topology_gate_config_hash,
    validate_topology_gate_paired_config,
    write_readiness_packet,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = _REPO_ROOT / "configs" / "benchmarks" / "issue_3465_topology_gate_paired.yaml"
_READINESS_SCRIPT = (
    _REPO_ROOT / "scripts" / "benchmark" / "build_issue3465_topology_gate_paired_readiness.py"
)


def _config() -> dict[str, object]:
    return load_topology_gate_paired_config(CONFIG_PATH)


def test_preregistered_config_builds_ready_readiness_packet() -> None:
    """The pre-run config is reproducible and ready for a paired run after #3463 packets."""

    packet = build_topology_gate_readiness_packet(_config())

    assert packet["benchmark_evidence"] is False
    assert packet["status"] == "ready_for_paired_run"
    assert packet["blockers"] == []
    assert packet["corrective_complete"] is True
    assert packet["corrective_waiver_recorded"] is False
    assert set(packet["corrective_required_capabilities"]) == CORRECTIVE_REQUIRED_CAPABILITIES
    assert set(packet["corrective_capabilities"]) == CORRECTIVE_REQUIRED_CAPABILITIES
    assert {entry["pr"] for entry in packet["corrective_packets"]} == {
        "4388",
        "4411",
        "4426",
        "4435",
        "4444",
    }
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
    arm_report = check_topology_gate_arms(config["arms"])  # type: ignore[arg-type,index]

    assert arm_report["complete"] is True
    assert arm_report["allowed_difference"] == "near_parity_diversity_gate_enabled"

    bad_config = copy.deepcopy(config)
    bad_config["arms"][1]["policy_config"]["static_clearance_weight"] = 0.9  # type: ignore[index]
    bad_report = check_topology_gate_arms(bad_config["arms"])  # type: ignore[arg-type,index]

    assert bad_report["complete"] is False
    assert bad_report["field_mismatches"] == [
        {
            "allowed_difference": "near_parity_diversity_gate_enabled",
            "unexpected_fields": ["static_clearance_weight"],
        }
    ]


def test_topology_gate_config_hash_is_pinned() -> None:
    """Arm policy-config changes update the pinned readiness hash."""

    config = _config()
    assert config["readiness"]["topology_gate_config_hash"] == topology_gate_config_hash(  # type: ignore[index]
        config["arms"]  # type: ignore[arg-type,index]
    )

    drifted = copy.deepcopy(config)
    drifted["readiness"]["topology_gate_config_hash"] = "0" * 64  # type: ignore[index]

    with pytest.raises(ValueError, match="topology_gate_config_hash"):
        validate_topology_gate_paired_config(drifted)


def test_corrective_complete_requires_required_packets() -> None:
    """Corrective completion must name integration capabilities, not just flip a boolean."""

    config = _config()
    config["readiness"]["corrective_packets"] = []  # type: ignore[index]

    with pytest.raises(ValueError, match="missing required capabilities"):
        validate_topology_gate_paired_config(config)


def test_missing_corrective_packet_stays_blocked() -> None:
    """The readiness packet still fails closed when corrective completion is not recorded."""

    config = _config()
    config["readiness"]["corrective_complete"] = False  # type: ignore[index]
    config["readiness"]["corrective_packets"] = []  # type: ignore[index]

    packet = build_topology_gate_readiness_packet(config)

    assert packet["status"] == "blocked_corrective_issue"
    assert packet["blockers"] == ["blocked_corrective_issue"]
    assert packet["corrective_capabilities"] == []


def test_ineligible_rows_fail_closed() -> None:
    """Fallback/degraded or other ineligible row declarations block the future run."""

    config = _config()
    config["readiness"]["known_ineligible_rows"] = [  # type: ignore[index]
        {"planner": "topology_guided_hybrid_rule_v0", "reason": "fallback"}
    ]
    packet = build_topology_gate_readiness_packet(config)

    assert packet["status"] == "blocked_ineligible_rows"
    assert packet["blockers"] == ["blocked_ineligible_rows"]


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
    assert '"status": "ready_for_paired_run"' in out_path.read_text(encoding="utf-8")


def test_cli_accepts_output_directory_outside_repo(tmp_path: Path) -> None:
    """The CLI writes a pre-run readiness packet without running a benchmark."""

    completed = subprocess.run(
        [
            sys.executable,
            str(_READINESS_SCRIPT),
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
    assert "status=ready_for_paired_run" in completed.stdout
    assert "row_check.complete=True" in completed.stdout
    assert (tmp_path / "issue_3465_topology_gate_paired_readiness.json").is_file()
