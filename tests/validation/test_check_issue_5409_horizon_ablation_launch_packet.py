"""Tests for the issue #5409 h500/h600 horizon-ablation launch packet + checker."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "configs/benchmarks/issue_5409_horizon_ablation_launch_packet.yaml"
SCRIPT = REPO_ROOT / "scripts/validation/check_issue_5409_horizon_ablation_launch_packet.py"

_SPEC = importlib.util.spec_from_file_location("_issue_5409_launch_check", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _load_packet() -> dict:
    return yaml.safe_load(PACKET.read_text(encoding="utf-8"))


def _write_packet(tmp_path: Path, packet: dict) -> Path:
    path = tmp_path / "packet.yaml"
    path.write_text(yaml.safe_dump(packet), encoding="utf-8")
    return path


def test_packet_passes_fail_closed_contract() -> None:
    """The checked-in packet is a valid fail-closed launch contract."""
    summary = _MODULE.validate_packet(_load_packet())

    assert summary["ok"] is True
    assert summary["issue"] == 5409
    assert summary["planner_count"] == 12
    assert summary["scenario_count"] == 48
    assert summary["resolved_seeds"] == [111, 112, 113]
    assert summary["rows_per_horizon"] == 1728
    assert summary["rows_total"] == 3456
    assert summary["scenario_matrix_hash"] == "c10df617a87c"
    assert summary["pair_is_valid"] is True
    assert summary["pair_mismatch_count"] == 0
    assert summary["arm_roles"] == ["h500", "h600"]
    assert summary["checkpoint_gate_status"] == "pending_submit_node_execution"
    assert summary["compute_submit_authorized"] is False


def test_row_arithmetic_closes() -> None:
    """12 planners x 48 scenarios x 3 seeds reconstructs the declared per-horizon row count."""
    matrix = _load_packet()["matrix"]
    product = matrix["planner_count"] * matrix["scenario_count"] * len(matrix["resolved_seeds"])
    assert product == matrix["rows_per_horizon"]
    assert matrix["rows_total"] == 2 * matrix["rows_per_horizon"]


def test_referenced_configs_exist_and_match_pinned_digests() -> None:
    """Both arm configs exist on disk and still hash to the digests the packet pins."""
    for arm in _load_packet()["arms"]:
        config = REPO_ROOT / arm["config"]
        assert config.is_file(), arm["config"]
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        assert digest == arm["config_sha256"], arm["role"]


def test_arm_configs_declare_the_expected_horizons() -> None:
    """The packet's horizon claim matches what each campaign config actually sets."""
    for arm in _load_packet()["arms"]:
        config = yaml.safe_load((REPO_ROOT / arm["config"]).read_text(encoding="utf-8"))
        assert config["horizon"] == arm["horizon"], arm["role"]
        assert config["preregistration"]["ablation_role"] == arm["role"]
        assert (
            config["preregistration"]["expected_scenario_matrix_hash"]
            == _load_packet()["matrix"]["expected_scenario_matrix_hash"]
        )


def test_referenced_scripts_exist_on_disk() -> None:
    """Every tool the packet routes an operator to is actually checked in."""
    packet = _load_packet()
    for rel in (
        packet["pair_validation"]["validator"],
        packet["environment_identity"]["entry_point"],
        packet["checkpoint_gate"]["script"],
    ):
        assert (REPO_ROOT / rel).is_file(), rel


def test_each_horizon_has_its_own_results_dir_and_receipt() -> None:
    """CAMERA_READY_RESULTS_DIR and the gate receipt are distinct and nested per horizon."""
    arms = _load_packet()["arms"]
    results = [arm["results_dir_template"] for arm in arms]
    reports = [arm["checkpoint_gate_report_template"] for arm in arms]

    assert len(set(results)) == 2
    assert len(set(reports)) == 2
    for arm, results_dir, report in zip(arms, results, reports, strict=True):
        assert results_dir.endswith(arm["campaign_id"])
        assert report.startswith(results_dir)
        assert report.endswith("checkpoint_staging.json")


def test_environment_identity_binds_the_shared_hashes() -> None:
    """The pair-invariant fingerprints are recorded, and the per-arm ones differ."""
    env = _load_packet()["environment_identity"]
    measured = env["measured"]

    for field_name in (
        "scenario_matrix_hash",
        "comparability_mapping_hash",
        "observation_noise_hash",
    ):
        assert field_name in env["shared_across_pair"]
        assert measured[field_name]
    assert measured["h500_config_hash"] != measured["h600_config_hash"]


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda p: p.update(schema_version="other.v1"), "schema drift"),
        (lambda p: p.update(no_benchmark_result_claim=False), "claims a result"),
        (lambda p: p.update(generating_commit="abc123"), "short commit"),
        (lambda p: p["matrix"].update(planner_count=7), "roster shrunk"),
        (lambda p: p["matrix"].update(resolved_seeds=[111, 112]), "seed budget changed"),
        (lambda p: p["matrix"].update(rows_per_horizon=1000), "row count drift"),
        (
            lambda p: p["matrix"].update(expected_scenario_matrix_hash="deadbeef0000"),
            "comparability hash drift",
        ),
        (lambda p: p["pair_validation"].update(is_valid=False), "pair not validated"),
        (lambda p: p["pair_validation"].update(mismatch_count=1), "pair mismatch"),
        (lambda p: p["arms"][1].update(horizon=500), "horizons no longer differ"),
        (lambda p: p["arms"][0].update(config_sha256="0" * 64), "config digest drift"),
        (
            lambda p: p["arms"][0].update(results_dir_template="/abs/results"),
            "results dir not templated",
        ),
        (
            lambda p: p["arms"][1].update(
                results_dir_template=p["arms"][0]["results_dir_template"]
            ),
            "shared results dir",
        ),
        (lambda p: p["checkpoint_gate"].update(mode="metadata_only"), "not submit-safe mode"),
        (lambda p: p["checkpoint_gate"].update(submit_safe_required=False), "gate optional"),
        (lambda p: p["checkpoint_gate"].update(status="passed"), "gate falsely marked done"),
        (lambda p: p["checkpoint_gate"].update(submits_slurm=True), "gate submits slurm"),
        (
            lambda p: p["checkpoint_gate"]["local_resolvability_check"].update(
                result="submit_safe"
            ),
            "resolvability probe claimed submit-safe",
        ),
        (
            lambda p: p["artifact_manifest"].update(paired_required=["paired_horizon_deltas.json"]),
            "matched-key completeness dropped",
        ),
        (
            lambda p: p["artifact_manifest"].update(raw_episode_jsonl_in_git=True),
            "raw rows committed",
        ),
        (
            lambda p: p["fail_closed_policy"].update(valid_row_statuses=["native", "fallback"]),
            "fallback counted as valid",
        ),
        (
            lambda p: p["fail_closed_policy"].update(hash_drift_blocks_comparison=False),
            "hash drift tolerated",
        ),
        (
            lambda p: p["environment_identity"].update(dependency_sync="uv sync"),
            "incomplete environment",
        ),
        (
            lambda p: p["environment_identity"].update(shared_across_pair=["scenario_matrix_hash"]),
            "unbound shared identity",
        ),
        (lambda p: p["preservation"].update(raw_artifacts_external=False), "no durable route"),
        (
            lambda p: p["execution_boundary"].update(compute_submit_authorized=True),
            "self-authorized compute",
        ),
        (
            lambda p: p["execution_boundary"].update(submit_slurm_from_this_packet=True),
            "self-authorized submission",
        ),
    ],
)
def test_packet_fails_closed_on_contract_violation(mutate, reason: str) -> None:
    """Every way the packet could become unsafe is rejected, not silently accepted."""
    packet = copy.deepcopy(_load_packet())
    mutate(packet)
    with pytest.raises(_MODULE.PacketError):
        _MODULE.validate_packet(packet)


def test_missing_required_section_is_rejected() -> None:
    """Dropping a whole contract section fails closed rather than defaulting."""
    for section in (
        "matrix",
        "pair_validation",
        "environment_identity",
        "checkpoint_gate",
        "artifact_manifest",
        "fail_closed_policy",
        "preservation",
        "execution_boundary",
    ):
        packet = copy.deepcopy(_load_packet())
        packet.pop(section)
        with pytest.raises(_MODULE.PacketError):
            _MODULE.validate_packet(packet)


def test_cli_reports_ready_for_the_checked_in_packet() -> None:
    """The CLI exits 0 and emits a machine-readable ready summary."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ready"
    assert payload["ok"] is True
    assert payload["rows_total"] == 3456


def test_cli_blocks_on_a_mutated_packet(tmp_path: Path) -> None:
    """A packet that self-authorizes compute exits 1 (blocked), not 0."""
    packet = copy.deepcopy(_load_packet())
    packet["execution_boundary"]["compute_submit_authorized"] = True
    path = _write_packet(tmp_path, packet)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--packet", str(path), "--json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert payload["ok"] is False


def test_cli_reports_malformed_for_a_missing_packet(tmp_path: Path) -> None:
    """A missing packet exits 2 (malformed) rather than being treated as empty."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--packet", str(tmp_path / "absent.yaml"), "--json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "malformed"
