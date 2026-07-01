"""Tests for the issue #3653 SNQI decision-disagreement packet checker."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "configs/benchmarks/issue_3653_snqi_decision_disagreement_packet.yaml"
SCRIPT = REPO_ROOT / "scripts/validation/check_issue_3653_snqi_decision_disagreement_packet.py"

_SPEC = importlib.util.spec_from_file_location("_issue_3653_snqi_packet_check", SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _load_packet() -> dict:
    payload = yaml.safe_load(PACKET.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_issue_3653_packet_passes_fail_closed_contract() -> None:
    """The checked packet is a no-submit, artifact-blocked application contract."""

    summary = _MODULE.validate_packet(_load_packet())

    assert summary["status"] == "ok"
    assert summary["issue"] == 3653
    assert summary["current_status"] == "blocked_missing_valid_campaign_episodes"
    assert summary["target_host"] == "imech039"
    assert summary["source_job_id"] == 13175
    assert summary["expected_episode_count"] == 8640
    assert summary["artifact_count"] == 5


def test_issue_3653_packet_rejects_claim_boundary_without_paper_guard() -> None:
    """The packet must not be broadened into paper-facing claims."""

    packet = _load_packet()
    packet["claim_boundary"] = (
        "Diagnostic packet only. This does not run a full benchmark campaign, "
        "does not submit Slurm/GPU work, and does not establish SNQI as a "
        "primary safety ranking."
    )

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "paper/dissertation" in str(exc)
    else:
        raise AssertionError("packet should reject missing paper claim guard")


def test_issue_3653_packet_rejects_compute_submission() -> None:
    """Compute submission remains outside this issue slice."""

    packet = _load_packet()
    packet["execution_boundary"]["compute_submit_authorized"] = True

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "compute_submit_authorized must be false" in str(exc)
    else:
        raise AssertionError("packet should reject compute submission authorization")


def test_issue_3653_packet_rejects_missing_decision_disagreement_artifact() -> None:
    """The empirical application must export the decision-disagreement CSV."""

    packet = _load_packet()
    packet["export"]["required_artifacts"].remove(
        "snqi_scalarization_sensitivity_decision_disagreement.csv"
    )

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "required_artifacts mismatch" in str(exc)
    else:
        raise AssertionError("packet should reject missing decision-disagreement export")


def test_issue_3653_check_cli_json() -> None:
    """The checker CLI returns a machine-readable success summary."""

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--packet", str(PACKET), "--json"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["issue"] == 3653
    assert payload["current_status"] == "blocked_missing_valid_campaign_episodes"
