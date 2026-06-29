"""Tests for the issue #3810 no-submit long-horizon launch packet."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "configs/benchmarks/issue_3810_long_horizon_snqi_launch_packet.yaml"
SCRIPT = REPO_ROOT / "scripts/validation/check_issue_3810_long_horizon_launch_packet.py"

_SPEC = importlib.util.spec_from_file_location("_issue_3810_packet_check", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _load_packet() -> dict:
    return yaml.safe_load(PACKET.read_text(encoding="utf-8"))


def test_issue_3810_packet_passes_fail_closed_contract() -> None:
    """The checked-in packet satisfies the issue #3810 launch contract."""
    summary = _MODULE.validate_packet(_load_packet())

    assert summary["ok"] is True
    assert summary["issue"] == 3810
    assert summary["max_episode_steps"] == 600
    assert summary["seed_count"] == 30
    assert summary["planner_count"] >= 10
    assert summary["compute_submit_authorized"] is False
    assert summary["slurm_job_id"] == "not_submitted"
    assert 13175 in summary["blocking_jobs"]


def test_issue_3810_packet_rejects_authorized_submit() -> None:
    """The packet validator rejects any direct compute authorization."""
    packet = _load_packet()
    packet["launch_packet"]["compute_submit_authorized"] = True

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "compute submit must be false" in str(exc)
    else:
        raise AssertionError("packet should reject compute submission authorization")


def test_issue_3810_packet_rejects_horizon_only_claim_boundary() -> None:
    """The packet keeps the multi-factor comparison caveat mandatory."""
    packet = _load_packet()
    packet["campaign"]["claim_boundary"] = "Launch packet only. It does not run a benchmark."

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "horizon-only" in str(exc)
    else:
        raise AssertionError("packet should reject missing comparison caveat")


def test_issue_3810_packet_rejects_low_exposure_success_evidence() -> None:
    """Low-exposure successes stay diagnostic instead of success evidence."""
    packet = _load_packet()
    packet["launch_packet"]["wait_it_out_guard"]["low_exposure_success_status"] = (
        "successful_evidence"
    )

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "low exposure must be diagnostic" in str(exc)
    else:
        raise AssertionError("packet should reject low-exposure success evidence")


def test_issue_3810_packet_rejects_null_list_fields_cleanly() -> None:
    """Explicit-null list fields fail closed with PacketError, not TypeError."""
    packet = _load_packet()
    packet["metrics"]["ids"] = None

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "SNQI metric required" in str(exc)
    else:
        raise AssertionError("packet should reject a null metrics.ids list")


def test_issue_3810_packet_rejects_blank_matrix_path() -> None:
    """A blank scenario matrix path must not pass the repo-relative check."""
    packet = _load_packet()
    packet["scenario_suite"]["matrix_path"] = ""

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "scenario matrix must be repo-relative" in str(exc)
    else:
        raise AssertionError("packet should reject a blank scenario matrix path")


def test_issue_3810_packet_rejects_missing_campaign_id() -> None:
    """A missing campaign id fails closed instead of escaping as KeyError."""
    packet = _load_packet()
    packet["campaign"].pop("id", None)

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "campaign.id required" in str(exc)
    else:
        raise AssertionError("packet should reject a missing campaign id")


def test_issue_3810_packet_cli_returns_2_on_malformed_packet(tmp_path) -> None:
    """The CLI converts a malformed packet into a clean exit 2, not a traceback."""
    bad_packet = tmp_path / "bad_packet.yaml"
    bad_packet.write_text("schema_version: research-campaign-manifest.v0.1\n", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--packet", str(bad_packet), "--json"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2, completed.stderr
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_issue_3810_packet_cli_json() -> None:
    """The command-line validator emits a compact JSON pass summary."""
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--packet", str(PACKET), "--json"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"compute_submit_authorized": false' in completed.stdout
    assert '"max_episode_steps": 600' in completed.stdout
