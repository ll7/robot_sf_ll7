"""Tests issue #3810 long-horizon analysis + retention packet contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "configs/benchmarks/issue_3810_long_horizon_snqi_launch_packet.yaml"
SCRIPT = REPO_ROOT / "scripts/validation/check_issue_3810_long_horizon_analysis_packet.py"

SPEC = importlib.util.spec_from_file_location("_issue_3810_analysis_packet_check", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_MODULE)


def _load_packet() -> dict:
    return yaml.safe_load(PACKET.read_text(encoding="utf-8"))


def test_issue_3810_analysis_packet_passes() -> None:
    """The checked-in packet satisfies the analysis and retention contract."""

    summary = _MODULE.validate_packet(_load_packet())
    assert summary["ok"] is True
    assert summary["issue"] == 3810
    assert summary["route_contract"] == "blocked_pending_submit_host_route_and_reconciliation"


def test_issue_3810_analysis_packet_rejects_bad_target_host() -> None:
    """The packet must keep the reconciled submit-host target."""

    packet = _load_packet()
    packet["analysis_and_retention_packet"]["route"]["target_host"] = "imech156-u"
    with pytest.raises(_MODULE.PacketError, match="target host must be imech039"):
        _MODULE.validate_packet(packet)


def test_issue_3810_analysis_packet_rejects_stale_private_ops_flags() -> None:
    """The packet must not advertise unsupported private-ops queue flags."""

    packet = _load_packet()
    packet["analysis_and_retention_packet"]["preflight"]["duplicate_check_command"] = (
        "/home/luttkule/git/robot_sf_ll7-private-ops/ops/jobs/scripts/route_job.sh "
        "--json --target-host imech039 --issue 3810"
    )
    with pytest.raises(_MODULE.PacketError, match="current queue_summary interface"):
        _MODULE.validate_packet(packet)


def test_issue_3810_analysis_packet_rejects_missing_scope_caveat() -> None:
    """The horizon report must retain the multi-factor comparison caveat."""

    packet = _load_packet()
    packet["analysis_and_retention_packet"]["horizon_sensitivity_report"][
        "required_scope_caveat"
    ] = "missing"
    with pytest.raises(_MODULE.PacketError, match="horizon scope caveat mismatch"):
        _MODULE.validate_packet(packet)


def test_issue_3810_analysis_packet_rejects_local_submission() -> None:
    """The analysis packet must remain no-submit on the local machine."""

    packet = _load_packet()
    packet["analysis_and_retention_packet"]["execution_contract"]["local_submission_allowed"] = True
    with pytest.raises(
        _MODULE.PacketError,
        match="execution_contract.local_submission_allowed must be false",
    ):
        _MODULE.validate_packet(packet)


def test_issue_3810_analysis_packet_rejects_missing_job_reconciliation_reason() -> None:
    """The decision gate must retain the active job reconciliation blocker."""

    packet = _load_packet()
    packet["analysis_and_retention_packet"]["execution_contract"]["decision_gate"]["reason"] = (
        "Issue remains open."
    )
    with pytest.raises(_MODULE.PacketError, match="decision gate must require job 13175"):
        _MODULE.validate_packet(packet)


def test_issue_3810_analysis_packet_rejects_incomplete_retention_manifest() -> None:
    """The retention manifest must preserve reviewable report and metadata paths."""

    packet = _load_packet()
    packet["analysis_and_retention_packet"]["retention"]["durable_manifests"] = [
        "docs/context/evidence/issue_3810_long_horizon_snqi/"
    ]
    with pytest.raises(
        _MODULE.PacketError, match="durable manifests must include reports directory"
    ):
        _MODULE.validate_packet(packet)
