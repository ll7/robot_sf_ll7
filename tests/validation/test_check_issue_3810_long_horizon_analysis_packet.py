"""Tests issue #3810 long-horizon analysis + retention packet contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
    summary = _MODULE.validate_packet(_load_packet())
    assert summary["ok"] is True
    assert summary["issue"] == 3810
    assert summary["route_contract"] == "blocked_pending_submit_host_route_and_reconciliation"


def test_issue_3810_analysis_packet_rejects_bad_target_host() -> None:
    packet = _load_packet()
    packet["analysis_and_retention_packet"]["route"]["target_host"] = "imech156-u"
    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "target host must be imech036" in str(exc)
    else:
        raise AssertionError("packet should reject non-imech036 route")


def test_issue_3810_analysis_packet_rejects_missing_scope_caveat() -> None:
    packet = _load_packet()
    packet["analysis_and_retention_packet"]["horizon_sensitivity_report"]["required_scope_caveat"] = "missing"
    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "horizon scope caveat mismatch" in str(exc)
    else:
        raise AssertionError("packet should reject wrong scope caveat")
