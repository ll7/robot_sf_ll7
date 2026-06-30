"""Tests issue #3808 TTC near-miss decision packet checker."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "configs/benchmarks/issue_3808_ttc_near_miss_decision_packet.yaml"
SCRIPT = REPO_ROOT / "scripts/validation/check_issue_3808_ttc_near_miss_decision_packet.py"

_SPEC = importlib.util.spec_from_file_location("_issue_3808_ttc_packet_check", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _load_packet() -> dict:
    payload = yaml.safe_load(PACKET.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_issue_3808_check_passes() -> None:
    """Committed packet contract and rendered fixtures satisfy checker."""
    packet = _load_packet()
    summary = _MODULE.validate_packet(packet)
    assert summary["status"] == "ok"
    assert summary["issue"] == 3808
    assert summary["fixture_count"] == 4


def test_issue_3808_check_rejects_broadened_fixture_contract(tmp_path: Path) -> None:
    """Unexpected packet fixture map should be rejected before any claim."""
    packet = _load_packet()
    packet["expected_fixtures"].pop("opening")
    bad_packet = tmp_path / "packet.yaml"
    bad_packet.write_text(yaml.safe_dump(packet), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--packet", str(bad_packet)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "expected_fixtures must exactly match" in completed.stderr + completed.stdout


def test_issue_3808_check_cli_json() -> None:
    """CLI returns machine-readable success summary."""
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
    assert payload["issue"] == 3808
    assert payload["fixture_count"] == 4
