"""Tests for the issue #5348 S30 attempt-6 PPO-resume launch packet + checker."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "configs/benchmarks/issue_5348_s30_attempt6_resume_launch_packet.yaml"
SCRIPT = REPO_ROOT / "scripts/validation/check_issue_5348_s30_resume_launch_packet.py"

_SPEC = importlib.util.spec_from_file_location("_issue_5348_resume_check", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _load_packet() -> dict:
    return yaml.safe_load(PACKET.read_text(encoding="utf-8"))


def test_packet_passes_fail_closed_contract() -> None:
    """The checked-in packet is a valid fail-closed resume contract."""
    summary = _MODULE.validate_packet(_load_packet())

    assert summary["ok"] is True
    assert summary["issue"] == 5348
    assert summary["gate_status"] == "satisfied"
    assert summary["job_id"] == "13376"
    assert summary["attempt"] == 6
    assert summary["clean_arm_count"] == 5
    assert summary["ppo_preserved"] == 562
    assert summary["ppo_remaining"] == 878
    assert summary["resume_after_episode"] == 562
    assert summary["restart_from_zero"] is False
    assert summary["total_preserved_episodes"] == 5 * 1440 + 562
    assert summary["arms_to_run"] == ["ppo"]
    assert summary["report_builder_contract_changed"] is False
    assert summary["compute_submit_authorized"] is False
    assert summary["status_until_run"] == "ready_for_resume_submission"


def test_packet_episode_arithmetic_closes() -> None:
    """Preserved + remaining PPO episodes reconstruct the full 1440-episode arm."""
    packet = _load_packet()
    resume_arm = packet["preservation"]["resume_arm"]
    assert resume_arm["episodes_preserved"] + resume_arm["episodes_remaining"] == 1440


def test_config_reference_exists_on_disk() -> None:
    """The packet points at the real six-arm S30 campaign config."""
    config = REPO_ROOT / _load_packet()["campaign"]["config"]
    assert config.is_file(), f"missing S30 campaign config: {config}"


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    [
        (
            lambda p: p["preservation"]["resume_arm"].__setitem__("restart_from_zero", True),
            "must not restart from zero",
        ),
        (
            lambda p: p["preservation"]["resume_arm"].__setitem__("resume_after_episode", 0),
            "resume must start after episode 562",
        ),
        (
            lambda p: p["preservation"]["resume_arm"].__setitem__("episodes_remaining", 900),
            "exactly 878 remaining",
        ),
        (
            lambda p: p["resume_execution"].__setitem__("compute_submit_authorized", True),
            "compute_submit_authorized must be false",
        ),
        (
            lambda p: p["resume_execution"].__setitem__("arms_to_run", ["ppo", "orca"]),
            "only the ppo arm may be resumed",
        ),
        (
            lambda p: p["aggregation"].__setitem__("report_builder_contract_changed", True),
            "report-builder contract must be unchanged",
        ),
        (
            lambda p: p["gating"].__setitem__("gate_status", "blocked"),
            "gate_status must be satisfied",
        ),
    ],
)
def test_packet_fails_closed_on_mutation(mutation, expected_fragment: str) -> None:
    """Each unsafe mutation must trip the fail-closed checker."""
    packet = copy.deepcopy(_load_packet())
    mutation(packet)
    with pytest.raises(_MODULE.PacketError) as excinfo:
        _MODULE.validate_packet(packet)
    assert expected_fragment in str(excinfo.value)


def test_cli_reports_ready() -> None:
    """The CLI exits 0 and reports ready on the checked-in packet."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--packet", str(PACKET), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "ready"
    assert payload["issue"] == 5348


def test_cli_fails_closed_on_missing_packet(tmp_path: Path) -> None:
    """A missing packet is malformed (exit 2), never silently ready."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--packet", str(tmp_path / "absent.yaml"), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["status"] == "malformed"
