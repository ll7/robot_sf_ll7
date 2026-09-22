"""Tests for strict PR-readiness artifact production and reconciliation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from scripts.dev.pr_ready_artifact_contract import reconcile_readiness_artifacts


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _termination_payload() -> dict[str, object]:
    return {"schema": "pr_ready_termination.v1", "status": "terminated"}


def test_prefixed_machine_json_is_rejected_from_byte_zero(tmp_path: Path) -> None:
    """A human preface cannot be accepted as a machine-readable artifact."""
    machine = tmp_path / "base-sensitive.json"
    machine.write_text('gate not required\n{"status": "passed"}\n', encoding="utf-8")

    result = reconcile_readiness_artifacts([machine])

    assert result["passed"] is False
    assert result["status"] == "unavailable"
    assert any(code.startswith("json_malformed:") for code in result["reason_codes"])


def test_terminated_machine_result_cannot_be_reconciled_as_passed(
    tmp_path: Path,
) -> None:
    """A termination receipt remains non-passing even when prose says passed."""
    machine = tmp_path / "pr_ready_termination.json"
    summary = tmp_path / "RESULT.md"
    log = tmp_path / "pr_ready_check.log"
    _write_json(machine, _termination_payload())
    summary.write_text("PR readiness status: passed\n", encoding="utf-8")
    log.write_text("All checks passed!\nPR readiness received SIGTERM.\n", encoding="utf-8")

    result = reconcile_readiness_artifacts([machine], human_summary=summary, command_log=log)

    assert result["status"] == "conflict"
    assert result["passed"] is False
    assert "readiness_status_disagreement" in result["reason_codes"]


def test_matching_machine_summary_and_log_are_passed(tmp_path: Path) -> None:
    """A complete, consistently reported readiness run can still pass."""
    machine = tmp_path / "pr_ready.json"
    summary = tmp_path / "RESULT.md"
    log = tmp_path / "pr_ready_check.log"
    _write_json(machine, {"schema": "pr_ready_freshness.v1", "status": "passed"})
    summary.write_text("PR readiness status: passed\n", encoding="utf-8")
    log.write_text("All checks passed!\n", encoding="utf-8")

    result = reconcile_readiness_artifacts([machine], human_summary=summary, command_log=log)

    assert result["status"] == "passed"
    assert result["passed"] is True
    assert result["reason_codes"] == []
