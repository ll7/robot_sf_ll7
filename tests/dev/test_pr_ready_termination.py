"""Tests for bounded PR-readiness termination receipts."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from scripts.dev.pr_ready_termination import TerminationContext, build_receipt, write_receipt

if TYPE_CHECKING:
    from pathlib import Path


def test_build_receipt_is_bounded_and_credential_free() -> None:
    """Receipt context is bounded and never includes command lines or environments."""
    receipt = build_receipt(
        TerminationContext(
            signal_number=15,
            phase="core_lane",
            lane="core",
            last_progress="x" * 1000,
            last_progress_at_utc="2026-09-03T06:00:00Z",
            cleanup_status="process_group_cleanup_unverified",
            mode="interim",
            controller_pid=os.getpid(),
            child_pid=os.getpid(),
            child_process_group_id=os.getpgrp(),
        )
    )

    assert receipt["schema"] == "pr_ready_termination.v1"
    assert receipt["signal"] == {"name": "SIGTERM", "number": 15, "exit_code": 143}
    assert len(receipt["last_progress"]["message"]) == 200
    assert receipt["security"] == {
        "command_line_included": False,
        "environment_included": False,
    }
    assert "command" not in receipt
    assert "environment" not in receipt
    assert receipt["resources"]["host"]["cpu_count"] is not None


def test_write_receipt_is_private_and_does_not_overwrite(tmp_path: Path) -> None:
    """Receipts are private files and an existing path is never replaced."""
    receipt = build_receipt(
        TerminationContext(
            signal_number=15,
            phase="preflight",
            lane="none",
            last_progress="preflight",
            last_progress_at_utc="2026-09-03T06:00:00Z",
            cleanup_status="no_child_active",
            mode="interim",
        )
    )
    output = tmp_path / "nested" / "termination.json"

    assert write_receipt(receipt, output) == output
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_receipt(receipt, output)
