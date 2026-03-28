"""Tests for the local PR-readiness freshness helper."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

SCRIPT = Path("scripts/dev/pr_ready_freshness.py")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_pr_ready_freshness_status_missing_stamp(tmp_path: Path) -> None:
    """Missing readiness evidence should fail closed."""
    stamp_path = tmp_path / "missing.json"

    result = _run(
        "status",
        "--branch",
        "codex/712-pr-open-skill",
        "--head-sha",
        "abc123",
        "--base-ref",
        "origin/main",
        "--stamp-path",
        str(stamp_path),
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["fresh"] is False
    assert payload["reason"] == "missing"


def test_pr_ready_freshness_write_then_status_succeeds(tmp_path: Path) -> None:
    """A matching, recent passed stamp should be treated as fresh."""
    stamp_path = tmp_path / "ready.json"

    write_result = _run(
        "write",
        "--branch",
        "codex/712-pr-open-skill",
        "--head-sha",
        "abc123",
        "--base-ref",
        "origin/main",
        "--stamp-path",
        str(stamp_path),
    )
    assert write_result.returncode == 0

    status_result = _run(
        "status",
        "--branch",
        "codex/712-pr-open-skill",
        "--head-sha",
        "abc123",
        "--base-ref",
        "origin/main",
        "--stamp-path",
        str(stamp_path),
    )

    assert status_result.returncode == 0
    payload = json.loads(status_result.stdout)
    assert payload["fresh"] is True
    assert payload["reason"] == "fresh"


def test_pr_ready_freshness_rejects_stale_or_mismatched_head(tmp_path: Path) -> None:
    """Freshness should fail closed on stale or non-matching evidence."""
    stamp_path = tmp_path / "ready.json"
    stale_payload = {
        "branch": "codex/712-pr-open-skill",
        "base_ref": "origin/main",
        "head_sha": "oldsha",
        "recorded_at_utc": (datetime.now(UTC) - timedelta(hours=30)).isoformat(),
        "status": "passed",
    }
    stamp_path.write_text(json.dumps(stale_payload), encoding="utf-8")

    result = _run(
        "status",
        "--branch",
        "codex/712-pr-open-skill",
        "--head-sha",
        "newsha",
        "--base-ref",
        "origin/main",
        "--stamp-path",
        str(stamp_path),
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["fresh"] is False
    assert payload["reason"] == "head_sha_mismatch"
