"""Tests for the local PR-readiness freshness helper."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

SCRIPT = Path("scripts/dev/pr_ready_freshness.py")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the PR-readiness freshness helper as a subprocess."""
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
    assert payload["stamp"]["tree_state"] in {"clean", "dirty"}


def test_pr_ready_freshness_write_records_explicit_tree_state(tmp_path: Path) -> None:
    """Readiness stamps should preserve whether the proof came from a dirty tree."""
    stamp_path = tmp_path / "ready.json"

    result = _run(
        "write",
        "--branch",
        "codex/1844-final-readiness",
        "--head-sha",
        "abc123",
        "--base-ref",
        "origin/main",
        "--tree-state",
        "dirty",
        "--stamp-path",
        str(stamp_path),
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["stamp"]["tree_state"] == "dirty"


def test_pr_ready_freshness_require_clean_rejects_dirty_write(tmp_path: Path) -> None:
    """Final readiness mode must fail closed instead of recording dirty-tree proof."""
    stamp_path = tmp_path / "ready.json"

    result = _run(
        "write",
        "--branch",
        "codex/1844-final-readiness",
        "--head-sha",
        "abc123",
        "--base-ref",
        "origin/main",
        "--tree-state",
        "dirty",
        "--require-clean-tree",
        "--stamp-path",
        str(stamp_path),
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["reason"] == "dirty_worktree"
    assert not stamp_path.exists()


def test_pr_ready_freshness_require_clean_rejects_dirty_stamp(tmp_path: Path) -> None:
    """Final freshness checks should not accept older interim dirty-tree stamps."""
    stamp_path = tmp_path / "ready.json"
    payload = {
        "branch": "codex/1844-final-readiness",
        "base_ref": "origin/main",
        "head_sha": "abc123",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "status": "passed",
        "tree_state": "dirty",
    }
    stamp_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(
        "status",
        "--branch",
        "codex/1844-final-readiness",
        "--head-sha",
        "abc123",
        "--base-ref",
        "origin/main",
        "--tree-state",
        "clean",
        "--require-clean-tree",
        "--stamp-path",
        str(stamp_path),
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["fresh"] is False
    assert payload["reason"] == "stamp_tree_state_not_clean"


def test_pr_ready_freshness_rejects_mismatched_head(tmp_path: Path) -> None:
    """Freshness should fail closed on a non-matching head SHA."""
    stamp_path = tmp_path / "ready.json"
    payload = {
        "branch": "codex/712-pr-open-skill",
        "base_ref": "origin/main",
        "head_sha": "oldsha",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "status": "passed",
    }
    stamp_path.write_text(json.dumps(payload), encoding="utf-8")

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


def test_pr_ready_freshness_rejects_stale_stamp(tmp_path: Path) -> None:
    """Freshness should fail closed on stale evidence."""
    stamp_path = tmp_path / "ready.json"
    stale_payload = {
        "branch": "codex/712-pr-open-skill",
        "base_ref": "origin/main",
        "head_sha": "abc123",
        "recorded_at_utc": (datetime.now(UTC) - timedelta(hours=30)).isoformat(),
        "status": "passed",
    }
    stamp_path.write_text(json.dumps(stale_payload), encoding="utf-8")

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
    assert payload["reason"] == "expired"


def test_pr_ready_freshness_write_records_base_sha(tmp_path: Path) -> None:
    """The readiness stamp should capture the validated base SHA (issue #5782)."""
    stamp_path = tmp_path / "ready.json"
    result = _run(
        "write",
        "--branch",
        "codex/712-pr-open-skill",
        "--head-sha",
        "abc123",
        "--base-ref",
        "origin/main",
        "--base-sha",
        "deadbeefcafe",
        "--stamp-path",
        str(stamp_path),
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["stamp"]["base_sha"] == "deadbeefcafe"


def test_pr_ready_freshness_status_reports_base_drift_as_advisory(tmp_path: Path) -> None:
    """Base drift is reported as a reviewable advisory, not a freshness failure.

    Issue #5782: a readiness stamp recorded against one origin/main commit remains
    fresh after origin/main advances, because the readiness gate already ran a
    base-drift recheck before recording the stamp and records it only when the drift
    is current or unrelated to the PR's changed paths. Failing status on any base
    movement would recreate the re-run friction the gate-level check exists to break,
    so status reports the drift as a visible advisory instead.
    """
    stamp_path = tmp_path / "ready.json"
    payload = {
        "branch": "codex/712-pr-open-skill",
        "base_ref": "origin/main",
        "base_sha": "oldsha0000000000000000000000000000000000",
        "head_sha": "abc123",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "status": "passed",
    }
    stamp_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(
        "status",
        "--branch",
        "codex/712-pr-open-skill",
        "--head-sha",
        "abc123",
        "--base-ref",
        "origin/main",
        "--base-sha",
        "newsha1111111111111111111111111111111111",
        "--stamp-path",
        str(stamp_path),
    )

    assert result.returncode == 0
    result_payload = json.loads(result.stdout)
    assert result_payload["fresh"] is True
    assert result_payload["reason"] == "fresh"
    # The drift must be surfaced for review even though the stamp stays fresh.
    drift = result_payload["base_drift"]
    assert drift["stamped_base_sha"].startswith("oldsha")
    assert drift["current_base_sha"].startswith("newsha")
    assert drift["base_ref"] == "origin/main"


def test_pr_ready_freshness_status_ignores_drift_when_sha_unknown(tmp_path: Path) -> None:
    """When neither the stamp nor the caller can resolve a base SHA, fall back to
    the base_ref string comparison (issue #5782 graceful degradation)."""
    stamp_path = tmp_path / "ready.json"
    payload = {
        "branch": "codex/712-pr-open-skill",
        "base_ref": "origin/main",
        "head_sha": "abc123",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "status": "passed",
    }
    stamp_path.write_text(json.dumps(payload), encoding="utf-8")

    # Caller passes no --base-sha (ref unresolvable locally); stamp has none either.
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

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["fresh"] is True
