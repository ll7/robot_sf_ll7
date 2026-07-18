"""Tests for the PR-gate worktree deterministic health/recreate guard."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from scripts.dev import gate_worktree_guard as guard
from scripts.dev.pr_gate_lease import (
    PRGateLease,
    lease_path,
    legacy_lease_path,
    save_lease,
)


@pytest.fixture
def mock_git_dirs(tmp_path: Path) -> None:
    """Mock git common dir and repo root to use a temp directory."""
    from scripts.dev import pr_gate_lease

    git_common = tmp_path / ".git"
    git_common.mkdir()

    with (
        patch.object(pr_gate_lease, "_git_common_dir", return_value=git_common),
        patch.object(pr_gate_lease, "_repo_root", return_value=tmp_path),
    ):
        yield tmp_path


def _write_active_lease(tmp_path: Path, wt_path: Path, **overrides: object) -> None:
    """Write an active (non-expired) lease recording ``wt_path`` as the worktree."""
    now = datetime.now(UTC)
    lease = PRGateLease(
        schema="pr_gate_lease.v1",
        created_at=now.isoformat(),
        expires_at=(now + timedelta(hours=2)).isoformat(),
        pr_number=overrides.get("pr_number", 5819),
        gate_id=overrides.get("gate_id", "gate-5819"),
        owner=overrides.get("owner", "auto-smart-routing"),
        last_heartbeat=now.isoformat(),
        worktree_path=str(wt_path),
        head_ref=overrides.get("head_ref"),
        head_sha=overrides.get("head_sha", "0123456789abcdef"),
    )
    save_lease(lease, lease_path(wt_path))


class TestVerifyGateWorktree:
    """Tests for verify_gate_worktree."""

    def test_healthy_when_path_exists(self, mock_git_dirs: Path) -> None:
        """An existing worktree path is reported healthy."""
        wt = mock_git_dirs / "gate-wt"
        wt.mkdir()

        health = guard.verify_gate_worktree(wt)

        assert health.exists is True
        assert health.classification == "healthy"
        assert health.cleanup_owner is None

    def test_missing_without_lease_reports_missing(self, mock_git_dirs: Path) -> None:
        """A missing path with no lease is reported missing, owner unknown."""
        wt = mock_git_dirs / "gone-wt"

        health = guard.verify_gate_worktree(wt)

        assert health.exists is False
        assert health.classification == "missing"
        assert health.cleanup_owner is None

    def test_missing_with_active_lease_reports_owner(self, mock_git_dirs: Path) -> None:
        """A missing path with a live lease reports the cleanup owner."""
        wt = mock_git_dirs / "gone-wt"
        _write_active_lease(mock_git_dirs, wt, owner="auto-smart-routing", pr_number=5819)

        health = guard.verify_gate_worktree(wt)

        assert health.exists is False
        assert health.classification == "missing"
        assert health.cleanup_owner is not None
        assert "owner=auto-smart-routing" in health.cleanup_owner
        assert "pr=#5819" in health.cleanup_owner
        assert health.lease_pr_number == 5819

    def test_missing_with_expired_lease_has_no_owner(self, mock_git_dirs: Path) -> None:
        """A missing path whose lease is expired is not attributed to an owner."""
        wt = mock_git_dirs / "gone-wt"
        now = datetime.now(UTC)
        lease = PRGateLease(
            schema="pr_gate_lease.v1",
            created_at=(now - timedelta(hours=3)).isoformat(),
            expires_at=(now - timedelta(hours=1)).isoformat(),
            pr_number=5819,
            gate_id="gate-5819",
            owner="auto-smart-routing",
            last_heartbeat=(now - timedelta(hours=3)).isoformat(),
            worktree_path=str(wt),
        )
        save_lease(lease, lease_path(wt))

        health = guard.verify_gate_worktree(wt)

        assert health.exists is False
        assert health.cleanup_owner is None


class TestRecreateGateWorktree:
    """Tests for recreate_gate_worktree."""

    def test_no_recreate_when_healthy(self, mock_git_dirs: Path) -> None:
        """An existing worktree is left untouched (recreated=False)."""
        wt = mock_git_dirs / "gate-wt"
        wt.mkdir()

        result = guard.recreate_gate_worktree(wt)

        assert result.recreated is False
        assert result.error is None

    def test_cannot_recreate_without_lease(self, mock_git_dirs: Path) -> None:
        """A missing path with no lease cannot be recreated safely."""
        wt = mock_git_dirs / "gone-wt"

        result = guard.recreate_gate_worktree(wt)

        assert result.recreated is False
        assert result.error is not None
        assert "no live lease" in result.error

    def test_recreate_from_lease_calls_worktree_add(self, mock_git_dirs: Path) -> None:
        """A missing path with a live lease recreates the worktree from its branch."""
        wt = mock_git_dirs / "gone-wt"
        _write_active_lease(
            mock_git_dirs,
            wt,
            owner="auto-smart-routing",
            head_ref="gate/branch",
            head_sha="head-sha",
        )

        calls: list[list[str]] = []

        def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            calls.append(list(args))
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="")

        with patch.object(guard, "_run_command", side_effect=fake_run):
            result = guard.recreate_gate_worktree(wt)

        assert result.recreated is True
        assert result.branch == "gate/branch"
        add_calls = [c for c in calls if c[:3] == ["git", "worktree", "add"]]
        assert add_calls, "expected git worktree add to be invoked"
        assert str(wt) in add_calls[0]
        assert add_calls[0] == ["git", "worktree", "add", "--force", str(wt), "gate/branch"]

    def test_recreate_failure_reports_error(self, mock_git_dirs: Path) -> None:
        """A failed git worktree add surfaces the git error deterministically."""
        wt = mock_git_dirs / "gone-wt"
        _write_active_lease(
            mock_git_dirs,
            wt,
            owner="auto-smart-routing",
            head_ref="gate/branch",
            head_sha="head-sha",
        )

        def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            if args[:3] == ["git", "worktree", "add"]:
                return subprocess.CompletedProcess(
                    args=args, returncode=1, stdout="", stderr="fatal: boom"
                )
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="")

        with patch.object(guard, "_run_command", side_effect=fake_run):
            result = guard.recreate_gate_worktree(wt)

        assert result.recreated is False
        assert "boom" in (result.error or "")


class TestEnsureGateWorktree:
    """Tests for ensure_gate_worktree."""

    def test_ensure_healthy_no_recreate(self, mock_git_dirs: Path) -> None:
        """Healthy worktree: ensure returns health with no recreate result."""
        wt = mock_git_dirs / "gate-wt"
        wt.mkdir()

        health, recreate = guard.ensure_gate_worktree(wt)

        assert health.exists is True
        assert recreate is None

    def test_ensure_missing_recreates(self, mock_git_dirs: Path) -> None:
        """Missing worktree with lease: ensure recreates and returns both results."""
        wt = mock_git_dirs / "gone-wt"
        _write_active_lease(
            mock_git_dirs,
            wt,
            owner="auto-smart-routing",
            head_ref="gate/branch",
            head_sha="head-sha",
        )

        def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="")

        with patch.object(guard, "_run_command", side_effect=fake_run):
            health, recreate = guard.ensure_gate_worktree(wt)

        assert health.exists is False
        assert recreate is not None and recreate.recreated is True

    def test_legacy_lease_is_matched_by_worktree_path(self, mock_git_dirs: Path) -> None:
        """A legacy shared lease is used only when it identifies this path."""
        wt = mock_git_dirs / "gone-wt"
        _write_active_lease(
            mock_git_dirs,
            wt,
            owner="legacy-gate",
            head_ref="gate/legacy",
            head_sha="legacy-sha",
        )
        hashed = lease_path(wt)
        hashed.unlink()
        legacy = PRGateLease(
            schema="pr_gate_lease.v1",
            created_at=datetime.now(UTC).isoformat(),
            expires_at=(datetime.now(UTC) + timedelta(hours=2)).isoformat(),
            pr_number=5819,
            gate_id="legacy-gate",
            owner="legacy-gate",
            last_heartbeat=datetime.now(UTC).isoformat(),
            worktree_path=str(wt),
            head_ref="gate/legacy",
            head_sha="legacy-sha",
        )
        save_lease(legacy, legacy_lease_path())

        health = guard.verify_gate_worktree(wt)

        assert health.cleanup_owner == "owner=legacy-gate; pr=#5819; gate=legacy-gate"


def test_guard_module_is_valid_json_serializable(mock_git_dirs: Path) -> None:
    """Health dataclass serializes cleanly to JSON."""
    from dataclasses import asdict

    wt = mock_git_dirs / "gate-wt"
    wt.mkdir()
    health = guard.verify_gate_worktree(wt)
    payload = json.loads(json.dumps(asdict(health)))
    assert payload["classification"] == "healthy"
