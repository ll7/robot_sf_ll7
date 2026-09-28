"""Tests for the PR-gate worktree deterministic health/recreate guard."""

from __future__ import annotations

import json
import os
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

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
        assert health.recovery["loss_boundary"] == guard.MISSING_STATE_LOSS_BOUNDARY
        assert health.recovery["local_state_restored"] is False

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
        assert health.branch is None
        assert health.head_sha == "0123456789abcdef"
        assert health.recovery == {
            "status": "partial",
            "branch_checkout": "available_from_lease",
            "local_state_restored": False,
            "loss_boundary": guard.MISSING_STATE_LOSS_BOUNDARY,
            "next_action": "recover_dirty_state_from_backup_or_handoff",
        }

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
        assert result.recovery["loss_boundary"] == guard.MISSING_STATE_LOSS_BOUNDARY
        assert result.recovery["local_state_restored"] is False

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
        assert result.head_sha == "head-sha"
        assert result.recovery["loss_boundary"] == guard.MISSING_STATE_LOSS_BOUNDARY
        assert result.recovery["local_state_restored"] is False
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

    def test_recreate_serializes_registry_mutation(self, mock_git_dirs: Path) -> None:
        """Recovery enters the shared lifecycle lock for add and lease refresh."""
        wt = mock_git_dirs / "gone-wt"
        _write_active_lease(
            mock_git_dirs,
            wt,
            owner="auto-smart-routing",
            head_ref="gate/branch",
            head_sha="head-sha",
        )
        lock_events: list[str] = []

        @contextmanager
        def fake_lock():
            lock_events.append("enter")
            yield
            lock_events.append("exit")

        def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="")

        with (
            patch.object(guard, "_run_command", side_effect=fake_run),
            patch("scripts.dev.pr_gate_lease.worktree_lifecycle_lock", fake_lock),
        ):
            result = guard.recreate_gate_worktree(wt)

        assert result.recreated is True
        assert lock_events == ["enter", "exit", "enter", "exit"]


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
        assert recreate.recovery["loss_boundary"] == guard.MISSING_STATE_LOSS_BOUNDARY

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


def _init_index_lock_worktree(path: Path) -> Path:
    """Create a small real Git worktree and return its resolved index-lock path."""
    path.mkdir()
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--git-path", "index.lock"],
        capture_output=True,
        text=True,
        check=True,
    )
    lock_path = Path(result.stdout.strip())
    if not lock_path.is_absolute():
        lock_path = path / lock_path
    return lock_path.resolve()


def _fake_proc_root(path: Path, *, owner: str, lock_path: Path | None = None) -> Path:
    """Create a minimal process table for deterministic lock-owner tests."""
    proc_root = path / "proc"
    if owner == "unavailable":
        return proc_root
    proc_root.mkdir()
    if owner == "active":
        descriptor_dir = proc_root / "123" / "fd"
        descriptor_dir.mkdir(parents=True)
        assert lock_path is not None
        (descriptor_dir / "9").symlink_to(lock_path)
    elif owner == "ambiguous":
        process_dir = proc_root / "456"
        process_dir.mkdir()
        (process_dir / "fd").write_text("uninspectable descriptor table", encoding="utf-8")
    elif owner == "proven_orphan":
        (proc_root / "1" / "fd").mkdir(parents=True)
    return proc_root


@pytest.mark.parametrize(
    ("owner", "expected_owner", "expected_pids"),
    [
        ("active", "active", [123]),
        ("proven_orphan", "proven_orphan", []),
        ("ambiguous", "ambiguous", []),
        ("unavailable", "unavailable", []),
    ],
)
def test_preflight_blocks_index_lock_and_preserves_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    owner: str,
    expected_owner: str,
    expected_pids: list[int],
) -> None:
    """Lock owner classes all fail closed without changing the lock or worktree."""
    worktree = tmp_path / "worktree"
    lock_path = _init_index_lock_worktree(worktree)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"preserve this lock\n")
    stale_time = datetime.now(UTC).timestamp() - 30
    os.utime(lock_path, (stale_time, stale_time))
    dirty_file = worktree / "untracked.txt"
    dirty_file.write_text("preserve worktree state\n", encoding="utf-8")
    initial_lock_stat = lock_path.stat()
    initial_lock_bytes = lock_path.read_bytes()
    initial_dirty_bytes = dirty_file.read_bytes()
    index_path = lock_path.parent / "index"
    initial_index_exists = index_path.exists()
    initial_index_bytes = index_path.read_bytes() if initial_index_exists else None
    proc_root = _fake_proc_root(tmp_path, owner=owner, lock_path=lock_path)

    with patch.object(guard, "PROC_ROOT", proc_root):
        exit_code = guard.main(["preflight", "--path", str(worktree), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code != 0
    assert payload["schema"] == "gate_worktree_preflight.v1"
    assert payload["status"] == "index_lock_present"
    inspection = payload["index_lock"]
    assert inspection["index_lock_path"] == str(lock_path)
    assert inspection["lock_exists"] is True
    assert inspection["lock_mtime_utc"].endswith("Z")
    assert inspection["lock_age_seconds"] >= 29
    assert inspection["lock_size_bytes"] == len(initial_lock_bytes)
    assert inspection["dirty_state"] == "dirty"
    assert inspection["owner_classification"] == expected_owner
    assert inspection["owner_pids"] == expected_pids
    assert payload["recovery_packet"]["index_lock_path"] == str(lock_path)
    assert payload["recovery_packet"]["next_action"] != "continue_local_refresh"
    after_lock_stat = lock_path.stat()
    assert lock_path.read_bytes() == initial_lock_bytes
    assert after_lock_stat.st_ino == initial_lock_stat.st_ino
    assert after_lock_stat.st_size == initial_lock_stat.st_size
    assert after_lock_stat.st_mtime_ns == initial_lock_stat.st_mtime_ns
    assert dirty_file.read_bytes() == initial_dirty_bytes
    assert index_path.exists() is initial_index_exists
    if initial_index_exists:
        assert index_path.read_bytes() == initial_index_bytes


def test_preflight_allows_absent_index_lock_without_recovery_packet(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An absent lock is the only preflight state that authorizes local work."""
    worktree = tmp_path / "worktree"
    lock_path = _init_index_lock_worktree(worktree)
    proc_root = _fake_proc_root(tmp_path, owner="proven_orphan")

    with (
        patch.object(guard, "PROC_ROOT", proc_root),
        patch.object(guard, "_proc_lock_owners", side_effect=AssertionError("unexpected scan")),
    ):
        exit_code = guard.main(["preflight", "--path", str(worktree), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "ready"
    assert payload["index_lock"]["index_lock_path"] == str(lock_path)
    assert payload["index_lock"]["lock_exists"] is False
    assert payload["index_lock"]["owner_classification"] == "absent"
    assert payload["recovery_packet"] is None
