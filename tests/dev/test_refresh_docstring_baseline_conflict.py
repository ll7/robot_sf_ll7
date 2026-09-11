"""Offline unit coverage for the docstring-baseline refresh helper (#9057)."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from scripts.dev import refresh_docstring_baseline_conflict as refresh

if TYPE_CHECKING:
    from pathlib import Path


def _completed(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


class FakeRunner:
    """Scripted subprocess runner covering git, gh, and validator commands."""

    def __init__(
        self,
        worktree: Path,
        *,
        branch: str = "docs/child",
        conflicts: tuple[str, ...] = (),
        fail_mode: str | None = None,
        dirty: bool = False,
        push_rc: int = 0,
        pr_head: str | None = None,
    ) -> None:
        """Record commands and return deterministic results for the refresh flow."""
        self.worktree = worktree
        self.branch = branch
        self.conflicts = conflicts
        self.fail_mode = fail_mode
        self.dirty = dirty
        self.push_rc = push_rc
        self.pr_head = pr_head if pr_head is not None else branch
        self.calls: list[tuple[str, ...]] = []

    def __call__(
        self,
        command: list[str],
        *,
        cwd: Path,
        timeout: int = refresh.DEFAULT_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(tuple(command))
        if command[0] == "gh":
            return _completed(
                command,
                stdout=json.dumps({"state": "OPEN", "headRefName": self.pr_head}),
            )
        if command[0] == sys.executable:
            return self._validator_result(command)
        return self._git_result(command)

    def _validator_result(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        mode = command[command.index("--mode") + 1]
        if mode == self.fail_mode:
            return _completed(command, returncode=1, stdout=f"{mode} failed")
        return _completed(command, stdout=f"{mode} ok")

    def _identity_result(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[1] == "symbolic-ref":
            return _completed(command, stdout=f"{self.branch}\n")
        if "--show-toplevel" in command:
            return _completed(command, stdout=str(self.worktree))
        return _completed(command, stdout="1" * 40)

    def _status_result(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        if self.dirty and "--untracked-files=no" in command:
            return _completed(command, stdout=" M fixture.txt\n")
        if "--" in command:
            return _completed(command, stdout=f" M {refresh.BASELINE_PATH}\n")
        return _completed(command)

    def _merge_result(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        if "--abort" in command:
            return _completed(command)
        return _completed(
            command,
            returncode=1 if self.conflicts else 0,
            stderr="CONFLICT" if self.conflicts else "",
        )

    def _git_result(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        subcommand = command[1]
        if subcommand in {"rev-parse", "symbolic-ref"}:
            return self._identity_result(command)
        if subcommand == "status":
            return self._status_result(command)
        if subcommand == "merge":
            return self._merge_result(command)
        if subcommand in {"fetch", "checkout", "add", "commit"}:
            return _completed(command)
        if subcommand == "diff":
            joined = "\n".join(self.conflicts)
            return _completed(command, stdout=f"{joined}\n" if joined else "")
        if subcommand == "push":
            return _completed(
                command,
                returncode=self.push_rc,
                stderr="rejected" if self.push_rc else "",
            )
        raise AssertionError(f"unexpected git command: {command}")

    def called(self, *prefix: str) -> bool:
        return any(call[: len(prefix)] == prefix for call in self.calls)


def _refresh(worktree: Path, runner: FakeRunner, **overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "worktree": worktree,
        "repo": "ll7/robot_sf_ll7",
        "pr": None,
        "branch": None,
        "base_ref": "origin/main",
        "no_push": False,
        "dry_run": False,
    }
    arguments.update(overrides)
    with patch.object(refresh, "_run", runner):
        return refresh.refresh(**arguments)  # type: ignore[arg-type]


def test_refresh_resolves_baseline_conflict_and_pushes(tmp_path: Path) -> None:
    """The baseline-only conflict path resolves, validates, commits, and pushes."""
    runner = FakeRunner(
        tmp_path,
        conflicts=(str(refresh.BASELINE_PATH),),
    )

    payload = _refresh(tmp_path, runner)

    assert payload["ok"] is True
    assert payload["pushed"] is True
    assert payload["baseline_changed"] is True
    assert runner.called("git", "checkout", "--theirs")
    assert runner.called("git", "push", "origin")
    assert not runner.called("git", "merge", "--abort")


def test_refresh_aborts_on_unexpected_conflict(tmp_path: Path) -> None:
    """A conflict outside the baseline aborts the merge without pushing."""
    runner = FakeRunner(
        tmp_path,
        conflicts=(str(refresh.BASELINE_PATH), "robot_sf/example.py"),
    )

    with pytest.raises(refresh.RefreshError, match="resolve manually"):
        _refresh(tmp_path, runner)

    assert runner.called("git", "merge", "--abort")
    assert not runner.called("git", "push")


def test_refresh_aborts_and_does_not_push_when_validation_fails(tmp_path: Path) -> None:
    """Failed ratchet validation restores the pre-merge state and never pushes."""
    runner = FakeRunner(
        tmp_path,
        conflicts=(str(refresh.BASELINE_PATH),),
        fail_mode="ratchet",
    )

    with pytest.raises(refresh.RefreshError, match="ratchet"):
        _refresh(tmp_path, runner)

    assert runner.called("git", "merge", "--abort")
    assert not runner.called("git", "commit")
    assert not runner.called("git", "push")


def test_refresh_refuses_dirty_worktree(tmp_path: Path) -> None:
    """Tracked local changes stop the refresh before any fetch or merge."""
    runner = FakeRunner(tmp_path, dirty=True)

    with pytest.raises(refresh.RefreshError, match="uncommitted"):
        _refresh(tmp_path, runner)

    assert not runner.called("git", "fetch")
    assert not runner.called("git", "merge")


def test_refresh_refuses_branch_mismatch(tmp_path: Path) -> None:
    """A requested branch that is not checked out stops the refresh."""
    runner = FakeRunner(tmp_path, branch="docs/actual")

    with pytest.raises(refresh.RefreshError, match="does not match checkout branch"):
        _refresh(tmp_path, runner, branch="docs/requested")

    assert not runner.called("git", "fetch")


def test_refresh_verifies_pr_identity(tmp_path: Path) -> None:
    """A PR whose head branch differs from the checkout is refused."""
    runner = FakeRunner(tmp_path, pr_head="docs/other")

    with pytest.raises(refresh.RefreshError, match="head branch"):
        _refresh(tmp_path, runner, pr=42)


def test_refresh_dry_run_does_not_merge(tmp_path: Path) -> None:
    """Dry-run mode reports the plan after fetching without mutating history."""
    runner = FakeRunner(tmp_path)

    payload = _refresh(tmp_path, runner, dry_run=True)

    assert payload["ok"] is True
    assert payload["status"] == "dry_run"
    assert runner.called("git", "fetch")
    assert not runner.called("git", "merge")


def test_refresh_push_failure_reports_error(tmp_path: Path) -> None:
    """A rejected push surfaces as a failure after local validation passed."""
    runner = FakeRunner(tmp_path, push_rc=1)

    with pytest.raises(refresh.RefreshError, match="push failed"):
        _refresh(tmp_path, runner)


def test_main_prints_error_payload_for_branch_mismatch(tmp_path: Path) -> None:
    """The CLI returns exit 1 and one JSON error payload for refused refreshes."""
    runner = FakeRunner(tmp_path, branch="docs/actual")

    with patch.object(refresh, "_run", runner):
        exit_code = refresh.main(["--worktree", str(tmp_path), "--branch", "docs/requested"])

    assert exit_code == 1
