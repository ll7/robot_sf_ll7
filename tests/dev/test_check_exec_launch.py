"""Tests for scripts/dev/check_exec_launch.py (issue #9321)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.dev import check_exec_launch as launcher
from scripts.dev.check_exec_launch import (
    check_launch_preflight,
    is_retry_allowlisted,
    run_with_launch_check,
)


def test_is_retry_allowlisted_git_read_only() -> None:
    """Allowlist correctly identifies read-only git commands."""
    assert is_retry_allowlisted(["git", "status"]) is True
    assert is_retry_allowlisted(["git", "ls-remote", "origin", "refs/heads/main"]) is True
    assert is_retry_allowlisted(["git", "rev-parse", "origin/main"]) is True
    assert is_retry_allowlisted(["git", "diff", "HEAD~1"]) is True
    assert is_retry_allowlisted(["git", "log", "-n", "1"]) is True
    assert is_retry_allowlisted(["git", "show", "HEAD"]) is True
    assert is_retry_allowlisted(["/usr/bin/git", "status"]) is True
    assert is_retry_allowlisted(["git", "-C", "/tmp", "status"]) is True
    assert is_retry_allowlisted(["git", "--git-dir=/repo/.git", "status"]) is True


def test_is_retry_allowlisted_git_mutating_rejected() -> None:
    """Mutating or checkout git commands are not allowlisted for automatic retry."""
    assert is_retry_allowlisted(["git", "commit", "-m", "test"]) is False
    assert is_retry_allowlisted(["git", "push", "origin", "main"]) is False
    assert is_retry_allowlisted(["git", "checkout", "main"]) is False
    assert is_retry_allowlisted(["git", "rebase", "main"]) is False
    assert is_retry_allowlisted(["git", "merge", "feat"]) is False
    assert is_retry_allowlisted(["git", "reset", "--hard"]) is False


def test_is_retry_allowlisted_gh_read_only() -> None:
    """Allowlist correctly identifies read-only gh commands."""
    assert is_retry_allowlisted(["gh", "pr", "view", "123"]) is True
    assert is_retry_allowlisted(["gh", "issue", "view", "456"]) is True
    assert is_retry_allowlisted(["gh", "pr", "checks", "123"]) is True
    assert is_retry_allowlisted(["gh", "pr", "list"]) is True
    assert is_retry_allowlisted(["gh", "issue", "list"]) is True
    assert is_retry_allowlisted(["gh", "api", "repos/ll7/robot_sf_ll7"]) is True
    assert is_retry_allowlisted(["gh", "api", "--method", "GET", "repos/ll7/robot_sf_ll7"]) is True
    assert is_retry_allowlisted(["gh", "api", "-X", "GET", "repos/ll7/robot_sf_ll7"]) is True


def test_is_retry_allowlisted_gh_mutating_rejected() -> None:
    """Mutating gh commands are not allowlisted for automatic retry."""
    assert is_retry_allowlisted(["gh", "pr", "create"]) is False
    assert is_retry_allowlisted(["gh", "pr", "merge", "123"]) is False
    assert is_retry_allowlisted(["gh", "pr", "edit", "123", "--add-label", "foo"]) is False
    assert (
        is_retry_allowlisted(["gh", "api", "--method", "POST", "repos/ll7/robot_sf_ll7"]) is False
    )
    assert is_retry_allowlisted(["gh", "api", "-X", "DELETE", "repos/ll7/robot_sf_ll7"]) is False


def test_is_retry_allowlisted_other_commands_rejected() -> None:
    """Arbitrary commands are not allowlisted for automatic retry."""
    assert is_retry_allowlisted(["python", "-c", "pass"]) is False
    assert is_retry_allowlisted(["bash", "-c", "ls"]) is False
    assert is_retry_allowlisted(["echo", "hello"]) is False
    assert is_retry_allowlisted([]) is False


def test_preflight_checks_success(tmp_path: Path) -> None:
    """Preflight passes for valid executable and directory."""
    passed, resolved, error = check_launch_preflight([sys.executable, "-c", "pass"], cwd=tmp_path)
    assert passed is True
    assert resolved is not None
    assert Path(resolved).is_file()
    assert error is None


def test_preflight_fails_on_nonexistent_cwd(tmp_path: Path) -> None:
    """Preflight fails if working directory does not exist."""
    bad_cwd = tmp_path / "does_not_exist"
    passed, resolved, error = check_launch_preflight([sys.executable], cwd=bad_cwd)
    assert passed is False
    assert resolved is None
    assert "working directory does not exist" in str(error)


def test_preflight_fails_on_nonexistent_executable_in_path(tmp_path: Path) -> None:
    """Preflight fails if executable is not found in PATH."""
    passed, resolved, error = check_launch_preflight(["nonexistent_binary_xyz_9321"], cwd=tmp_path)
    assert passed is False
    assert resolved is None
    assert "executable not found in PATH" in str(error)


def test_preflight_fails_on_nonexistent_explicit_executable_path(tmp_path: Path) -> None:
    """Preflight fails if explicit executable path does not exist."""
    bad_exe = tmp_path / "missing_tool"
    passed, resolved, error = check_launch_preflight([str(bad_exe)], cwd=tmp_path)
    assert passed is False
    assert resolved is None
    assert "executable path not found" in str(error)


def test_preflight_fails_on_empty_command(tmp_path: Path) -> None:
    """Preflight fails when command sequence is empty."""
    passed, _resolved, error = check_launch_preflight([], cwd=tmp_path)
    assert passed is False
    assert "empty" in str(error)


def test_run_with_launch_check_successful_execution(tmp_path: Path) -> None:
    """A valid command executes and returns a success receipt."""
    receipt = run_with_launch_check(
        [sys.executable, "-c", "print('output_ok')"],
        cwd=tmp_path,
        capture_output=True,
    )
    assert receipt.schema == launcher.SCHEMA
    assert receipt.phase == "execution"
    assert receipt.status == "ok"
    assert receipt.returncode == 0
    assert receipt.error is None
    assert receipt.retried is False
    assert receipt.suggested_retry is None
    assert receipt.stdout is not None and "output_ok" in receipt.stdout


def test_run_with_launch_check_command_failure_distinguished_from_launch(tmp_path: Path) -> None:
    """A command that exits non-zero is in phase execution with status failed, not launch_failed."""
    receipt = run_with_launch_check(
        [sys.executable, "-c", "import sys; sys.exit(42)"],
        cwd=tmp_path,
    )
    assert receipt.phase == "execution"
    assert receipt.status == "failed"
    assert receipt.returncode == 42
    assert receipt.suggested_retry is not None


def test_run_with_launch_check_preflight_failure_receipt(tmp_path: Path) -> None:
    """A preflight failure records phase preflight and status launch_failed without running."""
    receipt = run_with_launch_check(
        ["nonexistent_binary_xyz_9321", "arg1"],
        cwd=tmp_path,
        allow_retry=False,
    )
    assert receipt.phase == "preflight"
    assert receipt.status == "launch_failed"
    assert receipt.returncode is None
    assert receipt.error is not None and "executable not found" in receipt.error
    assert receipt.suggested_retry == "nonexistent_binary_xyz_9321 arg1"


def test_run_with_launch_check_retries_allowlisted_command_on_launch_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Allowlisted commands retry once when subprocess.Popen raises OSError."""
    attempts = 0

    class FakeProc:
        returncode = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            return "branch-sha", ""

    def mock_popen(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FileNotFoundError(2, "No such file or directory (transient CreateProcess)")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    receipt = run_with_launch_check(
        ["git", "ls-remote", "origin", "main"],
        cwd=tmp_path,
        capture_output=True,
        allow_retry=True,
        retry_delay_s=0.001,
    )

    assert attempts == 2
    assert receipt.retried is True
    assert receipt.phase == "execution"
    assert receipt.status == "ok"
    assert receipt.returncode == 0


def test_run_with_launch_check_non_allowlisted_command_does_not_retry_on_launch_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Non-allowlisted commands do not retry automatically; they surface suggested retry."""
    attempts = 0

    def mock_popen(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    receipt = run_with_launch_check(
        ["git", "push", "origin", "main"],
        cwd=tmp_path,
        allow_retry=True,
    )

    assert attempts == 1
    assert receipt.retried is False
    assert receipt.phase == "launch"
    assert receipt.status == "launch_failed"
    assert receipt.suggested_retry == "git push origin main"


def test_cli_json_and_receipt_file(tmp_path: Path) -> None:
    """CLI prints JSON and writes receipt file correctly."""
    receipt_file = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.dev.check_exec_launch",
            "--json",
            "--receipt-file",
            str(receipt_file),
            "--cwd",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            "print('cli_test')",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    stdout_data = json.loads(result.stdout)
    assert stdout_data["schema"] == launcher.SCHEMA
    assert stdout_data["status"] == "ok"
    assert stdout_data["phase"] == "execution"

    assert receipt_file.is_file()
    file_data = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert file_data == stdout_data


def test_cli_returns_127_on_launch_failure(tmp_path: Path) -> None:
    """CLI exits with 127 when launch fails."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.dev.check_exec_launch",
            "--cwd",
            str(tmp_path),
            "--",
            "nonexistent_command_9321",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 127
    assert "launch_failed" in result.stderr or "preflight" in result.stderr
