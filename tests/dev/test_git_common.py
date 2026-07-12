"""Tests for scripts/dev/git_common.py shared artifact-path helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts.dev.git_common import resolve_agent_artifact_dir, resolve_git_common_dir


def _completed(returncode: int = 0, stdout: str = "", stderr: str = ""):
    import subprocess

    return subprocess.CompletedProcess(
        args=["git"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_resolve_git_common_dir_returns_path_in_normal_checkout() -> None:
    """In a normal checkout, resolve_git_common_dir returns an absolute path."""
    result = resolve_git_common_dir()
    assert result is not None
    assert result.is_absolute()


def test_resolve_git_common_dir_returns_none_on_git_failure() -> None:
    """When git fails, resolve_git_common_dir returns None."""
    with patch("scripts.dev.git_common.subprocess.run", return_value=_completed(128)):
        assert resolve_git_common_dir() is None


def test_resolve_git_common_dir_returns_none_on_empty_output() -> None:
    """When git returns empty stdout, resolve_git_common_dir returns None."""
    with patch("scripts.dev.git_common.subprocess.run", return_value=_completed(0, "")):
        assert resolve_git_common_dir() is None


def test_resolve_agent_artifact_dir_contains_codex_agent_runs(tmp_path: Path) -> None:
    """The resolved artifact dir should sit under codex-agent-runs/active/."""
    fake_common = tmp_path / "shared-git"
    fake_common.mkdir()
    with patch(
        "scripts.dev.git_common.resolve_git_common_dir",
        return_value=fake_common,
    ):
        result = resolve_agent_artifact_dir("my-subdir", mkdir=False)

    assert result == fake_common / "codex-agent-runs" / "active" / "my-subdir"


def test_resolve_agent_artifact_dir_creates_directory_by_default(tmp_path: Path) -> None:
    """With mkdir=True (default), the directory should be created."""
    fake_common = tmp_path / "shared-git"
    fake_common.mkdir()
    with patch(
        "scripts.dev.git_common.resolve_git_common_dir",
        return_value=fake_common,
    ):
        result = resolve_agent_artifact_dir("created-subdir")

    assert result.is_dir()


def test_resolve_agent_artifact_dir_no_mkdir_when_disabled(tmp_path: Path) -> None:
    """With mkdir=False, the directory should not be created."""
    fake_common = tmp_path / "shared-git"
    fake_common.mkdir()
    with patch(
        "scripts.dev.git_common.resolve_git_common_dir",
        return_value=fake_common,
    ):
        result = resolve_agent_artifact_dir("no-create", mkdir=False)

    assert not result.exists()


def test_resolve_agent_artifact_dir_fallback_without_git(tmp_path: Path) -> None:
    """When git is unavailable, fall back to output/tmp/<subdir>."""
    with patch("scripts.dev.git_common.resolve_git_common_dir", return_value=None):
        result = resolve_agent_artifact_dir("fallback-test", mkdir=False)

    assert result == Path("output") / "tmp" / "fallback-test"


def test_resolve_agent_artifact_dir_requires_subdir_name() -> None:
    """An empty subdir name should not crash; the caller controls validation."""
    # The function accepts any string; just verify it returns a path.
    with patch("scripts.dev.git_common.resolve_git_common_dir", return_value=None):
        result = resolve_agent_artifact_dir("", mkdir=False)
    assert result == Path("output") / "tmp"
