"""Tests for the explicit base-sensitive changed-file selector."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.dev.base_sensitive_selector import (
    BASE_SENSITIVE,
    ORDINARY,
    SELECTOR_VERSION,
    UNKNOWN,
    classify_changed_files,
    find_base_sensitive_test_files,
)

SENSITIVE = ["tests/test_snapshot.py", "tests/test_tuple_contract.py"]


def test_selector_classifies_marker_file_intersection() -> None:
    result = classify_changed_files(
        ["robot_sf/runtime.py", "tests/test_snapshot.py"],
        sensitive_files=SENSITIVE,
    )

    assert result["status"] == BASE_SENSITIVE
    assert result["selector"] == SELECTOR_VERSION
    assert result["changed_sensitive_files"] == ["tests/test_snapshot.py"]


def test_selector_classifies_unrelated_change_as_ordinary() -> None:
    result = classify_changed_files(["scripts/dev/helper.py"], sensitive_files=SENSITIVE)

    assert result["status"] == ORDINARY
    assert result["changed_sensitive_files"] == []


def test_selector_fails_closed_when_inventory_is_missing() -> None:
    result = classify_changed_files(None, sensitive_files=SENSITIVE)

    assert result["status"] == UNKNOWN
    assert result["reason"] == "changed_file_inventory_unavailable"


def test_path_normalization_preserves_hidden_directory_names() -> None:
    """Normalization removes only an explicit relative prefix."""
    result = classify_changed_files(["./.agents/skills/example.md"], sensitive_files=SENSITIVE)

    assert result["status"] == ORDINARY
    assert result["changed_files"] == [".agents/skills/example.md"]


def _init_fixture_repo(repo_root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(repo_root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "init",
        ],
        check=True,
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_selection_is_repository_bound_and_ignores_nested_copies(tmp_path: Path) -> None:
    """Only Git-tracked test files are selected (issue #8025 regression)."""
    _init_fixture_repo(tmp_path)
    tracked = tmp_path / "tests" / "test_tracked_marker.py"
    _write(tracked, f"def test_x():\n    assert {BASE_SENSITIVE!r} in {BASE_SENSITIVE!r}\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tests/test_tracked_marker.py"], check=True)

    ignored_copies = [
        tmp_path / ".emdash" / "wt-a" / "tests" / "test_copy.py",
        tmp_path / ".worktrees" / "wt-b" / "tests" / "test_copy.py",
        tmp_path / "output" / "cache" / "tests" / "test_copy.py",
    ]
    for copy in ignored_copies:
        _write(copy, "def test_copy():\n    assert True\n")

    selected = find_base_sensitive_test_files(tmp_path)

    assert selected == ["tests/test_tracked_marker.py"]


def test_untracked_nested_worktree_test_is_excluded_even_with_marker(tmp_path: Path) -> None:
    """An untracked copy carrying the marker text is still excluded."""
    _init_fixture_repo(tmp_path)
    ignored_marker_copy = tmp_path / ".worktrees" / "wt" / "tests" / "test_marker_copy.py"
    _write(ignored_marker_copy, "base_sensitive = True\n")

    assert find_base_sensitive_test_files(tmp_path) == []


def test_git_failure_fails_closed(tmp_path: Path) -> None:
    """A broken Git index must fail closed instead of widening the selection."""
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests" / "test_x.py").write_text("base_sensitive\n", encoding="utf-8")
    (tmp_path / ".git").write_text("not a repository\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="git ls-files failed"):
        find_base_sensitive_test_files(tmp_path)


def test_real_repository_selection_contains_no_ignored_copies() -> None:
    """The real checkout's selection must never include ignored nested copies."""
    repo_root = Path(__file__).resolve().parents[2]
    selected = find_base_sensitive_test_files(repo_root)
    forbidden_prefixes = (".emdash/", ".worktrees/", "output/")
    assert selected, "selector found no base-sensitive files in the real repository"
    assert not any(path.startswith(forbidden_prefixes) for path in selected)
