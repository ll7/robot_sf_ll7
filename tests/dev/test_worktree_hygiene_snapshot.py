"""Tests for worktree hygiene snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.dev import worktree_hygiene_snapshot as snapshot

if TYPE_CHECKING:
    from pathlib import Path


def _result(stdout: str = "", stderr: str = "", returncode: int = 0):
    return snapshot.subprocess.CompletedProcess(
        args=["git"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_parse_worktree_porcelain_handles_branch_and_detached() -> None:
    """Parse branch and detached rows from porcelain worktree output."""
    rows = snapshot._parse_worktree_porcelain(
        "\n".join(
            [
                "worktree /repo/main",
                "HEAD aaa",
                "branch refs/heads/main",
                "",
                "worktree /repo/detached",
                "HEAD bbb",
                "detached",
                "",
            ]
        )
    )

    assert rows == [
        {"path": "/repo/main", "head_sha": "aaa", "branch": "main"},
        {"path": "/repo/detached", "head_sha": "bbb", "detached": "true"},
    ]


def test_classify_issues_reports_dirty_missing_upstream_and_drift() -> None:
    """Report all hygiene issue classes represented by one row."""
    assert snapshot._classify_issues(
        branch="feature",
        is_detached=False,
        dirty_entries=2,
        upstream=None,
        ahead=1,
        behind=3,
    ) == ["dirty", "missing_upstream", "ahead", "behind"]


def test_build_snapshot_filters_and_counts(monkeypatch, tmp_path: Path) -> None:
    """Filter worktrees and aggregate issue counts in the snapshot."""
    main = tmp_path / "main"
    feature = tmp_path / "feature"
    main.mkdir()
    feature.mkdir()

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del timeout
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(
                "\n".join(
                    [
                        f"worktree {main}",
                        "HEAD aaa",
                        "branch refs/heads/main",
                        "",
                        f"worktree {feature}",
                        "HEAD bbb",
                        "branch refs/heads/feature",
                        "",
                    ]
                )
            )
        if args == ["git", "status", "--porcelain"]:
            return _result(" M changed.py\n" if cwd == str(feature) else "")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            if cwd == str(feature):
                return _result("origin/feature\n")
            return _result("origin/main\n")
        if args == ["git", "rev-list", "--left-right", "--count", "HEAD...origin/feature"]:
            return _result("1\t2\n")
        if args == ["git", "rev-list", "--left-right", "--count", "HEAD...origin/main"]:
            return _result("0\t0\n")
        raise AssertionError(f"unexpected command: {args} cwd={cwd}")

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    result = snapshot.build_snapshot(filters=["feature"], worktree_limit=10)

    assert result.total_worktrees == 2
    assert result.included_worktrees == 1
    assert result.worktrees_truncated is False
    assert result.issue_counts == {"ahead": 1, "behind": 1, "dirty": 1}
    assert result.worktrees[0].branch == "feature"
    assert result.worktrees[0].dirty_entries == 1
    assert result.worktrees[0].ahead == 1
    assert result.worktrees[0].behind == 2


def test_repo_status_is_optional(monkeypatch, tmp_path: Path) -> None:
    """Include current checkout status only when requested."""
    main = tmp_path / "main"
    main.mkdir()

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del cwd, timeout
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(f"worktree {main}\nHEAD aaa\nbranch refs/heads/main\n")
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        if args == ["git", "status", "--short", "--branch"]:
            return _result("## main...origin/main\n M docs.md\n")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("origin/main\n")
        if args == ["git", "rev-list", "--left-right", "--count", "HEAD...origin/main"]:
            return _result("0\t4\n")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    result = snapshot.build_snapshot(include_repo_status=True)

    assert result.repo_status is not None
    assert result.repo_status.branch_status == "## main...origin/main"
    assert result.repo_status.dirty_entries == 1
    assert result.repo_status.behind == 4


def test_missing_worktree_path_marks_status_failed(monkeypatch, tmp_path: Path) -> None:
    """Classify missing worktree paths as status failures."""
    main = tmp_path / "main"
    missing = tmp_path / "missing"
    main.mkdir()

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del timeout
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(f"worktree {missing}\nHEAD aaa\nbranch refs/heads/gone\n")
        if args == ["git", "status", "--porcelain"]:
            return _result(stderr="missing", returncode=127)
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result(stderr="missing", returncode=127)
        raise AssertionError(f"unexpected command: {args} cwd={cwd}")

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    result = snapshot.build_snapshot()

    assert result.included_worktrees == 1
    assert result.issue_counts == {"missing_upstream": 1, "status_failed": 1}
    assert result.worktrees[0].dirty_entries == -1


def test_current_worktree_is_reported_when_truncated(monkeypatch, tmp_path: Path) -> None:
    """Preserve current worktree identity even when rows are truncated."""
    first = tmp_path / "first"
    current = tmp_path / "current"
    first.mkdir()
    current.mkdir()

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del timeout
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(
                "\n".join(
                    [
                        f"worktree {first}",
                        "HEAD aaa",
                        "branch refs/heads/first",
                        "",
                        f"worktree {current}",
                        "HEAD bbb",
                        "branch refs/heads/current",
                    ]
                )
            )
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("origin/first\n")
        if args == ["git", "rev-list", "--left-right", "--count", "HEAD...origin/first"]:
            return _result("0\t0\n")
        raise AssertionError(f"unexpected command: {args} cwd={cwd}")

    monkeypatch.chdir(current)
    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    result = snapshot.build_snapshot(worktree_limit=1)

    assert result.current_worktree == str(current)
    assert result.included_worktrees == 1
    assert result.worktrees_truncated is True
