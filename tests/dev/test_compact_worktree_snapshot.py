"""Tests for compact worktree bootstrap snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.dev import compact_worktree_snapshot as snapshot

if TYPE_CHECKING:
    from pathlib import Path


def _result(stdout: str = "", stderr: str = "", returncode: int = 0):
    return snapshot.subprocess.CompletedProcess(
        args=["git"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_parse_worktree_porcelain_extracts_compact_rows() -> None:
    """Porcelain worktree output should become branch/path/head rows."""
    rows = snapshot._parse_worktree_porcelain(
        "\n".join(
            [
                "worktree /repo/main",
                "HEAD abc123",
                "branch refs/heads/main",
                "",
                "worktree /repo/issue-2715",
                "HEAD def456",
                "branch refs/heads/issue-2715-compact-ci-worktree-snapshots",
                "",
            ]
        )
    )

    assert rows == [
        {"path": "/repo/main", "head": "abc123", "branch": "main"},
        {
            "path": "/repo/issue-2715",
            "head": "def456",
            "branch": "issue-2715-compact-ci-worktree-snapshots",
        },
    ]


def test_detect_worktrees_filters_by_issue_slug_and_reports_total(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Filtering should avoid dumping every linked worktree into parent context."""
    main = tmp_path / "main"
    issue = tmp_path / "issue-2715-compact-ci-worktree-snapshots"
    other = tmp_path / "other"
    main.mkdir()
    issue.mkdir()
    other.mkdir()
    (issue / ".git").write_text("gitdir: /repo/.git/worktrees/issue-2715\n", encoding="utf-8")

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del cwd, timeout
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(
                "\n".join(
                    [
                        f"worktree {main}",
                        "HEAD aaa",
                        "branch refs/heads/main",
                        "",
                        f"worktree {issue}",
                        "HEAD bbb",
                        "branch refs/heads/issue-2715-compact-ci-worktree-snapshots",
                        "",
                        f"worktree {other}",
                        "HEAD ccc",
                        "branch refs/heads/other-work",
                        "",
                    ]
                )
            )
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    worktrees, total, truncated = snapshot._detect_worktrees(
        current_path=issue,
        main_repo_root=main,
        filters=["2715"],
        limit=20,
    )

    assert total == 3
    assert truncated is False
    assert [row.branch for row in worktrees] == ["issue-2715-compact-ci-worktree-snapshots"]
    assert worktrees[0].is_current is True
    assert worktrees[0].is_fresh is True
    assert worktrees[0].bootstrap_required is True


def test_include_all_worktrees_does_not_slice_to_zero(monkeypatch, tmp_path: Path) -> None:
    """The include-all path should include rows instead of applying a zero limit."""
    worktree = tmp_path / "issue-1"
    worktree.mkdir()

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del cwd, timeout
        if args == ["git", "branch", "--show-current"]:
            return _result("issue-1\n")
        if args == ["git", "rev-parse", "HEAD"]:
            return _result("abc123\n")
        if args == ["git", "rev-parse", "--git-common-dir"]:
            return _result(str(tmp_path / ".git") + "\n")
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(
                "\n".join(
                    [
                        f"worktree {worktree}",
                        "HEAD abc123",
                        "branch refs/heads/issue-1",
                        "",
                    ]
                )
            )
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.chdir(worktree)
    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    result = snapshot.build_snapshot(include_all_worktrees=True)

    assert result.worktree_count == 1
    assert result.included_worktree_count == 1
    assert result.worktrees_truncated is False
