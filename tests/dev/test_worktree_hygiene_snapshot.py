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


def _lookup(status: str, reason: str | None = None):
    return snapshot.LookupState(status=status, reason=reason)


def _artifacts(classification: str = "none", status: str = "ok"):
    return [
        snapshot.ArtifactRootInspection(
            root="output",
            classification=classification,
            status=status,
        )
    ]


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


def test_current_worktree_ignores_malformed_rows(monkeypatch, tmp_path: Path) -> None:
    """Do not let rows without paths match the current checkout."""
    current = tmp_path / "current"
    current.mkdir()

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del timeout
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(
                "\n".join(
                    [
                        "HEAD malformed",
                        "",
                        f"worktree {current}",
                        "HEAD aaa",
                        "branch refs/heads/current",
                    ]
                )
            )
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("origin/current\n")
        if args == ["git", "rev-list", "--left-right", "--count", "HEAD...origin/current"]:
            return _result("0\t0\n")
        raise AssertionError(f"unexpected command: {args} cwd={cwd}")

    monkeypatch.chdir(current)
    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    result = snapshot.build_snapshot()

    assert result.current_worktree == str(current)


def test_retirement_plan_marks_clean_merged_worktree_removable(monkeypatch, tmp_path: Path) -> None:
    """A clean, merged, unclaimed row with no durable artifacts is removable."""
    main = tmp_path / "main"
    done = tmp_path / "done"
    main.mkdir()
    done.mkdir()

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del cwd, timeout
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(f"worktree {done}\nHEAD aaa\nbranch refs/heads/done\n")
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("origin/done\n")
        if args == ["git", "rev-list", "--left-right", "--count", "HEAD...origin/done"]:
            return _result("0\t0\n")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    result = snapshot.build_snapshot(
        include_retirement_plan=True,
        claim_lookup=lambda row: _lookup("inactive"),
        merge_lookup=lambda row: _lookup("merged"),
        artifact_lookup=lambda row: _artifacts(),
    )

    assert result.worktrees[0].retirement is not None
    assert result.worktrees[0].retirement.action == snapshot.RETIREMENT_REMOVABLE


def test_retirement_plan_preserves_dirty_worktree(monkeypatch, tmp_path: Path) -> None:
    """Dirty tracked or untracked status fails closed to preserve."""
    main = tmp_path / "main"
    dirty = tmp_path / "dirty"
    main.mkdir()
    dirty.mkdir()

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del cwd, timeout
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(f"worktree {dirty}\nHEAD aaa\nbranch refs/heads/dirty\n")
        if args == ["git", "status", "--porcelain"]:
            return _result(" M changed.py\n?? new.txt\n")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("origin/dirty\n")
        if args == ["git", "rev-list", "--left-right", "--count", "HEAD...origin/dirty"]:
            return _result("0\t0\n")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    result = snapshot.build_snapshot(
        include_retirement_plan=True,
        claim_lookup=lambda row: _lookup("inactive"),
        merge_lookup=lambda row: _lookup("merged"),
        artifact_lookup=lambda row: _artifacts(),
    )

    assert result.worktrees[0].retirement is not None
    assert result.worktrees[0].retirement.action == snapshot.RETIREMENT_PRESERVE
    assert "tracked or untracked status is dirty" in result.worktrees[0].retirement.reasons


def test_retirement_plan_preserves_ahead_worktree(monkeypatch, tmp_path: Path) -> None:
    """Unpushed commits are a preservation risk even if other checks are green."""
    main = tmp_path / "main"
    ahead = tmp_path / "ahead"
    main.mkdir()
    ahead.mkdir()

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del cwd, timeout
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(f"worktree {ahead}\nHEAD aaa\nbranch refs/heads/ahead\n")
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("origin/ahead\n")
        if args == ["git", "rev-list", "--left-right", "--count", "HEAD...origin/ahead"]:
            return _result("2\t0\n")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    result = snapshot.build_snapshot(
        include_retirement_plan=True,
        claim_lookup=lambda row: _lookup("inactive"),
        merge_lookup=lambda row: _lookup("merged"),
        artifact_lookup=lambda row: _artifacts(),
    )

    assert result.worktrees[0].retirement is not None
    assert result.worktrees[0].retirement.action == snapshot.RETIREMENT_PRESERVE
    assert "worktree has commits ahead of upstream" in result.worktrees[0].retirement.reasons


def test_retirement_plan_reviews_detached_worktree(monkeypatch, tmp_path: Path) -> None:
    """Detached rows need human review because branch/upstream meaning is missing."""
    main = tmp_path / "main"
    detached = tmp_path / "detached"
    main.mkdir()
    detached.mkdir()

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del cwd, timeout
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(f"worktree {detached}\nHEAD aaa\ndetached\n")
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    result = snapshot.build_snapshot(
        include_retirement_plan=True,
        claim_lookup=lambda row: _lookup("inactive"),
        merge_lookup=lambda row: _lookup("merged"),
        artifact_lookup=lambda row: _artifacts(),
    )

    assert result.worktrees[0].retirement is not None
    assert result.worktrees[0].retirement.action == snapshot.RETIREMENT_REVIEW
    assert "detached HEAD needs human review" in result.worktrees[0].retirement.reasons


def test_retirement_plan_reviews_missing_upstream(monkeypatch, tmp_path: Path) -> None:
    """Rows without upstreams are not safe-removal recommendations."""
    main = tmp_path / "main"
    missing = tmp_path / "missing"
    main.mkdir()
    missing.mkdir()

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del cwd, timeout
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(f"worktree {missing}\nHEAD aaa\nbranch refs/heads/missing\n")
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result(stderr="no upstream", returncode=128)
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    result = snapshot.build_snapshot(
        include_retirement_plan=True,
        claim_lookup=lambda row: _lookup("inactive"),
        merge_lookup=lambda row: _lookup("merged"),
        artifact_lookup=lambda row: _artifacts(),
    )

    assert result.worktrees[0].retirement is not None
    assert result.worktrees[0].retirement.action == snapshot.RETIREMENT_REVIEW
    assert "upstream is missing" in result.worktrees[0].retirement.reasons


def test_retirement_plan_reviews_unavailable_external_state(monkeypatch, tmp_path: Path) -> None:
    """Unavailable claim, merge, or artifact state prevents removable classification."""
    main = tmp_path / "main"
    uncertain = tmp_path / "uncertain"
    main.mkdir()
    uncertain.mkdir()

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del cwd, timeout
        if args == ["git", "worktree", "list", "--porcelain"]:
            return _result(f"worktree {uncertain}\nHEAD aaa\nbranch refs/heads/uncertain\n")
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("origin/uncertain\n")
        if args == ["git", "rev-list", "--left-right", "--count", "HEAD...origin/uncertain"]:
            return _result("0\t0\n")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.chdir(main)
    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    result = snapshot.build_snapshot(
        include_retirement_plan=True,
        claim_lookup=lambda row: _lookup("unavailable"),
        merge_lookup=lambda row: _lookup("unavailable"),
        artifact_lookup=lambda row: _artifacts("unavailable", "unavailable"),
    )

    assert result.worktrees[0].retirement is not None
    assert result.worktrees[0].retirement.action == snapshot.RETIREMENT_REVIEW
    assert "claim state unavailable" in result.worktrees[0].retirement.reasons
    assert "merge state unavailable" in result.worktrees[0].retirement.reasons
    assert "output artifact state unavailable" in result.worktrees[0].retirement.reasons


def test_output_root_inspection_preserves_durable_evidence(monkeypatch, tmp_path: Path) -> None:
    """Ignored evidence-like output paths are reported and classified as durable-required."""
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    row = snapshot.WorktreeHygiene(
        path=str(worktree),
        branch="issue-1",
        head_sha="aaa",
        is_current=False,
        is_detached=False,
        dirty_entries=0,
        upstream="origin/issue-1",
        ahead=0,
        behind=0,
    )

    def fake_run(args: list[str], *, cwd: str | None = None, timeout: int = 30):
        del cwd, timeout
        if args == ["git", "status", "--ignored", "--short", "-uall", "--", "output"]:
            return _result("!! output/benchmarks/run/episodes.jsonl\n")
        if args == ["git", "ls-files", "--", "output"]:
            return _result("")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(snapshot, "_run_command", fake_run)

    artifacts = snapshot._inspect_output_root(row)

    assert artifacts == [
        snapshot.ArtifactRootInspection(
            root="output",
            classification="durable_required",
            status="ok",
            ignored_entries=1,
            sample_paths=["output/benchmarks/run/episodes.jsonl"],
            reason="output/ contains evidence-like paths",
        )
    ]
