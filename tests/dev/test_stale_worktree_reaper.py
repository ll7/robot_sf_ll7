"""Tests for stale worktree reaper dry-run classification."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from scripts.dev import stale_worktree_reaper as reaper

if TYPE_CHECKING:
    from pathlib import Path


def _result(stdout: str = "", stderr: str = "", returncode: int = 0):
    return reaper.subprocess.CompletedProcess(
        args=["git"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _worktree_porcelain(*entries: tuple[str, str, str]) -> str:
    """Build porcelain output from (path, head, branch) tuples."""
    blocks = []
    for path, head, branch in entries:
        blocks.append(f"worktree {path}")
        blocks.append(f"HEAD {head}")
        if branch:
            blocks.append(f"branch refs/heads/{branch}")
        blocks.append("")
    return "\n".join(blocks)


def test_parse_worktree_porcelain_extracts_rows() -> None:
    """Porcelain output should yield path/head/branch dicts."""
    stdout = _worktree_porcelain(
        ("/repo/main", "aaa111", "main"),
        ("/repo/issue-99", "bbb222", "issue-99-fix"),
    )
    rows = reaper._parse_worktree_porcelain(stdout)
    assert len(rows) == 2
    assert rows[0]["path"] == "/repo/main"
    assert rows[0]["head_sha"] == "aaa111"
    assert rows[0]["branch"] == "main"
    assert rows[1]["branch"] == "issue-99-fix"


def test_classify_current_worktree_is_protected(tmp_path: Path) -> None:
    """The current worktree must never be classified as deletable."""
    wt = tmp_path / "worktree-current"
    wt.mkdir()
    candidate = reaper.classify_worktree(
        path=str(wt),
        branch="main",
        head_sha="abc",
        current_path=str(wt),
        skip_pr_check=True,
    )
    assert candidate.classification == "current"
    assert candidate.preservation_required == "current worktree"


def test_classify_clean_stale_candidate(tmp_path: Path) -> None:
    """A worktree with no risks should be classified as clean_stale."""
    main = tmp_path / "main"
    stale = tmp_path / "stale-branch"
    main.mkdir()
    stale.mkdir()

    def fake_run(args, *, cwd=None, timeout=30):
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("origin/stale-branch\n")
        if args == ["git", "log", "origin/stale-branch..HEAD", "--oneline"]:
            return _result("")
        if args == [
            "gh",
            "pr",
            "list",
            "--head",
            "stale-branch",
            "--state",
            "open",
            "--json",
            "number",
        ]:
            return _result("[]")
        if args == ["git", "status", "--ignored", "--short", "-uall"]:
            return _result("")
        return _result("", returncode=1)

    with patch.object(reaper, "_run_command", side_effect=fake_run):
        candidate = reaper.classify_worktree(
            path=str(stale),
            branch="stale-branch",
            head_sha="def",
            current_path=str(main),
            skip_pr_check=False,
        )
    assert candidate.classification == "clean_stale"
    assert candidate.risk_flags == []


def test_classify_dirty_worktree(tmp_path: Path) -> None:
    """A worktree with uncommitted changes should be classified as risky."""
    main = tmp_path / "main"
    dirty = tmp_path / "dirty-wt"
    main.mkdir()
    dirty.mkdir()

    def fake_run(args, *, cwd=None, timeout=30):
        if args == ["git", "status", "--porcelain"]:
            return _result(" M file.txt\n")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("origin/dirty-branch\n")
        if args == ["git", "log", "origin/dirty-branch..HEAD", "--oneline"]:
            return _result("")
        if args == [
            "gh",
            "pr",
            "list",
            "--head",
            "dirty-branch",
            "--state",
            "open",
            "--json",
            "number",
        ]:
            return _result("[]")
        if args == ["git", "status", "--ignored", "--short", "-uall"]:
            return _result("")
        return _result("", returncode=1)

    with patch.object(reaper, "_run_command", side_effect=fake_run):
        candidate = reaper.classify_worktree(
            path=str(dirty),
            branch="dirty-branch",
            head_sha="abc",
            current_path=str(main),
            skip_pr_check=False,
        )
    assert candidate.classification == "risky"
    assert "dirty" in candidate.risk_flags


def test_classify_unpushed_commits(tmp_path: Path) -> None:
    """Commits ahead of origin should be flagged as unpushed_commits."""
    main = tmp_path / "main"
    ahead = tmp_path / "ahead-wt"
    main.mkdir()
    ahead.mkdir()

    def fake_run(args, *, cwd=None, timeout=30):
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("origin/ahead-branch\n")
        if args == ["git", "log", "origin/ahead-branch..HEAD", "--oneline"]:
            return _result("abc1234 add feature\n")
        if args == [
            "gh",
            "pr",
            "list",
            "--head",
            "ahead-branch",
            "--state",
            "open",
            "--json",
            "number",
        ]:
            return _result("[]")
        if args == ["git", "status", "--ignored", "--short", "-uall"]:
            return _result("")
        return _result("", returncode=1)

    with patch.object(reaper, "_run_command", side_effect=fake_run):
        candidate = reaper.classify_worktree(
            path=str(ahead),
            branch="ahead-branch",
            head_sha="def",
            current_path=str(main),
            skip_pr_check=False,
        )
    assert candidate.classification == "risky"
    assert "unpushed_commits" in candidate.risk_flags


def test_classify_missing_upstream_as_unpushed_risk(tmp_path: Path) -> None:
    """A branch without an upstream should not be considered safe to reap."""
    main = tmp_path / "main"
    local_only = tmp_path / "local-only-wt"
    main.mkdir()
    local_only.mkdir()

    def fake_run(args, *, cwd=None, timeout=30):
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("", "no upstream", 128)
        if args == [
            "gh",
            "pr",
            "list",
            "--head",
            "local-only",
            "--state",
            "open",
            "--json",
            "number",
        ]:
            return _result("[]")
        if args == ["git", "status", "--ignored", "--short", "-uall"]:
            return _result("")
        return _result("", returncode=1)

    with patch.object(reaper, "_run_command", side_effect=fake_run):
        candidate = reaper.classify_worktree(
            path=str(local_only),
            branch="local-only",
            head_sha="def",
            current_path=str(main),
            skip_pr_check=False,
        )
    assert candidate.classification == "risky"
    assert "unpushed_commits" in candidate.risk_flags


def test_classify_open_pr_risk(tmp_path: Path) -> None:
    """An open PR should be flagged as open_pr."""
    main = tmp_path / "main"
    pr_wt = tmp_path / "pr-wt"
    main.mkdir()
    pr_wt.mkdir()

    def fake_run(args, *, cwd=None, timeout=30):
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("origin/pr-branch\n")
        if args == ["git", "log", "origin/pr-branch..HEAD", "--oneline"]:
            return _result("")
        if args == [
            "gh",
            "pr",
            "list",
            "--head",
            "pr-branch",
            "--state",
            "open",
            "--json",
            "number",
        ]:
            return _result('[{"number": 42}]')
        if args == ["git", "status", "--ignored", "--short", "-uall"]:
            return _result("")
        return _result("", returncode=1)

    with patch.object(reaper, "_run_command", side_effect=fake_run):
        candidate = reaper.classify_worktree(
            path=str(pr_wt),
            branch="pr-branch",
            head_sha="abc",
            current_path=str(main),
            skip_pr_check=False,
        )
    assert candidate.classification == "risky"
    assert "open_pr" in candidate.risk_flags


def test_skip_pr_check_is_conservative_risk(tmp_path: Path) -> None:
    """Skipping PR lookup should prevent a branch worktree from becoming deletable."""
    main = tmp_path / "main"
    stale = tmp_path / "stale-wt"
    main.mkdir()
    stale.mkdir()

    def fake_run(args, *, cwd=None, timeout=30):
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("origin/stale-branch\n")
        if args == ["git", "log", "origin/stale-branch..HEAD", "--oneline"]:
            return _result("")
        if args == ["git", "status", "--ignored", "--short", "-uall"]:
            return _result("")
        return _result("", returncode=1)

    with patch.object(reaper, "_run_command", side_effect=fake_run):
        candidate = reaper.classify_worktree(
            path=str(stale),
            branch="stale-branch",
            head_sha="abc",
            current_path=str(main),
            skip_pr_check=True,
        )
    assert candidate.classification == "risky"
    assert "pr_check_skipped" in candidate.risk_flags


def test_classify_ignored_output_risk(tmp_path: Path) -> None:
    """Ignored output files should be flagged as ignored_output."""
    main = tmp_path / "main"
    out_wt = tmp_path / "output-wt"
    main.mkdir()
    out_wt.mkdir()

    def fake_run(args, *, cwd=None, timeout=30):
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("origin/output-branch\n")
        if args == ["git", "log", "origin/output-branch..HEAD", "--oneline"]:
            return _result("")
        if args == [
            "gh",
            "pr",
            "list",
            "--head",
            "output-branch",
            "--state",
            "open",
            "--json",
            "number",
        ]:
            return _result("[]")
        if args == ["git", "status", "--ignored", "--short", "-uall"]:
            return _result("!! output/model_cache/\n!! output/videos/")
        return _result("", returncode=1)

    with patch.object(reaper, "_run_command", side_effect=fake_run):
        candidate = reaper.classify_worktree(
            path=str(out_wt),
            branch="output-branch",
            head_sha="abc",
            current_path=str(main),
            skip_pr_check=False,
        )
    assert candidate.classification == "risky"
    assert "ignored_output" in candidate.risk_flags


def test_apply_refuses_risky_candidates(tmp_path: Path) -> None:
    """apply_deletions must not remove worktrees with risk flags."""
    main = tmp_path / "main"
    risky = tmp_path / "risky-wt"
    main.mkdir()
    risky.mkdir()

    plan = reaper.ReaperPlan(
        schema=reaper.SCHEMA_VERSION,
        mode="dry_run",
        total_worktrees=2,
        current_worktree=str(main),
        candidates=[
            reaper.WorktreeCandidate(
                path=str(main),
                branch="main",
                head_sha="a",
                is_current=True,
                classification="current",
            ),
            reaper.WorktreeCandidate(
                path=str(risky),
                branch="risky",
                head_sha="b",
                is_current=False,
                classification="risky",
                risk_flags=["dirty"],
                preservation_required="risky: dirty",
            ),
        ],
        deletable=[],
        refused=[str(risky)],
        errors=[],
        audit_log=["classified risky"],
    )

    result = reaper.apply_deletions(plan)
    assert result.mode == "apply"
    assert str(risky) in result.refused
    assert result.errors == []
    assert any("refused risky candidate" in event for event in result.audit_log)


def test_dry_run_produces_no_deletion_command(tmp_path: Path, monkeypatch) -> None:
    """Dry-run mode must never invoke git worktree remove."""
    main = tmp_path / "main"
    stale = tmp_path / "stale-wt"
    main.mkdir()
    stale.mkdir()

    def fake_run(args, *, cwd=None, timeout=30):
        if args[:4] == ["git", "worktree", "list", "--porcelain"]:
            return _result(
                _worktree_porcelain(
                    (str(main), "aaa", "main"),
                    (str(stale), "bbb", "stale-branch"),
                )
            )
        if args == ["git", "status", "--porcelain"]:
            return _result("")
        if args == ["git", "rev-parse", "--abbrev-ref", "@{upstream}"]:
            return _result("origin/stale-branch\n")
        if args[0:4] == ["git", "log", "origin/stale-branch..HEAD", "--oneline"]:
            return _result("")
        if args[:4] == [
            "gh",
            "pr",
            "list",
            "--head",
            "stale-branch",
            "--state",
            "open",
            "--json",
            "number",
        ]:
            return _result("[]")
        if args == ["git", "status", "--ignored", "--short", "-uall"]:
            return _result("")
        if args == ["git", "worktree", "remove", str(stale)]:
            raise AssertionError("git worktree remove must NOT be called in dry-run mode")
        return _result("", returncode=1)

    monkeypatch.setattr(reaper, "_run_command", fake_run)
    plan = reaper.build_plan(skip_pr_check=False)
    assert plan.mode == "dry_run"
    assert str(stale) in plan.deletable
    assert plan.errors == []
    assert any(str(stale) in event for event in plan.audit_log)


def test_build_plan_dry_run_default(tmp_path: Path, monkeypatch) -> None:
    """build_plan should default to dry_run mode."""
    main = tmp_path / "main"
    main.mkdir()

    def fake_run(args, *, cwd=None, timeout=30):
        if args[:4] == ["git", "worktree", "list", "--porcelain"]:
            return _result(
                _worktree_porcelain(
                    (str(main), "aaa", "main"),
                )
            )
        return _result("", returncode=1)

    monkeypatch.setattr(reaper, "_run_command", fake_run)
    plan = reaper.build_plan(skip_pr_check=True, current_path=str(main))
    assert plan.mode == "dry_run"
    assert plan.total_worktrees == 1
    assert len(plan.candidates) == 1
    assert plan.candidates[0].classification == "current"
    assert plan.audit_log == [f"classified {main} as current"]
