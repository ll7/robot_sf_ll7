"""Tests for stale worktree reaper dry-run classification."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING
from unittest.mock import patch

from scripts.dev import stale_worktree_reaper as reaper

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import FakeSubprocess


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


def _setup_reaper_mock(
    fake_subprocess: FakeSubprocess,
    *,
    branch: str = "stale-branch",
    status_porcelain: str = "",
    upstream: str | None = None,
    log: str = "",
    pr_list: str = "[]",
    ignored: str = "",
    git_common_dir: str | None = None,
) -> FakeSubprocess:
    fake_subprocess.register(["git", "status", "--porcelain"], _result(status_porcelain))
    if upstream == "no upstream":
        fake_subprocess.register(
            ["git", "rev-parse", "--abbrev-ref", "@{upstream}"], _result("", "no upstream", 128)
        )
    else:
        up_val = f"origin/{branch}\n" if upstream is None else upstream
        fake_subprocess.register(
            ["git", "rev-parse", "--abbrev-ref", "@{upstream}"], _result(up_val)
        )
    fake_subprocess.register(["git", "log"], _result(log))
    fake_subprocess.register(["gh", "pr", "list"], _result(pr_list))
    fake_subprocess.register(["git", "status", "--ignored"], _result(ignored))
    if git_common_dir:
        fake_subprocess.register(["git", "rev-parse", "--git-common-dir"], _result(git_common_dir))
    fake_subprocess.set_default(_result("", returncode=1))
    return fake_subprocess


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


def test_classify_clean_stale_candidate(tmp_path: Path, fake_subprocess: FakeSubprocess) -> None:
    """A worktree with no risks should be classified as clean_stale."""
    main = tmp_path / "main"
    stale = tmp_path / "stale-branch"
    main.mkdir()
    stale.mkdir()

    _setup_reaper_mock(fake_subprocess, branch="stale-branch")

    with patch.object(reaper, "_run_command", side_effect=fake_subprocess):
        candidate = reaper.classify_worktree(
            path=str(stale),
            branch="stale-branch",
            head_sha="def",
            current_path=str(main),
            skip_pr_check=False,
        )
    assert candidate.classification == "clean_stale"
    assert candidate.risk_flags == []


def test_classify_dirty_worktree(tmp_path: Path, fake_subprocess: FakeSubprocess) -> None:
    """A worktree with uncommitted changes should be classified as risky."""
    main = tmp_path / "main"
    dirty = tmp_path / "dirty-wt"
    main.mkdir()
    dirty.mkdir()

    _setup_reaper_mock(fake_subprocess, branch="dirty-branch", status_porcelain=" M file.txt\n")

    with patch.object(reaper, "_run_command", side_effect=fake_subprocess):
        candidate = reaper.classify_worktree(
            path=str(dirty),
            branch="dirty-branch",
            head_sha="abc",
            current_path=str(main),
            skip_pr_check=False,
        )
    assert candidate.classification == "risky"
    assert "dirty" in candidate.risk_flags


def test_classify_unpushed_commits(tmp_path: Path, fake_subprocess: FakeSubprocess) -> None:
    """Commits ahead of origin should be flagged as unpushed_commits."""
    main = tmp_path / "main"
    ahead = tmp_path / "ahead-wt"
    main.mkdir()
    ahead.mkdir()

    _setup_reaper_mock(fake_subprocess, branch="ahead-branch", log="abc1234 add feature\n")

    with patch.object(reaper, "_run_command", side_effect=fake_subprocess):
        candidate = reaper.classify_worktree(
            path=str(ahead),
            branch="ahead-branch",
            head_sha="def",
            current_path=str(main),
            skip_pr_check=False,
        )
    assert candidate.classification == "risky"
    assert "unpushed_commits" in candidate.risk_flags


def test_classify_missing_upstream_as_unpushed_risk(
    tmp_path: Path, fake_subprocess: FakeSubprocess
) -> None:
    """A branch without an upstream should not be considered safe to reap."""
    main = tmp_path / "main"
    local_only = tmp_path / "local-only-wt"
    main.mkdir()
    local_only.mkdir()

    _setup_reaper_mock(fake_subprocess, branch="local-only", upstream="no upstream")

    with patch.object(reaper, "_run_command", side_effect=fake_subprocess):
        candidate = reaper.classify_worktree(
            path=str(local_only),
            branch="local-only",
            head_sha="def",
            current_path=str(main),
            skip_pr_check=False,
        )
    assert candidate.classification == "risky"
    assert "unpushed_commits" in candidate.risk_flags


def test_classify_open_pr_risk(tmp_path: Path, fake_subprocess: FakeSubprocess) -> None:
    """An open PR should be flagged as open_pr."""
    main = tmp_path / "main"
    pr_wt = tmp_path / "pr-wt"
    main.mkdir()
    pr_wt.mkdir()

    _setup_reaper_mock(fake_subprocess, branch="pr-branch", pr_list='[{"number": 42}]')

    with patch.object(reaper, "_run_command", side_effect=fake_subprocess):
        candidate = reaper.classify_worktree(
            path=str(pr_wt),
            branch="pr-branch",
            head_sha="abc",
            current_path=str(main),
            skip_pr_check=False,
        )
    assert candidate.classification == "risky"
    assert "open_pr" in candidate.risk_flags


def test_skip_pr_check_is_conservative_risk(
    tmp_path: Path, fake_subprocess: FakeSubprocess
) -> None:
    """Skipping PR lookup should prevent a branch worktree from becoming deletable."""
    main = tmp_path / "main"
    stale = tmp_path / "stale-wt"
    main.mkdir()
    stale.mkdir()

    _setup_reaper_mock(fake_subprocess, branch="stale-branch")

    with patch.object(reaper, "_run_command", side_effect=fake_subprocess):
        candidate = reaper.classify_worktree(
            path=str(stale),
            branch="stale-branch",
            head_sha="abc",
            current_path=str(main),
            skip_pr_check=True,
        )
    assert candidate.classification == "risky"
    assert "pr_check_skipped" in candidate.risk_flags


def test_classify_ignored_output_risk(tmp_path: Path, fake_subprocess: FakeSubprocess) -> None:
    """Ignored output files should be flagged as ignored_output."""
    main = tmp_path / "main"
    out_wt = tmp_path / "output-wt"
    main.mkdir()
    out_wt.mkdir()

    _setup_reaper_mock(
        fake_subprocess, branch="output-branch", ignored="!! output/model_cache/\n!! output/videos/"
    )

    with patch.object(reaper, "_run_command", side_effect=fake_subprocess):
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


def test_dry_run_produces_no_deletion_command(
    tmp_path: Path, monkeypatch, fake_subprocess: FakeSubprocess
) -> None:
    """Dry-run mode must never invoke git worktree remove."""
    main = tmp_path / "main"
    stale = tmp_path / "stale-wt"
    main.mkdir()
    stale.mkdir()

    fake_subprocess.register(
        ["git", "worktree", "list", "--porcelain"],
        _result(
            _worktree_porcelain(
                (str(main), "aaa", "main"),
                (str(stale), "bbb", "stale-branch"),
            )
        ),
    )
    _setup_reaper_mock(fake_subprocess, branch="stale-branch")
    fake_subprocess.register(
        ["git", "worktree", "remove"],
        AssertionError("git worktree remove must NOT be called in dry-run mode"),
    )

    monkeypatch.setattr(reaper, "_run_command", fake_subprocess)
    plan = reaper.build_plan(skip_pr_check=False)
    assert plan.mode == "dry_run"
    assert str(stale) in plan.deletable
    assert plan.errors == []
    assert any(str(stale) in event for event in plan.audit_log)


def test_build_plan_dry_run_default(
    tmp_path: Path, monkeypatch, fake_subprocess: FakeSubprocess
) -> None:
    """build_plan should default to dry_run mode."""
    main = tmp_path / "main"
    main.mkdir()

    fake_subprocess.register(
        ["git", "worktree", "list", "--porcelain"],
        _result(_worktree_porcelain((str(main), "aaa", "main"))),
    )
    fake_subprocess.set_default(_result("", returncode=1))

    monkeypatch.setattr(reaper, "_run_command", fake_subprocess)
    plan = reaper.build_plan(skip_pr_check=True, current_path=str(main))
    assert plan.mode == "dry_run"
    assert plan.total_worktrees == 1
    assert len(plan.candidates) == 1
    assert plan.candidates[0].classification == "current"
    assert plan.audit_log == [f"classified {main} as current"]


def test_classify_active_pr_gate_lease_risk(
    tmp_path: Path, monkeypatch, fake_subprocess: FakeSubprocess
) -> None:
    """A worktree with an active PR-gate lease should be flagged as risky."""
    main = tmp_path / "main"
    lease_wt = tmp_path / "lease-wt"
    main.mkdir()
    lease_wt.mkdir()

    _setup_reaper_mock(
        fake_subprocess, branch="lease-branch", git_common_dir=str(tmp_path / ".git")
    )

    def fake_has_active_lease(path: str) -> bool:
        return True

    monkeypatch.setattr(reaper, "_has_active_pr_gate_lease", fake_has_active_lease)
    monkeypatch.setattr(reaper, "_run_command", fake_subprocess)

    candidate = reaper.classify_worktree(
        path=str(lease_wt),
        branch="lease-branch",
        head_sha="abc",
        current_path=str(main),
        skip_pr_check=False,
    )
    assert candidate.classification == "risky"
    assert "active_pr_gate_lease" in candidate.risk_flags


def test_classify_no_pr_gate_lease_not_risky(
    tmp_path: Path, monkeypatch, fake_subprocess: FakeSubprocess
) -> None:
    """A worktree without an active PR-gate lease should not get the flag."""
    main = tmp_path / "main"
    stale = tmp_path / "stale-wt"
    main.mkdir()
    stale.mkdir()

    _setup_reaper_mock(
        fake_subprocess, branch="stale-branch", git_common_dir=str(tmp_path / ".git")
    )

    def fake_has_active_lease(path: str) -> bool:
        return False

    monkeypatch.setattr(reaper, "_has_active_pr_gate_lease", fake_has_active_lease)
    monkeypatch.setattr(reaper, "_run_command", fake_subprocess)

    candidate = reaper.classify_worktree(
        path=str(stale),
        branch="stale-branch",
        head_sha="abc",
        current_path=str(main),
        skip_pr_check=False,
    )
    assert candidate.classification == "clean_stale"
    assert "active_pr_gate_lease" not in candidate.risk_flags


def test_classify_unreadable_pr_gate_lease_risk(
    tmp_path: Path, monkeypatch, fake_subprocess: FakeSubprocess
) -> None:
    """A worktree with an unreadable/malformed lease file should be flagged as risky."""
    main = tmp_path / "main"
    stale = tmp_path / "stale-wt"
    main.mkdir()
    stale.mkdir()

    git_common = tmp_path / ".git"
    git_common.mkdir()

    _setup_reaper_mock(fake_subprocess, branch="stale-branch", git_common_dir=str(git_common))
    monkeypatch.setattr(reaper, "_run_command", fake_subprocess)

    from scripts.dev.pr_gate_lease import lease_path

    with patch("scripts.dev.pr_gate_lease._git_common_dir", return_value=git_common):
        l_path = lease_path(stale)
        l_path.write_text("{invalid json\n")

    with patch("scripts.dev.pr_gate_lease._git_common_dir", return_value=git_common):
        candidate = reaper.classify_worktree(
            path=str(stale),
            branch="stale-branch",
            head_sha="abc",
            current_path=str(main),
            skip_pr_check=False,
        )

    assert candidate.classification == "risky"
    assert "unreadable_pr_gate_lease" in candidate.risk_flags


def test_classify_legacy_pr_gate_lease_risk(
    tmp_path: Path, monkeypatch, fake_subprocess: FakeSubprocess
) -> None:
    """A live pre-isolation lease protects worktrees during the path migration."""
    main = tmp_path / "main"
    stale = tmp_path / "stale-wt"
    main.mkdir()
    stale.mkdir()

    git_common = tmp_path / ".git"
    git_common.mkdir()

    _setup_reaper_mock(fake_subprocess, branch="stale-branch", git_common_dir=str(git_common))
    monkeypatch.setattr(reaper, "_run_command", fake_subprocess)

    from scripts.dev.pr_gate_lease import create_lease, lease_path, legacy_lease_path

    with (
        patch("scripts.dev.pr_gate_lease._git_common_dir", return_value=git_common),
        patch("scripts.dev.pr_gate_lease._repo_root", return_value=stale),
    ):
        lease = create_lease(pr_number=5736)
        new_path = lease_path(stale)
        new_path.unlink()
        legacy_lease_path().write_text(json.dumps(asdict(lease)) + "\n")

        candidate = reaper.classify_worktree(
            path=str(stale),
            branch="stale-branch",
            head_sha="abc",
            current_path=str(main),
            skip_pr_check=False,
        )

    assert candidate.classification == "risky"
    assert "active_pr_gate_lease" in candidate.risk_flags
