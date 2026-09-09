"""Tests for stale worktree reaper dry-run classification."""

from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from scripts.dev import stale_worktree_reaper as reaper

if TYPE_CHECKING:
    from tests.conftest import FakeSubprocess


REPO_NAME = "ll7/robot_sf_ll7"
VERIFIED_PR_NUMBER = 8663


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run one Git command against a disposable fixture repository."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


class _FakeGitHubTransport:
    """Offline read-only GitHub transport with optional response sequences."""

    def __init__(
        self, metadata: dict[str, object] | list[dict[str, object]], *, open_prs: object = None
    ):
        self.calls: list[list[str]] = []
        self._metadata = metadata if isinstance(metadata, list) else [metadata]
        self._open_prs = [] if open_prs is None else open_prs

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        if args[1:3] == ["pr", "list"]:
            return _result(stdout=json.dumps(self._open_prs))
        if args[1:2] == ["api"]:
            metadata = self._metadata.pop(0) if len(self._metadata) > 1 else self._metadata[0]
            if isinstance(metadata, subprocess.CompletedProcess):
                return metadata
            return _result(stdout=json.dumps(metadata))
        return _result(stderr="unexpected GitHub command", returncode=1)


class _IgnoredRaceTransport(_FakeGitHubTransport):
    """Create ignored target content during the apply-time PR-detail read."""

    def __init__(self, metadata: dict[str, object], target: Path):
        super().__init__(metadata)
        self._target = target

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        result = super().__call__(args)
        api_calls = sum(call[1:2] == ["api"] for call in self.calls)
        if args[1:2] == ["api"] and api_calls == 2:
            late_output = self._target / "output" / "created-during-read.txt"
            late_output.parent.mkdir()
            late_output.write_text("must be preserved\n", encoding="utf-8")
        return result


def _commit_tree(repo: Path, tree_sha: str, message: str, *parents: str) -> str:
    """Create a disposable commit object without changing a checked-out ref."""
    args = ["commit-tree", tree_sha]
    for parent in parents:
        args.extend(["-p", parent])
    args.extend(["-m", message])
    return _git(repo, *args).stdout.strip()


@contextmanager
def _null_lock():
    """Stand in for the lifecycle lock while exercising disposable repositories."""
    yield


def _verified_fixture(tmp_path: Path) -> dict[str, object]:
    """Create a clean merged-tree-shaped Git fixture with a deleted upstream ref."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    repo = tmp_path / "fixture-repo"
    target = tmp_path / "worktree with spaces"
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(repo)], check=True, capture_output=True
    )
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Fixture User")
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    _git(repo, "remote", "add", "origin", str(remote))
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    (repo / ".gitignore").write_text("output/\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt", ".gitignore")
    _git(repo, "commit", "-m", "base")
    base_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "worktree", "add", "-b", "feature/verified", str(target), "main")
    _git(target, "config", "user.email", "fixture@example.invalid")
    _git(target, "config", "user.name", "Fixture User")
    (target / "tracked.txt").write_text("squashed result\n", encoding="utf-8")
    _git(target, "add", "tracked.txt")
    _git(target, "commit", "-m", "feature")
    head_sha = _git(target, "rev-parse", "HEAD").stdout.strip()
    tree_sha = _git(target, "rev-parse", "HEAD^{tree}").stdout.strip()
    merge_sha = _commit_tree(repo, tree_sha, "squash merge", base_sha)
    _git(repo, "reset", "--hard", merge_sha)
    _git(repo, "update-ref", "refs/remotes/origin/main", merge_sha)
    _git(repo, "push", "origin", f"{merge_sha}:refs/heads/main")
    _git(target, "config", "branch.feature/verified.remote", "origin")
    _git(target, "config", "branch.feature/verified.merge", "refs/heads/feature/verified")
    metadata = {
        "number": VERIFIED_PR_NUMBER,
        "state": "closed",
        "merged_at": "2026-09-09T01:47:00Z",
        "merge_commit_sha": merge_sha,
        "base": {
            "ref": "main",
            "sha": base_sha,
            "repo": {"full_name": REPO_NAME},
        },
        "head": {
            "ref": "feature/verified",
            "sha": head_sha,
            "repo": {"full_name": REPO_NAME},
        },
    }
    return {
        "repo": repo,
        "remote": remote,
        "target": target,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "tree_sha": tree_sha,
        "merge_sha": merge_sha,
        "metadata": metadata,
    }


def _verified_plan(
    fixture: dict[str, object],
    transport: _FakeGitHubTransport,
    *,
    target_path: str | None = None,
    branch: str = "feature/verified",
    head_sha: str | None = None,
) -> reaper.ReaperPlan:
    """Build one explicit verified plan for a disposable fixture."""
    repo = fixture["repo"]
    target = fixture["target"]
    assert isinstance(repo, Path)
    assert isinstance(target, Path)
    expected_head = fixture["head_sha"] if head_sha is None else head_sha
    assert isinstance(expected_head, str)
    return reaper.build_plan(
        current_path=str(repo),
        target_path=str(target) if target_path is None else target_path,
        verified_pr_number=VERIFIED_PR_NUMBER,
        verified_branch=branch,
        verified_head_sha=expected_head,
        repo=REPO_NAME,
        github_transport=transport,
    )


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


@pytest.mark.parametrize(
    "args",
    [
        ["gh", "api", "repos/ll7/robot_sf_ll7/pulls/1", "--method", "DELETE"],
        ["gh", "api", "repos/ll7/robot_sf_ll7/issues/1"],
        ["gh", "api", "repos/ll7/robot_sf_ll7/pulls/0"],
        ["gh", "pr", "close", "1"],
    ],
)
def test_github_read_transport_refuses_non_read_commands(args: list[str]) -> None:
    """The injectable/default GitHub boundary permits only bounded read shapes."""
    result = reaper._github_read(args, transport=lambda _: pytest.fail("transport was called"))

    assert result.returncode == 126
    assert "permits" in result.stderr


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


def test_verified_merged_tree_is_explicit_and_applies_only_after_revalidation(
    tmp_path: Path,
) -> None:
    """Exact Git/PR proof permits only the target fixture removal and preserves its branch."""
    fixture = _verified_fixture(tmp_path)
    metadata = fixture["metadata"]
    assert isinstance(metadata, dict)
    transport = _FakeGitHubTransport(metadata)
    repo = fixture["repo"]
    target = fixture["target"]
    head_sha = fixture["head_sha"]
    assert isinstance(repo, Path)
    assert isinstance(target, Path)
    assert isinstance(head_sha, str)

    with patch.object(reaper, "_worktree_lease_state", return_value=(None, None)):
        default_plan = reaper.build_plan(
            current_path=str(repo),
            target_path=str(target),
            skip_pr_check=True,
        )
        assert str(target) in default_plan.refused
        assert "unpushed_commits" in default_plan.candidates[0].risk_flags

        plan = _verified_plan(fixture, transport)
        assert plan.errors == []
        assert plan.verification_mode == reaper.VERIFIED_MERGE_MODE
        assert plan.deletable == [str(target)]
        evidence = plan.candidates[0].verification
        assert evidence["head_sha"] == head_sha
        assert evidence["merge_method_scope"] == "method_agnostic_exact_tree"
        assert evidence["merge_commit_sha"] == fixture["merge_sha"]
        assert evidence["main_sha"] == fixture["merge_sha"]
        assert evidence["worktree_tree_sha"] == fixture["tree_sha"]
        assert evidence["merge_tree_sha"] == fixture["tree_sha"]
        assert evidence["main_tree_sha"] == fixture["tree_sha"]
        assert evidence["risk_discharged"] == "missing origin upstream only"
        assert target.is_dir()

        lock_events: list[str] = []

        @contextmanager
        def fake_lock():
            lock_events.append("enter")
            yield
            lock_events.append("exit")

        with patch("scripts.dev.pr_gate_lease.worktree_lifecycle_lock", fake_lock):
            applied = reaper.apply_deletions(plan, github_transport=transport)

    assert applied.errors == []
    assert applied.deletable == []
    assert str(target) not in applied.refused
    assert lock_events == ["enter", "exit"]
    assert not target.exists()
    assert _git(repo, "rev-parse", "refs/heads/feature/verified").stdout.strip() == head_sha
    assert _git(repo, "cat-file", "-e", f"{head_sha}^{{commit}}").returncode == 0
    assert _git(repo, "rev-parse", "refs/heads/main").stdout.strip() == fixture["merge_sha"]
    assert all(call[0:2] == ["gh", "pr"] or call[0:2] == ["gh", "api"] for call in transport.calls)
    assert all("--method" not in call for call in transport.calls)
    assert len(transport.calls) == 4


def test_verified_merged_tree_accepts_regular_merge_shape(tmp_path: Path) -> None:
    """The proof is intentionally merge-method agnostic when tree identity is exact."""
    fixture = _verified_fixture(tmp_path)
    repo = fixture["repo"]
    target = fixture["target"]
    base_sha = fixture["base_sha"]
    head_sha = fixture["head_sha"]
    tree_sha = fixture["tree_sha"]
    assert isinstance(repo, Path)
    assert isinstance(target, Path)
    assert isinstance(base_sha, str)
    assert isinstance(head_sha, str)
    assert isinstance(tree_sha, str)

    regular_merge_sha = _commit_tree(
        repo,
        tree_sha,
        "regular merge shape",
        base_sha,
        head_sha,
    )
    _git(repo, "reset", "--hard", regular_merge_sha)
    _git(repo, "update-ref", "refs/remotes/origin/main", regular_merge_sha)
    _git(repo, "push", "--force", "origin", f"{regular_merge_sha}:refs/heads/main")
    metadata = json.loads(json.dumps(fixture["metadata"]))
    assert isinstance(metadata, dict)
    metadata["merge_commit_sha"] = regular_merge_sha
    transport = _FakeGitHubTransport(metadata)

    with patch.object(reaper, "_worktree_lease_state", return_value=(None, None)):
        plan = _verified_plan(fixture, transport)

    assert plan.errors == []
    assert plan.deletable == [str(target)]
    assert plan.candidates[0].verification["merge_method_scope"] == ("method_agnostic_exact_tree")


@pytest.mark.parametrize(
    "failure",
    [
        "detached",
        "unmerged",
        "wrong_pr",
        "wrong_repo",
        "wrong_branch",
        "wrong_head",
        "missing_merge_object",
        "different_tree",
        "not_ancestor",
        "stale_remote_main",
        "remote_unavailable",
        "lookup_error",
        "ambiguous_open_pr",
    ],
)
def test_verified_negative_matrix_fails_closed(  # noqa: C901, PLR0915 - each named negative mutates one proof input
    tmp_path: Path,
    failure: str,
) -> None:
    """Every missing, stale, ambiguous, or mismatched proof refuses the fixture target."""
    fixture = _verified_fixture(tmp_path)
    metadata = json.loads(json.dumps(fixture["metadata"]))
    assert isinstance(metadata, dict)
    open_prs: object = []
    if failure == "detached":
        target = fixture["target"]
        assert isinstance(target, Path)
        _git(target, "checkout", "--detach", "HEAD")
    elif failure == "unmerged":
        metadata["state"] = "open"
        metadata["merged_at"] = None
    elif failure == "wrong_pr":
        metadata["number"] = VERIFIED_PR_NUMBER + 1
    elif failure == "wrong_repo":
        metadata["base"]["repo"]["full_name"] = "other/repository"
    elif failure == "wrong_branch":
        metadata["head"]["ref"] = "feature/other"
    elif failure == "wrong_head":
        metadata["head"]["sha"] = "1" * 40
    elif failure == "missing_merge_object":
        metadata["merge_commit_sha"] = "f" * 40
    elif failure == "different_tree":
        repo = fixture["repo"]
        base_sha = fixture["base_sha"]
        assert isinstance(repo, Path)
        assert isinstance(base_sha, str)
        base_tree = _git(repo, "rev-parse", f"{base_sha}^{{tree}}").stdout.strip()
        metadata["merge_commit_sha"] = _commit_tree(repo, base_tree, "different tree", base_sha)
    elif failure == "not_ancestor":
        repo = fixture["repo"]
        tree_sha = fixture["tree_sha"]
        assert isinstance(repo, Path)
        assert isinstance(tree_sha, str)
        metadata["merge_commit_sha"] = _commit_tree(repo, tree_sha, "unrelated merge-shaped object")
    elif failure == "stale_remote_main":
        repo = fixture["repo"]
        base_sha = fixture["base_sha"]
        assert isinstance(repo, Path)
        assert isinstance(base_sha, str)
        _git(repo, "update-ref", "refs/remotes/origin/main", base_sha)
    elif failure == "remote_unavailable":
        repo = fixture["repo"]
        assert isinstance(repo, Path)
        _git(repo, "remote", "set-url", "origin", str(repo / "missing-remote.git"))
    elif failure == "lookup_error":
        metadata = [_result(stderr="HTTP 503", returncode=1)]
    elif failure == "ambiguous_open_pr":
        open_prs = {"number": 42}

    transport = _FakeGitHubTransport(metadata, open_prs=open_prs)
    repo = fixture["repo"]
    target = fixture["target"]
    assert isinstance(repo, Path)
    assert isinstance(target, Path)
    with patch.object(reaper, "_worktree_lease_state", return_value=(None, None)):
        plan = _verified_plan(fixture, transport)

    assert plan.deletable == []
    assert str(target) in plan.refused or plan.errors
    assert target.is_dir()
    if failure == "ambiguous_open_pr":
        assert "unreadable_open_pr_state" in plan.candidates[0].risk_flags
    else:
        assert "verified_merge_refused" in plan.candidates[0].risk_flags


def test_verified_refuses_authoritative_remote_branch_reappearance(tmp_path: Path) -> None:
    """A remote branch that is absent locally still blocks verified cleanup."""
    fixture = _verified_fixture(tmp_path)
    repo = fixture["repo"]
    remote = fixture["remote"]
    target = fixture["target"]
    head_sha = fixture["head_sha"]
    assert isinstance(repo, Path)
    assert isinstance(remote, Path)
    assert isinstance(target, Path)
    assert isinstance(head_sha, str)
    _git(repo, "push", "origin", f"{head_sha}:refs/heads/feature/verified")
    _git(repo, "update-ref", "-d", "refs/remotes/origin/feature/verified")
    metadata = fixture["metadata"]
    assert isinstance(metadata, dict)
    transport = _FakeGitHubTransport(metadata)

    with patch.object(reaper, "_worktree_lease_state", return_value=(None, None)):
        plan = _verified_plan(fixture, transport)

    assert plan.deletable == []
    assert str(target) in plan.refused
    assert any(
        "authoritative origin candidate ref still exists" in event for event in plan.audit_log
    )
    assert target.is_dir()


def test_verified_apply_refuses_ignored_content_created_after_final_remote_read(
    tmp_path: Path,
) -> None:
    """Apply-time preservation recheck catches ignored content created during the last read."""
    fixture = _verified_fixture(tmp_path)
    repo = fixture["repo"]
    target = fixture["target"]
    metadata = fixture["metadata"]
    assert isinstance(repo, Path)
    assert isinstance(target, Path)
    assert isinstance(metadata, dict)
    transport = _IgnoredRaceTransport(metadata, target)

    with patch.object(reaper, "_worktree_lease_state", return_value=(None, None)):
        plan = _verified_plan(fixture, transport)
    assert plan.deletable == [str(target)]

    with (
        patch.object(reaper, "_worktree_lease_state", return_value=(None, None)),
        patch("scripts.dev.pr_gate_lease.worktree_lifecycle_lock", _null_lock),
    ):
        result = reaper.apply_deletions(plan, github_transport=transport)

    preserved = target / "output" / "created-during-read.txt"
    assert result.errors == []
    assert str(target) in result.refused
    assert any("apply-time preservation recheck" in event for event in result.audit_log)
    assert target.is_dir()
    assert preserved.read_text(encoding="utf-8") == "must be preserved\n"


def test_verified_never_published_and_existing_upstream_are_refused(tmp_path: Path) -> None:
    """Verified mode cannot turn an unconfigured or still-present upstream into proof."""
    for existing_ref in (False, True):
        fixture = _verified_fixture(tmp_path / ("existing" if existing_ref else "missing"))
        repo = fixture["repo"]
        target = fixture["target"]
        head_sha = fixture["head_sha"]
        assert isinstance(repo, Path)
        assert isinstance(target, Path)
        assert isinstance(head_sha, str)
        if existing_ref:
            _git(repo, "update-ref", "refs/remotes/origin/feature/verified", head_sha)
        else:
            _git(target, "config", "--unset-all", "branch.feature/verified.remote")
        metadata = fixture["metadata"]
        assert isinstance(metadata, dict)
        transport = _FakeGitHubTransport(metadata)
        with patch.object(reaper, "_worktree_lease_state", return_value=(None, None)):
            plan = _verified_plan(fixture, transport)
        assert plan.deletable == []
        assert str(target) in plan.refused
        assert "unpushed_commits" in plan.candidates[0].risk_flags
        assert target.is_dir()


@pytest.mark.parametrize("content_kind", ["dirty", "untracked", "ignored"])
def test_verified_existing_content_gates_refuse(tmp_path: Path, content_kind: str) -> None:
    """Dirty, untracked, and ignored fixture content remains a preservation refusal."""
    fixture = _verified_fixture(tmp_path)
    target = fixture["target"]
    assert isinstance(target, Path)
    if content_kind == "dirty":
        (target / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    elif content_kind == "untracked":
        (target / "new.txt").write_text("untracked\n", encoding="utf-8")
    else:
        (target / "output").mkdir()
        (target / "output" / "ignored.txt").write_text("ignored\n", encoding="utf-8")
    metadata = fixture["metadata"]
    assert isinstance(metadata, dict)
    transport = _FakeGitHubTransport(metadata)
    with patch.object(reaper, "_worktree_lease_state", return_value=(None, None)):
        plan = _verified_plan(fixture, transport)
    assert plan.deletable == []
    assert str(target) in plan.refused
    assert ("ignored_output" if content_kind == "ignored" else "dirty") in plan.candidates[
        0
    ].risk_flags
    assert target.is_dir()


@pytest.mark.parametrize("lease_state", ["active", "unreadable"])
def test_verified_lease_states_refuse(tmp_path: Path, lease_state: str) -> None:
    """Active and unreadable leases continue to block verified cleanup."""
    fixture = _verified_fixture(tmp_path)
    metadata = fixture["metadata"]
    target = fixture["target"]
    assert isinstance(metadata, dict)
    assert isinstance(target, Path)
    lease = SimpleNamespace(
        gate_id="task-8663", owner="worker", pr_number=8663, expires_at="future"
    )
    state = (lease_state, lease if lease_state == "active" else None)
    transport = _FakeGitHubTransport(metadata)
    with patch.object(reaper, "_worktree_lease_state", return_value=state):
        plan = _verified_plan(fixture, transport)
    assert plan.deletable == []
    assert str(target) in plan.refused
    assert f"{lease_state}_pr_gate_lease" in plan.candidates[0].risk_flags
    assert target.is_dir()


def test_verified_exact_path_rejects_symlink_and_boundary_aliases(tmp_path: Path) -> None:
    """Only the exact registered canonical path is accepted, including for spaced names."""
    fixture = _verified_fixture(tmp_path)
    target = fixture["target"]
    assert isinstance(target, Path)
    alias = tmp_path / "target-alias"
    alias.symlink_to(target, target_is_directory=True)
    neighbor = Path(f"{target}-neighbor")
    metadata = fixture["metadata"]
    assert isinstance(metadata, dict)
    for path in (alias, neighbor):
        transport = _FakeGitHubTransport(metadata)
        with patch.object(reaper, "_worktree_lease_state", return_value=(None, None)):
            plan = _verified_plan(fixture, transport, target_path=str(path))
        assert plan.deletable == []
        assert plan.errors
        assert target.is_dir()


def test_verified_current_worktree_is_protected(tmp_path: Path) -> None:
    """The explicit verified request cannot override the current-worktree gate."""
    fixture = _verified_fixture(tmp_path)
    repo = fixture["repo"]
    assert isinstance(repo, Path)
    metadata = fixture["metadata"]
    assert isinstance(metadata, dict)
    transport = _FakeGitHubTransport(metadata)
    with patch.object(reaper, "_worktree_lease_state", return_value=(None, None)):
        plan = _verified_plan(fixture, transport, target_path=str(repo))
    assert plan.errors == []
    assert plan.deletable == []
    assert plan.candidates[0].classification == "current"
    assert repo.is_dir()
    assert transport.calls == []


def test_verified_apply_refuses_head_and_ignored_drift(tmp_path: Path) -> None:
    """Apply-time identity and ignored-state drift refuses removal after a valid plan."""
    fixture = _verified_fixture(tmp_path)
    repo = fixture["repo"]
    target = fixture["target"]
    assert isinstance(repo, Path)
    assert isinstance(target, Path)
    metadata = fixture["metadata"]
    assert isinstance(metadata, dict)

    transport = _FakeGitHubTransport(metadata)
    with patch.object(reaper, "_worktree_lease_state", return_value=(None, None)):
        plan = _verified_plan(fixture, transport)
    (target / "tracked.txt").write_text("head drift\n", encoding="utf-8")
    _git(target, "add", "tracked.txt")
    _git(target, "commit", "-m", "head drift")
    lock_events: list[str] = []

    @contextmanager
    def fake_lock():
        lock_events.append("enter")
        yield
        lock_events.append("exit")

    with (
        patch.object(reaper, "_worktree_lease_state", return_value=(None, None)),
        patch("scripts.dev.pr_gate_lease.worktree_lifecycle_lock", fake_lock),
    ):
        result = reaper.apply_deletions(plan, github_transport=transport)
    assert result.errors == []
    assert str(target) in result.refused
    assert any("head" in event for event in result.audit_log)
    assert target.is_dir()
    assert lock_events == ["enter", "exit"]

    fixture = _verified_fixture(tmp_path / "ignored-drift")
    repo = fixture["repo"]
    target = fixture["target"]
    metadata = fixture["metadata"]
    assert isinstance(repo, Path)
    assert isinstance(target, Path)
    assert isinstance(metadata, dict)
    transport = _FakeGitHubTransport(metadata)
    with patch.object(reaper, "_worktree_lease_state", return_value=(None, None)):
        plan = _verified_plan(fixture, transport)
    (target / "output").mkdir()
    (target / "output" / "late.txt").write_text("late ignored\n", encoding="utf-8")
    with (
        patch.object(reaper, "_worktree_lease_state", return_value=(None, None)),
        patch("scripts.dev.pr_gate_lease.worktree_lifecycle_lock", fake_lock),
    ):
        result = reaper.apply_deletions(plan, github_transport=transport)
    assert result.errors == []
    assert str(target) in result.refused
    assert any("ignored" in event for event in result.audit_log)
    assert target.is_dir()


def test_verified_apply_refuses_metadata_drift_under_lock(tmp_path: Path) -> None:
    """A fresh authoritative PR read that differs from the plan cannot authorize removal."""
    fixture = _verified_fixture(tmp_path)
    metadata = fixture["metadata"]
    target = fixture["target"]
    assert isinstance(metadata, dict)
    assert isinstance(target, Path)
    drifted = json.loads(json.dumps(metadata))
    drifted["merged_at"] = "2026-09-09T02:00:00Z"
    transport = _FakeGitHubTransport([metadata, drifted])
    with patch.object(reaper, "_worktree_lease_state", return_value=(None, None)):
        plan = _verified_plan(fixture, transport)
        assert plan.deletable == [str(target)]
        with patch("scripts.dev.pr_gate_lease.worktree_lifecycle_lock", _null_lock):
            result = reaper.apply_deletions(plan, github_transport=transport)
    assert result.errors == []
    assert str(target) in result.refused
    assert any("drifted" in event for event in result.audit_log)
    assert target.is_dir()


def test_verified_apply_refuses_new_lease_acquired_at_lock_boundary(tmp_path: Path) -> None:
    """A lease acquired after planning wins under the shared lifecycle lock."""
    fixture = _verified_fixture(tmp_path)
    metadata = fixture["metadata"]
    target = fixture["target"]
    assert isinstance(metadata, dict)
    assert isinstance(target, Path)
    transport = _FakeGitHubTransport(metadata)
    lease_active = False
    lease = SimpleNamespace(
        gate_id="new-task", owner="new-owner", pr_number=None, expires_at="future"
    )

    def lease_state(_path: str):
        return ("active", lease) if lease_active else (None, None)

    with patch.object(reaper, "_worktree_lease_state", side_effect=lease_state):
        plan = _verified_plan(fixture, transport)
        assert plan.deletable == [str(target)]

    @contextmanager
    def race_lock():
        nonlocal lease_active
        lease_active = True
        yield

    with (
        patch.object(reaper, "_worktree_lease_state", side_effect=lease_state),
        patch("scripts.dev.pr_gate_lease.worktree_lifecycle_lock", race_lock),
    ):
        result = reaper.apply_deletions(plan, github_transport=transport)
    assert result.errors == []
    assert str(target) in result.refused
    assert target.is_dir()
    assert len(transport.calls) == 2


def test_verified_apply_refuses_deleted_target_path(tmp_path: Path) -> None:
    """A target removed between planning and apply is never recreated or removed again."""
    fixture = _verified_fixture(tmp_path)
    repo = fixture["repo"]
    target = fixture["target"]
    assert isinstance(repo, Path)
    assert isinstance(target, Path)
    metadata = fixture["metadata"]
    assert isinstance(metadata, dict)
    transport = _FakeGitHubTransport(metadata)
    with patch.object(reaper, "_worktree_lease_state", return_value=(None, None)):
        plan = _verified_plan(fixture, transport)
    _git(repo, "worktree", "remove", str(target))
    assert not target.exists()
    with (
        patch.object(reaper, "_worktree_lease_state", return_value=(None, None)),
        patch("scripts.dev.pr_gate_lease.worktree_lifecycle_lock", _null_lock),
    ):
        result = reaper.apply_deletions(plan, github_transport=transport)
    assert result.errors == []
    assert str(target) in result.refused
    assert not target.exists()
    assert (
        _git(repo, "rev-parse", "refs/heads/feature/verified").stdout.strip() == fixture["head_sha"]
    )


def test_verified_apply_refuses_registered_path_drift(tmp_path: Path) -> None:
    """A worktree moved after planning is no longer the exact registered target."""
    fixture = _verified_fixture(tmp_path)
    repo = fixture["repo"]
    target = fixture["target"]
    assert isinstance(repo, Path)
    assert isinstance(target, Path)
    metadata = fixture["metadata"]
    assert isinstance(metadata, dict)
    transport = _FakeGitHubTransport(metadata)
    with patch.object(reaper, "_worktree_lease_state", return_value=(None, None)):
        plan = _verified_plan(fixture, transport)

    moved = tmp_path / "moved-after-plan"
    _git(repo, "worktree", "move", str(target), str(moved))
    assert not target.exists()
    assert moved.is_dir()
    with (
        patch.object(reaper, "_worktree_lease_state", return_value=(None, None)),
        patch("scripts.dev.pr_gate_lease.worktree_lifecycle_lock", _null_lock),
    ):
        result = reaper.apply_deletions(plan, github_transport=transport)

    assert result.errors == []
    assert str(target) in result.refused
    assert moved.is_dir()
    assert any("registered" in event for event in result.audit_log)
    assert (
        _git(repo, "rev-parse", "refs/heads/feature/verified").stdout.strip() == fixture["head_sha"]
    )


def test_verified_apply_refuses_tracking_ref_reappearance(tmp_path: Path) -> None:
    """A deleted origin tracking ref reappearing after planning invalidates proof."""
    fixture = _verified_fixture(tmp_path)
    repo = fixture["repo"]
    target = fixture["target"]
    head_sha = fixture["head_sha"]
    assert isinstance(repo, Path)
    assert isinstance(target, Path)
    assert isinstance(head_sha, str)
    metadata = fixture["metadata"]
    assert isinstance(metadata, dict)
    transport = _FakeGitHubTransport(metadata)
    with patch.object(reaper, "_worktree_lease_state", return_value=(None, None)):
        plan = _verified_plan(fixture, transport)

    _git(repo, "update-ref", "refs/remotes/origin/feature/verified", head_sha)
    with (
        patch.object(reaper, "_worktree_lease_state", return_value=(None, None)),
        patch("scripts.dev.pr_gate_lease.worktree_lifecycle_lock", _null_lock),
    ):
        result = reaper.apply_deletions(plan, github_transport=transport)

    assert result.errors == []
    assert str(target) in result.refused
    assert target.is_dir()
    assert any("upstream" in event or "tracking" in event for event in result.audit_log)
    assert _git(repo, "rev-parse", "refs/heads/feature/verified").stdout.strip() == head_sha
