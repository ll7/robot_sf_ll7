"""Regression coverage for issue #8553 active-worktree cleanup races."""

from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.dev import gate_worktree_guard as guard
from scripts.dev import stale_worktree_reaper as reaper
from scripts.dev.pr_gate_lease import PRGateLease, create_lease, lease_path, save_lease, status

REPO_ROOT = Path(__file__).resolve().parents[2]
CREATE_WORKTREE = REPO_ROOT / "scripts" / "dev" / "create_worktree.sh"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _repo_with_worktree(tmp_path: Path, branch: str) -> tuple[Path, Path, str]:
    repo = tmp_path / "repo"
    worktree = tmp_path / "worktree"
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(repo, "config", "user.name", "Issue 8553 Test")
    _git(repo, "config", "user.email", "issue-8553@example.invalid")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "fixture")
    _git(repo, "worktree", "add", "-b", branch, str(worktree), "main")
    head_sha = _git(repo, "rev-parse", branch).stdout.strip()
    return repo, worktree, head_sha


def _clean_candidate_plan(
    repo: Path, worktree: Path, branch: str, head_sha: str
) -> reaper.ReaperPlan:
    candidate = reaper.WorktreeCandidate(
        path=str(worktree),
        branch=branch,
        head_sha=head_sha,
        is_current=False,
        classification="clean_stale",
    )
    return reaper.ReaperPlan(
        schema=reaper.SCHEMA_VERSION,
        mode="dry_run",
        total_worktrees=2,
        current_worktree=str(repo),
        candidates=[candidate],
        deletable=[str(worktree)],
        refused=[],
        errors=[],
        audit_log=[f"classified {worktree} as clean_stale"],
    )


def test_creator_task_id_acquires_path_lease_before_return(tmp_path: Path, monkeypatch) -> None:
    """The canonical shell creator publishes ownership even when no receipt is requested."""
    repo = tmp_path / "creator-repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(repo, "config", "user.name", "Issue 8553 Test")
    _git(repo, "config", "user.email", "issue-8553@example.invalid")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "fixture")
    worktree = tmp_path / "creator-worktree"

    result = subprocess.run(
        [
            str(CREATE_WORKTREE),
            "--path",
            str(worktree),
            "--branch",
            "cycle-180-created",
            "--base",
            "HEAD",
            "--minimum-free-bytes",
            "0",
            "--task-id",
            "cycle-180-created",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    monkeypatch.chdir(repo)
    lease_status = status(worktree_path=worktree)
    assert lease_status["active"] is True
    assert lease_status["lease"]["gate_id"] == "cycle-180-created"
    assert lease_status["lease"]["owner"] == "cycle-180-created"
    assert lease_status["lease"]["worktree_path"] == str(worktree.resolve())


def test_apply_refuses_lease_acquired_after_plan(tmp_path: Path, monkeypatch) -> None:
    """A task that claims a worktree after planning wins over cleanup deterministically."""
    branch = "cycle-180-review"
    repo, worktree, head_sha = _repo_with_worktree(tmp_path, branch)
    monkeypatch.chdir(repo)
    plan = _clean_candidate_plan(repo, worktree, branch, head_sha)

    create_lease(
        gate_id="cycle-180-review",
        owner="cycle-180-review",
        ttl_hours=1,
        worktree_path=worktree,
    )

    result = reaper.apply_deletions(plan)

    assert worktree.is_dir()
    assert str(worktree) in result.refused
    assert str(worktree) not in result.deletable
    refusal = next(event for event in result.audit_log if "refused live lease candidate" in event)
    assert "task=cycle-180-review" in refusal


def test_expired_lease_allows_cleanup_and_preserves_branch_ref(tmp_path: Path, monkeypatch) -> None:
    """A crashed owner's expired lease stops blocking cleanup while the branch ref survives."""
    branch = "cycle-180-stale-owner"
    repo, worktree, head_sha = _repo_with_worktree(tmp_path, branch)
    monkeypatch.chdir(repo)
    now = datetime.now(UTC)
    expired = PRGateLease(
        schema="pr_gate_lease.v1",
        created_at=(now - timedelta(hours=3)).isoformat(),
        expires_at=(now - timedelta(hours=1)).isoformat(),
        pr_number=None,
        gate_id="cycle-180-stale-owner",
        owner="cycle-180-stale-owner",
        last_heartbeat=(now - timedelta(hours=2)).isoformat(),
        worktree_path=str(worktree),
        head_ref=branch,
        head_sha=head_sha,
    )
    save_lease(expired, lease_path(worktree))
    plan = _clean_candidate_plan(repo, worktree, branch, head_sha)

    result = reaper.apply_deletions(plan)

    assert not worktree.exists()
    assert str(worktree) not in result.refused
    assert result.errors == []
    assert _git(repo, "show-ref", "--verify", f"refs/heads/{branch}").returncode == 0
    assert any(event == f"removed {worktree}" for event in result.audit_log)


def test_missing_leased_worktree_emits_owner_and_recreates_from_branch(
    tmp_path: Path, monkeypatch
) -> None:
    """A vanished path with a surviving branch yields a structured recoverable handoff."""
    branch = "cycle-180-recovery"
    repo, worktree, _head_sha = _repo_with_worktree(tmp_path, branch)
    monkeypatch.chdir(repo)
    create_lease(
        gate_id="cycle-180-recovery",
        owner="cycle-180-recovery",
        ttl_hours=1,
        worktree_path=worktree,
    )

    shutil.rmtree(worktree)
    assert _git(repo, "show-ref", "--verify", f"refs/heads/{branch}").returncode == 0

    health = guard.verify_gate_worktree(worktree)
    recreated = guard.recreate_gate_worktree(worktree, ttl_hours=1)

    assert health.exists is False
    assert health.classification == "missing"
    assert health.cleanup_owner is not None
    assert "owner=cycle-180-recovery" in health.cleanup_owner
    assert "gate=cycle-180-recovery" in health.cleanup_owner
    assert recreated.recreated is True
    assert recreated.branch == branch
    assert worktree.is_dir()
    assert _git(worktree, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == branch
