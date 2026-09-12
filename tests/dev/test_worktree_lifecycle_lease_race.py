"""Regression coverage for active-worktree cleanup and task-lease races."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.dev import gate_worktree_guard as guard
from scripts.dev import stale_worktree_reaper as reaper
from scripts.dev import worktree_receipt
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


@pytest.mark.parametrize("force_python", (False, True), ids=("flock-cli", "portable-python"))
@pytest.mark.parametrize("with_receipt", (False, True), ids=("no-receipt", "receipt"))
@pytest.mark.parametrize("safe_path", (False, True), ids=("default-path", "safe-path"))
def test_creator_task_id_acquires_path_lease_before_return(
    tmp_path: Path, monkeypatch, force_python: bool, with_receipt: bool, safe_path: bool
) -> None:
    """Creation outside the source repo works without implicit Python import paths."""
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
    base_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    worktree = tmp_path / "creator-worktree"
    receipt_path = tmp_path / "creator-receipt.json"
    receipt_args = ["--receipt", str(receipt_path)] if with_receipt else []

    # An editable install can expose another checkout's scripts package even
    # after PYTHONPATH is removed. Keep the real child CLI independent of site.
    stub_bin = tmp_path / "stub-bin"
    stub_bin.mkdir()
    python_stub = stub_bin / "python3"
    python_stub.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "import sys\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}, '-S', *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PATH"] = f"{stub_bin}:{environment['PATH']}"
    if safe_path:
        # Also exclude scripts/dev: a sibling-import fallback still relies on
        # Python implicitly prepending the directory of the executed script.
        environment["PYTHONSAFEPATH"] = "1"
    else:
        environment.pop("PYTHONSAFEPATH", None)
    if force_python:
        environment["ROBOT_SF_WORKTREE_FORCE_PYTHON_LOCK"] = "1"
    else:
        environment.pop("ROBOT_SF_WORKTREE_FORCE_PYTHON_LOCK", None)

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
            *receipt_args,
            "--exec",
            "python3",
            "-c",
            "import os; assert 'PYTHONPATH' not in os.environ; "
            "assert 'ROBOT_SF_WORKTREE_LOCK_FD' not in os.environ",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "No module named 'scripts'" not in result.stderr
    monkeypatch.chdir(repo)
    lease_status = status(worktree_path=worktree)
    assert lease_status["active"] is True
    assert lease_status["lease"]["schema"] == "pr_gate_lease.v1"
    assert lease_status["lease"]["gate_id"] == "cycle-180-created"
    assert lease_status["lease"]["owner"] == "cycle-180-created"
    assert lease_status["lease"]["worktree_path"] == str(worktree.resolve())
    assert lease_status["lease"]["head_ref"] == "cycle-180-created"
    assert lease_status["lease"]["head_sha"] == base_sha
    assert _git(worktree, "rev-parse", "HEAD").stdout.strip() == base_sha

    if with_receipt:
        monkeypatch.chdir(worktree)
        receipt_check = worktree_receipt.check_receipt(receipt_path)
        assert receipt_check.ok, receipt_check.failure
        assert receipt_check.task_id == lease_status["lease"]["gate_id"]
        assert receipt_check.schema == "delegated_worktree_receipt.v1"
        assert receipt_check.expected_base_commit == receipt_check.current_commit == base_sha
        assert receipt_check.expected_ref == "refs/heads/cycle-180-created"
        assert receipt_check.expected_common_git_dir == str((repo / ".git").resolve())


def test_lease_cli_rejects_inherited_descriptor_for_another_file(tmp_path: Path) -> None:
    """Direct CLI imports must retain the repository-lock identity check."""
    repo = tmp_path / "repo"
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    (repo / ".git" / "robot-sf-create-worktree.lock").touch()
    with (tmp_path / "unrelated.lock").open("w", encoding="utf-8") as unrelated_lock:
        lock_fd = unrelated_lock.fileno()
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(CREATE_WORKTREE.with_name("pr_gate_lease.py")),
                "create",
                "--worktree",
                str(repo),
                "--gate-id",
                "issue-8610-invalid-lock",
            ],
            cwd=repo,
            env={**os.environ, "ROBOT_SF_WORKTREE_LOCK_FD": str(lock_fd)},
            pass_fds=(lock_fd,),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "inherited lock descriptor does not identify the repository lock file" in result.stderr
    assert not list((repo / ".git").glob(".pr-gate-lease-*.json"))


@pytest.mark.parametrize("force_python", (False, True), ids=("flock-cli", "portable-python"))
def test_creator_task_id_failure_rolls_back_worktree_and_branch(
    tmp_path: Path, monkeypatch, force_python: bool
) -> None:
    """A failed lease handoff cannot leave an unleased checkout or orphan branch."""
    repo = tmp_path / "creator-repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(repo, "config", "user.name", "Issue 8577 Test")
    _git(repo, "config", "user.email", "issue-8577@example.invalid")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "fixture")
    unrelated_branch = "unrelated-branch"
    _git(repo, "branch", unrelated_branch)
    main_sha = _git(repo, "rev-parse", "refs/heads/main").stdout.strip()
    unrelated_sha = _git(repo, "rev-parse", f"refs/heads/{unrelated_branch}").stdout.strip()
    worktree = tmp_path / "creator-worktree"
    branch = "cycle-181-failed-lease"

    stub_bin = tmp_path / "stub-bin"
    stub_bin.mkdir()
    python_stub = stub_bin / "python3"
    python_stub.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "import sys\n"
        "if (\n"
        "    len(sys.argv) > 2\n"
        "    and sys.argv[1].endswith('/pr_gate_lease.py')\n"
        "    and sys.argv[2] == 'create'\n"
        "):\n"
        "    print('injected lease handoff failure', file=sys.stderr)\n"
        "    raise SystemExit(73)\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PATH"] = f"{stub_bin}:{environment['PATH']}"
    if force_python:
        environment["ROBOT_SF_WORKTREE_FORCE_PYTHON_LOCK"] = "1"
    else:
        environment.pop("ROBOT_SF_WORKTREE_FORCE_PYTHON_LOCK", None)

    result = subprocess.run(
        [
            str(CREATE_WORKTREE),
            "--path",
            str(worktree),
            "--branch",
            branch,
            "--base",
            "HEAD",
            "--minimum-free-bytes",
            "0",
            "--task-id",
            "cycle-181-failed-lease",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 73, result.stdout + result.stderr
    assert "injected lease handoff failure" in result.stderr
    assert not worktree.exists()
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"
    assert _git(repo, "rev-parse", "refs/heads/main").stdout.strip() == main_sha
    assert _git(repo, "rev-parse", f"refs/heads/{unrelated_branch}").stdout.strip() == unrelated_sha
    assert not _git(repo, "branch", "--list", branch).stdout.strip()
    assert branch not in _git(repo, "worktree", "list", "--porcelain").stdout
    monkeypatch.chdir(repo)
    assert status(worktree_path=worktree)["active"] is False


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

    with monkeypatch.context() as isolated:
        isolated.setattr(reaper, "_read_unpushed_state", lambda _path, _branch: (False, None))
        isolated.setattr(
            reaper,
            "_read_strict_open_pr_state",
            lambda _branch, *, repo, transport: (False, None, []),
        )
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
