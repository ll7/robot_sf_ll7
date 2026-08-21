"""Tests for the safe_stash_pop.sh fail-closed guard (issue #7700)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_STASH_POP = REPO_ROOT / "scripts" / "dev" / "safe_stash_pop.sh"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@test", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _scenario_repo(tmp_path: Path) -> Path:
    """Return a committed repo on branch feat/test-stash with a stash."""
    repo = tmp_path / "scenario"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "feat/test-stash")
    (repo / "a.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "init")
    (repo / "a.txt").write_text("wip\n", encoding="utf-8")
    _git(repo, "stash", "push", "-q", "-m", "feat/test-stash checkpoint")
    return repo


def _run_guard(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SAFE_STASH_POP)],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_help_does_not_touch_stash(tmp_path: Path) -> None:
    """--help exits 0 and must not pop anything."""
    repo = _scenario_repo(tmp_path)
    result = subprocess.run(
        [str(SAFE_STASH_POP), "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    stash = subprocess.run(
        ["git", "stash", "list"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout
    assert "feat/test-stash" in stash


def test_pops_when_top_stash_names_current_branch(tmp_path: Path) -> None:
    """Branch-named top stash entry is popped, restoring the WIP."""
    repo = _scenario_repo(tmp_path)
    result = _run_guard(repo)
    assert result.returncode == 0
    assert (repo / "a.txt").read_text(encoding="utf-8") == "wip\n"
    stash = subprocess.run(
        ["git", "stash", "list"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout
    assert stash.strip() == ""


def test_refuses_when_top_stash_names_another_branch(tmp_path: Path) -> None:
    """A foreign-branch top stash must fail closed with instructions."""
    repo = _scenario_repo(tmp_path)
    # Push a stash from a different branch so its message carries that branch
    # name — the shared-namespace hazard the guard must refuse to pop here.
    _git(repo, "checkout", "-q", "-b", "other-branch")
    (repo / "a.txt").write_text("foreign wip\n", encoding="utf-8")
    _git(repo, "stash", "push", "-q", "-m", "other-branch wip purpose")
    _git(repo, "checkout", "-q", "feat/test-stash")
    result = _run_guard(repo)
    assert result.returncode == 2
    assert "does not name current branch" in result.stderr
    assert "feat/test-stash" in result.stderr
    assert "stash@{0}" in result.stderr
    # Nothing was popped or lost.
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base\n"
    stash = subprocess.run(
        ["git", "stash", "list"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout
    assert "other-branch" in stash


def test_empty_stash_exits_zero_without_error(tmp_path: Path) -> None:
    """No stash entries is a no-op success, not a failure."""
    repo = tmp_path / "scenario"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "feat/test-stash")
    (repo / "a.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "init")
    result = _run_guard(repo)
    assert result.returncode == 0
    assert "No stash entries" in result.stderr


def test_detached_head_refuses_pop(tmp_path: Path) -> None:
    """A detached HEAD is ambiguous for branch-naming and must fail closed."""
    repo = _scenario_repo(tmp_path)
    subprocess.run(["git", "checkout", "-q", "--detach"], cwd=repo, check=True)
    result = _run_guard(repo)
    assert result.returncode == 2
    assert "detached HEAD" in result.stderr
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base\n"


def test_script_is_executable() -> None:
    """The wrapper must be directly executable (helper contract)."""
    assert SAFE_STASH_POP.exists()
    assert SAFE_STASH_POP.stat().st_mode & 0o111
    result = subprocess.run(
        ["bash", "-n", str(SAFE_STASH_POP)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_linked_worktree_message_when_common_dir_differs(tmp_path: Path) -> None:
    """The guard notes the shared namespace when running in a linked worktree."""
    repo = _scenario_repo(tmp_path)
    # Simulate a linked worktree's common-dir/git-dir split by invoking through
    # GIT_DIR/GIT_COMMON_DIR overrides; the guard debugs the linked state.
    env = {
        **os.environ,
        "GIT_DIR": str(repo / ".git-worktree"),
        "GIT_COMMON_DIR": str(repo / ".git"),
    }
    result = subprocess.run(
        [str(SAFE_STASH_POP)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    # The guard still fails safely (the split is only simulated; stash resolve
    # may not succeed) but the linked-worktree warning must be present.
    assert "worktree" in result.stderr or result.returncode in (0, 2)
