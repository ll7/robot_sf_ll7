"""Subprocess coverage for protected review worktrees and merge probes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CREATE_WORKTREE = REPO_ROOT / "scripts" / "dev" / "create_worktree.sh"
GUARD = REPO_ROOT / "scripts" / "dev" / "review_worktree_guard.py"
HOOK = REPO_ROOT / "scripts" / "dev" / "git_hooks" / "pre-push"


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
    )


def _fixture_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--initial-branch=main", str(repo))
    _git(repo, "config", "user.name", "review-guard-test")
    _git(repo, "config", "user.email", "review-guard@example.invalid")
    (repo / "scripts/dev/git_hooks").mkdir(parents=True)
    shutil.copy2(GUARD, repo / "scripts/dev/review_worktree_guard.py")
    shutil.copy2(HOOK, repo / "scripts/dev/git_hooks/pre-push")
    (repo / "fixture.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    _git(tmp_path, "init", "--bare", str(remote))
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "origin", "main")
    return repo, remote


def _configure(worktree: Path, mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GUARD), "configure", "--worktree", str(worktree), "--mode", mode],
        cwd=worktree,
        capture_output=True,
        text=True,
        check=False,
    )


def _integrate(worktree: Path, source_ref: str = "origin/main") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "integrate",
            "--worktree",
            str(worktree),
            "--source-ref",
            source_ref,
            "--remote",
            "origin",
        ],
        cwd=worktree,
        capture_output=True,
        text=True,
        check=False,
    )


def _remove_worktree(repo: Path, worktree: Path, branch: str) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    subprocess.run(
        ["git", "branch", "-D", branch],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def _remote_refs(worktree: Path) -> str:
    common_git_dir = _git(
        worktree,
        "rev-parse",
        "--path-format=absolute",
        "--git-common-dir",
    ).stdout.strip()
    return _git(
        worktree,
        "--git-dir",
        common_git_dir,
        "ls-remote",
        "--refs",
        "origin",
    ).stdout


def _common_git(worktree: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a remote-read command with the common config, not review barriers."""
    common_git_dir = _git(
        worktree,
        "rev-parse",
        "--path-format=absolute",
        "--git-common-dir",
    ).stdout.strip()
    return _git(worktree, "--git-dir", common_git_dir, *args)


def test_create_review_worktree_blocks_refspecs_and_no_verify(tmp_path: Path) -> None:
    """The explicit-refspec incident reproduction fails before remote mutation."""
    repo, remote = _fixture_repo(tmp_path)
    worktree = tmp_path / "review"
    branch = "review/guard"
    try:
        _git(repo, "config", "remote.origin.pushurl", str(remote))
        created = subprocess.run(
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
                "--mode",
                "review",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        assert created.returncode == 0, created.stderr
        assert (
            _git(worktree, "config", "--get", "robot-sf.worktree-mode").stdout.strip() == "review"
        )
        configured_again = _configure(worktree, "review")
        assert configured_again.returncode == 0, configured_again.stderr
        effective_pushurls = _git(
            worktree,
            "remote",
            "get-url",
            "--all",
            "--push",
            "origin",
        ).stdout.splitlines()
        assert str(remote) not in effective_pushurls
        blocked_fetch = _git(worktree, "fetch", "origin", "main", check=False)
        assert blocked_fetch.returncode != 0, blocked_fetch.stdout + blocked_fetch.stderr
        before = _remote_refs(worktree)
        expected = _git(worktree, "rev-parse", "HEAD").stdout.strip()
        attempts = (
            ["push", "origin", "HEAD:refs/heads/explicit"],
            [
                "push",
                f"--force-with-lease=refs/heads/main:{expected}",
                "origin",
                "HEAD:refs/heads/force-explicit",
            ],
            ["push", "--no-verify", "origin", "HEAD:refs/heads/no-verify"],
            ["push", "--no-verify", str(remote), "HEAD:refs/heads/url-explicit"],
            [
                "push",
                "--no-verify",
                os.path.relpath(remote, worktree),
                "HEAD:refs/heads/relative-url",
            ],
        )
        for command in attempts:
            result = _git(worktree, *command, check=False)
            assert result.returncode != 0, result.stdout + result.stderr
        assert _remote_refs(worktree) == before

        restored = _configure(worktree, "implementation")
        assert restored.returncode == 0, restored.stderr
        allowed = _git(worktree, "push", "origin", "HEAD:refs/heads/implementation", check=False)
        assert allowed.returncode == 0, allowed.stdout + allowed.stderr
        assert "refs/heads/implementation" in _remote_refs(worktree)
    finally:
        _remove_worktree(repo, worktree, branch)


def test_review_mode_blocks_a_remote_added_after_configuration(tmp_path: Path) -> None:
    """The push-only catch-all also covers a newly configured remote."""
    repo, _remote = _fixture_repo(tmp_path)
    second_remote = tmp_path / "second-remote.git"
    _git(tmp_path, "init", "--bare", str(second_remote))
    _git(repo, "push", str(second_remote), "main:refs/heads/main")
    worktree = tmp_path / "review-new-remote"
    branch = "review/new-remote"
    try:
        _git(repo, "worktree", "add", "--no-track", "-b", branch, str(worktree), "HEAD")
        configured = _configure(worktree, "review")
        assert configured.returncode == 0, configured.stderr
        _git(worktree, "remote", "add", "mirror", str(second_remote))
        _git(worktree, "config", "--worktree", "remote.mirror.pushurl", str(second_remote))
        result = _git(
            worktree,
            "push",
            "--no-verify",
            "mirror",
            "HEAD:refs/heads/new-remote-bypass",
            check=False,
        )
        assert result.returncode != 0, result.stdout + result.stderr
        refs = _common_git(worktree, "ls-remote", "--refs", str(second_remote)).stdout
        assert "refs/heads/new-remote-bypass" not in refs
    finally:
        _remove_worktree(repo, worktree, branch)


def test_ordinary_implementation_worktree_remains_pushable(tmp_path: Path) -> None:
    """The default worktree mode does not install the review-only boundary."""
    repo, _remote = _fixture_repo(tmp_path)
    worktree = tmp_path / "implementation"
    branch = "implementation/guard"
    try:
        _git(repo, "worktree", "add", "--no-track", "-b", branch, str(worktree), "HEAD")
        mode = _git(worktree, "config", "--get", "robot-sf.worktree-mode", check=False)
        assert mode.returncode != 0
        allowed = _git(worktree, "push", "origin", "HEAD:refs/heads/implementation", check=False)
        assert allowed.returncode == 0, allowed.stdout + allowed.stderr
    finally:
        _remove_worktree(repo, worktree, branch)


def test_review_configuration_rejects_a_symlinked_worktree_config(tmp_path: Path) -> None:
    """The guard must not follow a linked worktree config symlink while mutating metadata."""
    repo, _remote = _fixture_repo(tmp_path)
    worktree = tmp_path / "review-symlink"
    branch = "review/symlink"
    target = tmp_path / "config-target"
    try:
        _git(repo, "worktree", "add", "--no-track", "-b", branch, str(worktree), "HEAD")
        _git(worktree, "config", "extensions.worktreeConfig", "true")
        config_path = Path(
            _git(
                worktree,
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                "config.worktree",
            ).stdout.strip()
        )
        target.write_text("", encoding="utf-8")
        config_path.symlink_to(target)

        result = _configure(worktree, "review")

        assert result.returncode != 0
        assert "must not be a symlink" in result.stdout
        assert target.read_text(encoding="utf-8") == ""
    finally:
        _remove_worktree(repo, worktree, branch)


def test_integration_aborts_and_compares_remote_refs(tmp_path: Path) -> None:
    """A successful synthetic merge leaves HEAD, status, and remote refs unchanged."""
    repo, _remote = _fixture_repo(tmp_path)
    worktree = tmp_path / "review"
    branch = "review/integration"
    try:
        _git(repo, "worktree", "add", "--no-track", "-b", branch, str(worktree), "HEAD")
        (repo / "main-only.txt").write_text("main\n", encoding="utf-8")
        _git(repo, "add", "main-only.txt")
        _git(repo, "commit", "-m", "advance main")
        _git(repo, "push", "origin", "main")
        _git(worktree, "fetch", "origin", "main")
        configured = _configure(worktree, "review")
        assert configured.returncode == 0, configured.stderr
        before_head = _git(worktree, "rev-parse", "HEAD").stdout.strip()
        before_refs = _remote_refs(worktree)

        result = _integrate(worktree)
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["merge_returncode"] == 0
        assert payload["abort_attempted"] is True
        assert payload["abort_returncode"] == 0
        assert payload["head_before"] == before_head == payload["head_after"]
        assert payload["status_before"] == payload["status_after"] == ""
        assert payload["merge_head_after"] is None
        assert payload["remote_refs_unchanged"] is True
        assert _remote_refs(worktree) == before_refs
    finally:
        _remove_worktree(repo, worktree, branch)


def test_conflicting_integration_is_aborted_and_reported(tmp_path: Path) -> None:
    """A conflict is diagnostic failure, but cleanup and remote comparison still run."""
    repo, _remote = _fixture_repo(tmp_path)
    worktree = tmp_path / "review-conflict"
    branch = "review/conflict"
    try:
        _git(repo, "worktree", "add", "--no-track", "-b", branch, str(worktree), "HEAD")
        (worktree / "fixture.txt").write_text("feature\n", encoding="utf-8")
        _git(worktree, "add", "fixture.txt")
        _git(worktree, "commit", "-m", "feature change")
        (repo / "fixture.txt").write_text("main change\n", encoding="utf-8")
        _git(repo, "add", "fixture.txt")
        _git(repo, "commit", "-m", "main change")
        _git(repo, "push", "origin", "main")
        _git(worktree, "fetch", "origin", "main")
        configured = _configure(worktree, "review")
        assert configured.returncode == 0, configured.stderr
        before_refs = _remote_refs(worktree)

        result = _integrate(worktree)
        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert payload["ok"] is False
        assert payload["merge_returncode"] != 0
        assert payload["abort_attempted"] is True
        assert payload["abort_returncode"] == 0
        assert payload["status_after"] == ""
        assert payload["merge_head_after"] is None
        assert payload["remote_refs_unchanged"] is True
        assert _remote_refs(worktree) == before_refs
    finally:
        _remove_worktree(repo, worktree, branch)


def test_hook_is_executable_and_shell_valid() -> None:
    """The tracked Git hook is directly runnable by Git."""
    assert os.access(HOOK, os.X_OK)
    result = subprocess.run(
        ["bash", "-n", str(HOOK)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
