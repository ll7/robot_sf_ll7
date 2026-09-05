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
        allowed_fetch = _git(worktree, "fetch", "origin", "main", check=False)
        assert allowed_fetch.returncode == 0, allowed_fetch.stdout + allowed_fetch.stderr
        allowed_ls_remote = _git(worktree, "ls-remote", "origin", check=False)
        assert allowed_ls_remote.returncode == 0, (
            allowed_ls_remote.stdout + allowed_ls_remote.stderr
        )
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
    """The push-only catch-all covers a remote added after configuration."""
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
        result = _git(
            worktree,
            "push",
            "--no-verify",
            "mirror",
            "HEAD:refs/heads/new-remote-bypass",
            check=False,
        )
        assert result.returncode != 0, result.stdout + result.stderr
        refs = _git(worktree, "ls-remote", "--refs", str(second_remote)).stdout
        assert "refs/heads/new-remote-bypass" not in refs
    finally:
        _remove_worktree(repo, worktree, branch)


def test_review_worktree_allows_ls_remote_and_fetch_while_blocking_pushes(tmp_path: Path) -> None:
    """Review mode separates push rejection from read-only ls-remote and fetch (issue #8321)."""
    repo, _remote = _fixture_repo(tmp_path)
    worktree = tmp_path / "review-read-urls"
    branch = "review/read-urls"
    try:
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

        # Read-only probe: git ls-remote origin refs/heads/*
        ls_remote = _git(worktree, "ls-remote", "origin", "refs/heads/*")
        assert "refs/heads/main" in ls_remote.stdout

        # Read-only fetch: git fetch origin main
        allowed_fetch = _git(worktree, "fetch", "origin", "main", check=False)
        assert allowed_fetch.returncode == 0, allowed_fetch.stdout + allowed_fetch.stderr

        # Push rejection: ordinary push is blocked
        push_normal = _git(worktree, "push", "origin", "HEAD:refs/heads/normal-push", check=False)
        assert push_normal.returncode != 0

        # Push rejection: --no-verify push is blocked
        push_no_verify = _git(
            worktree, "push", "--no-verify", "origin", "HEAD:refs/heads/no-verify-push", check=False
        )
        assert push_no_verify.returncode != 0

        # Remote refs remain unmutated
        refs = _git(worktree, "ls-remote", "--refs", "origin").stdout
        assert "refs/heads/normal-push" not in refs
        assert "refs/heads/no-verify-push" not in refs
    finally:
        _remove_worktree(repo, worktree, branch)


def test_review_mode_blocks_preconfigured_custom_transport(tmp_path: Path, monkeypatch) -> None:
    """A common-config custom transport cannot bypass the review barrier."""
    repo, _remote = _fixture_repo(tmp_path)
    helper_dir = tmp_path / "custom-transport-bin"
    helper_dir.mkdir()
    invoked = tmp_path / "custom-transport-invoked"
    helper = helper_dir / "git-remote-foo"
    helper.write_text(
        '#!/bin/sh\nprintf invoked > "$REVIEW_GUARD_CUSTOM_HELPER_SENTINEL"\nexit 97\n',
        encoding="utf-8",
    )
    helper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{helper_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("REVIEW_GUARD_CUSTOM_HELPER_SENTINEL", str(invoked))
    _git(repo, "config", "protocol.foo.allow", "always")
    worktree = tmp_path / "review-custom-transport"
    branch = "review/custom-transport"
    try:
        _git(repo, "worktree", "add", "--no-track", "-b", branch, str(worktree), "HEAD")
        configured = _configure(worktree, "review")
        assert configured.returncode == 0, configured.stderr
        assert _git(worktree, "config", "--get", "protocol.foo.allow").stdout.strip() == "never"
        _git(worktree, "remote", "add", "custom", "foo::custom-target")
        before = _remote_refs(worktree)

        result = _git(
            worktree,
            "push",
            "--no-verify",
            "custom",
            "HEAD:refs/heads/custom-protocol-bypass",
            check=False,
        )

        assert result.returncode != 0, result.stdout + result.stderr
        assert not invoked.exists(), result.stdout + result.stderr
        assert _remote_refs(worktree) == before
        restored = _configure(worktree, "implementation")
        assert restored.returncode == 0, restored.stderr
        assert _git(worktree, "config", "--get", "protocol.foo.allow").stdout.strip() == "always"
    finally:
        _remove_worktree(repo, worktree, branch)


def test_create_review_worktree_bootstraps_from_invoking_checkout(tmp_path: Path) -> None:
    """A base without the new helper can still be protected without dirtying the target."""
    repo, _remote = _fixture_repo(tmp_path)
    _git(repo, "rm", "scripts/dev/git_hooks/pre-push", "scripts/dev/review_worktree_guard.py")
    _git(repo, "commit", "-m", "fixture base predates review guard")
    worktree = tmp_path / "review-bootstrap"
    branch = "review/bootstrap"
    try:
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
        assert created.returncode == 0, created.stdout + created.stderr
        assert not (worktree / "scripts/dev/review_worktree_guard.py").exists()
        hook_root = (REPO_ROOT / "scripts/dev/git_hooks").resolve()
        assert _git(worktree, "config", "--get", "core.hooksPath").stdout.strip() == str(hook_root)
        assert (
            _git(worktree, "config", "--get", "robot-sf.worktree-mode").stdout.strip() == "review"
        )
        blocked = _git(
            worktree,
            "push",
            "origin",
            "HEAD:refs/heads/bootstrap-blocked",
            check=False,
        )
        assert blocked.returncode != 0, blocked.stdout + blocked.stderr
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
        before_orig_head = _git(worktree, "rev-parse", "origin/main").stdout.strip()
        _git(worktree, "update-ref", "ORIG_HEAD", before_orig_head)
        before_refs = _remote_refs(worktree)

        result = _integrate(worktree)
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["merge_returncode"] == 0
        assert payload["abort_attempted"] is True
        assert payload["abort_returncode"] == 0
        assert payload["head_before"] == before_head == payload["head_after"]
        assert payload["orig_head_before"] == before_orig_head == payload["orig_head_after"]
        assert payload["orig_head_restore_returncode"] == 0
        assert _git(worktree, "rev-parse", "ORIG_HEAD").stdout.strip() == before_orig_head
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
        orig_head = _git(worktree, "rev-parse", "-q", "--verify", "ORIG_HEAD", check=False)
        before_orig_head = orig_head.stdout.strip() or None
        before_refs = _remote_refs(worktree)

        result = _integrate(worktree)
        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert payload["ok"] is False
        assert payload["merge_returncode"] != 0
        assert payload["abort_attempted"] is True
        assert payload["abort_returncode"] == 0
        assert payload["orig_head_before"] == before_orig_head
        assert payload["orig_head_after"] == before_orig_head
        assert payload["orig_head_restore_returncode"] == 0
        restored_orig_head = _git(worktree, "rev-parse", "-q", "--verify", "ORIG_HEAD", check=False)
        assert (restored_orig_head.stdout.strip() or None) == before_orig_head
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
