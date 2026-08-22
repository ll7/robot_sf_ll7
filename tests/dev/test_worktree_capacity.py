"""Regression coverage for capacity-guarded worktree creation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts.dev import check_worktree_capacity as capacity

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_CAPACITY = REPO_ROOT / "scripts" / "dev" / "check_worktree_capacity.py"
CREATE_WORKTREE = REPO_ROOT / "scripts" / "dev" / "create_worktree.sh"


def test_capacity_inspects_existing_parent_without_creating_target(tmp_path: Path) -> None:
    target = tmp_path / "new-worktree"
    result = capacity.inspect_capacity(
        target,
        minimum_free_bytes=100,
        disk_usage=lambda _path: SimpleNamespace(free=200),
    )

    assert result.allowed
    assert result.filesystem_path == str(tmp_path)
    assert not target.exists()


def test_capacity_blocks_low_space_before_worktree_creation(tmp_path: Path) -> None:
    result = capacity.inspect_capacity(
        tmp_path / "new-worktree",
        minimum_free_bytes=201,
        disk_usage=lambda _path: SimpleNamespace(free=200),
    )

    assert result.status == "blocked"
    assert result.reason == "available space is below the worktree safety threshold"


def test_reclaim_inventory_is_descriptive_only(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "output").mkdir()
    (repo / ".worktrees").mkdir()
    shm = tmp_path / "shm"
    shm.mkdir()
    (shm / "issue-123-worker").mkdir()
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    inventory = capacity.build_reclaim_inventory(
        repo,
        shm_root=shm,
        size_fn=lambda _path: 4096,
    )

    assert {entry.category for entry in inventory} >= {"output", "worktrees", "shared-memory"}
    assert all(entry.guidance and entry.status in {"review", "absent"} for entry in inventory)
    assert all(entry.size_status in {"ok", "absent"} for entry in inventory)
    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert after == before


def test_directory_size_reports_success_with_bounded_command(monkeypatch, tmp_path: Path) -> None:
    completed = subprocess.CompletedProcess(
        args=["du"], returncode=0, stdout="4\t/path\n", stderr=""
    )
    calls: list[dict[str, object]] = []

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append({"args": args, **kwargs})
        return completed

    monkeypatch.setattr(capacity.subprocess, "run", fake_run)

    result = capacity._directory_size_bytes(tmp_path, timeout_seconds=1.5)

    assert result == capacity.DirectorySizeResult(bytes=4096, status="ok")
    assert calls[0]["timeout"] == 1.5
    assert calls[0]["args"] == (["du", "-sk", "--", str(tmp_path)],)


def test_directory_size_reports_timeout_without_hanging(monkeypatch, tmp_path: Path) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="du", timeout=0.01)

    monkeypatch.setattr(capacity.subprocess, "run", fake_run)

    result = capacity._directory_size_bytes(tmp_path, timeout_seconds=0.01)

    assert result.bytes is None
    assert result.status == "timeout"
    assert "0.01s" in (result.reason or "")


def test_directory_size_reports_unavailable_without_claiming_zero(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("du missing")

    monkeypatch.setattr(capacity.subprocess, "run", fake_run)

    result = capacity._directory_size_bytes(tmp_path)

    assert result == capacity.DirectorySizeResult(
        bytes=None,
        status="unavailable",
        reason="du could not determine the candidate size",
    )


def test_inventory_preserves_timeout_and_unavailable_evidence(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "output").mkdir()

    def size_fn(path: Path) -> capacity.DirectorySizeResult:
        if path == repo / "output":
            return capacity.DirectorySizeResult(bytes=None, status="timeout", reason="test timeout")
        return capacity.DirectorySizeResult(
            bytes=None, status="unavailable", reason="test unavailable"
        )

    inventory = capacity.build_reclaim_inventory(repo, size_fn=size_fn, shm_root=tmp_path / "shm")
    output_entry = next(entry for entry in inventory if entry.path == str(repo / "output"))

    assert output_entry.bytes is None
    assert output_entry.size_status == "timeout"
    assert output_entry.size_reason == "test timeout"


def test_capacity_cli_emits_json_and_inventory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(CHECK_CAPACITY),
            "--path",
            str(repo / "new-worktree"),
            "--minimum-free-bytes",
            "0",
            "--inventory",
            "--repo-root",
            str(repo),
            "--shm-root",
            str(tmp_path / "missing-shm"),
            "--size-timeout-seconds",
            "0.01",
            "--json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["capacity"]["status"] == "pass"
    assert payload["capacity"]["requested_path"].endswith("new-worktree")
    assert isinstance(payload["inventory"], list)
    assert all("size_status" in entry and "size_reason" in entry for entry in payload["inventory"])


def test_create_worktree_dry_run_does_not_invoke_git(tmp_path: Path) -> None:
    target = tmp_path / "new-worktree"
    result = subprocess.run(
        [
            str(CREATE_WORKTREE),
            "--path",
            str(target),
            "--branch",
            "test/capacity-dry-run",
            "--minimum-free-bytes",
            "0",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "git worktree add was not invoked" in result.stdout
    assert not target.exists()


def test_create_worktree_executes_command_inside_new_worktree(tmp_path: Path) -> None:
    target = tmp_path / "new-worktree"
    branch = "test/exec-in-worktree"
    try:
        result = subprocess.run(
            [
                str(CREATE_WORKTREE),
                "--path",
                str(target),
                "--branch",
                branch,
                "--minimum-free-bytes",
                "0",
                "--exec",
                "git",
                "rev-parse",
                "--show-toplevel",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines()[-1] == str(target)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(target)],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ["git", "worktree", "prune"], cwd=REPO_ROOT, capture_output=True, check=False
        )
        subprocess.run(
            ["git", "branch", "-D", branch], cwd=REPO_ROOT, capture_output=True, check=False
        )


def test_create_worktree_exec_requires_command(tmp_path: Path) -> None:
    target = tmp_path / "new-worktree"
    result = subprocess.run(
        [
            str(CREATE_WORKTREE),
            "--path",
            str(target),
            "--branch",
            "test/exec-missing-command",
            "--minimum-free-bytes",
            "0",
            "--exec",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--exec requires a command" in result.stderr
    assert not target.exists()


def test_create_worktree_refuses_low_space_before_git(tmp_path: Path) -> None:
    target = tmp_path / "new-worktree"
    result = subprocess.run(
        [
            str(CREATE_WORKTREE),
            "--path",
            str(target),
            "--branch",
            "test/capacity-blocked",
            "--minimum-free-bytes",
            str(2**63),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "available space is below" in result.stdout
    assert not target.exists()


def test_create_worktree_scripts_are_shell_valid_and_executable() -> None:
    assert os.access(CREATE_WORKTREE, os.X_OK)
    result = subprocess.run(
        ["bash", "-n", str(CREATE_WORKTREE)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_create_worktree_recovers_orphan_branch_before_add(tmp_path: Path) -> None:
    """An unregistered branch at the target path is removed before worktree add."""
    branch = "test/orphan-recover"
    # Create the orphan branch at origin/main (ref exists, no registered worktree);
    # -f keeps the test idempotent across partial runs. Pointing at origin/main
    # guarantees the branch is an ancestor of the default base ref.
    subprocess.run(
        ["git", "branch", "-f", branch, "origin/main"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )

    try:
        target = tmp_path / "new-worktree"
        result = subprocess.run(
            [
                str(CREATE_WORKTREE),
                "--path",
                str(target),
                "--branch",
                branch,
                "--minimum-free-bytes",
                "0",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert f"removing orphan branch '{branch}'" in result.stderr
        assert target.is_dir()
        assert (
            f"[{branch}]"
            in subprocess.run(
                ["git", "worktree", "list"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(tmp_path / "new-worktree")],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ["git", "worktree", "prune"], cwd=REPO_ROOT, capture_output=True, check=False
        )
        subprocess.run(
            ["git", "branch", "-D", branch], cwd=REPO_ROOT, capture_output=True, check=False
        )


def test_create_worktree_hints_when_orphan_branch_diverged(tmp_path: Path) -> None:
    """An orphan branch that is not an ancestor of the base gets a recovery hint."""
    branch = "test/orphan-diverged"
    # Create a synthetic root commit that cannot be an ancestor of the base ref.
    tree = subprocess.run(
        ["git", "mktree"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        input="",
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "commit-tree", tree, "-m", "orphan diverged"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Robot SF test",
            "GIT_AUTHOR_EMAIL": "robot-sf-test@example.invalid",
            "GIT_COMMITTER_NAME": "Robot SF test",
            "GIT_COMMITTER_EMAIL": "robot-sf-test@example.invalid",
        },
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch_result = subprocess.run(
        ["git", "branch", "-f", branch, commit],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert branch_result.returncode == 0, branch_result.stderr

    try:
        target = tmp_path / "new-worktree"
        result = subprocess.run(
            [
                str(CREATE_WORKTREE),
                "--path",
                str(target),
                "--branch",
                branch,
                "--minimum-free-bytes",
                "0",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
        assert "recover manually with" in result.stderr
        assert "git branch -D" in result.stderr
        assert not target.exists()
    finally:
        subprocess.run(
            ["git", "branch", "-D", branch], cwd=REPO_ROOT, capture_output=True, check=False
        )
