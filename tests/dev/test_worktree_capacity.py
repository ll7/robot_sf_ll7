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
    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert after == before


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
