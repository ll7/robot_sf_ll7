"""Regression coverage for capacity-guarded worktree creation."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.dev import check_worktree_capacity as capacity
from scripts.dev import worktree_creation_lock
from tests.support.environment_guards import git_identity_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_CAPACITY = REPO_ROOT / "scripts" / "dev" / "check_worktree_capacity.py"
CREATE_WORKTREE = REPO_ROOT / "scripts" / "dev" / "create_worktree.sh"
WORKTREE_CREATION_LOCK = REPO_ROOT / "scripts" / "dev" / "worktree_creation_lock.py"
# Test hook honored by create_worktree.sh: force the portable Python fcntl
# fallback even when the flock CLI is installed (issue #8488).
FORCE_PYTHON_LOCK_ENV = {"ROBOT_SF_WORKTREE_FORCE_PYTHON_LOCK": "1"}


def _unique_branch(tmp_path: Path, name: str) -> str:
    """Return a repository-global branch name unique to this test invocation.

    Concurrent readiness runs or xdist workers each get a distinct process id,
    and every invocation gets its own ``tmp_path``.  Hashing the complete
    resolved path avoids collisions from truncated temporary-directory names
    or PID reuse across PID namespaces (issue #7804).
    """
    path_digest = hashlib.sha256(str(tmp_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"test/{name}-{path_digest}-{os.getpid()}"


def _worktree_target(tmp_path: Path, branch: str) -> Path:
    """Give Git's shared worktree-admin directory a test-unique basename."""
    basename_digest = hashlib.sha256(branch.encode("utf-8")).hexdigest()[:10]
    return tmp_path / f"new-worktree-{basename_digest}"


def _cleanup_owned_worktree(target: Path, branch: str) -> None:
    """Remove only the worktree, ref, and config entries owned by a fixture."""
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(target)],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    subprocess.run(["git", "branch", "-D", branch], cwd=REPO_ROOT, capture_output=True, check=False)
    for suffix in ("remote", "merge"):
        subprocess.run(
            ["git", "config", "--unset-all", f"branch.{branch}.{suffix}"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )


def _create_orphan_branch(branch: str) -> None:
    """Create the orphan fixture without writing implicit upstream config."""
    subprocess.run(
        ["git", "branch", "--no-track", "-f", branch, "origin/main"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )


def _assert_no_branch_upstream(branch: str) -> None:
    """Ensure the test fixture did not add branch-specific config entries."""
    for suffix in ("remote", "merge"):
        result = subprocess.run(
            ["git", "config", "--get", f"branch.{branch}.{suffix}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1, (
            f"unexpected upstream config for {branch}: "
            f"branch.{branch}.{suffix}={result.stdout.strip()!r} {result.stderr.strip()!r}"
        )


def _run_orphan_recovery_child() -> None:
    """Exercise the real orphan-recovery path from an independent process."""
    with tempfile.TemporaryDirectory(prefix="robot-sf-worktree-child-") as temp_dir:
        temp_root = Path(temp_dir)
        branch = _unique_branch(temp_root, "orphan-recover")
        target = _worktree_target(temp_root, branch)
        print(json.dumps({"branch": branch, "target": str(target)}), flush=True)
        try:
            _create_orphan_branch(branch)
            _assert_no_branch_upstream(branch)
            if os.environ.get("ROBOT_SF_TEST_CHILD_EXIT_AFTER_SETUP") == "1":
                os._exit(17)
            if os.environ.get("ROBOT_SF_TEST_CHILD_PAUSE_AFTER_SETUP") == "1":
                time.sleep(60)
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
            _cleanup_owned_worktree(target, branch)


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
    branch = _unique_branch(tmp_path, "exec-in-worktree")
    target = _worktree_target(tmp_path, branch)
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
        upstream = subprocess.run(
            [
                "git",
                "-C",
                str(target),
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert upstream.returncode != 0
        assert "no upstream configured" in upstream.stderr
    finally:
        _cleanup_owned_worktree(target, branch)


def test_create_worktree_exec_requires_command(tmp_path: Path) -> None:
    target = tmp_path / "new-worktree"
    result = subprocess.run(
        [
            str(CREATE_WORKTREE),
            "--path",
            str(target),
            "--branch",
            _unique_branch(tmp_path, "exec-missing-command"),
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
    branch = _unique_branch(tmp_path, "orphan-recover")
    try:
        # --no-track keeps this fixture from writing shared upstream config;
        # origin/main remains the base commit and guarantees ancestor recovery.
        _create_orphan_branch(branch)
        _assert_no_branch_upstream(branch)
        target = _worktree_target(tmp_path, branch)
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
        _cleanup_owned_worktree(_worktree_target(tmp_path, branch), branch)


def test_create_worktree_python_fallback_serializes_against_lock_holder(
    tmp_path: Path,
) -> None:
    """The portable fallback must wait for the same shared lock file identity."""
    fcntl = pytest.importorskip("fcntl")
    branch = _unique_branch(tmp_path, "py-fallback-locked")
    target = _worktree_target(tmp_path, branch)
    common_dir = Path(
        subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    lock_path = common_dir / "robot-sf-create-worktree.lock"
    process: subprocess.Popen[str] | None = None
    try:
        _create_orphan_branch(branch)
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            process = subprocess.Popen(
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
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, **FORCE_PYTHON_LOCK_ENV},
            )
            time.sleep(0.5)
            assert process.poll() is None, "fallback creator did not wait for the lock"
            assert not target.exists()
            assert (
                subprocess.run(
                    ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
                    cwd=REPO_ROOT,
                    check=False,
                ).returncode
                == 0
            )
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        stdout, stderr = process.communicate(timeout=120)
        assert process.returncode == 0, f"stdout={stdout!r} stderr={stderr!r}"
        assert "portable lock" in stderr
        assert target.is_dir()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        _cleanup_owned_worktree(target, branch)


def test_create_worktree_lock_covers_orphan_recovery_and_add(tmp_path: Path) -> None:
    """The repository lock must cover branch cleanup through worktree registration."""
    fcntl = pytest.importorskip("fcntl")
    branch = _unique_branch(tmp_path, "locked-orphan-recover")
    target = _worktree_target(tmp_path, branch)
    common_dir = Path(
        subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    lock_path = common_dir / "robot-sf-create-worktree.lock"
    process: subprocess.Popen[str] | None = None
    try:
        _create_orphan_branch(branch)
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            process = subprocess.Popen(
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
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            time.sleep(0.5)
            assert process.poll() is None, "creator did not wait for the repository lock"
            assert not target.exists()
            assert (
                subprocess.run(
                    ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
                    cwd=REPO_ROOT,
                    check=False,
                ).returncode
                == 0
            )
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        stdout, stderr = process.communicate(timeout=60)
        assert process.returncode == 0, f"stdout={stdout!r} stderr={stderr!r}"
        assert target.is_dir()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        _cleanup_owned_worktree(target, branch)


def test_create_worktree_python_fallback_concurrent_creations_serialize(
    tmp_path: Path,
) -> None:
    """Two competing fallback creations must both complete and register cleanly."""
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_branch = _unique_branch(first_root, "py-fallback-race")
    second_branch = _unique_branch(second_root, "py-fallback-race")
    first_target = _worktree_target(first_root, first_branch)
    second_target = _worktree_target(second_root, second_branch)
    assert first_branch != second_branch
    processes: list[subprocess.Popen[str]] = []
    try:
        for branch, target in (
            (first_branch, first_target),
            (second_branch, second_target),
        ):
            processes.append(
                subprocess.Popen(
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
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env={**os.environ, **FORCE_PYTHON_LOCK_ENV},
                )
            )
        for process in processes:
            stdout, stderr = process.communicate(timeout=180)
            assert process.returncode == 0, f"stdout={stdout!r} stderr={stderr!r}"
            assert "portable lock" in stderr
        assert first_target.is_dir()
        assert second_target.is_dir()
        registered = subprocess.run(
            ["git", "worktree", "list"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert f"[{first_branch}]" in registered
        assert f"[{second_branch}]" in registered
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
        _cleanup_owned_worktree(first_target, first_branch)
        _cleanup_owned_worktree(second_target, second_branch)


def test_create_worktree_python_fallback_refuses_low_space(tmp_path: Path) -> None:
    """Failed capacity admission still refuses on the portable fallback path."""
    target = tmp_path / "new-worktree"
    result = subprocess.run(
        [
            str(CREATE_WORKTREE),
            "--path",
            str(target),
            "--branch",
            _unique_branch(tmp_path, "py-fallback-blocked"),
            "--minimum-free-bytes",
            str(2**63),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **FORCE_PYTHON_LOCK_ENV},
    )

    assert result.returncode == 2
    assert "available space is below" in result.stdout
    assert not target.exists()


def test_create_worktree_rejects_direct_locked_transaction(tmp_path: Path) -> None:
    """The internal re-entry flag cannot bypass the repository lock."""
    branch = _unique_branch(tmp_path, "direct-locked-transaction")
    target = _worktree_target(tmp_path, branch)
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
                "--__locked-transaction",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 2
        assert "internal" in result.stderr
        assert "repository lock" in result.stderr
        assert not target.exists()
    finally:
        _cleanup_owned_worktree(target, branch)


def test_create_worktree_locked_transaction_keeps_capacity_gate(tmp_path: Path) -> None:
    """The internal re-entry flag cannot skip the capacity safety gate."""
    branch = _unique_branch(tmp_path, "direct-locked-capacity")
    target = _worktree_target(tmp_path, branch)
    try:
        result = subprocess.run(
            [
                str(CREATE_WORKTREE),
                "--path",
                str(target),
                "--branch",
                branch,
                "--minimum-free-bytes",
                str(2**63),
                "--__locked-transaction",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 2
        assert "available space is below" in result.stdout
        assert not target.exists()
    finally:
        _cleanup_owned_worktree(target, branch)


def test_create_worktree_without_flock_cli_uses_python_fallback(tmp_path: Path) -> None:
    """A PATH without flock (e.g. macOS) must fall back instead of exiting 2."""
    stub_bin = tmp_path / "stub-bin"
    stub_bin.mkdir()
    for tool in ("bash", "sh", "git", "git-lfs", "python3", "dirname", "cat", "grep", "du"):
        resolved = shutil.which(tool)
        assert resolved is not None, f"test host is missing required tool: {tool}"
        (stub_bin / tool).symlink_to(resolved)
    assert shutil.which("flock", path=str(stub_bin)) is None, "stub PATH must hide flock"
    branch = _unique_branch(tmp_path, "no-flock-fallback")
    target = _worktree_target(tmp_path, branch)
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
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PATH": str(stub_bin)},
        )
        assert result.returncode == 0, result.stderr
        assert "portable lock" in result.stderr
        assert "flock is required" not in result.stderr
        assert target.is_dir()
    finally:
        _cleanup_owned_worktree(target, branch)


def test_create_worktree_python_fallback_runs_exec_exactly_once(tmp_path: Path) -> None:
    """--exec must survive the fallback re-entry and run exactly once."""
    branch = _unique_branch(tmp_path, "py-fallback-exec")
    target = _worktree_target(tmp_path, branch)
    marker = tmp_path / "exec-marker.txt"
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
                "bash",
                "-c",
                f"echo ran >> {marker}",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, **FORCE_PYTHON_LOCK_ENV},
        )
        assert result.returncode == 0, result.stderr
        assert marker.is_file(), "fallback must run the --exec command"
        assert marker.read_text().splitlines() == ["ran"]
    finally:
        _cleanup_owned_worktree(target, branch)


def test_create_worktree_python_fallback_exec_runs_after_lock_release(tmp_path: Path) -> None:
    """Portable creation releases the lock before --exec while keeping receipt checks."""
    pytest.importorskip("fcntl")
    branch = _unique_branch(tmp_path, "py-fallback-exec-unlocked")
    target = _worktree_target(tmp_path, branch)
    marker = tmp_path / "exec-lock-marker.txt"
    receipt_path = tmp_path / "exec.receipt.json"
    common_dir = Path(
        subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    lock_path = common_dir / "robot-sf-create-worktree.lock"
    probe_code = (
        "import fcntl\n"
        "import sys\n"
        "from pathlib import Path\n"
        "with open(sys.argv[1], 'a+') as lock_file:\n"
        "    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "Path(sys.argv[2]).write_text('unlocked\\n', encoding='utf-8')\n"
        "assert Path(sys.argv[3]).is_file()\n"
    )
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
                "--receipt",
                str(receipt_path),
                "--task-id",
                "issue-8498",
                "--exec",
                sys.executable,
                "-c",
                probe_code,
                str(lock_path),
                str(marker),
                str(receipt_path),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, **FORCE_PYTHON_LOCK_ENV},
        )

        assert result.returncode == 0, result.stderr
        assert marker.read_text(encoding="utf-8") == "unlocked\n"
        assert receipt_path.is_file()
    finally:
        _cleanup_owned_worktree(target, branch)


def test_worktree_creation_lock_helper_contract(tmp_path: Path) -> None:
    """The portable lock helper validates usage and propagates child status."""
    assert os.access(WORKTREE_CREATION_LOCK, os.X_OK)
    compile_result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(WORKTREE_CREATION_LOCK)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    lock_path = tmp_path / "test-creation.lock"
    usage = subprocess.run(
        [sys.executable, str(WORKTREE_CREATION_LOCK), str(lock_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert usage.returncode == 2
    assert "usage" in usage.stderr

    propagated = subprocess.run(
        [
            sys.executable,
            str(WORKTREE_CREATION_LOCK),
            str(lock_path),
            "--",
            sys.executable,
            "-c",
            "import sys; sys.exit(7)",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert propagated.returncode == 7
    assert lock_path.exists()

    signaled = subprocess.run(
        [
            sys.executable,
            str(WORKTREE_CREATION_LOCK),
            str(lock_path),
            "--",
            sys.executable,
            "-c",
            "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert signaled.returncode == 128 + signal.SIGTERM


def test_worktree_creation_lock_forwards_signal_and_keeps_lock_until_child_exit(
    tmp_path: Path,
) -> None:
    """A terminating helper must not release the lock while its mutation child lives."""
    lock_path = tmp_path / "test-signal.lock"
    started = tmp_path / "child.started"
    signal_marker = tmp_path / "child.signal"
    release = tmp_path / "child.release"
    contender_marker = tmp_path / "contender.acquired"
    child_code = (
        "import os, signal, sys, time\n"
        "from pathlib import Path\n"
        "started, signal_marker, release = map(Path, sys.argv[1:])\n"
        "def record_signal(signum, _frame):\n"
        "    signal_marker.write_text(str(signum), encoding='utf-8')\n"
        "signal.signal(signal.SIGTERM, record_signal)\n"
        "started.write_text(str(os.getpid()), encoding='utf-8')\n"
        "while not release.exists():\n"
        "    time.sleep(0.02)\n"
    )
    holder = subprocess.Popen(
        [
            sys.executable,
            str(WORKTREE_CREATION_LOCK),
            str(lock_path),
            "--",
            sys.executable,
            "-c",
            child_code,
            str(started),
            str(signal_marker),
            str(release),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    contender: subprocess.Popen[str] | None = None
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            assert holder.poll() is None, "lock helper exited before its child started"
            time.sleep(0.02)
        assert started.is_file(), "lock helper child did not start"
        child_pid = int(started.read_text(encoding="utf-8"))

        holder.send_signal(signal.SIGTERM)
        time.sleep(0.2)
        assert holder.poll() is None, "helper released the lock before its child exited"
        deadline = time.monotonic() + 5
        while not signal_marker.exists() and time.monotonic() < deadline:
            assert holder.poll() is None, "helper exited before forwarding SIGTERM"
            time.sleep(0.02)
        assert signal_marker.read_text(encoding="utf-8") == str(signal.SIGTERM)

        contender = subprocess.Popen(
            [
                sys.executable,
                str(WORKTREE_CREATION_LOCK),
                str(lock_path),
                "--",
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')",
                str(contender_marker),
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.2)
        assert contender.poll() is None, "child mutation was still alive but the lock was released"

        release.touch()
        assert holder.wait(timeout=10) == 128 + signal.SIGTERM
        assert contender.wait(timeout=10) == 0
        assert contender_marker.read_text(encoding="utf-8") == "ok"
    finally:
        release.touch()
        if holder.poll() is None:
            holder.terminate()
            try:
                holder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                holder.kill()
                holder.wait(timeout=5)
        if contender is not None and contender.poll() is None:
            contender.kill()
            contender.wait(timeout=5)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_worktree_creation_lock_child_descriptor_survives_helper_sigkill(
    tmp_path: Path,
) -> None:
    """An uncatchable helper exit must not release a live child's inherited lock."""
    lock_path = tmp_path / "test-sigkill.lock"
    started = tmp_path / "child.started"
    release = tmp_path / "child.release"
    finished = tmp_path / "child.finished"
    contender_marker = tmp_path / "contender.acquired"
    child_code = (
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "started, release, finished = map(Path, sys.argv[1:])\n"
        "started.write_text(str(os.getpid()), encoding='utf-8')\n"
        "while not release.exists():\n"
        "    time.sleep(0.02)\n"
        "finished.touch()\n"
    )
    holder = subprocess.Popen(
        [
            sys.executable,
            str(WORKTREE_CREATION_LOCK),
            str(lock_path),
            "--",
            sys.executable,
            "-c",
            child_code,
            str(started),
            str(release),
            str(finished),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    contender: subprocess.Popen[bytes] | None = None
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            assert holder.poll() is None, "lock helper exited before its child started"
            time.sleep(0.02)
        assert started.is_file(), "lock helper child did not start"
        child_pid = int(started.read_text(encoding="utf-8"))

        holder.kill()
        assert holder.wait(timeout=5) == -signal.SIGKILL
        contender = subprocess.Popen(
            [
                sys.executable,
                str(WORKTREE_CREATION_LOCK),
                str(lock_path),
                "--",
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')",
                str(contender_marker),
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.2)
        assert contender.poll() is None, "helper SIGKILL released a live child's lock"

        release.touch()
        deadline = time.monotonic() + 5
        while not finished.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert finished.is_file(), "child did not finish after the release marker"
        assert contender.wait(timeout=10) == 0
        assert contender_marker.read_text(encoding="utf-8") == "ok"
    finally:
        release.touch()
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=5)
        if contender is not None and contender.poll() is None:
            contender.kill()
            contender.wait(timeout=5)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(
    ("lock_errno", "expected_returncode"),
    ((errno.EAGAIN, 75), (errno.EACCES, 75), (errno.EIO, 2)),
)
def test_worktree_creation_lock_helper_non_blocking_errno_contract(
    monkeypatch,
    tmp_path: Path,
    lock_errno: int,
    expected_returncode: int,
) -> None:
    """Only documented non-blocking contention errno values map to exit 75."""
    fcntl = pytest.importorskip("fcntl")

    def fake_flock(_file_descriptor: int, operation: int) -> None:
        assert operation & fcntl.LOCK_NB
        raise OSError(lock_errno, "simulated lock contention")

    monkeypatch.setattr(fcntl, "flock", fake_flock)
    result = worktree_creation_lock.run(
        ["--non-blocking", str(tmp_path / "test-errno.lock"), "--", sys.executable, "-c", "pass"]
    )

    assert result == expected_returncode


def test_worktree_creation_lock_helper_non_blocking_reports_contention(
    tmp_path: Path,
) -> None:
    """--non-blocking exits 75 on a held lock and runs the child when free."""
    lock_path = tmp_path / "test-nonblocking.lock"
    held_marker = tmp_path / "test-nonblocking-held"
    holder = subprocess.Popen(
        [
            sys.executable,
            str(WORKTREE_CREATION_LOCK),
            str(lock_path),
            "--",
            sys.executable,
            "-c",
            "from pathlib import Path; import sys, time; Path(sys.argv[1]).touch(); time.sleep(30)",
            str(held_marker),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while not held_marker.exists() and holder.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert held_marker.exists(), "holder did not acquire the lock before running the child"
        contended = subprocess.run(
            [
                sys.executable,
                str(WORKTREE_CREATION_LOCK),
                "--non-blocking",
                str(lock_path),
                "--",
                sys.executable,
                "-c",
                "pass",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert contended.returncode == 75, contended.stderr
    finally:
        holder.terminate()
        holder.wait(timeout=15)
    free = subprocess.run(
        [
            sys.executable,
            str(WORKTREE_CREATION_LOCK),
            "--non-blocking",
            str(lock_path),
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert free.returncode == 0, free.stderr


def test_create_worktree_hints_when_orphan_branch_diverged(tmp_path: Path) -> None:
    """An orphan branch that is not an ancestor of the base gets a recovery hint."""
    branch = _unique_branch(tmp_path, "orphan-diverged")
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
        env=git_identity_environment(),
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
        target = _worktree_target(tmp_path, branch)
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


def test_unique_branch_names_differ_per_invocation(tmp_path: Path) -> None:
    """Two independent invocations must never share or delete a branch ref."""
    first_tmp = tmp_path / "abcdefghij-first"
    second_tmp = tmp_path / "abcdefghij-second"
    first_tmp.mkdir()
    second_tmp.mkdir()
    first = _unique_branch(first_tmp, "isolation")
    second = _unique_branch(second_tmp, "isolation")

    assert first != second
    assert first.startswith("test/isolation-")
    assert second.startswith("test/isolation-")
    assert first.endswith(str(os.getpid()))
    assert second.endswith(str(os.getpid()))
    first_digest = hashlib.sha256(str(first_tmp.resolve()).encode("utf-8")).hexdigest()[:10]
    assert first == f"test/isolation-{first_digest}-{os.getpid()}"


def test_owned_cleanup_does_not_prune_unrelated_worktrees(monkeypatch, tmp_path: Path) -> None:
    """Fixture cleanup must address exact paths and refs, never repository-global state."""
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    target = tmp_path / "owned-worktree"
    branch = "test/owned-cleanup"

    _cleanup_owned_worktree(target, branch)

    assert calls == [
        ["git", "worktree", "remove", "--force", str(target)],
        ["git", "branch", "-D", branch],
        ["git", "config", "--unset-all", f"branch.{branch}.remote"],
        ["git", "config", "--unset-all", f"branch.{branch}.merge"],
    ]


def _run_child_and_cleanup(
    env_overrides: dict[str, str], *, timeout: float
) -> tuple[dict[str, str], int, str]:
    child_code = (
        "from tests.dev.test_worktree_capacity import _run_orphan_recovery_child; "
        "_run_orphan_recovery_child()"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child_code],
        cwd=REPO_ROOT,
        env={**os.environ, **env_overrides},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        stdout, stderr = process.communicate()

    assert stdout, f"child did not publish owned-artifact identity: {stderr}"
    record = json.loads(stdout.splitlines()[0])
    _cleanup_owned_worktree(Path(record["target"]), record["branch"])
    ref = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{record['branch']}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    assert ref.returncode == 1, f"leaked child branch: {record['branch']}"
    _assert_no_branch_upstream(record["branch"])
    worktree_list = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert record["target"] not in worktree_list
    assert timed_out or process.returncode != 0, "child unexpectedly completed successfully"
    return record, process.returncode, stderr


def test_parent_cleanup_handles_failed_child_without_finally() -> None:
    """A child exiting after setup cannot leak its exact branch or config."""
    _run_child_and_cleanup({"ROBOT_SF_TEST_CHILD_EXIT_AFTER_SETUP": "1"}, timeout=60)


def test_parent_cleanup_handles_timed_out_child_without_finally() -> None:
    """A timed-out child is killed, then its parent removes the owned artifacts."""
    _run_child_and_cleanup({"ROBOT_SF_TEST_CHILD_PAUSE_AFTER_SETUP": "1"}, timeout=5)


def test_concurrent_orphan_recovery_uses_independent_processes() -> None:
    """Two real child processes must recover and clean up distinct orphan branches."""
    child_code = (
        "from tests.dev.test_worktree_capacity import _run_orphan_recovery_child; "
        "_run_orphan_recovery_child()"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", child_code],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    results: list[tuple[str, str, int]] = []
    for process in processes:
        try:
            stdout, stderr = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        results.append((stdout, stderr, process.returncode))

    records: list[dict[str, str]] = []
    missing_output: list[str] = []
    for stdout, stderr, returncode in results:
        if stdout:
            records.append(json.loads(stdout.splitlines()[0]))
        else:
            missing_output.append(stderr)

    for record in records:
        _cleanup_owned_worktree(Path(record["target"]), record["branch"])

    assert not missing_output, f"child did not publish cleanup identity: {missing_output}"

    for stdout, stderr, returncode in results:
        assert returncode == 0, f"child failed with stdout={stdout!r} stderr={stderr!r}"

    branches = [record["branch"] for record in records]
    assert len(set(branches)) == 2, records
    worktree_list = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for record in records:
        branch = record["branch"]
        ref = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        assert ref.returncode == 1, f"leaked branch ref: {branch}"
        _assert_no_branch_upstream(branch)
        assert record["target"] not in worktree_list


def test_unique_branch_names_are_clean_git_refs(tmp_path: Path) -> None:
    """Per-invocation branch names must be valid, branchable git refs."""
    branch = _unique_branch(tmp_path, "ref-validity")
    created = subprocess.run(
        ["git", "check-ref-format", "--branch", branch],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert created.returncode == 0, branch

    subprocess.run(["git", "branch", branch], cwd=REPO_ROOT, check=True, capture_output=True)
    try:
        listed = subprocess.run(
            ["git", "branch", "--list", branch],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert branch in listed
    finally:
        subprocess.run(
            ["git", "branch", "-D", branch], cwd=REPO_ROOT, capture_output=True, check=False
        )
