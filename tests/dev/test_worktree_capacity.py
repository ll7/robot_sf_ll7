"""Regression coverage for capacity-guarded worktree creation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from scripts.dev import check_worktree_capacity as capacity

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_CAPACITY = REPO_ROOT / "scripts" / "dev" / "check_worktree_capacity.py"
CREATE_WORKTREE = REPO_ROOT / "scripts" / "dev" / "create_worktree.sh"


def _unique_branch(tmp_path: Path, name: str) -> str:
    """Return a repository-global branch name unique to this test invocation.

    Concurrent readiness runs or xdist workers each get a distinct process id,
    and every invocation gets its own ``tmp_path``; combining both yields a
    collision-free ref that one invocation owns and cleans up (issue #7804).
    """
    path_part = tmp_path.name.replace("-", "").replace("_", "")[:10] or "default"
    return f"test/{name}-{path_part}-{os.getpid()}"


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
        target = temp_root / "new-worktree"
        print(json.dumps({"branch": branch, "target": str(target)}), flush=True)
        try:
            _create_orphan_branch(branch)
            _assert_no_branch_upstream(branch)
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
    branch = _unique_branch(tmp_path, "exec-in-worktree")
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


def test_unique_branch_names_differ_per_invocation(tmp_path: Path) -> None:
    """Two independent invocations must never share or delete a branch ref."""
    first = _unique_branch(tmp_path, "isolation")
    second_tmp = tmp_path / "another-invocation"
    second_tmp.mkdir()
    second = _unique_branch(second_tmp, "isolation")

    assert first != second
    assert first.startswith("test/isolation-")
    assert second.startswith("test/isolation-")
    assert first.endswith(str(os.getpid()))
    assert second.endswith(str(os.getpid()))


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
            raise AssertionError(f"orphan-recovery child timed out: {stderr}") from None
        results.append((stdout, stderr, process.returncode))

    records: list[dict[str, str]] = []
    for stdout, stderr, returncode in results:
        assert returncode == 0, f"child failed with stdout={stdout!r} stderr={stderr!r}"
        records.append(json.loads(stdout.splitlines()[0]))

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
