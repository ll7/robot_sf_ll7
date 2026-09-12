"""Regression tests for delegated-worker worktree receipts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.dev import worktree_receipt


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _fixture_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(tmp_path, "init", "--initial-branch=main", str(repo))
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "fixture.txt").write_text("original\n", encoding="utf-8")
    _git(repo, "add", "fixture.txt")
    _git(repo, "commit", "-m", "fixture")
    assigned = tmp_path / "assigned"
    wrong = tmp_path / "wrong"
    _git(repo, "worktree", "add", "--detach", str(assigned), "HEAD")
    _git(repo, "worktree", "add", "--detach", str(wrong), "HEAD")
    return repo, assigned, wrong


def test_receipt_passes_only_in_assigned_linked_worktree(tmp_path: Path, monkeypatch) -> None:
    """A receipt binds checkout, ref, common Git directory, and base."""
    _repo, assigned, wrong = _fixture_repo(tmp_path)
    receipt = worktree_receipt.create_receipt(assigned, task_id="issue-8310", base_ref="HEAD")
    receipt_path = tmp_path / "receipt.json"
    worktree_receipt._write_atomic(receipt_path, receipt)

    monkeypatch.chdir(assigned)
    good = worktree_receipt.check_receipt(receipt_path)
    monkeypatch.chdir(wrong)
    bad = worktree_receipt.check_receipt(receipt_path)

    assert good.ok
    assert not bad.ok
    assert bad.failure and "worktree mismatch" in bad.failure
    assert (wrong / "fixture.txt").read_text(encoding="utf-8") == "original\n"
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["task_id"] == "issue-8310"


def test_wrong_worktree_fails_before_fixture_mutation(tmp_path: Path) -> None:
    """The delegated entry-point guard rejects the wrong worktree before a write."""
    _repo, assigned, wrong = _fixture_repo(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    helper = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "worktree_receipt.py"
    worktree_receipt._write_atomic(
        receipt_path,
        worktree_receipt.create_receipt(assigned, task_id="task-8310", base_ref="HEAD"),
    )
    result = subprocess.run(
        [
            sys.executable,
            str(helper),
            "check",
            "--receipt",
            str(receipt_path),
            "--json",
        ],
        cwd=wrong,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["failure"].startswith("worktree mismatch")
    assert not (wrong / "worker-mutated.txt").exists()


def test_receipt_is_immutable_and_rejects_symlink(tmp_path: Path) -> None:
    """Receipt creation does not overwrite or follow a symlink target."""
    _repo, assigned, _wrong = _fixture_repo(tmp_path)
    receipt = worktree_receipt.create_receipt(assigned, task_id="task", base_ref="HEAD")
    existing = tmp_path / "existing.json"
    existing.write_text("keep\n", encoding="utf-8")
    try:
        worktree_receipt._write_atomic(existing, receipt)
    except ValueError as exc:
        assert "overwrite" in str(exc)
    else:
        raise AssertionError("existing receipt was overwritten")
    link = tmp_path / "link.json"
    link.symlink_to(existing)
    try:
        worktree_receipt._write_atomic(link, receipt)
    except ValueError as exc:
        assert "symlink" in str(exc)
    else:
        raise AssertionError("symlink receipt was followed")


def test_create_worktree_exec_checks_opt_in_receipt(tmp_path: Path) -> None:
    """The canonical creator emits and checks a receipt before ``--exec``."""
    repo, _assigned, _wrong = _fixture_repo(tmp_path)
    target = tmp_path / "created"
    receipt_path = repo / "created.receipt.json"
    branch = "worker/issue-8310"
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "create_worktree.sh"
    try:
        result = subprocess.run(
            [
                str(script),
                "--path",
                str(target),
                "--branch",
                branch,
                "--base",
                "HEAD",
                "--minimum-free-bytes",
                "0",
                "--receipt",
                receipt_path.name,
                "--task-id",
                "task-8310",
                "--exec",
                "python",
                "-c",
                "from pathlib import Path; Path('worker-started.txt').write_text('ok\\n')",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert receipt_path.is_file()
        assert (target / "worker-started.txt").read_text(encoding="utf-8") == "ok\n"
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(target)],
            cwd=repo,
            capture_output=True,
            check=False,
        )
        subprocess.run(["git", "branch", "-D", branch], cwd=repo, capture_output=True, check=False)


def test_receipt_scope_flags_cross_scope_commit_and_state(tmp_path: Path, monkeypatch) -> None:
    """A declared scope rejects cross-scope commits, staged files, and untracked files."""
    _repo, assigned, _wrong = _fixture_repo(tmp_path)
    receipt = worktree_receipt.create_receipt(
        assigned,
        task_id="issue-9115",
        base_ref="HEAD",
        allowed_paths=["scripts/dev/**"],
    )
    receipt_path = tmp_path / "receipt-scope.json"
    worktree_receipt._write_atomic(receipt_path, receipt)

    (assigned / "unrelated.txt").write_text("contamination\n", encoding="utf-8")
    _git(assigned, "add", "unrelated.txt")
    _git(assigned, "commit", "-m", "contamination")
    (assigned / "staged.py").write_text("staged\n", encoding="utf-8")
    _git(assigned, "add", "staged.py")
    (assigned / "untracked.md").write_text("untracked\n", encoding="utf-8")

    monkeypatch.chdir(assigned)
    result = worktree_receipt.check_receipt(receipt_path)

    assert not result.ok
    assert result.failure and "scope violations" in result.failure
    codes = {finding.split(":", 1)[0] for finding in result.scope_findings}
    assert codes == {"cross_scope_commit", "staged_out_of_scope", "untracked_out_of_scope"}
    assert receipt["allowed_paths"] == ["scripts/dev/**"]


def test_receipt_scope_allows_in_scope_commits_and_main_merges(tmp_path: Path, monkeypatch) -> None:
    """In-scope work and an intentional current-main merge stay valid."""
    repo, assigned, _wrong = _fixture_repo(tmp_path)
    receipt = worktree_receipt.create_receipt(
        assigned,
        task_id="issue-9115",
        base_ref="HEAD",
        allowed_paths=["scripts/dev/**", "tests/dev/**"],
    )
    receipt_path = tmp_path / "receipt-scope-ok.json"
    worktree_receipt._write_atomic(receipt_path, receipt)

    feature = assigned / "scripts" / "dev" / "feature.py"
    feature.parent.mkdir(parents=True, exist_ok=True)
    feature.write_text("feature\n", encoding="utf-8")
    _git(assigned, "add", "scripts/dev/feature.py")
    _git(assigned, "commit", "-m", "in-scope feature")

    incoming = repo / "robot_sf" / "incoming.py"
    incoming.parent.mkdir(parents=True, exist_ok=True)
    incoming.write_text("incoming\n", encoding="utf-8")
    _git(repo, "add", "robot_sf/incoming.py")
    _git(repo, "commit", "-m", "main advance")
    _git(assigned, "merge", "--no-ff", "main", "-m", "Merge main")

    monkeypatch.chdir(assigned)
    result = worktree_receipt.check_receipt(receipt_path)

    assert result.ok, result.failure
    assert result.scope_findings == ()
    assert result.allowed_paths == ("scripts/dev/**", "tests/dev/**")


def test_receipt_scope_absent_keeps_identity_only_behavior(tmp_path: Path, monkeypatch) -> None:
    """A receipt without scope does not inspect commits, staged, or untracked paths."""
    _repo, assigned, _wrong = _fixture_repo(tmp_path)
    receipt = worktree_receipt.create_receipt(assigned, task_id="issue-9115", base_ref="HEAD")
    receipt_path = tmp_path / "receipt-no-scope.json"
    worktree_receipt._write_atomic(receipt_path, receipt)

    (assigned / "unrelated.txt").write_text("allowed without scope\n", encoding="utf-8")
    _git(assigned, "add", "unrelated.txt")
    _git(assigned, "commit", "-m", "unrelated")

    monkeypatch.chdir(assigned)
    result = worktree_receipt.check_receipt(receipt_path)

    assert result.ok, result.failure
    assert result.scope_findings == ()
