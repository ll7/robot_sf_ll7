"""Tests for the base-drift check (scripts/dev/check_base_drift.py, issue #5782)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "dev" / "check_base_drift.py"


def _git(repo: Path, *args: str) -> str:
    """Run a git command in *repo* and return stdout."""
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_in(repo: Path, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run the base-drift helper inside *repo* and return the completed process.

    Always requests machine-readable JSON output so tests can assert on the
    structured result regardless of the human-readable status message path.
    """
    cmd = [sys.executable, str(SCRIPT), "--json", *args]
    return subprocess.run(
        cmd,
        cwd=str(repo),
        check=False,
        capture_output=True,
        text=True,
        input=stdin,
    )


def _init_repo(repo: Path) -> None:
    """Create an empty committed git repo in *repo*."""
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "init",
        ],
        cwd=repo,
        check=True,
    )


def _build_drift_repo(tmp_path: Path, *, main_touches_pr_file: bool) -> tuple[Path, str]:
    """Build a repo with a PR branch and an advanced origin/main.

    Returns (repo, validated_base_sha). The PR branch sits on top of
    ``validated_base_sha``; ``origin/main`` is advanced one commit past it.
    When *main_touches_pr_file* is True the advanced main commit edits the PR's
    own changed file (shared.py); otherwise it edits an unrelated file so drift
    does not intersect the PR's changed-file set.
    """
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "shared.py").write_text("print('base')\n", encoding="utf-8")
    (repo / "unrelated.py").write_text("print('base')\n", encoding="utf-8")
    _init_repo(repo)
    (repo / "shared.py").write_text("print('base commit')\n", encoding="utf-8")
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"], cwd=repo, check=True
    )
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base"],
        cwd=repo,
        check=True,
    )
    base_sha = _git(repo, "rev-parse", "HEAD")

    # PR branch commit touches shared.py.
    (repo / "shared.py").write_text("print('pr change')\n", encoding="utf-8")
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"], cwd=repo, check=True
    )
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "pr"],
        cwd=repo,
        check=True,
    )
    pr_commit = _git(repo, "rev-parse", "HEAD")

    # Advance origin/main one commit past the base.
    subprocess.run(["git", "branch", "main", base_sha], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "-q", "main"], cwd=repo, check=True)
    if main_touches_pr_file:
        (repo / "shared.py").write_text("print('unrelated main change')\n", encoding="utf-8")
    else:
        (repo / "unrelated.py").write_text("print('unrelated main change')\n", encoding="utf-8")
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"], cwd=repo, check=True
    )
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "main advance"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=repo, check=True)

    # Return HEAD to the PR commit.
    subprocess.run(["git", "checkout", "-q", pr_commit], cwd=repo, check=True)
    return repo, base_sha


def test_current_base_reports_no_drift(tmp_path: Path) -> None:
    """When validated SHA matches the current base, status is current (exit 0)."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    _init_repo(repo)
    base_sha = _git(repo, "rev-parse", "HEAD")
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", base_sha], cwd=repo, check=True
    )
    result = _run_in(repo, "--base-ref", "origin/main", "--validated-base-sha", base_sha)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "current"


def test_unrelated_drift_recommends_reuse(tmp_path: Path) -> None:
    """Drift that avoids PR-changed files yields reuse_recommended (exit 0)."""
    repo, base_sha = _build_drift_repo(tmp_path, main_touches_pr_file=False)
    result = _run_in(
        repo,
        "--base-ref",
        "origin/main",
        "--validated-base-sha",
        base_sha,
        "--changed-files",
        "/dev/stdin",
        stdin="shared.py\n",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "reuse_recommended"
    assert payload["affected_file_count"] == 0
    assert "reuse" in payload["message"].lower()


def test_drift_touching_pr_files_fails_closed(tmp_path: Path) -> None:
    """Drift intersecting PR-changed files fails closed (exit 1) and names files."""
    repo, base_sha = _build_drift_repo(tmp_path, main_touches_pr_file=True)
    result = _run_in(
        repo,
        "--base-ref",
        "origin/main",
        "--validated-base-sha",
        base_sha,
        "--changed-files",
        "/dev/stdin",
        stdin="shared.py\n",
    )
    assert result.returncode == 1, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "revalidate_required"
    assert payload["affected_file_count"] >= 1
    assert "shared.py" in payload["affected_files"]


def test_unresolved_base_ref_is_indeterminate(tmp_path: Path) -> None:
    """A base ref that does not resolve locally yields indeterminate (exit 2)."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    result = _run_in(repo, "--base-ref", "does/not/exist", "--validated-base-sha", "abc123")
    assert result.returncode == 2, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "indeterminate"


def test_identical_sha_is_current_without_drift_compute(tmp_path: Path) -> None:
    """When validated SHA equals current base, status is current (exit 0)."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    base_sha = _git(repo, "rev-parse", "HEAD")
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", base_sha], cwd=repo, check=True
    )
    result = _run_in(repo, "--base-ref", "origin/main", "--validated-base-sha", base_sha)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "current"
