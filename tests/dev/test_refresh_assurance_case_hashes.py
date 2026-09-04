"""Contract tests for the working-tree release-assurance hash helper."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/dev/refresh_assurance_case_hashes.py"
CASE = Path("docs/context/evidence/issue_4683_release_assurance_case_example.json")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a Git command in a temporary repository."""
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _make_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal tracked release-assurance example."""
    repo = tmp_path / "repo"
    source = repo / "docs" / "RELEASE.md"
    case = repo / CASE
    source.parent.mkdir(parents=True)
    case.parent.mkdir(parents=True)
    source.write_text("release v1\n", encoding="utf-8")
    case.write_text(
        json.dumps(
            {
                "evidence": [
                    {
                        "path": "docs/RELEASE.md",
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Assurance hash test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "initial")
    return repo, source


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the helper CLI in a temporary repository."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def test_check_write_cycle_updates_only_stale_hash(tmp_path: Path) -> None:
    """Check fails closed, write repairs the hash, and the diff stays one-line."""
    repo, source = _make_repo(tmp_path)
    source.write_text("release v2\n", encoding="utf-8")

    check = _run(repo, "--check")
    assert check.returncode == 1
    assert json.loads(check.stdout) == {
        "case": CASE.as_posix(),
        "mismatched_paths": ["docs/RELEASE.md"],
        "status": "mismatch",
    }

    write = _run(repo, "--write")
    assert write.returncode == 0
    assert json.loads(write.stdout) == {
        "case": CASE.as_posix(),
        "status": "updated",
        "updated_paths": ["docs/RELEASE.md"],
    }
    diff = _git(repo, "diff", "--", CASE.as_posix()).stdout
    changed_lines = [
        line
        for line in diff.splitlines()
        if (line.startswith("+") or line.startswith("-")) and not line.startswith(("+++", "---"))
    ]
    assert len(changed_lines) == 2
    assert _run(repo, "--check").returncode == 0


def test_check_rejects_untracked_source_path(tmp_path: Path) -> None:
    """The helper must not hash an untracked path into release evidence."""
    repo, _source = _make_repo(tmp_path)
    case = repo / CASE
    payload = json.loads(case.read_text(encoding="utf-8"))
    payload["evidence"][0]["path"] = "docs/UNTRACKED.md"
    (repo / "docs/UNTRACKED.md").write_text("not tracked\n", encoding="utf-8")
    case.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = _run(repo, "--check")

    assert result.returncode == 2
    assert "Evidence path is not tracked" in result.stderr
