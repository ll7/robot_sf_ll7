"""Tests for safe generated-output cleanup."""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.dev.clean_generated_output import clean_paths


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in the temporary repository."""
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def test_clean_paths_preserves_tracked_files_and_removes_generated(tmp_path: Path) -> None:
    """Cleanup should delete generated files without dirtying tracked output files."""
    repo = tmp_path / "repo"
    tracked_dir = repo / "output" / "repos"
    tracked_dir.mkdir(parents=True)
    (tracked_dir / "README.md").write_text("tracked\n", encoding="utf-8")
    (tracked_dir / "generated.json").write_text("{}\n", encoding="utf-8")
    (repo / ".gitignore").write_text("output/repos/generated.json\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _git(repo, "add", ".gitignore", "output/repos/README.md")
    _git(repo, "-c", "user.name=CI", "-c", "user.email=ci@example.com", "commit", "-m", "seed")

    assert clean_paths([Path("output/repos")], cwd=repo) == 0

    assert (tracked_dir / "README.md").exists()
    assert not (tracked_dir / "generated.json").exists()
    assert _git(repo, "status", "--short").stdout.strip() == ""


def test_clean_paths_removes_untracked_output_tree(tmp_path: Path) -> None:
    """Fully untracked generated trees can be removed wholesale."""
    repo = tmp_path / "repo"
    generated_dir = repo / "output" / "coverage"
    generated_dir.mkdir(parents=True)
    (generated_dir / "coverage.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _git(
        repo,
        "-c",
        "user.name=CI",
        "-c",
        "user.email=ci@example.com",
        "commit",
        "--allow-empty",
        "-m",
        "seed",
    )

    assert clean_paths([Path("output/coverage")], cwd=repo) == 0

    assert not generated_dir.exists()
