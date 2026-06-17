"""Tests for safe generated-output cleanup."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.dev.clean_generated_output import clean_paths

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "clean_generated_output.py"


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


def test_clean_paths_preserves_force_tracked_file_in_ignored_directory(tmp_path: Path) -> None:
    """Ignored directories may contain force-tracked files that cleanup must preserve."""
    repo = tmp_path / "repo"
    ignored_dir = repo / "output" / "reports"
    ignored_dir.mkdir(parents=True)
    (ignored_dir / "keep.md").write_text("tracked\n", encoding="utf-8")
    (ignored_dir / "generated.json").write_text("{}\n", encoding="utf-8")
    (repo / ".gitignore").write_text("output/reports/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _git(repo, "add", ".gitignore")
    _git(repo, "add", "-f", "output/reports/keep.md")
    _git(repo, "-c", "user.name=CI", "-c", "user.email=ci@example.com", "commit", "-m", "seed")

    assert clean_paths([Path("output/reports")], cwd=repo) == 0

    assert (ignored_dir / "keep.md").exists()
    assert not (ignored_dir / "generated.json").exists()
    assert _git(repo, "status", "--short").stdout.strip() == ""


def test_clean_paths_removes_nested_empty_generated_directories(tmp_path: Path) -> None:
    """Cleanup should remove parent directories that become empty during traversal."""
    repo = tmp_path / "repo"
    tracked_dir = repo / "output" / "repos"
    nested_dir = tracked_dir / "tmp" / "nested"
    nested_dir.mkdir(parents=True)
    (tracked_dir / "README.md").write_text("tracked\n", encoding="utf-8")
    (nested_dir / "generated.json").write_text("{}\n", encoding="utf-8")
    (repo / ".gitignore").write_text("output/repos/tmp/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _git(repo, "add", ".gitignore", "output/repos/README.md")
    _git(repo, "-c", "user.name=CI", "-c", "user.email=ci@example.com", "commit", "-m", "seed")

    assert clean_paths([Path("output/repos")], cwd=repo) == 0

    assert (tracked_dir / "README.md").exists()
    assert not (tracked_dir / "tmp").exists()
    assert _git(repo, "status", "--short").stdout.strip() == ""


def test_cli_preserves_tracked_output_evidence_and_cleans_generated_files(
    tmp_path: Path,
) -> None:
    """The real cleanup CLI should not delete tracked evidence under output/."""
    repo = tmp_path / "repo"
    evidence_dir = repo / "output" / "evidence"
    scratch_dir = repo / "output" / "scratch"
    evidence_dir.mkdir(parents=True)
    scratch_dir.mkdir(parents=True)
    tracked_evidence = evidence_dir / "summary.json"
    ignored_generated = evidence_dir / "run.log"
    untracked_generated = scratch_dir / "scratch.txt"
    tracked_evidence.write_text('{"claim": "preserved"}\n', encoding="utf-8")
    ignored_generated.write_text("local log\n", encoding="utf-8")
    untracked_generated.write_text("temporary\n", encoding="utf-8")
    (repo / ".gitignore").write_text("output/evidence/*.log\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _git(repo, "add", ".gitignore", "output/evidence/summary.json")
    _git(repo, "-c", "user.name=CI", "-c", "user.email=ci@example.com", "commit", "-m", "seed")

    # Run the trusted local CLI through the active test interpreter in a temp repo.
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "output"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.startswith("cleaned_generated_paths=")
    assert tracked_evidence.read_text(encoding="utf-8") == '{"claim": "preserved"}\n'
    assert not ignored_generated.exists()
    assert not untracked_generated.exists()
    assert not scratch_dir.exists()
    assert _git(repo, "status", "--short").stdout.strip() == ""
