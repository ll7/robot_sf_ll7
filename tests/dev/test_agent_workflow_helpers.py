"""Tests for compact delegated-agent self-review helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_SCRIPT = REPO_ROOT / "scripts" / "review-agent-run.sh"
SUMMARY_SCRIPT = REPO_ROOT / "scripts" / "summarize-agent-runs.py"


def git(*args: str, cwd: Path) -> None:
    """Run one Git setup command for an isolated temporary repository."""
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def linked_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return a main checkout, linked checkout, and common run root."""
    repo = tmp_path / "repo"
    git("init", "--initial-branch=main", str(repo), cwd=tmp_path)
    git("config", "user.name", "test", cwd=repo)
    git("config", "user.email", "test@example.invalid", cwd=repo)
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    git("add", "README.md", cwd=repo)
    git("commit", "-m", "fixture", cwd=repo)
    linked = tmp_path / "linked"
    git("worktree", "add", "--detach", str(linked), "HEAD", cwd=repo)
    common_git_dir = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=linked,
            text=True,
        ).strip()
    )
    run_root = common_git_dir / "codex-agent-runs"
    run_root.mkdir()
    return repo, linked, run_root


def write_complete_run(run_root: Path, name: str = "run-001") -> Path:
    """Write the minimum complete compact worker bundle."""
    run_dir = run_root / name
    run_dir.mkdir()
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "provider": "test-provider",
                "model": "test-model",
                "task_class": "read_only_review",
                "worker_status": 0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "status.txt").write_text("validated\n", encoding="utf-8")
    (run_dir / "diffstat.txt").write_text("1 file changed\n", encoding="utf-8")
    (run_dir / "changed_files.txt").write_text("scripts/example.py\n", encoding="utf-8")
    return run_dir


def test_review_agent_run_reads_complete_bundle_from_linked_worktree(
    linked_repo: tuple[Path, Path, Path],
) -> None:
    """The helper resolves artifacts through the common Git directory."""
    _, linked, run_root = linked_repo
    run_dir = write_complete_run(run_root)
    result = subprocess.run(
        ["bash", str(REVIEW_SCRIPT), "--run-dir", str(run_dir)],
        cwd=linked,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "result=reviewed" in result.stdout
    notes = list((run_root / "notes" / "inbox").glob("*.md"))
    assert len(notes) == 1
    assert "test-provider" in notes[0].read_text(encoding="utf-8")


def test_review_agent_run_latest_selects_complete_bundle(
    linked_repo: tuple[Path, Path, Path],
) -> None:
    """--latest works from a linked worktree without a worktree-local run root."""
    _, linked, run_root = linked_repo
    write_complete_run(run_root, "20260812T010000Z-complete")
    result = subprocess.run(
        ["bash", str(REVIEW_SCRIPT), "--latest"],
        cwd=linked,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "20260812T010000Z-complete" in result.stdout


def test_review_agent_run_reports_missing_compact_artifacts(
    linked_repo: tuple[Path, Path, Path],
) -> None:
    """Missing artifacts produce an explicit incomplete result and private note."""
    _, linked, run_root = linked_repo
    run_dir = run_root / "incomplete"
    run_dir.mkdir()
    (run_dir / "result.json").write_text("{}\n", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(REVIEW_SCRIPT), "--run-dir", str(run_dir)],
        cwd=linked,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "incomplete-artifact" in result.stdout
    note = next((run_root / "notes" / "inbox").glob("*.md"))
    assert "missing_required_artifacts" in note.read_text(encoding="utf-8")


def test_summarize_agent_runs_notes_only_uses_common_git_dir(
    linked_repo: tuple[Path, Path, Path],
) -> None:
    """--notes-only summarizes inbox notes without reading worker logs."""
    _, linked, run_root = linked_repo
    note_root = run_root / "notes" / "inbox"
    note_root.mkdir(parents=True)
    (note_root / "lesson.md").write_text(
        "---\nobservation_class: tooling\nconfidence: high\n---\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(SUMMARY_SCRIPT), "--notes-only"],
        cwd=linked,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Workflow notes: 1" in result.stdout
    assert "observation_class=tooling" in result.stdout
    assert "Delegated runs:" not in result.stdout
