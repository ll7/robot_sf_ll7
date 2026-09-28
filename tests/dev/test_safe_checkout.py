"""Contract tests for the safe-checkout dirty-tree guard (issue #9445)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "scripts" / "dev" / "safe_checkout.sh"


def _run(*argv: str | Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(HELPER), *[str(a) for a in argv]],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _git(*argv: str, cwd: Path) -> None:
    subprocess.run(["git", *argv], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def dirty_repo(tmp_path: Path) -> Path:
    """Return a temp git repo with one committed file carrying an uncommitted edit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-b", "main", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    (repo / "tracked.md").write_text("committed\n", encoding="utf-8")
    _git("add", "tracked.md", cwd=repo)
    _git("commit", "-m", "base", cwd=repo)
    (repo / "tracked.md").write_text("uncommitted edit\n", encoding="utf-8")
    return repo


def test_clean_tree_proceeds(dirty_repo: Path) -> None:
    """A clean guard scope lets the checkout through untouched."""
    _git("checkout", "--", "tracked.md", cwd=dirty_repo)
    proc = _run("--", "tracked.md", cwd=dirty_repo)

    assert proc.returncode == 0
    assert (dirty_repo / "tracked.md").read_text(encoding="utf-8") == "committed\n"


def test_dirty_tree_is_refused_without_backup(dirty_repo: Path) -> None:
    """Dirty tracked files block the restore and stay untouched."""
    proc = _run("--", "tracked.md", cwd=dirty_repo)

    assert proc.returncode == 2
    assert "tracked.md" in proc.stderr
    assert (dirty_repo / "tracked.md").read_text(encoding="utf-8") == "uncommitted edit\n"


def test_backup_snapshots_before_restore(dirty_repo: Path, tmp_path: Path) -> None:
    """--backup writes an applicable patch, then the restore proceeds."""
    backup = tmp_path / "work.patch"
    proc = _run("--backup", backup, "--", "tracked.md", cwd=dirty_repo)

    assert proc.returncode == 0
    assert backup.stat().st_size > 0
    _git("apply", str(backup), cwd=dirty_repo)  # replays the discarded edit
    assert (dirty_repo / "tracked.md").read_text(encoding="utf-8") == "uncommitted edit\n"


def test_path_scope_ignores_outside_dirt(dirty_repo: Path) -> None:
    """Dirt outside an explicit --path scope does not block the restore."""
    (dirty_repo / "other.md").write_text("new\n", encoding="utf-8")
    _git("add", "other.md", cwd=dirty_repo)
    _git("commit", "-m", "other", cwd=dirty_repo)
    (dirty_repo / "other.md").write_text("dirty\n", encoding="utf-8")
    _git("checkout", "--", "tracked.md", cwd=dirty_repo)

    proc = _run("--path", "tracked.md", "--", "tracked.md", cwd=dirty_repo)

    assert proc.returncode == 0
    assert (dirty_repo / "other.md").read_text(encoding="utf-8") == "dirty\n"


def test_missing_checkout_args_is_usage_error(tmp_path: Path) -> None:
    """No checkout args after -- is a usage error before any git read."""
    proc = _run("--", cwd=tmp_path)

    assert proc.returncode == 2


def test_outside_git_repo_fails_closed(tmp_path: Path) -> None:
    """Running outside a repository fails closed without touching anything."""
    proc = _run("--", "tracked.md", cwd=tmp_path)

    assert proc.returncode == 2
    assert "not inside a git repository" in proc.stderr
