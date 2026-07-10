"""Regression test: the repo `.gitignore` must ignore a `.venv` symlink.

Fresh linked worktrees may point `.venv` at the main checkout's virtualenv via a
symlink (a shared-venv runner setup). A directory-only ignore rule (`.venv/`,
with a trailing slash) matches directories but *not* symlinks, so the symlink
showed up as untracked (`?? .venv`). See issue #5027.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_repo_gitignore_venv_patterns() -> list[str]:
    """Return the virtual-environment ignore patterns from the repo `.gitignore`."""
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() in {".venv", "venv"}]


def test_repo_gitignore_uses_slashless_venv_patterns() -> None:
    """The repo ships `.venv`/`venv` without a trailing slash so symlinks match."""
    patterns = _read_repo_gitignore_venv_patterns()
    assert ".venv" in patterns, "`.gitignore` must contain slashless `.venv` (see #5027)"
    assert "venv" in patterns, "`.gitignore` must contain slashless `venv` (see #5027)"


def test_venv_symlink_is_ignored_by_repo_patterns(tmp_path: Path) -> None:
    """A `.venv` symlink is ignored under the repo's virtual-env patterns.

    Behavioral check independent of the repo working tree: build a scratch git
    repo whose `.gitignore` carries only the repo's venv patterns, create a
    `.venv` symlink, and assert `git check-ignore` matches it. Under the old
    directory-only rule (`.venv/`) this returns non-zero (not ignored).
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)

    patterns = _read_repo_gitignore_venv_patterns()
    (repo / ".gitignore").write_text("\n".join(patterns) + "\n", encoding="utf-8")

    real_venv = repo / "shared-venv"
    real_venv.mkdir()
    (repo / ".venv").symlink_to(real_venv)

    result = subprocess.run(
        ["git", "check-ignore", ".venv"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "`.venv` symlink should be ignored but git check-ignore reported it as "
        f"tracked (returncode={result.returncode}, stdout={result.stdout!r})"
    )
