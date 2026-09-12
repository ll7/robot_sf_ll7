"""Resolve Git-common-dir paths for agent-run artifacts.

In a linked worktree ``.git`` is a file, not a directory, so writing to a
literal ``.git/codex-agent-runs/...`` path fails.  Use
:func:`resolve_agent_artifact_dir` to obtain the correct absolute path via
``git rev-parse --git-common-dir``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def resolve_git_common_dir() -> Path | None:
    """Return the absolute path to the Git common directory, or ``None``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip())
    return None


def resolve_agent_artifact_dir(subdir: str, *, mkdir: bool = True) -> Path:
    """Return the absolute path to a codex-agent-runs artifact subdirectory.

    Parameters
    ----------
    subdir:
        Name of the subdirectory under
        ``<git-common-dir>/codex-agent-runs/active/``.
    mkdir:
        If ``True`` (the default), create the directory if it does not exist.

    Returns
    -------
    Path
        Absolute path to the resolved artifact directory.  Falls back to
        ``output/tmp/<subdir>`` when git is unavailable.
    """
    common_dir = resolve_git_common_dir()
    if common_dir is not None:
        artifact_dir = common_dir / "codex-agent-runs" / "active" / subdir
    else:
        artifact_dir = Path(__file__).resolve().parents[2] / "output" / "tmp" / subdir
    if mkdir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir
