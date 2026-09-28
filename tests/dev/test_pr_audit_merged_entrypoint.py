"""Regression tests for the pr_audit_merged direct-execution entrypoint (issue #9337)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _bare_python() -> str:
    """Return an interpreter without the project venv on its path.

    The project venv exposes an editable top-level ``scripts`` package, which
    masks the missing repo-root bootstrap this test guards. The base
    interpreter plus ``-S`` reproduces a bare worktree invocation.
    """
    return getattr(sys, "_base_executable", None) or sys.executable


def _run(*argv: str) -> subprocess.CompletedProcess[str]:
    """Run an audit entrypoint form without ambient path help."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.pop("UV_PYTHON", None)
    return subprocess.run(
        [_bare_python(), "-S", *argv],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


def test_direct_path_help_exits_zero() -> None:
    """Direct path execution must resolve the scripts package (issue #9337)."""
    completed = _run("scripts/dev/pr_audit_merged.py", "--help")
    assert completed.returncode == 0, completed.stderr
    assert "post-merge audit" in completed.stdout.lower()


def test_module_help_still_exits_zero() -> None:
    """Module invocation must keep working after the bootstrap change."""
    completed = _run("-m", "scripts.dev.pr_audit_merged", "--help")
    assert completed.returncode == 0, completed.stderr
    assert "post-merge audit" in completed.stdout.lower()
