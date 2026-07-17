"""Regression tests for the fast-pysf readiness preflight."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "check_fast_pysf_runtime.py"


def _run_with_fake_package(tmp_path: Path, forces_source: str) -> subprocess.CompletedProcess[str]:
    package = tmp_path / "pysocialforce"
    package.mkdir()
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    (package / "forces.py").write_text(forces_source, encoding="utf-8")

    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_missing_gil_context_reports_environment_repair(tmp_path: Path) -> None:
    """A stale PySocialForce install fails with the targeted repair command."""
    result = _run_with_fake_package(tmp_path, "def social_force():\n    return None\n")

    assert result.returncode == 1
    assert "social_force_gil_releasing_context is missing or not callable" in result.stderr
    assert "uv sync --all-extras --reinstall-package robot-sf" in result.stderr


def test_non_callable_gil_context_fails_closed(tmp_path: Path) -> None:
    """A present but non-callable API symbol must not satisfy the runtime contract."""
    result = _run_with_fake_package(tmp_path, "social_force_gil_releasing_context = None\n")

    assert result.returncode == 1
    assert "is missing or not callable" in result.stderr


def test_import_error_reports_environment_repair(tmp_path: Path) -> None:
    """An import failure reports the deterministic environment repair command."""
    result = _run_with_fake_package(tmp_path, "raise ImportError('dependency missing')\n")

    assert result.returncode == 1
    assert "could not import pysocialforce.forces" in result.stderr
    assert "uv sync --all-extras --reinstall-package robot-sf" in result.stderr


def test_current_fast_pysf_runtime_passes() -> None:
    """The repository-supported environment exposes the threaded rollout API."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "fast-pysf runtime preflight passed" in result.stdout
