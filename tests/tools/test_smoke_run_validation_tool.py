"""Smoke-test the OpenCode run_validation entrypoint and command map."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path("scripts/validation/smoke_run_validation.js")


def _node_supports_es_modules() -> bool:
    """Return whether the available Node runtime can run the ESM smoke script."""
    node = shutil.which("node")
    if node is None:
        return False
    result = subprocess.run(
        [node, "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    version = result.stdout.strip().lstrip("v")
    major = int(version.split(".", 1)[0]) if version.split(".", 1)[0].isdigit() else 0
    return major >= 18


def test_run_validation_smoke_script_passes() -> None:
    """Verify the run_validation tool targets execute the expected mapped scripts.

    This matters because the OpenCode repo helper is an execution surface used by
    agent workflows, and a command-map drift here would silently break repo-local
    validation entrypoints.
    """
    if not _node_supports_es_modules():
        pytest.skip("Node >= 18 is required for the ESM run_validation smoke script.")

    result = subprocess.run(
        ["node", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr or result.stdout
