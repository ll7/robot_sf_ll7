"""Smoke-test the OpenCode run_validation entrypoint and command map."""

from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path("scripts/validation/smoke_run_validation.js")


def test_run_validation_smoke_script_passes() -> None:
    """Verify the run_validation tool targets execute the expected mapped scripts.

    This matters because the OpenCode repo helper is an execution surface used by
    agent workflows, and a command-map drift here would silently break repo-local
    validation entrypoints.
    """

    result = subprocess.run(
        ["node", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
