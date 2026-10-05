"""Admit browser harnesses only on the declared Node runtime."""

import os
import re
import subprocess

import pytest

MINIMUM_NODE_MAJOR = 22


def require_node_runtime() -> str:
    """Skip with a precise reason when Node cannot run the ESM browser assets."""
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
            env={"PATH": os.environ.get("PATH", "")},
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("Browser runtime requires Node >= 22; Node version is unavailable")
    version = result.stdout.strip()
    match = re.fullmatch(r"v(\d+)\.\d+\.\d+", version)
    if match is None or int(match[1]) < MINIMUM_NODE_MAJOR:
        pytest.skip("Browser runtime requires Node >= 22; installed runtime is unsupported")
    return version
