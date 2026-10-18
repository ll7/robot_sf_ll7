"""Admit browser harnesses only on the declared Node runtime."""

import os
import re
import subprocess

import pytest

MINIMUM_NODE_VERSION = (22, 7)


def require_node_runtime() -> str:
    """Require ESM syntax detection; only an absent local Node may skip."""
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
            env={"PATH": os.environ.get("PATH", "")},
        )
    except FileNotFoundError:
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail("Browser runtime requires Node >= 22.7; Node is missing in CI")
        pytest.skip("Browser runtime requires Node >= 22.7; Node is absent locally")
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.fail(f"Browser runtime requires Node >= 22.7; Node version probe failed: {exc}")
    version = result.stdout.strip()
    match = re.fullmatch(r"v(\d+)\.(\d+)\.\d+", version)
    if match is None or (int(match[1]), int(match[2])) < MINIMUM_NODE_VERSION:
        pytest.fail(
            f"Browser runtime requires Node >= 22.7; installed runtime is unsupported: {version}"
        )
    return version
