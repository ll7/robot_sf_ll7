#!/usr/bin/env python3
"""Check that the installed fast-pysf runtime matches the threaded rollout API."""

from __future__ import annotations

import importlib
import sys

EXPECTED_SYMBOL = "social_force_gil_releasing_context"
REPAIR_COMMAND = "uv sync --all-extras --reinstall-package robot-sf"


def check_fast_pysf_runtime() -> str | None:
    """Return an actionable error when the required fast-pysf API is unavailable."""
    try:
        forces = importlib.import_module("pysocialforce.forces")
    except ImportError as exc:
        return f"could not import pysocialforce.forces ({exc})"

    if not callable(getattr(forces, EXPECTED_SYMBOL, None)):
        return f"pysocialforce.forces.{EXPECTED_SYMBOL} is missing"
    return None


def main() -> int:
    """Run the fast-pysf readiness check and print repair guidance on failure."""
    error = check_fast_pysf_runtime()
    if error is None:
        print("fast-pysf runtime preflight passed")
        return 0

    print("fast-pysf runtime preflight failed: " + error, file=sys.stderr)
    print(
        "The installed PySocialForce package is older than this checkout's threaded rollout API.",
        file=sys.stderr,
    )
    print(
        f"Refresh the environment with `{REPAIR_COMMAND}`, then rerun readiness.", file=sys.stderr
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
