#!/usr/bin/env python3
"""Check that the installed PySocialForce API matches the codebase contract.

This avoids collection failures in environments with stale installed packages.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Check that the social_force_gil_releasing_context API is importable."""
    try:
        from pysocialforce.forces import social_force_gil_releasing_context  # noqa: F401
    except (ImportError, AttributeError) as exc:
        print(f"ImportError: {exc}", file=sys.stderr)
        print(
            "Stale PySocialForce installation detected in virtual environment.\n"
            "The installed package lacks the 'social_force_gil_releasing_context' API.\n"
            "To repair:\n"
            "  uv sync --all-extras --reinstall-package robot-sf",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
