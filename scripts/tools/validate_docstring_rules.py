"""Guard script to enforce docstring-specific Ruff rules."""

from __future__ import annotations

import argparse
import subprocess

DOCSTRING_RULES = [
    "D100",
    "D101",
    "D102",
    "D103",
    "D104",
    "D105",
    "D106",
    "D107",
    "D201",
    "D417",
    "D419",
]


def main() -> int:
    """Run Ruff with the docstring rule set and return its exit status."""
    parser = argparse.ArgumentParser(description="Validate docstring rules with Ruff.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Paths to lint (defaults to repository root).",
    )
    args = parser.parse_args()
    cmd = [
        "ruff",
        "check",
        f"--select={','.join(DOCSTRING_RULES)}",
        *args.paths,
    ]
    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
