"""Simulation integration layer for Robot SF.

This package provides glue code that connects the high‑level gymnasium
environments to the optimized SocialForce pedestrian physics provided by the
`fast-pysf` git submodule. Import-time guard logic below surfaces a clear and
actionable error message when the submodule has not been initialized (common
after a fresh clone without `git submodule update --init --recursive`).

The goal is to fail fast with guidance rather than raising obscure import
errors deeper in the call stack.
"""

from __future__ import annotations

from pathlib import Path


def _assert_fast_pysf_initialized() -> None:
    """Raise a RuntimeError if the fast-pysf submodule looks uninitialized.

    Heuristic: repository root contains a `fast-pysf` directory with expected
    Python package `pysocialforce`. If either directory or an expected file is
    missing we assume the user forgot to initialize submodules.
    """

    repo_root = Path(__file__).resolve().parents[2]
    submodule_dir = repo_root / "fast-pysf"
    expected_pkg = submodule_dir / "pysocialforce"

    if not submodule_dir.exists() or not expected_pkg.exists():
        raise RuntimeError(
            "fast-pysf submodule not initialized or incomplete.\n"
            "Expected directory: 'fast-pysf/pysocialforce'.\n"
            "Fix by running: \n"
            "  git submodule update --init --recursive\n\n"
            "If this was intentional (e.g., lightweight docs build), avoid importing 'robot_sf.sim' modules."
        )


# Execute guard on import so downstream modules can assume availability.
_assert_fast_pysf_initialized()

__all__ = ["_assert_fast_pysf_initialized"]
