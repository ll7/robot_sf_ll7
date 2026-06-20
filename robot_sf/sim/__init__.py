"""Simulation integration layer for Robot SF.

This package provides glue code that connects the high‑level gymnasium
environments to the optimized SocialForce pedestrian physics provided by the
`fast-pysf` git subtree. Import-time guard logic below surfaces a clear and
actionable error message when the subtree has not been initialized.

The goal is to fail fast with guidance rather than raising obscure import
errors deeper in the call stack.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _has_installed_pysocialforce() -> bool:
    """Return whether an installed ``pysocialforce`` package is importable."""

    return importlib.util.find_spec("pysocialforce") is not None


def _assert_fast_pysf_initialized(repo_root: Path | None = None) -> None:
    """Raise a RuntimeError if the fast-pysf submodule looks uninitialized.

    Heuristic: repository root contains a `fast-pysf` directory with expected
    Python package `pysocialforce`. Installed wheels do not include the source
    checkout's `fast-pysf` directory, so they may satisfy this contract through
    the declared package dependency instead.
    """

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    submodule_dir = repo_root / "fast-pysf"
    expected_pkg = submodule_dir / "pysocialforce"
    is_source_checkout = (repo_root / "pyproject.toml").exists()

    if submodule_dir.exists() and expected_pkg.exists():
        return
    if not is_source_checkout and _has_installed_pysocialforce():
        return

    raise RuntimeError(
        "fast-pysf submodule not initialized or incomplete.\n"
        "Expected directory: 'fast-pysf/pysocialforce'.\n"
        "Fix source checkouts by running: \n"
        "  git submodule update --init --recursive\n\n"
        "Installed wheels should include a usable 'pysocialforce' package. "
        "If this was intentional (e.g., lightweight docs build), avoid importing "
        "'robot_sf.sim' modules.",
    )


# Execute guard on import so downstream modules can assume availability.
_assert_fast_pysf_initialized()

__all__ = ["_assert_fast_pysf_initialized"]
