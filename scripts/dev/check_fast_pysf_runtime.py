#!/usr/bin/env python3
"""Check that the installed PySocialForce API matches the codebase contract.

This avoids collection failures in environments with stale installed packages.
"""

from __future__ import annotations

import importlib
import sys
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FAST_PYSF_SOURCE_PACKAGE = REPO_ROOT / "fast-pysf" / "pysocialforce"
EXPECTED_SYMBOL = "social_force_gil_releasing_context"
REPAIR_COMMAND = "uv sync --all-extras --reinstall-package robot-sf"
WORKTREE_RECOVERY_COMMAND = (
    "scripts/dev/run_worktree_shared_venv.sh --recover-stale-fast-pysf -- <command>"
)


def _package_files(package_path: Path) -> tuple[Path, ...]:
    """Return cache-free relative files that make up one Python package."""
    files = []
    for path in package_path.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(package_path)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(relative)
    return tuple(sorted(files))


def _package_digest(package_path: Path) -> str:
    """Hash package paths and contents while ignoring interpreter caches."""
    digest = sha256()
    for relative in _package_files(package_path):
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((package_path / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _same_path(left: Path, right: Path) -> bool:
    """Compare paths after resolving worktree and editable-install links."""
    return left.resolve() == right.resolve()


def check_fast_pysf_package_coherence(
    source_package: Path,
    installed_package: Path,
) -> str | None:
    """Return an actionable error when installed and checkout packages differ."""
    if not source_package.is_dir():
        return f"checkout fast-pysf package is missing: {source_package}"
    if not installed_package.is_dir():
        return f"installed pysocialforce package is missing: {installed_package}"
    if _same_path(source_package, installed_package):
        return None

    try:
        source_digest = _package_digest(source_package)
        installed_digest = _package_digest(installed_package)
    except OSError as exc:
        return f"could not compare checkout and installed pysocialforce packages ({exc})"

    if source_digest == installed_digest:
        return None
    return (
        "installed pysocialforce package is stale relative to this checkout "
        f"(checkout sha256={source_digest[:12]}, installed sha256={installed_digest[:12]})"
    )


def _imported_package_path() -> Path | None:
    """Return the package directory selected by the active Python import path."""
    package = sys.modules.get("pysocialforce")
    package_paths = getattr(package, "__path__", ())
    for package_path in package_paths:
        return Path(package_path)
    return None


def check_fast_pysf_runtime() -> str | None:
    """Return an actionable error when the required fast-pysf API is unavailable."""
    try:
        forces = importlib.import_module("pysocialforce.forces")
    except ImportError as exc:
        return f"could not import pysocialforce.forces ({exc})"

    if not callable(getattr(forces, EXPECTED_SYMBOL, None)):
        return f"pysocialforce.forces.{EXPECTED_SYMBOL} is missing or not callable"

    installed_package = _imported_package_path()
    if installed_package is None:
        return "could not locate the active pysocialforce package directory"
    return check_fast_pysf_package_coherence(FAST_PYSF_SOURCE_PACKAGE, installed_package)


def main() -> int:
    """Run the fast-pysf readiness check and print repair guidance on failure."""
    error = check_fast_pysf_runtime()
    if error is None:
        print("fast-pysf runtime preflight passed")
        return 0

    print("fast-pysf runtime preflight failed: " + error, file=sys.stderr)
    print(
        "The active PySocialForce environment does not satisfy this checkout's fast-pysf API.",
        file=sys.stderr,
    )
    print(
        "From a linked worktree, rerun the command with "
        f"`{WORKTREE_RECOVERY_COMMAND}`; it refreshes only a worktree-local environment "
        "under the capacity and recovery-lock gates.",
        file=sys.stderr,
    )
    print(
        f"To repair an explicitly owned environment, run `{REPAIR_COMMAND}` in that checkout, "
        "then rerun readiness.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
