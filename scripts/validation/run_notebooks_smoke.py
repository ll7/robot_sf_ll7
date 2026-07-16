"""Execute the CPU-only quickstart notebooks headless via nbconvert.

This is the CI smoke for ``notebooks/`` (Issue #5798). It executes every
``notebooks/*.ipynb`` top-to-bottom under a headless kernel
(``SDL_VIDEODRIVER=dummy``) and fails closed if any cell errors or nbconvert
exits non-zero.

The notebooks themselves are deterministic and write their small artifacts under
``output/notebooks/`` (git-ignored); this script only checks that they execute
cleanly.

Usage:
    uv run python scripts/validation/run_notebooks_smoke.py
    uv run python scripts/validation/run_notebooks_smoke.py --list
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
PER_NOTEBOOK_TIMEOUT = 600  # seconds; these are tiny CPU episodes


def discover_notebooks() -> list[Path]:
    """Return the sorted list of notebook paths under ``notebooks/``."""
    if not NOTEBOOKS_DIR.is_dir():
        return []
    return sorted(NOTEBOOKS_DIR.glob("*.ipynb"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Execute the CPU-only quickstart notebooks headless via nbconvert.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the notebooks that would be executed without running them.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=PER_NOTEBOOK_TIMEOUT,
        help=f"Per-notebook execution timeout in seconds (default: {PER_NOTEBOOK_TIMEOUT})",
    )
    return parser.parse_args(argv)


def _headless_env() -> dict[str, str]:
    """Return an environment forcing headless execution for the kernel."""
    env = dict(os.environ)
    # Headless SDL (no window) is required for any pygame/SDL code paths.
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    # Do NOT set MPLBACKEND=Agg: the notebooks re-arm the inline backend per
    # plotting cell; forcing Agg here would suppress inline figure output.
    return env


def run_notebook(notebook: Path, *, timeout: int) -> tuple[bool, str]:
    """Execute a single notebook in-place headless. Returns (ok, summary)."""
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--stdout",  # write executed notebook to stdout instead of mutating the file
        f"--ExecutePreprocessor.timeout={timeout}",
        str(notebook),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=_headless_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    summary = (
        f"exit={proc.returncode} "
        f"{proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ''}".strip()
    )
    return proc.returncode == 0, summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run the notebooks smoke. Returns a process exit code."""
    args = parse_args(argv)
    notebooks = discover_notebooks()
    if not notebooks:
        print(f"No notebooks found under {NOTEBOOKS_DIR.relative_to(REPO_ROOT)}/", file=sys.stderr)
        return 1

    if args.list:
        print(f"Notebooks ({len(notebooks)}):")
        for nb in notebooks:
            print(f"  {nb.relative_to(REPO_ROOT)}")
        return 0

    print(f"Executing {len(notebooks)} notebook(s) headless via nbconvert...")
    failures: list[str] = []
    for nb in notebooks:
        rel = nb.relative_to(REPO_ROOT)
        ok, summary = run_notebook(nb, timeout=args.timeout)
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {rel}  ({summary})")
        if not ok:
            failures.append(str(rel))

    if failures:
        print(f"\n{len(failures)} notebook(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\nAll {len(notebooks)} notebook(s) executed cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
