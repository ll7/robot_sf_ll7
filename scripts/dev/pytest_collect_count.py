#!/usr/bin/env python3
"""Report the exact collected-test count for validation metadata (issue #9268).

PR validation sections must cite the actual collected count at the exact head,
not a remembered number (the #9267 follow-up: body said 19, the module
collected 18). Run this helper and paste its output verbatim::

    uv run python scripts/dev/pytest_collect_count.py tests/benchmark/test_step_trace_reset_provenance.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_COUNT_RE = re.compile(r"(\d+)\s+tests?\s+collected")


def collect_count(paths: list[str], repo_root: Path) -> tuple[int, str]:
    """Return the collected-test count and the raw summary line."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *paths],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    output = (proc.stdout + proc.stderr).strip()
    match = _COUNT_RE.search(output)
    if match is None:
        raise RuntimeError(f"could not parse collected count from:\n{output[-500:]}")
    return int(match.group(1)), match.group(0)


def main() -> int:
    """Print ``<count> <path...> @ <head>`` for validation metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Test paths to count")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    try:
        count, summary = collect_count(args.paths, repo_root)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    print(f"{count} {' '.join(args.paths)} @ {head} ({summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
