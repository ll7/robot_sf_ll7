#!/usr/bin/env python3
"""Fail-closed release-readiness / claim-audit preflight checklist (issue #3081).

Read-only preflight over a declarative release checklist. It reports, per item,
whether a declared release prerequisite is ``complete`` or ``blocked`` (with
explicit gaps), and derives an overall ``passed``/``blocked`` status that is
``passed`` only when every prerequisite is satisfied.

It makes no benchmark, metric, or research claim and performs no publication: it
does not tag, upload, regenerate artifacts, close issues, or *declare* release
readiness. A passing preflight only means no blocking gaps were found among the
declared prerequisites; a maintainer still owns the readiness decision.

Examples:
    # Print a Markdown report against the current checkout.
    uv run python scripts/tools/release_preflight_check.py

    # Emit JSON and fail the process when any prerequisite is blocked (CI gate).
    uv run python scripts/tools/release_preflight_check.py --format json --fail-on-blocked
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.release_preflight import (
    ReleasePreflightError,
    evaluate_release_preflight,
    load_release_preflight_checklist,
    render_markdown,
)

DEFAULT_CHECKLIST = Path("configs/benchmarks/releases/release_july_2026_preflight_issue_3081.yaml")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checklist",
        type=Path,
        default=DEFAULT_CHECKLIST,
        help="Path to the release preflight checklist YAML/JSON definition.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve declared artifact paths.",
    )
    parser.add_argument(
        "--format",
        choices=("md", "json"),
        default="md",
        help="Output format (default: md).",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Exit non-zero when any checklist item is blocked.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = _parse_args(argv)
    try:
        checklist = load_release_preflight_checklist(args.checklist)
    except (OSError, ReleasePreflightError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = evaluate_release_preflight(checklist, args.repo_root)

    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(render_markdown(result))

    if args.fail_on_blocked and result["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
