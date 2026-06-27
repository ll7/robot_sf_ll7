#!/usr/bin/env python3
"""List research-engine packages and their artifact/prerequisite readiness (issue #3057).

Read-only preflight over the research-package registry declared for epic #3057. It reports,
for each package, which required tracked artifacts are present vs. missing and whether its
declared prerequisite packages are satisfied, deriving a fail-closed ready/blocked status.

This makes no benchmark, metric, or research claim; it schedules nothing and runs no
campaign. It only inspects existing config/contract metadata on disk.

Examples:
    # Print a Markdown table to stdout against the current checkout.
    uv run python scripts/tools/research_package_preflight.py

    # Emit JSON and fail the process when any package is blocked (CI/preflight gate).
    uv run python scripts/tools/research_package_preflight.py --format json --fail-on-blocked
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.research.package_registry import (
    evaluate_registry_preflight,
    load_registry,
    render_markdown,
)

DEFAULT_REGISTRY = Path("configs/research/research_package_registry_issue_3057.yaml")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Path to the research-package registry YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Directory that repo-relative artifact paths are resolved against.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file to write the rendered report to (otherwise stdout).",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Exit non-zero when any package is blocked (use as a preflight gate).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    try:
        registry = load_registry(args.registry)
    except (OSError, ValueError) as exc:
        print(f"failed to load registry {args.registry}: {exc}", file=sys.stderr)
        return 2

    report = evaluate_registry_preflight(registry, repo_root=args.repo_root)
    rendered = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else render_markdown(report)
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)

    if args.fail_on_blocked and report["summary"]["blocked_count"] > 0:
        blocked = [p["package_id"] for p in report["packages"] if p["status"] == "blocked"]
        print(f"blocked packages: {', '.join(blocked)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
