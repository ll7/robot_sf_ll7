#!/usr/bin/env python3
"""
Compare coverage against baseline and generate warnings.

This CLI tool compares current coverage.json against a baseline and generates
warnings in various formats for CI/CD integration. Non-blocking by default
(exits 0 on coverage decrease), but can optionally fail via --fail-on-decrease.

Usage:
    python scripts/coverage/compare_coverage.py \\
        --current coverage.json \\
        --baseline .coverage-baseline.json \\
        --threshold 1.0 \\
        --format github

Exit Codes:
    0: Success (even if coverage decreased - warnings only)
    1: Fatal error (missing files, invalid data)
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

from robot_sf.coverage_tools.baseline_comparator import (
    compare,
    generate_warning,
    load_baseline,
)


def main() -> int:
    """Main entry point for coverage comparison CLI."""
    parser = ArgumentParser(description="Compare coverage against baseline (non-blocking)")
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("coverage.json"),
        help="Path to current coverage.json (default: coverage.json)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(".coverage-baseline.json"),
        help="Path to baseline coverage JSON (default: .coverage-baseline.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Warning threshold in percentage points (default: 1.0)",
    )
    parser.add_argument(
        "--format",
        choices=["github", "terminal", "json"],
        default="terminal",
        help="Output format (default: terminal)",
    )
    parser.add_argument(
        "--fail-on-decrease",
        action="store_true",
        help="Exit with code 1 if coverage decreased (default: always exit 0)",
    )

    args = parser.parse_args()

    # Load baseline (optional - missing baseline is OK)
    baseline = load_baseline(args.baseline)

    # Compare current against baseline
    try:
        delta = compare(args.current, baseline, threshold=args.threshold)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Generate and output warning
    warning = generate_warning(delta, format_type=args.format)
    if warning:
        print(warning)

    # Optional: fail on decrease
    if args.fail_on_decrease and delta.has_decrease:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
