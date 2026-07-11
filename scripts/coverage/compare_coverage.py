#!/usr/bin/env python3
"""
Compare coverage against a baseline and enforce an optional absolute floor.

This CLI tool compares current coverage.json against a baseline and generates
warnings in various formats for CI/CD integration. Baseline decreases are
non-blocking by default, but callers can fail on a decrease or an absolute
minimum total coverage percentage.

Usage:
    python scripts/coverage/compare_coverage.py \
        --current coverage.json \
        --absolute-only \
        --minimum-total 85.0 \
        --format github

Exit Codes:
    0: Success (even if coverage decreased - warnings only)
    1: Coverage policy failure or fatal input error
    2: Invalid command-line arguments
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

from robot_sf.coverage_tools.baseline_comparator import (
    compare,
    generate_warning,
    load_baseline,
)


def percentage(value: str) -> float:
    """Parse a percentage constrained to the inclusive range 0..100."""
    parsed = float(value)
    if not 0.0 <= parsed <= 100.0:
        msg = "percentage must be between 0 and 100"
        raise ValueError(msg)
    return parsed


def main() -> int:
    """Main entry point for coverage comparison CLI."""
    parser = ArgumentParser(description="Compare coverage and optionally enforce policy")
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("coverage.json"),
        help="Path to current coverage.json (default: coverage.json)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("coverage/.coverage-baseline.json"),
        help="Path to baseline coverage JSON (default: coverage/.coverage-baseline.json)",
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
    parser.add_argument(
        "--minimum-total",
        type=percentage,
        help="Fail if total measured coverage is below this percentage",
    )
    parser.add_argument(
        "--absolute-only",
        action="store_true",
        help="Check only --minimum-total without loading or reporting a baseline",
    )

    args = parser.parse_args()
    if args.absolute_only and args.minimum_total is None:
        parser.error("--absolute-only requires --minimum-total")

    # Absolute-only enforcement deliberately does not depend on the advisory
    # cache baseline. Missing baselines remain acceptable in comparison mode.
    baseline = None if args.absolute_only else load_baseline(args.baseline)

    # Compare current against baseline
    try:
        delta = compare(args.current, baseline, threshold=args.threshold)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Generate and output warning
    warning = "" if args.absolute_only else generate_warning(delta, format_type=args.format)
    floor_failed = args.minimum_total is not None and delta.current_coverage < args.minimum_total

    if args.format == "json" and args.minimum_total is not None:
        payload = json.loads(warning) if warning else {"current_coverage": delta.current_coverage}
        payload["minimum_total_coverage"] = args.minimum_total
        payload["absolute_floor_passed"] = not floor_failed
        print(json.dumps(payload, indent=2))
    elif warning:
        print(warning)

    if floor_failed:
        message = (
            f"Total coverage {delta.current_coverage:.2f}% is below the required "
            f"{args.minimum_total:.2f}% absolute floor."
        )
        if args.format == "github":
            print(f"::error title=Absolute Coverage Floor Failed::{message}")
        elif args.format == "terminal":
            print(f"ERROR: {message}", file=sys.stderr)
        return 1

    # Optional: fail on decrease
    if args.fail_on_decrease and delta.has_decrease:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
