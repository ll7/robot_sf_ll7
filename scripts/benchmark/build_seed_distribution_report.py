#!/usr/bin/env python3
"""Build a seed distribution report from existing campaign artifacts.

Normalizes seed-level benchmark outputs into the seed_distribution_report.v1
common schema. Runs entirely on local artifacts -- no simulations, no SLURM,
no campaigns are launched.

Usage:
    uv run python scripts/benchmark/build_seed_distribution_report.py \\
        --campaign-root output/example-campaign \\
        --out-json output/example-campaign/reports/seed_distribution_report.json \\
        --out-md output/example-campaign/reports/seed_distribution_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robot_sf.benchmark.seed_distribution_report import (
    build_seed_distribution_report,
    validate_schema_version,
    write_report,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a seed_distribution_report.v1 from existing campaign artifacts.",
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        help="Path to the campaign root directory containing seed-level artifacts.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Path to write the JSON report.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Path to write the Markdown summary report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Exit code: 0 on success, 1 on missing/invalid inputs, 2 on write failure.
    """
    args = _parse_args(argv)

    if args.out_json is None and args.out_md is None:
        print(
            "error: at least one of --out-json or --out-md is required",
            file=sys.stderr,
        )
        return 1

    try:
        report = build_seed_distribution_report(args.campaign_root)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    validate_schema_version(report)

    try:
        written = write_report(report, out_json=args.out_json, out_md=args.out_md)
    except (OSError, ValueError) as exc:
        print(f"error writing report: {exc}", file=sys.stderr)
        return 2

    for kind, path in written.items():
        print(f"wrote {kind}: {path}")

    surface_count = len(report.get("surfaces", []))
    print(f"seed_distribution_report.v1: {surface_count} surfaces normalized")

    return 0


if __name__ == "__main__":
    sys.exit(main())
