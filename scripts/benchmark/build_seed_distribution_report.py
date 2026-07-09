#!/usr/bin/env python3
"""Build a unified seed-distribution report from existing campaign artifacts.

This script consumes existing seed-level artifacts (seed_variability_by_scenario.json,
headline_ci_rank_stability.json) and emits a normalized seed_distribution_report.v1
JSON and optional Markdown summary without running any simulations.

Usage:

    uv run python scripts/benchmark/build_seed_distribution_report.py \\
        --campaign-root output/benchmarks/camera_ready/<campaign_id> \\
        --out-json output/.../reports/seed_distribution_report.json \\
        --out-md output/.../reports/seed_distribution_report.md

Exit codes:

- 0: report built successfully (may contain advisory diagnostics).
- 1: no supported seed-level artifacts found or other hard error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.seed_distribution_report import (
    DEFAULT_INSUFFICIENT_SEED_THRESHOLD,
    DEFAULT_WIDE_INTERVAL_THRESHOLD,
    build_report_from_campaign_dir,
    format_report_markdown,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build seed_distribution_report.v1 from campaign artifacts."
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        help="Path to the campaign output directory (containing reports/).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Path for the output JSON report.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Path for the output Markdown summary.",
    )
    parser.add_argument(
        "--primary-metric",
        default="success",
        help="Primary metric for surface point estimates (default: success).",
    )
    parser.add_argument(
        "--insufficient-seed-threshold",
        type=int,
        default=DEFAULT_INSUFFICIENT_SEED_THRESHOLD,
        help="Seed count below which insufficient_seed_count is flagged.",
    )
    parser.add_argument(
        "--wide-interval-threshold",
        type=float,
        default=DEFAULT_WIDE_INTERVAL_THRESHOLD,
        help="CI half-width above which wide_interval is flagged.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build seed distribution report from campaign artifacts."""
    args = _parse_args(argv)

    try:
        report = build_report_from_campaign_dir(
            args.campaign_root,
            primary_metric=args.primary_metric,
            insufficient_seed_threshold=args.insufficient_seed_threshold,
            wide_interval_threshold=args.wide_interval_threshold,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=False),
            encoding="utf-8",
        )
        print(f"Wrote JSON report to {args.out_json}")

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(format_report_markdown(report), encoding="utf-8")
        print(f"Wrote Markdown summary to {args.out_md}")

    if not args.out_json and not args.out_md:
        print(report.to_json())

    return 0


if __name__ == "__main__":
    sys.exit(main())
