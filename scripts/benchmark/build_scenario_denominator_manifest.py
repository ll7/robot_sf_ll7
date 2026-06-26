#!/usr/bin/env python3
"""Build or check scenario denominator manifests from benchmark configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.scenario_denominator_manifest import (
    DenominatorManifestError,
    build_scenario_denominator_manifest,
    check_denominator_table,
    check_manifest,
    denominator_table_rows,
    write_denominator_table,
    write_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help="Benchmark campaign config YAML. Repeat to combine multiple canonical configs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON manifest output path. If omitted without checks, manifest prints to stdout.",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=None,
        help="Optional CSV per-family x planner denominator table output path.",
    )
    parser.add_argument(
        "--check-manifest",
        type=Path,
        default=None,
        help="Existing manifest to compare with generated config-derived manifest.",
    )
    parser.add_argument(
        "--check-table",
        type=Path,
        default=None,
        help="Existing CSV, JSON, YAML, or Markdown denominator table to compare with manifest rows.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the scenario denominator manifest CLI."""

    args = _build_parser().parse_args(argv)
    try:
        manifest = build_scenario_denominator_manifest(args.config)
        if args.check_manifest is not None:
            check_manifest(manifest, args.check_manifest)
        if args.check_table is not None:
            check_denominator_table(manifest, args.check_table)
        if args.output is not None:
            path = write_manifest(manifest, args.output)
            print(f"wrote scenario denominator manifest: {path}")
        elif args.check_manifest is None and args.check_table is None:
            print(json.dumps(manifest, indent=2, sort_keys=True))
        if args.table is not None:
            path = write_denominator_table(manifest, args.table)
            print(f"wrote scenario denominator table: {path}")
        if args.check_manifest is not None or args.check_table is not None:
            rows = len(denominator_table_rows(manifest))
            print(f"scenario denominator checks passed ({rows} family x planner rows)")
    except (DenominatorManifestError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"scenario denominator manifest failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
