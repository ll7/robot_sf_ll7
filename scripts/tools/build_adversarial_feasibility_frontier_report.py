#!/usr/bin/env python3
"""Build a fixture-first frontier report from declared round evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.adversarial.feasibility_frontier_report import (
    FrontierReportError,
    write_frontier_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate persisted planner/falsifier round evidence and write a "
            "machine-readable report, Markdown summary, and frontier figure."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Persisted evidence JSON bundle")
    parser.add_argument("--out-dir", required=True, type=Path, help="New or empty report directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the report builder and return a shell exit code."""
    args = _parser().parse_args(argv)
    try:
        report = write_frontier_report(args.input, args.out_dir)
    except (FrontierReportError, OSError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "schema_version": report["schema_version"],
                "experiment_id": report["experiment_id"],
                "round_count": report["round_count"],
                "input_sha256": report["input_sha256"],
                "output_dir": str(args.out_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
