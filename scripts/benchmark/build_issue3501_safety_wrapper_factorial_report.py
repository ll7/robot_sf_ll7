#!/usr/bin/env python3
"""Build issue #3501 paired safety-wrapper factorial report artifacts."""

from __future__ import annotations

import argparse

from robot_sf.benchmark.safety_wrapper_ablation_manifest import load_safety_wrapper_ablation_rows
from robot_sf.benchmark.safety_wrapper_factorial_report import (
    build_safety_wrapper_factorial_report,
    write_safety_wrapper_factorial_report,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", required=True, help="Completed issue #3501 ablation rows JSON/JSONL."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for summary.json, per_planner_effects.csv, and README.md.",
    )
    return parser.parse_args()


def main() -> int:
    """Build report artifacts and fail closed when rows are incomplete."""

    args = parse_args()
    rows = load_safety_wrapper_ablation_rows(args.rows)
    report = build_safety_wrapper_factorial_report(rows)
    paths = write_safety_wrapper_factorial_report(report, args.out)
    print(
        "issue_3501_safety_wrapper_factorial_report "
        f"status={report['status']} summary={paths['summary']}"
    )
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
