#!/usr/bin/env python3
"""Run the issue #9348 three-width doorway application preflight."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_sf.benchmark.three_width_doorway_application import (
    DEFAULT_MANIFEST_PATH,
    run_three_width_preflight,
    write_preflight_report,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    """Run the bounded oracle-first preflight and write its report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_REPO_ROOT / DEFAULT_MANIFEST_PATH,
        help="versioned three-width application manifest",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="output path for the review-marked preflight report",
    )
    parser.add_argument(
        "--variants-dir",
        type=Path,
        default=None,
        help="optional directory in which to retain generated variant maps and scenarios",
    )
    args = parser.parse_args()
    report = run_three_width_preflight(args.manifest, output_dir=args.variants_dir)
    write_preflight_report(report, args.out_json)
    print(
        f"issue #9348 preflight: variants={len(report['variants'])} "
        f"oracle_available={report['checks']['oracle_available_for_every_variant']} "
        f"go={report['go']} report={args.out_json}"
    )
    return 0 if report["go"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
