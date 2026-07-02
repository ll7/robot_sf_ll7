#!/usr/bin/env python3
"""Build paired safety and comfort release-gate reports from existing rows."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from robot_sf.benchmark.release_gates import (
    build_release_gate_report,
    load_metric_rows,
    load_release_gate_spec,
    write_release_gate_report,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json", type=Path, required=True, help="Existing summary rows JSON."
    )
    parser.add_argument("--gate-spec", type=Path, required=True, help="Release-gate YAML spec.")
    parser.add_argument("--output-json", type=Path, required=True, help="Report JSON output path.")
    parser.add_argument(
        "--output-csv", type=Path, help="Optional pass/fail matrix CSV output path."
    )
    parser.add_argument("--output-md", type=Path, help="Optional Markdown report output path.")
    parser.add_argument(
        "--generated-at-utc",
        help="Optional deterministic ISO-8601 timestamp for reviewable fixtures.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the release-gate report builder."""

    args = parse_args()
    rows = load_metric_rows(args.input_json)
    gates = load_release_gate_spec(args.gate_spec)
    command = shlex.join(sys.argv)
    report = build_release_gate_report(
        rows,
        gates,
        input_path=args.input_json,
        gate_spec_path=args.gate_spec,
        command=command,
        generated_at_utc=args.generated_at_utc,
    )
    paths = write_release_gate_report(
        report,
        json_path=args.output_json,
        csv_path=args.output_csv,
        markdown_path=args.output_md,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "json_path": str(paths["json"]),
                "csv_path": str(paths.get("csv", "")),
                "markdown_path": str(paths.get("markdown", "")),
                "matrix_rows": len(report["matrix_rows"]),
                "status_counts": report["status_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
