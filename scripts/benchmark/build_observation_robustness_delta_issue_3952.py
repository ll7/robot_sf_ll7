#!/usr/bin/env python3
"""Build issue #3952 observation robustness delta artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_sf.benchmark.robustness_delta import (
    build_robustness_delta_report,
    write_report_csv,
    write_report_json,
    write_report_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-jsonl", type=Path, required=True)
    parser.add_argument("--perturbed-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build JSON, CSV, and Markdown robustness reports."""
    args = build_parser().parse_args(argv)
    report = build_robustness_delta_report(
        nominal_jsonl=args.nominal_jsonl,
        perturbed_jsonl=args.perturbed_jsonl,
    )
    write_report_json(report, args.output_json)
    write_report_csv(report, args.output_csv)
    write_report_markdown(report, args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
