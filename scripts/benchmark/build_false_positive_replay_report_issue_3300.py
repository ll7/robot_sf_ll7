#!/usr/bin/env python3
"""Build issue #3300 false-positive actor-injection replay artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_sf.benchmark.false_positive_replay_report import (
    build_false_positive_replay_report,
    write_false_positive_replay_csv,
    write_false_positive_replay_json,
    write_false_positive_replay_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-jsonl", type=Path, required=True)
    parser.add_argument("--perturbed-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--replay-mode",
        choices=("executable", "trace_derived"),
        default="executable",
        help="Classify executable replay separately from trace-only diagnostics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build JSON, CSV, and Markdown false-positive replay reports."""
    args = build_parser().parse_args(argv)
    report = build_false_positive_replay_report(
        nominal_jsonl=args.nominal_jsonl,
        perturbed_jsonl=args.perturbed_jsonl,
        replay_mode=args.replay_mode,
    )
    write_false_positive_replay_json(report, args.output_json)
    write_false_positive_replay_csv(report, args.output_csv)
    write_false_positive_replay_markdown(report, args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
