#!/usr/bin/env python3
"""Run the issue #4973 deterministic CPU corridor acceptance harness."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_sf.research.zanlungo_corridor_acceptance import (
    load_acceptance_config,
    run_acceptance,
    write_acceptance_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the validation command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research/issue_4973_zanlungo_corridor_acceptance.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/issue_4973_zanlungo_corridor_acceptance"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the configured harness and return nonzero when acceptance is not met."""
    args = build_parser().parse_args(argv)
    report = run_acceptance(load_acceptance_config(args.config))
    paths = write_acceptance_report(report, args.output_dir)
    print(f"acceptance_met={report['acceptance_met']}")
    print(f"summary={paths['summary_json']}")
    return 0 if report["acceptance_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
