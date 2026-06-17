#!/usr/bin/env python3
"""Generate JSON and Markdown reports from an ``odd_hazard_coverage.v1`` config.

The input is a checked-in YAML/JSON config, not a local benchmark campaign output.
Every weakly_covered, blocked, or absent row carries an explicit gap reason so that
benchmark wording cannot claim uncovered ODD or hazard classes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.odd_hazard_coverage_matrix import (
    generate_json_report,
    generate_markdown_report,
    load_odd_hazard_coverage_matrix,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_command(args: argparse.Namespace) -> str:
    """Return a reproducible command string for provenance."""

    return (
        "uv run python scripts/tools/generate_odd_hazard_coverage_matrix.py "
        f"--config {args.config.as_posix()} "
        f"--out-json {args.out_json.as_posix()} "
        f"--out-md {args.out_md.as_posix()} "
        f"--repo-root {args.repo_root.as_posix()}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ODD hazard coverage matrix report generator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/odd_hazard_coverage.v1.yaml"),
        help="Path to the odd_hazard_coverage.v1 YAML/JSON config.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="Output path for the generated JSON report.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        required=True,
        help="Output path for the generated Markdown report.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root used to resolve relative config references.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        matrix = load_odd_hazard_coverage_matrix(args.config)
        command = _build_command(args)
        json_report = generate_json_report(matrix, repo_root=args.repo_root, command=command)
        markdown_report = generate_markdown_report(
            matrix, repo_root=args.repo_root, command=command
        )
    except Exception as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(json_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.out_md.write_text(markdown_report, encoding="utf-8")

    sys.stdout.write(
        json.dumps(
            {
                "json_report": args.out_json.as_posix(),
                "markdown_report": args.out_md.as_posix(),
                "reference_valid": json_report["summary"]["reference_valid"],
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
