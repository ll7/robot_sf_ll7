"""Build an analysis-only Flint surface candidate from a canonical report pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.flint_chart import (
    FlintChartContractError,
    build_surface,
    load_json,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the surface-builder argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON surface input")
    parser.add_argument("--output", type=Path, required=True, help="JSON candidate output")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate and summarize without writing the candidate output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build one candidate surface or return a fail-closed diagnostic."""
    args = build_parser().parse_args(argv)
    try:
        surface = build_surface(load_json(args.input))
        if not args.check_only:
            write_json(args.output, surface)
        print(
            json.dumps(
                {
                    "status": "validated_not_promoted",
                    "surface_id": surface["surface_id"],
                    "source_context": surface["source_context"],
                    "compared_cells": surface["parity"]["compared_cells"],
                    "output": None if args.check_only else str(args.output),
                    "claim_boundary": surface["claim_boundary"],
                },
                sort_keys=True,
            )
        )
        return 0
    except FlintChartContractError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
