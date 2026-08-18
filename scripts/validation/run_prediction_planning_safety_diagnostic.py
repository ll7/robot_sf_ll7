#!/usr/bin/env python3
"""Run the deterministic fixture diagnostic for issue #7317."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.safety.prediction_planning_safety import (
    build_fixture_diagnostic_report,
    validate_prediction_planning_safety_report,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the fixture diagnostic command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build and validate the fixture-only prediction/planning/runtime safety "
            "diagnostic for issue #7317."
        )
    )
    parser.add_argument("--seed", type=int, default=7317, help="Deterministic fixture seed.")
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=0.8,
        help="Held-out empirical coverage target in (0, 1).",
    )
    parser.add_argument(
        "--hard-floor-m",
        type=float,
        default=0.3,
        help="Immutable deterministic safety margin floor in metres.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Without it, the report is written to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build, validate, and emit the deterministic fixture report."""
    args = build_arg_parser().parse_args(argv)
    report = build_fixture_diagnostic_report(
        seed=args.seed,
        coverage_target=args.coverage_target,
        hard_floor_m=args.hard_floor_m,
    )
    payload = report.to_dict()
    validate_prediction_planning_safety_report(payload)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(json.dumps({"status": "valid", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
