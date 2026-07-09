#!/usr/bin/env python3
"""Validate draft scenario YAML before training or benchmark workflows consume it."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.tools.scenario_authoring import (
    configure_authoring_tool_logging,
    validate_scenario_file,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the scenario validation CLI parser."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n  uv run python scripts/tools/validate_scenario.py <scenario_config.yaml>"
        ),
    )
    parser.add_argument("scenario_config", type=Path, help="Scenario YAML file to validate.")
    parser.add_argument(
        "--allow-legacy-metadata",
        action="store_true",
        help="Do not require metadata.authoring; useful for checking older maintained configs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show loader and map-parser logs during validation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run scenario authoring validation and print actionable errors."""

    args = _build_parser().parse_args(argv)
    configure_authoring_tool_logging(verbose=args.verbose)
    report = validate_scenario_file(
        args.scenario_config,
        require_authoring_metadata=not args.allow_legacy_metadata,
    )
    if report.ok:
        print(f"OK: validated {report.scenario_count} scenario(s) in {args.scenario_config}")
        print("Status: draft validation only; not benchmark evidence.")
        return 0

    print(f"FAILED: found {len(report.issues)} issue(s) in {args.scenario_config}")
    for issue in report.issues:
        print(f"- {issue.format()}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
