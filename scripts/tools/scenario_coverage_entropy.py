#!/usr/bin/env python3
"""Build a config-only scenario coverage entropy report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.runner import load_scenario_matrix
from robot_sf.benchmark.scenario_coverage import (
    build_scenario_coverage_report,
    write_scenario_coverage_report,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path, help="Scenario matrix YAML file.")
    parser.add_argument("--output-json", type=Path, help="Optional JSON report output path.")
    parser.add_argument(
        "--output-markdown", type=Path, help="Optional Markdown report output path."
    )
    return parser.parse_args()


def main() -> int:
    """Run the coverage report builder."""
    args = parse_args()
    scenarios = load_scenario_matrix(args.matrix)
    report = build_scenario_coverage_report(scenarios, source=str(args.matrix))
    write_scenario_coverage_report(
        report,
        json_path=args.output_json,
        markdown_path=args.output_markdown,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "scenario_count": report["summary"]["scenario_count"],
                "coverage_entropy": report["summary"]["coverage_entropy"],
                "output_json": str(args.output_json) if args.output_json else None,
                "output_markdown": str(args.output_markdown) if args.output_markdown else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
