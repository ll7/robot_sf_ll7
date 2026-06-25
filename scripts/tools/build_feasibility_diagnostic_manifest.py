#!/usr/bin/env python3
"""Build a dry-run feasibility diagnostic manifest for issue #3484.

This tool prepares the universally-failing scenario-family diagnostic worklist. It
does not run planners, certify route feasibility, or produce benchmark evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.scenario_certification.failure_cause import build_feasibility_diagnostic_manifest
from robot_sf.training.scenario_loader import load_scenarios


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_config", type=Path, help="Scenario YAML manifest to inspect.")
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help=(
            "Scenario family/archetype to include. May be supplied multiple times. "
            "Defaults to all families in the manifest."
        ),
    )
    parser.add_argument("--output", type=Path, help="Write JSON report to this path.")
    return parser


def main() -> int:
    """Build and emit the diagnostic planning manifest."""

    args = _build_parser().parse_args()
    scenarios = load_scenarios(args.scenario_config)
    manifest = build_feasibility_diagnostic_manifest(
        scenarios,
        source=args.scenario_config.as_posix(),
        family_ids=args.family,
    )
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
