#!/usr/bin/env python3
"""Fail-closed readiness check for scenario-horizon Results evidence (issue #3266).

Reports whether a re-exported scenario-horizon campaign artifact is valid
benchmark evidence, diagnostic-only, or blocked by a missing artifact. This is a
diagnostic-only status check: it never reruns a campaign and never promotes
evidence.

Exit codes:

- ``0`` -- artifact is valid benchmark evidence.
- ``2`` -- artifact is diagnostic-only (non-success rows or unresolved SNQI caveat).
- ``3`` -- artifact is blocked (missing or unparseable).

Example::

    uv run python scripts/validation/check_scenario_horizon_results_readiness.py \
        docs/context/evidence/issue_3203_scenario_horizon_reexport_2026-06-20/reports/campaign_table.md
"""

from __future__ import annotations

import argparse
import json
import sys

from robot_sf.benchmark.scenario_horizon_readiness import (
    BLOCKED,
    VALID,
    classify_scenario_horizon_readiness,
)

#: Map readiness verdicts to canonical fail-closed exit codes.
_EXIT_CODES = {VALID: 0, BLOCKED: 3}
_DIAGNOSTIC_EXIT_CODE = 2


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "artifact",
        help="Path to a re-exported campaign_table.md or campaign-summary JSON.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the verdict as JSON instead of human-readable text.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the readiness check and return a fail-closed exit code.

    Returns:
        ``0`` for valid, ``2`` for diagnostic-only, ``3`` for blocked.
    """
    args = _build_parser().parse_args(argv)
    readiness = classify_scenario_horizon_readiness(args.artifact)

    if args.json:
        print(json.dumps(readiness.to_payload(), indent=2, sort_keys=True))
    else:
        print(f"scenario-horizon Results readiness: {readiness.status}")
        print(f"  artifact: {readiness.artifact}")
        print(f"  planner_rows: {readiness.planner_rows}")
        print(f"  ppo_status: {readiness.ppo_status}")
        print(f"  snqi_contract_status: {readiness.snqi_contract_status}")
        if readiness.blockers:
            print("  blockers:")
            for blocker in readiness.blockers:
                print(f"    - {blocker}")

    return _EXIT_CODES.get(readiness.status, _DIAGNOSTIC_EXIT_CODE)


if __name__ == "__main__":
    sys.exit(main())
