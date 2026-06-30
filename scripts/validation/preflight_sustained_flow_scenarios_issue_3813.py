#!/usr/bin/env python3
"""Validate the issue #3813 sustained-flow scenario scaffold.

This is a CPU-only static preflight. It proves the continuous-spawn variants are
enumerable and fail closed as metadata-only scaffold entries; it does not run a
benchmark campaign or submit compute.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.scenario_certification.sustained_flow import (
    DEFAULT_SUSTAINED_FLOW_SCENARIO_SET,
    preflight_sustained_flow_scenario_set,
    sustained_flow_preflight_to_dict,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-set",
        type=Path,
        default=DEFAULT_SUSTAINED_FLOW_SCENARIO_SET,
        help="Scenario matrix to validate.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the full JSON report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the sustained-flow scaffold preflight CLI.

    Returns:
        Process exit code: zero when the scaffold conforms, non-zero otherwise.
    """

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    report = preflight_sustained_flow_scenario_set(args.scenario_set)
    payload = sustained_flow_preflight_to_dict(report)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        result = "PASS" if report.conforms else "FAIL"
        print(f"{result}: {payload['scenario_set']}")
        print(f"variants: {payload['variant_count']}")
        print(f"runtime_support: {payload['runtime_support']}")
        print(f"benchmark_evidence: {payload['benchmark_evidence']}")
        for error in payload["errors"]:
            print(f"ERROR: {error}")

    return 0 if report.conforms else 1


if __name__ == "__main__":
    raise SystemExit(main())
