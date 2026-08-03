#!/usr/bin/env python3
"""Run the issue #6641 runtime radius-binding canary (benchmark 6600 Gate 1).

Plain-language summary
----------------------
This is the command-line entry point for the radius-binding canary. It loads one
geometry-sensitive scenario, checks the selected collision-envelope radius
propagates consistently to all five binding surfaces (simulator collision
geometry, obstacle and pedestrian contact logic, feasibility/oracle, metric
metadata and output rows, and planner inputs), and writes a machine-readable
go/no-go verdict. It is a small bounded diagnostic, never a production sweep.

Exit code is 0 when the verdict is ``go`` and non-zero (fail-closed) on any
``no-go`` surface or operational error.

Example
-------
    uv run python scripts/benchmark/run_radius_binding_canary_issue_6641.py \
        --scenario configs/scenarios/canary_corridor.yaml \
        --robot-radius 0.3 --ped-radius 0.4 --out verdict.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.radius_binding_canary import (
    DEFAULT_RADIUS_TOLERANCE_M,
    DEFAULT_SCAN_STEP_M,
    DEFAULT_SCENARIO_REL,
    DEFAULT_SELECTED_PED_RADIUS_M,
    DEFAULT_SELECTED_ROBOT_RADIUS_M,
    VERDICT_GO,
    canary_verdict_to_dict,
    run_radius_binding_canary,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the canary command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        type=Path,
        default=DEFAULT_SCENARIO_REL,
        help=(f"Geometry-sensitive scenario manifest path (default: {DEFAULT_SCENARIO_REL})."),
    )
    parser.add_argument(
        "--robot-radius",
        type=float,
        default=DEFAULT_SELECTED_ROBOT_RADIUS_M,
        help="Selected robot collision-envelope radius in metres.",
    )
    parser.add_argument(
        "--ped-radius",
        type=float,
        default=DEFAULT_SELECTED_PED_RADIUS_M,
        help="Selected pedestrian radius in metres.",
    )
    parser.add_argument(
        "--tolerance-m",
        type=float,
        default=DEFAULT_RADIUS_TOLERANCE_M,
        help="Absolute radius tolerance for accepting a binding (metres).",
    )
    parser.add_argument(
        "--scan-step-m",
        type=float,
        default=DEFAULT_SCAN_STEP_M,
        help="Step size for the differential radius/distance scans (metres).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the machine-readable JSON verdict.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the canary and return the process exit code."""
    args = parse_args(argv)
    scenario_path = args.scenario
    verdict = run_radius_binding_canary(
        scenario_path=scenario_path,
        selected_robot_radius_m=args.robot_radius,
        selected_ped_radius_m=args.ped_radius,
        tolerance_m=args.tolerance_m,
        scan_step_m=args.scan_step_m,
    )
    payload = canary_verdict_to_dict(verdict)
    # Reject any accidental non-finite value instead of emitting JavaScript-style
    # NaN/Infinity tokens that are not valid machine-readable JSON.
    text = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        out_path = args.out if args.out.is_absolute() else _REPO_ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return 0 if verdict.verdict == VERDICT_GO else 1


if __name__ == "__main__":
    sys.exit(main())
