"""CLI for the ORCA-residual learned-policy lane readiness/preflight surface (issue #1358).

Thin wrapper over :func:`robot_sf.benchmark.orca_residual_lane_readiness.assess_lane_readiness`.
It inventories local lane scaffolding (prerequisites), informational command shapes (routes),
and the declared external gates (blockers) without executing training, submitting SLURM, or
altering planner behavior.

Exit codes:
  0  local scaffolding is handoff-complete (``blocked_on_followup``); only the declared
     external gates (child #1475, SLURM, durable artifacts) remain.
  2  a required local surface is missing or the lineage packet failed validation
     (``prerequisites_incomplete``).

Examples:
  uv run python scripts/tools/orca_residual_lane_readiness.py
  uv run python scripts/tools/orca_residual_lane_readiness.py --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.orca_residual_lane_readiness import assess_lane_readiness

_STATUS_HANDOFF_READY = "blocked_on_followup"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root for resolving relative paths (default: current directory).",
    )
    parser.add_argument(
        "--no-validate-packet",
        action="store_true",
        help="Skip the canonical lineage-packet validation (structural presence checks only).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full readiness report as JSON.",
    )
    return parser


def _print_human(report: dict) -> None:
    """Print a compact human-readable summary of the readiness report."""
    status = report["overall_status"]
    print(f"ORCA-residual lane readiness (#1358): {status.upper()}")
    print("- prerequisites:")
    for prereq in report["prerequisites"]:
        marker = "ok" if prereq["present"] and not prereq["messages"] else "MISSING/INVALID"
        print(f"   * {prereq['key']}: {marker} ({prereq['path']})")
        for message in prereq["messages"]:
            print(f"       - {message}")
    print("- blockers (expected external gates):")
    for blocker in report["blockers"]:
        print(f"   * {blocker['key']} [{blocker['gate']}]: {blocker['reason']}")
    print("- routes (informational, not executed):")
    for route in report["routes"]:
        print(f"   * {route['key']}: {route['command_shape']}")
    if report["errors"]:
        print("- errors:")
        for err in report["errors"]:
            print(f"   * {err}")


def main(argv: list[str] | None = None) -> int:
    """Assess lane readiness and return a shell-friendly exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    report = assess_lane_readiness(
        args.repo_root,
        validate_packet=not args.no_validate_packet,
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)

    return 0 if report["overall_status"] == _STATUS_HANDOFF_READY else 2


if __name__ == "__main__":
    raise SystemExit(main())
