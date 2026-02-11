#!/usr/bin/env python3
"""Inspect SVG maps for parser-facing route and zone issues.

This tool is intended for fast local debugging when a map "looks fine" but
behaves unexpectedly after parsing. It reports route-only mode usage, route
index consistency, obstacle-interior crossings, and risky SVG path commands.

Examples:
    # Inspect one SVG file
    uv run python scripts/validation/svg_inspect.py maps/svg_maps/classic_crossing.svg

    # Inspect all classic maps and fail on warnings or errors
    uv run python scripts/validation/svg_inspect.py maps/svg_maps --pattern "classic_*.svg" --strict warning

    # Write JSON report for tooling/CI analysis
    uv run python scripts/validation/svg_inspect.py maps/svg_maps --json output/validation/svg_inspection.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from robot_sf.common.logging import configure_logging
from robot_sf.maps.verification.svg_inspection import SvgInspectionReport, inspect_svg_targets


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Raw command-line arguments excluding executable name.

    Returns:
        argparse.Namespace: Parsed CLI options.
    """

    parser = argparse.ArgumentParser(
        description="Inspect SVG route/zones and parser-facing geometry issues.",
    )
    parser.add_argument(
        "target",
        type=Path,
        nargs="?",
        default=Path("maps/svg_maps"),
        help="SVG file or directory to inspect (default: maps/svg_maps)",
    )
    parser.add_argument(
        "--pattern",
        default="*.svg",
        help="Glob pattern when target is a directory (default: *.svg)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional JSON output path for full reports.",
    )
    parser.add_argument(
        "--strict",
        choices=["none", "warning", "error"],
        default="none",
        help="Exit non-zero on minimum severity threshold (default: none).",
    )
    parser.add_argument(
        "--show-routes",
        action="store_true",
        help="Print detailed per-route summaries.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def _print_report(report: SvgInspectionReport, show_routes: bool) -> None:
    """Print a human-readable summary for a single SVG report.

    Args:
        report: SVG inspection report.
        show_routes: Whether to print per-route details.
    """

    logger.info("SVG: {}", report.svg_file)
    logger.info(
        "  size={}x{} robot_routes={} ped_routes={} robot_zones=({}/{}) ped_zones=({}/{})",
        report.map_width,
        report.map_height,
        report.robot_routes,
        report.ped_routes,
        report.robot_spawn_zones,
        report.robot_goal_zones,
        report.ped_spawn_zones,
        report.ped_goal_zones,
    )
    logger.info(
        "  modes: ped_route_only={} robot_route_only={}",
        report.ped_route_only_mode,
        report.robot_route_only_mode,
    )

    if show_routes:
        for route in report.routes:
            logger.info(
                "  route kind={} label={} id={} wps={} len={:.2f} crosses_obstacle={} cmds={}",
                route.kind,
                route.label,
                route.path_id or "-",
                route.waypoint_count,
                route.route_length,
                route.crosses_obstacle_interior,
                route.commands,
            )

    if not report.findings:
        logger.info("  findings: none")
        return

    for finding in report.findings:
        prefix = f"{finding.severity.upper()} {finding.code}"
        if finding.path_id:
            logger.info("  {} (path={}): {}", prefix, finding.path_id, finding.message)
        else:
            logger.info("  {}: {}", prefix, finding.message)


def _should_fail(reports: list[SvgInspectionReport], strict: str) -> bool:
    """Evaluate strict-threshold failure condition.

    Args:
        reports: Inspection reports.
        strict: Strictness threshold (`none`, `warning`, or `error`).

    Returns:
        bool: True when strict policy requires non-zero exit code.
    """

    if strict == "none":
        return False
    threshold = {"warning": 1, "error": 2}[strict]
    return any(report.max_severity_rank() >= threshold for report in reports)


def main(argv: list[str] | None = None) -> int:
    """Run SVG inspection CLI.

    Args:
        argv: Optional CLI argument list.

    Returns:
        int: Process exit code.
    """

    args = _parse_args(argv or [])
    configure_logging(verbose=args.verbose)

    reports = inspect_svg_targets(args.target, pattern=args.pattern)
    for report in reports:
        _print_report(report, show_routes=args.show_routes)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([report.to_dict() for report in reports], indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote JSON report: {}", args.json)

    return 1 if _should_fail(reports, args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
