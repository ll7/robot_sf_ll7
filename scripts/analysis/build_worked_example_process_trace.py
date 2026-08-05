#!/usr/bin/env python3
"""Build deterministic ``worked_example_process_trace.v1`` artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_sf.analysis_workbench.interaction_coordinates import (
    ConflictZoneSpec,
    RouteSpec,
    write_worked_example_process_trace,
)


def main() -> int:
    """Run the process-trace CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a renderer-neutral worked_example_process_trace.v1 diagnostic from a "
            "simulation_trace_export.v1 JSON artifact."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input trace export JSON.")
    parser.add_argument("--out", required=True, type=Path, help="Output process trace JSON.")
    parser.add_argument("--focal-actor-id", help="Explicit actor ID for the focal encounter.")
    parser.add_argument("--pair-input", type=Path, help="Optional second trace for compatibility.")
    parser.add_argument("--route-id", help="Registered route ID.")
    parser.add_argument("--route-start", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--route-end", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--conflict-zone-id", help="Registered conflict zone ID.")
    parser.add_argument("--conflict-center", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--conflict-radius-m", type=float)
    args = parser.parse_args()

    route = None
    if args.route_id or args.route_start or args.route_end:
        if not (args.route_id and args.route_start and args.route_end):
            parser.error("--route-id, --route-start, and --route-end must be provided together")
        route = RouteSpec(
            route_id=args.route_id,
            start=(args.route_start[0], args.route_start[1]),
            end=(args.route_end[0], args.route_end[1]),
        )

    conflict_zone = None
    if args.conflict_zone_id or args.conflict_center or args.conflict_radius_m is not None:
        if not (
            args.conflict_zone_id and args.conflict_center and args.conflict_radius_m is not None
        ):
            parser.error(
                "--conflict-zone-id, --conflict-center, and --conflict-radius-m "
                "must be provided together"
            )
        conflict_zone = ConflictZoneSpec(
            zone_id=args.conflict_zone_id,
            center=(args.conflict_center[0], args.conflict_center[1]),
            radius_m=args.conflict_radius_m,
        )

    output = write_worked_example_process_trace(
        args.input,
        args.out,
        route=route,
        conflict_zone=conflict_zone,
        focal_actor_id=args.focal_actor_id,
        pair_input_path=args.pair_input,
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
