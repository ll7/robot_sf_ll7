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
    parser.add_argument(
        "--pair-comparison-grain",
        choices=("matched_planner_pair", "matched_realization_pair"),
        help="Declared comparison grain for --pair-input compatibility gates.",
    )
    parser.add_argument(
        "--encounter-report",
        type=Path,
        help="Optional canonical near_miss_encounter.v1 report used to bind the focal interval.",
    )
    parser.add_argument("--route-id", help="Registered route ID.")
    parser.add_argument("--route-provenance-id", help="Provenance ID for the registered route.")
    parser.add_argument("--route-registry-checksum", help="Checksum for the registered route.")
    parser.add_argument("--route-start", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--route-end", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--conflict-zone-id", help="Registered conflict zone ID.")
    parser.add_argument(
        "--conflict-provenance-id",
        help="Provenance ID for the registered conflict zone.",
    )
    parser.add_argument(
        "--conflict-registry-checksum",
        help="Checksum for the registered conflict zone.",
    )
    parser.add_argument("--conflict-center", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--conflict-radius-m", type=float)
    args = parser.parse_args()
    if args.pair_input is not None and args.pair_comparison_grain is None:
        parser.error("--pair-comparison-grain is required when --pair-input is provided")

    route = None
    if (
        args.route_id
        or args.route_start
        or args.route_end
        or args.route_provenance_id
        or args.route_registry_checksum
    ):
        if not (
            args.route_id
            and args.route_start
            and args.route_end
            and args.route_provenance_id
            and args.route_registry_checksum
        ):
            parser.error(
                "--route-id, --route-provenance-id, --route-registry-checksum, "
                "--route-start, and --route-end must be provided together"
            )
        route = RouteSpec(
            route_id=args.route_id,
            start=(args.route_start[0], args.route_start[1]),
            end=(args.route_end[0], args.route_end[1]),
            provenance_id=args.route_provenance_id,
            registry_checksum=args.route_registry_checksum,
        )

    conflict_zone = None
    if (
        args.conflict_zone_id
        or args.conflict_center
        or args.conflict_radius_m is not None
        or args.conflict_provenance_id
        or args.conflict_registry_checksum
    ):
        if not (
            args.conflict_zone_id
            and args.conflict_center
            and args.conflict_radius_m is not None
            and args.conflict_provenance_id
            and args.conflict_registry_checksum
        ):
            parser.error(
                "--conflict-zone-id, --conflict-provenance-id, "
                "--conflict-registry-checksum, --conflict-center, and "
                "--conflict-radius-m must be provided together"
            )
        conflict_zone = ConflictZoneSpec(
            zone_id=args.conflict_zone_id,
            center=(args.conflict_center[0], args.conflict_center[1]),
            radius_m=args.conflict_radius_m,
            provenance_id=args.conflict_provenance_id,
            registry_checksum=args.conflict_registry_checksum,
        )

    output = write_worked_example_process_trace(
        args.input,
        args.out,
        route=route,
        conflict_zone=conflict_zone,
        focal_actor_id=args.focal_actor_id,
        pair_input_path=args.pair_input,
        encounter_report_path=args.encounter_report,
        pair_comparison_grain=args.pair_comparison_grain,
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
