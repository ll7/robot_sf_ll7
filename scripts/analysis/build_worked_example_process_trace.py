#!/usr/bin/env python3
"""Build deterministic ``worked_example_process_trace.v1`` artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_sf.analysis_workbench.interaction_coordinates import (
    WorkedExampleProcessTraceValidationError,
    load_registered_conflict_zone_spec,
    load_registered_route_spec,
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
    parser.add_argument(
        "--focal-encounter-id",
        help="Explicit canonical encounter ID selected across the full encounter report.",
    )
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
    parser.add_argument(
        "--geometry-registry",
        type=Path,
        help="Versioned process_trace_geometry_registry.v1 JSON artifact.",
    )
    parser.add_argument("--route-entry-id", help="Unique route entry ID in --geometry-registry.")
    parser.add_argument(
        "--conflict-zone-entry-id",
        help="Unique conflict-zone entry ID in --geometry-registry.",
    )
    args = parser.parse_args()
    if args.pair_input is not None and args.pair_comparison_grain is None:
        parser.error("--pair-comparison-grain is required when --pair-input is provided")

    if (args.route_entry_id or args.conflict_zone_entry_id) and args.geometry_registry is None:
        parser.error("--geometry-registry is required for route/conflict entry selection")
    if args.geometry_registry is not None and not (
        args.route_entry_id or args.conflict_zone_entry_id
    ):
        parser.error("--geometry-registry requires at least one entry ID")
    try:
        route = (
            load_registered_route_spec(args.geometry_registry, args.route_entry_id)
            if args.geometry_registry is not None and args.route_entry_id
            else None
        )
        conflict_zone = (
            load_registered_conflict_zone_spec(args.geometry_registry, args.conflict_zone_entry_id)
            if args.geometry_registry is not None and args.conflict_zone_entry_id
            else None
        )
    except WorkedExampleProcessTraceValidationError as exc:
        parser.error(str(exc))

    output = write_worked_example_process_trace(
        args.input,
        args.out,
        route=route,
        conflict_zone=conflict_zone,
        focal_actor_id=args.focal_actor_id,
        focal_encounter_id=args.focal_encounter_id,
        pair_input_path=args.pair_input,
        encounter_report_path=args.encounter_report,
        pair_comparison_grain=args.pair_comparison_grain,
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
