"""Run the config-first smoke for the public goal-candidate provider.

The report is implementation-integrity and candidate-coverage evidence only.  It does
not measure posterior accuracy, calibration, route preference, or planner quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf.prediction.goal_candidate_coverage import (
    OracleGoalTruth,
    evaluate_goal_candidate_coverage,
)
from robot_sf.prediction.goal_candidate_provider import (
    CandidatePathMode,
    GoalCandidateProviderConfig,
    GoalCandidateSource,
    PublicGoalCandidateRecord,
    generate_goal_candidates,
)

Point = tuple[float, float]


def _record(
    source: GoalCandidateSource,
    source_id: str,
    position: Point,
    *,
    route_signature: str | None = None,
    path: tuple[Point, ...] = (),
) -> PublicGoalCandidateRecord:
    """Build a public final-destination fixture record.

    Returns:
        Immutable public source record.
    """

    return PublicGoalCandidateRecord(
        source=source,
        source_id=source_id,
        position=position,
        route_signature=route_signature,
        path_points=path,
        path_mode=CandidatePathMode.PLANNER_PATH if path else CandidatePathMode.NONE,
        provenance_refs=(f"smoke:{source.value}",),
    )


def _candidate_payload(result) -> list[dict[str, Any]]:
    """Serialize candidate fields relevant to the research contract.

    Returns:
        Candidate receipt rows sorted by stable ID.
    """

    return [
        {
            "candidate_id": candidate.id,
            "role": candidate.role.value,
            "source": candidate.source,
            "position_xy": list(candidate.position) if candidate.position is not None else None,
            "direction_xy": list(candidate.direction) if candidate.direction is not None else None,
            "path_tangent_xy": list(candidate.path_tangent)
            if candidate.path_tangent is not None
            else None,
            "path_mode": candidate.path_mode,
            "feasibility_status": candidate.feasibility_status,
            "prior_weight": candidate.prior_weight,
            "parent_destination_id": candidate.parent_destination_id,
            "route_signature": candidate.route_signature,
            "provenance_refs": list(candidate.provenance_refs),
        }
        for candidate in sorted(result.candidate_set.candidates, key=lambda item: item.id)
    ]


def _scenario_records() -> dict[
    str, tuple[tuple[PublicGoalCandidateRecord, ...], tuple[Point, ...]]
]:
    """Return public records and static obstacle polygons for all smoke fixtures.

    Returns:
        Mapping from fixture name to ``(records, obstacles)``.
    """

    return {
        "open_room": ((), ()),
        "straight_corridor": (
            (
                _record(
                    GoalCandidateSource.CORRIDOR_ENDPOINT,
                    "corridor-east",
                    (12.0, 0.0),
                    path=((0.0, 0.0), (12.0, 0.0)),
                ),
                _record(
                    GoalCandidateSource.CORRIDOR_ENDPOINT,
                    "corridor-west",
                    (-12.0, 0.0),
                    path=((0.0, 0.0), (-12.0, 0.0)),
                ),
            ),
            (),
        ),
        "doorway": (
            (
                _record(
                    GoalCandidateSource.DOOR_OR_EXIT,
                    "door-left",
                    (12.0, 0.0),
                    route_signature="left-door",
                    path=((0.0, 0.0), (5.0, 2.0), (12.0, 0.0)),
                ),
                _record(
                    GoalCandidateSource.DOOR_OR_EXIT,
                    "door-right",
                    (12.0, 0.0),
                    route_signature="right-door",
                    path=((0.0, 0.0), (5.0, -2.0), (12.0, 0.0)),
                ),
            ),
            (((5.8, -0.5), (6.2, -0.5), (6.2, 0.5), (5.8, 0.5)),),
        ),
        "crossing": (
            tuple(
                _record(
                    GoalCandidateSource.CROSSING_ENTRY_EXIT,
                    f"exit-{index}",
                    point,
                )
                for index, point in enumerate(
                    ((12.0, 0.0), (0.0, 12.0), (-12.0, 0.0), (0.0, -12.0))
                )
            ),
            (),
        ),
        "multiple_homotopies": (
            (
                _record(
                    GoalCandidateSource.PEDESTRIAN_ROUTE_TERMINAL,
                    "route-left",
                    (12.0, 0.0),
                    route_signature="left",
                    path=((0.0, 0.0), (5.0, 2.0), (12.0, 0.0)),
                ),
                _record(
                    GoalCandidateSource.PEDESTRIAN_ROUTE_TERMINAL,
                    "route-right",
                    (12.0, 0.0),
                    route_signature="right",
                    path=((0.0, 0.0), (5.0, -2.0), (12.0, 0.0)),
                ),
                _record(
                    GoalCandidateSource.PEDESTRIAN_ROUTE_TERMINAL,
                    "route-middle",
                    (12.0, 0.0),
                    route_signature="middle",
                    path=((0.0, 0.0), (6.0, 0.0), (12.0, 0.0)),
                ),
            ),
            (),
        ),
        "blocked_destination": (
            (
                _record(
                    GoalCandidateSource.MAP_DESTINATION_ZONE,
                    "blocked-east",
                    (12.0, 0.0),
                    path=((0.0, 0.0), (12.0, 0.0)),
                ),
            ),
            (((5.0, -1.0), (7.0, -1.0), (7.0, 1.0), (5.0, 1.0)),),
        ),
        "true_goal_absent": (
            (
                _record(
                    GoalCandidateSource.MAP_DESTINATION_ZONE,
                    "public-east",
                    (12.0, 0.0),
                ),
            ),
            (),
        ),
    }


def _coverage_for(name: str, result) -> dict[str, Any] | None:
    """Evaluate only the smoke cases with deliberately supplied oracle truth."""

    if name == "true_goal_absent":
        return evaluate_goal_candidate_coverage(
            result.candidate_set,
            OracleGoalTruth(final_position=(-12.0, 0.0)),
        ).to_dict()
    if name == "doorway":
        return evaluate_goal_candidate_coverage(
            result.candidate_set,
            OracleGoalTruth(
                final_position=(12.0, 0.0),
                direction=(0.0, 1.0),
                route_signature="left-door",
                observed_position_global=(0.0, 0.0),
            ),
        ).to_dict()
    return None


def build_smoke_report() -> dict[str, Any]:
    """Build the deterministic fixture report with runtime diagnostics.

    Returns:
        JSON-safe report for implementation-integrity and coverage inspection.
    """

    config = GoalCandidateProviderConfig(homotopy_count=2, open_ray_count=4)
    scenarios = _scenario_records()
    rows: dict[str, Any] = {}
    for name, (records, obstacles) in scenarios.items():
        result = generate_goal_candidates(
            records,
            config=config,
            observed_position_global=(0.0, 0.0),
            obstacles=obstacles,
        )
        rows[name] = {
            "candidate_set_digest": result.candidate_set_digest,
            "candidate_count": len(result.candidate_set.candidates),
            "candidate_cap": config.active_waypoint_cap
            + config.final_destination_cap
            + config.open_ray_cap
            + int(config.unknown_enabled),
            "candidate_rows": _candidate_payload(result),
            "source_statuses": [status.to_dict() for status in result.source_statuses],
            "rejected_records": [record.to_dict() for record in result.rejected_records],
            "coverage": _coverage_for(name, result),
            "runtime_ms": result.runtime_ms,
            "map_digest": result.map_digest,
            "config_hash": result.config_hash,
        }
    return {
        "schema_version": "goal_candidate_provider_smoke.v1",
        "claim_boundary": "implementation_integrity_and_candidate_coverage_only",
        "config": config.to_dict(),
        "scenarios": rows,
    }


def _parse_args() -> argparse.Namespace:
    """Parse smoke output options.

    Returns:
        Parsed command-line options.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, help="Optional path for the JSON receipt")
    return parser.parse_args()


def main() -> None:
    """Run the smoke and print its JSON receipt."""

    args = _parse_args()
    report = build_smoke_report()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
