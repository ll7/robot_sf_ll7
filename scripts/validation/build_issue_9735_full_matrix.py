#!/usr/bin/env python3
"""Build a compact ten-class VV-5 diagnostic matrix from four packet receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ORDER = (
    "reset_pedestrian_overlap",
    "respawn_onto_robot",
    "robot_start_inside_wall_radius",
    "per_cell_wall_force_sum",
    "centre_distance_threshold_and_wrong_braking",
    "infeasible_gap",
    "planner_only_radius_halved",
    "planner_goal_direction_flipped",
    "pedestrian_robot_force_disabled",
    "collision_total_doubled",
)
NEW_SCRIPT_PATHS = {
    "scripts/validation/run_issue_9735_geometry_radius_diagnostic.py",
    "scripts/validation/run_issue_9735_force_diagnostic.py",
    "scripts/validation/run_issue_9735_braking_goal_diagnostic.py",
}


def _sha(path: Path) -> str:
    """Return the SHA-256 of one raw packet."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_sources(packets: dict[str, dict[str, Any]], source_commit: str) -> dict[str, str]:
    """Verify packet source/config bytes against their source commit or new scripts."""
    repo = Path(__file__).resolve().parents[2]
    sources = dict(packets["first"]["source"]["sha256"])
    for name in ("geometry_radius", "force", "remaining"):
        for path, expected in packets[name]["source_sha256"].items():
            if path in sources and sources[path] != expected:
                raise ValueError(f"conflicting packet source digest: {path}")
            sources[path] = expected
    for path, expected in sources.items():
        content = (
            (repo / path).read_bytes()
            if path in NEW_SCRIPT_PATHS
            else subprocess.check_output(["git", "show", f"{source_commit}:{path}"], cwd=repo)
        )
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"packet source mismatch: {path}")
    return sources


def _first_case(case: dict[str, Any]) -> dict[str, Any]:
    """Compact the original four-class checker results."""
    fault = case["fault"]
    return {
        "fault": fault,
        "packet": "first",
        "check": case["check"],
        "clean": case["control"],
        "mutant": case["mutant"],
        "detected": bool(case["detected"]),
        "boundary": (
            "Synthetic respawn event plus matching collision, without placement execution."
            if fault == "respawn_onto_robot"
            else "Synthetic checker fixture; no nominal episode."
        ),
    }


def _geometry_case(fault: str, case: dict[str, Any]) -> dict[str, Any]:
    """Compact the route and planner-radius diagnostic results."""
    checks = case["detected_by"]
    if fault == "infeasible_gap":
        clean = case["control"]
        mutant = case["mutant"]
        return {
            "fault": fault,
            "packet": "geometry_radius",
            "check": "route_clearance + spawn_matrix_preflight",
            "clean": {
                "route_gate": clean["route_gate"],
                "preflight_status": clean["preflight"]["overall_status"],
                "oracle_geometric_feasible": clean["oracle"]["geometric"][
                    "route_geometrically_feasible"
                ],
                "oracle_full_status": clean["oracle"]["status"],
            },
            "mutant": {
                "route_gate": mutant["route_gate"],
                "preflight_status": mutant["preflight"]["overall_status"],
                "oracle_geometric_feasible": mutant["oracle"]["geometric"][
                    "route_geometrically_feasible"
                ],
                "oracle_full_status": mutant["oracle"]["status"],
            },
            "detected": bool(checks["route_clearance"] and checks["spawn_matrix_preflight"]),
            "inconclusive_checks": ["feasibility_oracle_full_verdict"],
            "boundary": case["oracle_boundary"],
        }
    return {
        "fault": fault,
        "packet": "geometry_radius",
        "check": "prediction_planner physical-unit audit",
        "clean": case["control"],
        "mutant": case["mutant"],
        "detected": bool(checks["planner_unit_audit_rule"]),
        "missed_checks": ["release_row_anomalies"],
        "boundary": "Synthetic planner config; no planner episode. Geometry checks use simulator radius.",
    }


def _force_case(case: dict[str, Any]) -> dict[str, Any]:
    """Compact the wall-force and pedestrian-force diagnostic results."""
    fault = case["fault"]
    if fault == "per_cell_wall_force_sum":
        clean = case["clean"]
        mutant = case["mutant"]
        return {
            "fault": fault,
            "packet": "force",
            "check": "social_force obstacle-force resolution invariance (relative change <= 0.25)",
            "clean": {
                "version": clean["version"],
                "coarse_force_xy": clean["coarse_force_xy"],
                "fine_force_xy": clean["fine_force_xy"],
                "relative_change": clean["relative_change"],
                "pass": clean["invariance_pass"],
            },
            "mutant": {
                "version": mutant["version"],
                "coarse_force_xy": mutant["coarse_force_xy"],
                "fine_force_xy": mutant["fine_force_xy"],
                "relative_change": mutant["relative_change"],
                "pass": mutant["invariance_pass"],
            },
            "detected": bool(case["detected"]),
            "boundary": case["boundary"],
        }
    return {
        "fault": fault,
        "packet": "force",
        "check": "robot_force_reductions with no expected-active roster binding",
        "clean": case["clean"],
        "mutant": case["mutant"],
        "detected": bool(case["detected"]),
        "proposed_check": "Bind approved per-scenario PRF-active state to runtime config and component roster across exposed steps; allow legitimate zero force.",
        "boundary": case["boundary"],
    }


def _remaining_case(case: dict[str, Any]) -> dict[str, Any]:
    """Compact the hybrid-braking and goal-flip diagnostic results."""
    if case["fault"] == "centre_distance_threshold_and_wrong_braking":
        return {
            "fault": case["fault"],
            "packet": "remaining",
            "check": "hybrid clearance cap and real-drive stopping-distance invariant",
            "clean": case["clean"],
            "mutant": case["mutant"],
            "samples": case["samples"],
            "detected": bool(case["detected"]),
            "boundary": case["boundary"],
        }
    return {
        "fault": case["fault"],
        "packet": "remaining",
        "check": "paired pedestrian-free baseline regression",
        "clean": case["clean"],
        "mutant": case["mutant"],
        "baseline_goal": case["baseline_goal"],
        "detected": bool(case["detected"]),
        "boundary": case["boundary"],
    }


def _gather_cases(packets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compact all packet classes, rejecting a repeated fault identity."""
    cases: dict[str, dict[str, Any]] = {}

    def insert(case: dict[str, Any]) -> None:
        fault = case["fault"]
        if fault in cases:
            raise ValueError(f"duplicate fault class: {fault}")
        cases[fault] = case

    for case in packets["first"]["results"]:
        insert(_first_case(case))
    for fault, case in packets["geometry_radius"]["cases"].items():
        insert(_geometry_case(fault, case))
    for case in packets["force"]["cases"]:
        insert(_force_case(case))
    for case in packets["remaining"]["cases"]:
        insert(_remaining_case(case))
    return cases


def main() -> None:
    """Read four packets, verify all ten classes, and write a compact matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "geometry-radius", "force", "remaining"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "first": args.first,
        "geometry_radius": args.geometry_radius,
        "force": args.force,
        "remaining": args.remaining,
    }
    packets = {name: json.loads(path.read_text()) for name, path in paths.items()}
    source_sha256 = _verified_sources(packets, args.source_commit)
    for name in ("geometry_radius", "force", "remaining"):
        key = "source_commit" if name == "geometry_radius" else "head"
        if packets[name][key] != args.source_commit:
            raise ValueError(f"{name} source commit does not match the requested source")
    cases = _gather_cases(packets)
    if set(cases) != set(ORDER) or len(cases) != len(ORDER):
        raise ValueError("packet fault roster differs from the ten-class #9735 contract")
    matrix = [cases[fault] for fault in ORDER]
    detected = sum(case["detected"] for case in matrix)
    misses = [case["fault"] for case in matrix if not case["detected"]]
    if detected != 8 or misses != ["pedestrian_robot_force_disabled", "collision_total_doubled"]:
        raise ValueError(f"unexpected sensitivity or misses: {detected}/10, {misses}")
    report = {
        "schema_version": "issue_9735_full_fault_matrix_diagnostic.v1",
        "evidence_tier": "diagnostic_fixture_only",
        "source_commit": args.source_commit,
        "packet_sha256": {name: _sha(path) for name, path in paths.items()},
        "source_sha256": source_sha256,
        "claim_boundary": "Ten synthetic fixture classes on one source; differing checks and some one-seed/component-only probes. No nominal release rows or final 0.0.8 gate sensitivity.",
        "sensitivity": {
            "detected": detected,
            "injected": len(matrix),
            "fraction": detected / len(matrix),
        },
        "missed": misses,
        "matrix": matrix,
        "release_admitted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "report": str(args.output),
                "sha256": _sha(args.output),
                "sensitivity": report["sensitivity"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
