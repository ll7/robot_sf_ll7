#!/usr/bin/env python3
"""Build a compact ten-class VV-5 diagnostic matrix from four packet receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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

# Packet-specific manifests are mandatory, even when another packet binds the same path.
REQUIRED_SOURCE_PATHS = {
    "first": (
        "scripts/validation/run_issue_9735_fault_injection.py",
        "robot_sf/sim/spawn_validation.py",
        "robot_sf/benchmark/spawn_validity.py",
        "robot_sf/analysis_workbench/release_row_anomalies.py",
        "maps/svg_maps/classic_head_on_corridor.svg",
    ),
    "geometry_radius": (
        "scripts/validation/run_issue_9735_geometry_radius_diagnostic.py",
        "robot_sf/benchmark/spawn_preflight.py",
        "robot_sf/benchmark/camera_ready/_route_clearance.py",
        "robot_sf/scenario_certification/feasibility_oracle.py",
        "robot_sf/benchmark/map_runner/map_runner.py",
        "robot_sf/planner/socnav_prediction.py",
        "robot_sf/analysis_workbench/release_row_anomalies.py",
        "tests/metamorphic/test_planner_unit_consistency.py",
        "configs/algos/prediction_planner_camera_ready.yaml",
        "configs/scenarios/single/francis2023_narrow_doorway.yaml",
        "maps/svg_maps/francis2023/francis2023_narrow_doorway.svg",
    ),
    "force": (
        "robot_sf/planner/socnav_social_force.py",
        "robot_sf/planner/socnav_base.py",
        "robot_sf/sim/simulator.py",
        "robot_sf/ped_npc/ped_robot_force.py",
        "robot_sf/benchmark/metrics.py",
    ),
    "remaining": (
        "configs/algos/hybrid_rule_v3_teb_like_rollout.yaml",
        "configs/algos/hybrid_rule_v4_clearance_braking.yaml",
        "robot_sf/planner/hybrid_rule_local_planner.py",
        "robot_sf/planner/socnav_social_force.py",
        "robot_sf/benchmark/map_runner/map_runner.py",
        "robot_sf/analysis_workbench/release_row_anomalies.py",
        "robot_sf/robot/differential_drive.py",
    ),
}


def _sha(path: Path) -> str:
    """Return the SHA-256 of one raw packet."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _immutable_commit(repo: Path, value: str, label: str) -> str:
    """Require an existing commit object named by its full immutable SHA."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"resolvable immutable {label} commit required")
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{value}^{{commit}}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or result.stdout.strip() != value:
        raise ValueError(f"resolvable immutable {label} commit required")
    return value


def _verified_sources(
    packets: dict[str, dict[str, Any]], source_commit: str, *, producer_commit: str | None = None
) -> dict[str, str]:
    """Verify complete manifests; historical producers require an explicit immutable pin."""
    repo = Path(__file__).resolve().parents[2]
    _immutable_commit(repo, source_commit, "source")
    if producer_commit is not None:
        _immutable_commit(repo, producer_commit, "producer")
    sources: dict[str, str] = {}

    def insert(path: str, expected: str) -> None:
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"invalid packet source digest: {path}")
        if path in sources and sources[path] != expected:
            raise ValueError(f"conflicting packet source digest: {path}")
        sources[path] = expected

    for name, required in REQUIRED_SOURCE_PATHS.items():
        manifest = (
            packets[name]["source"]["sha256"] if name == "first" else packets[name]["source_sha256"]
        )
        if not isinstance(manifest, dict) or set(manifest) != set(required):
            raise ValueError(f"required source roster differs for packet: {name}")
        for path, expected in manifest.items():
            insert(path, expected)
    insert(
        "scripts/validation/run_issue_9735_force_diagnostic.py",
        packets["force"]["fixture_script_sha256"],
    )
    insert(
        "scripts/validation/run_issue_9735_braking_goal_diagnostic.py",
        packets["remaining"]["script_sha256"],
    )
    for path, expected in sources.items():
        ref = producer_commit if path in NEW_SCRIPT_PATHS else source_commit
        content = (
            (repo / path).read_bytes()
            if ref is None
            else subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=repo)
        )
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"packet source mismatch: {path}")
    return sources


def _boolean(value: Any) -> bool:
    """Accept JSON booleans only, without Python truthiness coercion."""
    if type(value) is not bool:
        raise ValueError("detection/outcome must be a JSON boolean")
    return value


def _outcome(claimed: Any, clean_pass: Any, mutant_pass: Any) -> None:
    """Require a valid clean control and a detection flag consistent with outcomes."""
    claimed = _boolean(claimed)
    clean_pass = _boolean(clean_pass)
    mutant_pass = _boolean(mutant_pass)
    if not clean_pass or claimed != (clean_pass and not mutant_pass):
        raise ValueError("claimed detection contradicts clean/mutant outcome")


def _validate_gap(clean: dict, mutant: dict, checks: dict) -> None:
    """Validate each geometry verdict without promoting the inconclusive full oracle."""
    for side in (clean, mutant):
        if side["route_gate"] not in ("pass", "blocked") or side["preflight"][
            "overall_status"
        ] not in ("valid", "blocked"):
            raise ValueError("unknown geometry outcome")
    _outcome(
        checks["route_clearance"], clean["route_gate"] == "pass", mutant["route_gate"] == "pass"
    )
    _outcome(
        checks["spawn_matrix_preflight"],
        clean["preflight"]["overall_status"] == "valid",
        mutant["preflight"]["overall_status"] == "valid",
    )
    for side in (clean, mutant):
        _boolean(side["oracle"]["geometric"]["route_geometrically_feasible"])
    geometric = (
        clean["oracle"]["geometric"]["route_geometrically_feasible"]
        and not mutant["oracle"]["geometric"]["route_geometrically_feasible"]
    )
    full = (
        clean["oracle"]["status"] == "feasible"
        and mutant["oracle"]["status"] == "infeasible_by_construction"
    )
    for key, expected in (
        ("feasibility_oracle_geometric_subcheck", geometric),
        ("feasibility_oracle_full_verdict", full),
    ):
        if key in checks and checks[key] != expected:
            raise ValueError("claimed oracle detection contradicts outcome")


def _validate_radius(clean: dict, mutant: dict, checks: dict) -> None:
    """Check physical-unit and row-anomaly flags against their paired outcomes."""
    for side in (clean, mutant):
        if not isinstance(side["unit_audit_violations"], list):
            raise ValueError("unit audit outcome must be a list")
    _outcome(
        checks["planner_unit_audit_rule"],
        not clean["unit_audit_violations"],
        not mutant["unit_audit_violations"],
    )
    clean_pass = not _boolean(clean["row_anomaly_gate_blocked"])
    mutant_pass = not _boolean(mutant["row_anomaly_gate_blocked"])
    if "release_row_anomalies" in checks:
        _outcome(checks["release_row_anomalies"], clean_pass, mutant_pass)


def _validate_outcomes(packets: dict[str, dict[str, Any]]) -> None:
    """Validate every scored relationship before compacting or counting detections."""
    for case in packets["first"]["results"]:
        field = "gate_blocked" if case["fault"] == "collision_total_doubled" else "invalid_run"
        _outcome(
            case["detected"],
            not _boolean(case["control"][field]),
            not _boolean(case["mutant"][field]),
        )
    for fault, case in packets["geometry_radius"]["cases"].items():
        for value in case["detected_by"].values():
            _boolean(value)
        validate = _validate_gap if fault == "infeasible_gap" else _validate_radius
        validate(case["control"], case["mutant"], case["detected_by"])
    for case in packets["force"]["cases"]:
        field = (
            "invariance_pass"
            if case["fault"] == "per_cell_wall_force_sum"
            else "metrics_validator_pass"
        )
        _outcome(case["detected"], case["clean"][field], case["mutant"][field])
    for case in packets["remaining"]["cases"]:
        _outcome(case["detected"], case["clean"]["pass"], case["mutant"]["pass"])
        if case["fault"] == "planner_goal_direction_flipped":
            for side in (case["clean"], case["mutant"]):
                if _boolean(side["pass"]) != (not _boolean(side["gate_blocked"])):
                    raise ValueError("goal gate contradicts pass outcome")
            if (
                case["detected"]
                and "pedestrian_free_baseline_regression" not in case["mutant"]["finding_ids"]
            ):
                raise ValueError("goal detection lacks regression outcome")


def _first_case(case: dict[str, Any]) -> dict[str, Any]:
    """Compact the original four-class checker results."""
    fault = case["fault"]
    return {
        "fault": fault,
        "packet": "first",
        "check": case["check"],
        "clean": case["control"],
        "mutant": case["mutant"],
        "detected": _boolean(case["detected"]),
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
            "detected": _boolean(checks["route_clearance"])
            and _boolean(checks["spawn_matrix_preflight"]),
            "inconclusive_checks": ["feasibility_oracle_full_verdict"],
            "boundary": case["oracle_boundary"],
        }
    return {
        "fault": fault,
        "packet": "geometry_radius",
        "check": "prediction_planner physical-unit audit",
        "clean": case["control"],
        "mutant": case["mutant"],
        "detected": _boolean(checks["planner_unit_audit_rule"]),
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
            "detected": _boolean(case["detected"]),
            "boundary": case["boundary"],
        }
    return {
        "fault": fault,
        "packet": "force",
        "check": "robot_force_reductions with no expected-active roster binding",
        "clean": case["clean"],
        "mutant": case["mutant"],
        "detected": _boolean(case["detected"]),
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
            "detected": _boolean(case["detected"]),
            "boundary": case["boundary"],
        }
    return {
        "fault": case["fault"],
        "packet": "remaining",
        "check": "paired pedestrian-free baseline regression",
        "clean": case["clean"],
        "mutant": case["mutant"],
        "baseline_goal": case["baseline_goal"],
        "detected": _boolean(case["detected"]),
        "boundary": case["boundary"],
    }


def _gather_cases(packets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compact all packet classes, rejecting a repeated fault identity."""
    _validate_outcomes(packets)
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


def _packet_heads(packets: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Keep execution HEADs as provenance; content hashes bind tested source."""
    heads = {
        "geometry_radius": packets["geometry_radius"]["source_commit"],
        "force": packets["force"]["head"],
        "remaining": packets["remaining"]["head"],
    }
    if any(
        not isinstance(head, str) or re.fullmatch(r"[0-9a-f]{40}", head) is None
        for head in heads.values()
    ):
        raise ValueError("packet execution head must be a full Git SHA")
    return heads


def main() -> None:
    """Read four packets, verify all ten classes, and write a compact matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "geometry-radius", "force", "remaining"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--producer-commit",
        help="Full immutable Git SHA for historical producer bytes; omitted means current producer files",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "first": args.first,
        "geometry_radius": args.geometry_radius,
        "force": args.force,
        "remaining": args.remaining,
    }
    packets = {name: json.loads(path.read_text()) for name, path in paths.items()}
    source_sha256 = _verified_sources(
        packets, args.source_commit, producer_commit=args.producer_commit
    )
    packet_heads = _packet_heads(packets)
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
        "producer_commit": args.producer_commit,
        "packet_execution_heads": packet_heads,
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
