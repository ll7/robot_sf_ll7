#!/usr/bin/env python3
"""Run a bounded VV-5 fault-injection matrix on throwaway inputs.

This diagnostic command never patches production code or writes benchmark rows. Each
fault is selected with ``--fault`` (the default selects all implemented faults).
Detection requires a passing clean control and a failing mutant on the same check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from robot_sf.analysis_workbench.release_row_anomalies import analyze_release_rows
from robot_sf.benchmark.spawn_validity import build_spawn_validity
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.sim.spawn_validation import reset_spawn_clearance

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "maps/svg_maps/classic_head_on_corridor.svg"
CHECKS = ("spawn_validity", "release_row_anomalies")
FAULTS = (
    "reset_pedestrian_overlap",
    "respawn_onto_robot",
    "robot_start_inside_wall_radius",
    "collision_total_doubled",
)
DEFERRED_FAULTS = (
    "per_cell_wall_force_sum",
    "centre_distance_threshold_and_wrong_braking",
    "infeasible_gap",
    "planner_only_radius_halved",
    "planner_goal_direction_flipped",
    "pedestrian_robot_force_disabled",
)
PROPOSED_CHECK = (
    "Recompute total_collision_count and collisions from typed component counts and the "
    "event ledger; block any mismatch before release aggregation."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source() -> dict[str, Any]:
    paths = [
        Path(__file__),
        ROOT / "robot_sf/sim/spawn_validation.py",
        ROOT / "robot_sf/benchmark/spawn_validity.py",
        ROOT / "robot_sf/analysis_workbench/release_row_anomalies.py",
        MAP,
    ]
    return {
        "sha256": {path.relative_to(ROOT).as_posix(): _sha256(path) for path in paths},
    }


def _simulator(map_def: Any, *, robot_xy: tuple[float, float], ped_xy: tuple[float, float]) -> Any:
    return SimpleNamespace(
        map_def=map_def,
        config=SimpleNamespace(ped_radius=0.4),
        robots=[SimpleNamespace(pose=(robot_xy, 0.0), config=SimpleNamespace(radius=1.0))],
        ped_pos=[ped_xy],
    )


def _spawn_fault(fault: str, map_def: Any) -> dict[str, Any]:
    clean = reset_spawn_clearance(_simulator(map_def, robot_xy=(15.0, 20.0), ped_xy=(20.0, 20.0)))
    if fault == "reset_pedestrian_overlap":
        mutated = reset_spawn_clearance(
            _simulator(map_def, robot_xy=(15.0, 20.0), ped_xy=(15.5, 20.0))
        )
        clean_validity = build_spawn_validity(clean, [])
        mutant_validity = build_spawn_validity(mutated, [])
    elif fault == "robot_start_inside_wall_radius":
        mutated = reset_spawn_clearance(
            _simulator(map_def, robot_xy=(15.0, 2.2), ped_xy=(20.0, 20.0))
        )
        clean_validity = build_spawn_validity(clean, [])
        mutant_validity = build_spawn_validity(mutated, [])
    elif fault == "respawn_onto_robot":
        mutated = clean
        clean_validity = build_spawn_validity(clean, [], dt_seconds=0.1)
        mutant_validity = build_spawn_validity(
            mutated,
            [{"group_id": "fixture", "ped_rows": [0], "step": 5}],
            collision_events=[
                {
                    "collision_partner_type": "pedestrian",
                    "contact_partner_ids": [0],
                    "collision_time": 0.5,
                }
            ],
            dt_seconds=0.1,
        )
    else:
        raise ValueError(f"unsupported spawn fault: {fault}")
    clean_blocked = bool(clean_validity["invalid_run"])
    mutant_blocked = bool(mutant_validity["invalid_run"])
    return {
        "fault": fault,
        "check": "spawn_validity",
        "control": {
            "overlap": clean["overlap"],
            "invalid_run": clean_blocked,
        },
        "mutant": {
            "overlap": mutated["overlap"],
            "pedestrian_overlap": mutated["pedestrian_overlap"],
            "obstacle_overlap": mutated["obstacle_overlap"],
            "invalid_run": mutant_blocked,
            "respawn_overlap_collisions": len(mutant_validity["respawn_overlap_collisions"]),
        },
        "detected": not clean_blocked and mutant_blocked,
    }


def _row(planner: str, *, collision: bool, total: float) -> dict[str, Any]:
    return {
        "episode_id": f"fixture:111:{planner}",
        "scenario_id": "vv5-fixture",
        "seed": 111,
        "algo": planner,
        "steps": 100,
        "status": "collision" if collision else "success",
        "outcome": {
            "route_complete": not collision,
            "collision_event": collision,
            "timeout_event": False,
        },
        "integrity": {"effective_view": {"observation_ped_count": 1}},
        "event_ledger": {"exact_events": {"invalid_run": False}},
        "metrics": {
            "success": 0.0 if collision else 1.0,
            "ped_collision_count": total,
            "obstacle_collision_count": 0.0,
            "agent_collision_count": 0.0,
            "total_collision_count": total,
            "collisions": total,
        },
    }


def _metric_fault() -> dict[str, Any]:
    control = [
        _row("goal", collision=False, total=0.0),
        _row("social_force", collision=True, total=1.0),
    ]
    mutant = json.loads(json.dumps(control))
    mutant[1]["metrics"]["total_collision_count"] = 2.0
    mutant[1]["metrics"]["collisions"] = 2.0
    config = {
        "require_preflight": False,
        "pedestrian_aware_planners": [],
        "min_planners_per_cell": 2,
    }
    source = {"release_id": "synthetic-vv5", "planner_ids": ["goal", "social_force"]}
    clean_report = analyze_release_rows(control, config=config, source=source)
    mutant_report = analyze_release_rows(mutant, config=config, source=source)
    clean_blocked = bool(clean_report["gate"]["blocked"])
    mutant_blocked = bool(mutant_report["gate"]["blocked"])
    return {
        "fault": "collision_total_doubled",
        "check": "release_row_anomalies",
        "control": {
            "gate_blocked": clean_blocked,
            "finding_ids": sorted(item["detector_id"] for item in clean_report["findings"]),
            "total_collision_count": 1.0,
        },
        "mutant": {
            "gate_blocked": mutant_blocked,
            "finding_ids": sorted(item["detector_id"] for item in mutant_report["findings"]),
            "total_collision_count": 2.0,
            "component_sum": 1.0,
        },
        "detected": not clean_blocked and mutant_blocked,
        "proposed_check": PROPOSED_CHECK,
    }


def build_report(selected_faults: tuple[str, ...] = FAULTS) -> dict[str, Any]:
    """Run selected controlled mutants and return a deterministic detection matrix."""
    if not selected_faults or set(selected_faults) - set(FAULTS):
        raise ValueError("select at least one implemented fault")
    map_def = (
        convert_map(str(MAP))
        if any(fault != "collision_total_doubled" for fault in selected_faults)
        else None
    )
    results = [
        _metric_fault() if fault == "collision_total_doubled" else _spawn_fault(fault, map_def)
        for fault in FAULTS
        if fault in selected_faults
    ]
    detected = sum(result["detected"] for result in results)
    matrix = {
        result["fault"]: {
            check: ("detected" if result["detected"] else "missed")
            if result["check"] == check
            else "not_applicable"
            for check in CHECKS
        }
        for result in results
    }
    return {
        "schema_version": "issue_9735_fault_injection.v1",
        "review_marker": "AI-GENERATED NEEDS-REVIEW",
        "evidence_tier": "diagnostic_fixture_only",
        "claim_boundary": "Fixture-level checker sensitivity only; no nominal release or planner-behaviour claim.",
        "source": _source(),
        "selected_faults": [result["fault"] for result in results],
        "checks": list(CHECKS),
        "matrix": matrix,
        "sensitivity": {
            "detected": detected,
            "injected": len(results),
            "fraction": detected / len(results),
        },
        "results": results,
        "missed": [result["fault"] for result in results if not result["detected"]],
        "not_injected": list(DEFERRED_FAULTS),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact human review report from the machine matrix."""
    lines = [
        "<!-- AI-GENERATED (robot_sf#9735) - NEEDS-REVIEW -->",
        "# VV-5 fault-injection diagnostic (#9735)",
        "",
        report["claim_boundary"],
        "",
        f"Detection: {report['sensitivity']['detected']}/{report['sensitivity']['injected']} "
        "injected fixtures. Exact checker and map SHA-256 values appear below.",
        "",
        "| Fault | Spawn-validity check | Release-row anomalies |",
        "| --- | --- | --- |",
    ]
    for fault in report["selected_faults"]:
        result = report["matrix"][fault]
        lines.append(
            f"| `{fault}` | {result['spawn_validity']} | {result['release_row_anomalies']} |"
        )
    lines.extend(["", "## Misses and coverage boundary", ""])
    for result in report["results"]:
        if not result["detected"]:
            lines.append(
                f"- `{result['fault']}`: {result.get('proposed_check', 'investigate checker gap')}"
            )
    lines.append("- The collision-total miss is tracked as release-blocking issue #9855.")
    lines.append(
        "- The six remaining issue faults were not injected in this packet: "
        + ", ".join(f"`{fault}`" for fault in report["not_injected"])
        + "."
    )
    lines.extend(
        [
            "",
            "The controls and mutants use the production checker functions on synthetic inputs. "
            "This does not prove scenario-matrix coverage, runtime wiring, or causal attribution.",
            "",
            "## Source digests",
            "",
        ]
    )
    lines.extend(f"- `{path}`: `{digest}`" for path, digest in report["source"]["sha256"].items())
    return "\n".join(lines) + "\n"


def main() -> int:
    """Generate or verify the requested diagnostic report files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault", action="append", choices=FAULTS)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    selected = tuple(args.fault) if args.fault else FAULTS
    report = build_report(selected)
    content = {
        args.out_json: json.dumps(report, indent=2, sort_keys=True) + "\n",
        args.out_md: render_markdown(report),
    }
    if args.check:
        for path, expected in content.items():
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                parser.error(f"stale or missing report: {path}")
    else:
        for path, value in content.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")
    print(f"VV-5 diagnostic: {report['sensitivity']['detected']}/{len(report['results'])} detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
