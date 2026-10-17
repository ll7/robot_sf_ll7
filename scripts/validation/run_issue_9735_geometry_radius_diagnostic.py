#!/usr/bin/env python3
"""Throwaway #9735 geometry/radius injections; never writes release inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from robot_sf.analysis_workbench.release_row_anomalies import analyze_release_rows
from robot_sf.benchmark.camera_ready._route_clearance import (
    RouteClearanceError,
    _assert_route_clearance_feasible,
    _build_route_clearance_warnings,
)
from robot_sf.benchmark.map_runner.map_runner import _build_socnav_config
from robot_sf.benchmark.spawn_preflight import _check_release_scenario
from robot_sf.scenario_certification.feasibility_oracle import (
    FeasibilityOracleConfig,
    feasibility_verdict_to_dict,
    make_envelope_scenario,
    run_feasibility_oracle,
)
from robot_sf.training.scenario_loader import load_scenarios
from tests.metamorphic.test_planner_unit_consistency import DRIVE, _field_violations


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path, out: Path, *, gap: float) -> tuple[Path, Path]:
    source_map = root / "maps/svg_maps/francis2023/francis2023_narrow_doorway.svg"
    source_scenario = root / "configs/scenarios/single/francis2023_narrow_doorway.yaml"
    data = source_map.read_text()
    if (
        'x="15" y="1" width="1" height="3"' not in data
        or 'x="15" y="6" width="1" height="3"' not in data
    ):
        raise RuntimeError("source doorway rectangles changed; _fixture transform unreviewed")
    low_h = (8.0 - gap) / 2.0
    high_y = 1.0 + low_h + gap
    data = data.replace(
        'x="15" y="1" width="1" height="3"', f'x="15" y="1" width="1" height="{low_h:g}"'
    )
    data = data.replace(
        'x="15" y="6" width="1" height="3"', f'x="15" y="{high_y:g}" width="1" height="{low_h:g}"'
    )
    map_path = out / f"doorway_gap_{gap:g}m.svg"
    map_path.write_text(data)
    loaded = yaml.safe_load(source_scenario.read_text())
    scenario = loaded["scenarios"][0]
    scenario["name"] = f"vv5_doorway_gap_{gap:g}m"
    scenario["map_file"] = str(map_path)
    scenario["single_pedestrians"] = []
    scenario["seeds"] = [111]
    scenario["simulation_config"]["ped_density"] = 0.0
    scenario["robot_config"] = {"radius": 1.0}
    scenario["metadata"] = {"diagnostic_fixture": "issue_9735_throwaway"}
    path = out / f"scenario_gap_{gap:g}m.yaml"
    path.write_text(yaml.safe_dump({"scenarios": [scenario]}, sort_keys=True))
    return path, map_path


def _geometry_case(path: Path) -> dict:
    scenario = load_scenarios(path)[0]
    warnings = _build_route_clearance_warnings([dict(scenario)])
    try:
        _assert_route_clearance_feasible(warnings)
        route_gate = "pass"
    except RouteClearanceError:
        route_gate = "blocked"
    row = _check_release_scenario((dict(scenario), str(path), (111,), 0.1, 1, 0.1, False))["rows"][
        0
    ]
    oracle = run_feasibility_oracle(
        make_envelope_scenario(scenario, envelope_radius_m=1.0),
        config=FeasibilityOracleConfig(scenario_path=path, rollout_seed=111),
        envelope_radius_m=1.0,
    )
    oracle_dict = feasibility_verdict_to_dict(oracle, issue="9735")
    return {
        "route_gate": route_gate,
        "route_warnings": warnings,
        "preflight": {
            key: row.get(key)
            for key in (
                "overall_status",
                "robot_radius_m",
                "reset_clearance",
                "footprint_reachability",
                "passage_width",
                "respawn_safety",
                "cell_error",
            )
        },
        "oracle": {
            key: oracle_dict.get(key) for key in ("status", "feasible", "geometric", "completion")
        },
    }


def _radius_case(config_path: Path) -> dict:
    # prediction_planner is in the 14-arm release roster. Its own predictive
    # radius enters collision-risk scoring, independent of simulator geometry.
    cfg = _build_socnav_config(yaml.safe_load(config_path.read_text()))
    violations = list(
        _field_violations(
            "vv5_prediction_planner_fixture",
            "predictive_robot_radius",
            (cfg.predictive_robot_radius,),
        )
    )
    rows = [
        {
            "scenario_id": "vv5_radius_fixture",
            "seed": 111,
            "algo": planner,
            "steps": 100,
            "status": "success",
            "outcome": {"route_complete": True, "collision_event": False, "timeout_event": False},
            "integrity": {"effective_view": {"observation_ped_count": 1}},
            "event_ledger": {"exact_events": {"invalid_run": False}},
            "metrics": {
                "success": 1.0,
                "ped_collision_count": 0.0,
                "obstacle_collision_count": 0.0,
                "agent_collision_count": 0.0,
                "total_collision_count": 0.0,
                "collisions": 0.0,
            },
            "planner_config": {"predictive_robot_radius": cfg.predictive_robot_radius}
            if planner == "prediction_planner"
            else {},
        }
        for planner in ("goal", "prediction_planner")
    ]
    anomalies = analyze_release_rows(
        rows,
        config={
            "require_preflight": False,
            "pedestrian_aware_planners": [],
            "min_planners_per_cell": 2,
        },
        source={
            "release_id": "synthetic-vv5-radius",
            "planner_ids": ["goal", "prediction_planner"],
        },
    )
    return {
        "drive_radius_m": DRIVE.radius,
        "planner_radius_m": cfg.predictive_robot_radius,
        "unit_audit_violations": violations,
        "row_anomaly_gate_blocked": anomalies["gate"]["blocked"],
        "row_anomaly_findings": sorted(item["detector_id"] for item in anomalies["findings"]),
    }


def main() -> None:
    """Run the selected bounded diagnostic and write a checksummed report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--fault",
        action="append",
        choices=("infeasible_gap", "planner_only_radius_halved"),
        required=True,
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fixtures = out / "fixtures"
    fixtures.mkdir(exist_ok=True)
    report = {
        "schema": "issue_9735_fault_injection_extension.v1",
        "evidence_tier": "diagnostic_fixture_only",
        "source_commit": __import__("subprocess")
        .check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True)
        .strip(),
        "selected_faults": args.fault,
        "cases": {},
    }
    if "infeasible_gap" in args.fault:
        clean, clean_map = _fixture(root, fixtures, gap=3.0)
        mutant, mutant_map = _fixture(root, fixtures, gap=1.8)
        control = _geometry_case(clean)
        injected = _geometry_case(mutant)
        report["cases"]["infeasible_gap"] = {
            "control": control,
            "mutant": injected,
            "detected_by": {
                "route_clearance": control["route_gate"] == "pass"
                and injected["route_gate"] == "blocked",
                "spawn_matrix_preflight": control["preflight"]["overall_status"] == "valid"
                and injected["preflight"]["overall_status"] == "blocked",
                "feasibility_oracle_geometric_subcheck": control["oracle"]["geometric"][
                    "route_geometrically_feasible"
                ]
                is True
                and injected["oracle"]["geometric"]["route_geometrically_feasible"] is False,
                "feasibility_oracle_full_verdict": control["oracle"]["status"] == "feasible"
                and injected["oracle"]["status"] == "infeasible_by_construction",
            },
            "fixture_sha256": {p.name: _sha(p) for p in (clean, clean_map, mutant, mutant_map)},
            "oracle_boundary": "The geometric subcheck sees the injected gap. The full clean oracle is blocked by unrelated ancillary missing-data status and is not counted as a full-oracle detection.",
        }
    if "planner_only_radius_halved" in args.fault:
        source_config = root / "configs/algos/prediction_planner_camera_ready.yaml"
        clean_config = fixtures / "prediction_planner_radius_1m.yaml"
        mutant_config = fixtures / "prediction_planner_radius_0p5m.yaml"
        for path, radius in ((clean_config, 1.0), (mutant_config, 0.5)):
            values = yaml.safe_load(source_config.read_text())
            values["predictive_robot_radius"] = radius
            path.write_text(yaml.safe_dump(values, sort_keys=True))
        control = _radius_case(clean_config)
        injected = _radius_case(mutant_config)
        report["cases"]["planner_only_radius_halved"] = {
            "control": control,
            "mutant": injected,
            "detected_by": {
                "planner_unit_audit_rule": not control["unit_audit_violations"]
                and bool(injected["unit_audit_violations"]),
                "release_row_anomalies": not control["row_anomaly_gate_blocked"]
                and injected["row_anomaly_gate_blocked"],
            },
            "check_boundary": "The geometry preflight/oracle check simulator radius. The unit audit rule is the relevant planner-config detector. The row detector sees a synthetic planner_config provenance field but no trajectory; this _fixture exercises production prediction_planner config parsing, not a full release episode.",
            "fixture_sha256": {p.name: _sha(p) for p in (clean_config, mutant_config)},
            "historical_source_radius_m": yaml.safe_load(source_config.read_text())[
                "predictive_robot_radius"
            ],
        }
    sources = [
        Path(__file__),
        root / "robot_sf/benchmark/spawn_preflight.py",
        root / "robot_sf/benchmark/camera_ready/_route_clearance.py",
        root / "robot_sf/scenario_certification/feasibility_oracle.py",
        root / "robot_sf/benchmark/map_runner/map_runner.py",
        root / "robot_sf/planner/socnav_prediction.py",
        root / "robot_sf/analysis_workbench/release_row_anomalies.py",
        root / "tests/metamorphic/test_planner_unit_consistency.py",
        root / "configs/algos/prediction_planner_camera_ready.yaml",
        root / "configs/scenarios/single/francis2023_narrow_doorway.yaml",
        root / "maps/svg_maps/francis2023/francis2023_narrow_doorway.svg",
    ]
    report["source_sha256"] = {
        str(p.relative_to(root)) if p.is_relative_to(root) else p.name: _sha(p) for p in sources
    }
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": str(path),
                "case_detection": {k: v["detected_by"] for k, v in report["cases"].items()},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
