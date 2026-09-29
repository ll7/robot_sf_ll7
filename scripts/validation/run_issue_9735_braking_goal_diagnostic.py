#!/usr/bin/env python3
"""Two controlled VV-5 mutants and a consolidated diagnostic-only matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from robot_sf.analysis_workbench.release_row_anomalies import analyze_release_rows
from robot_sf.benchmark.map_runner.map_runner import _goal_policy
from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    build_hybrid_rule_local_planner_config,
    stopping_distance,
)
from robot_sf.planner.socnav import (
    SOCIAL_FORCE_PLANNER_RESOLUTION_INDEPENDENT_V2,
    SocialForcePlannerAdapter,
    SocNavPlannerConfig,
)
from robot_sf.robot.differential_drive import DifferentialDriveSettings

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "configs/algos/hybrid_rule_v3_teb_like_rollout.yaml"
V4 = ROOT / "configs/algos/hybrid_rule_v4_clearance_braking.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hybrid_case() -> dict:
    drive = DifferentialDriveSettings()
    v3 = HybridRuleLocalPlannerAdapter(
        build_hybrid_rule_local_planner_config(yaml.safe_load(V3.read_text()))
    )
    v4 = HybridRuleLocalPlannerAdapter(
        build_hybrid_rule_local_planner_config(yaml.safe_load(V4.read_text()))
    )
    v4.bind_env(SimpleNamespace(simulator=SimpleNamespace(robots=[SimpleNamespace(config=drive)])))
    rows = []
    for clearance in (0.05, 2.05):
        center = clearance + 1.0 + 0.4
        state = {
            "robot_pos": np.array([0.0, 0.0]),
            "heading": 0.0,
            "current_speed": 0.0,
            "goal": np.array([20.0, 0.0]),
            "ped_pos": np.array([[center, 0.0]]),
            "ped_vel": np.array([[0.0, 0.0]]),
            "robot_radius": 1.0,
            "ped_radius": 0.4,
            "dt": 0.1,
            "observation": {},
        }
        clean_cap = float(v4._v4_human_speed_cap(state))
        mutant_cap = float(v3._human_speed_cap(center))
        rows.append(
            {
                "surface_clearance_m": clearance,
                "center_distance_m": center,
                "clean_v4_cap_mps": clean_cap,
                "mutant_v3_cap_mps": mutant_cap,
                "clean_stopping_distance_m": stopping_distance(
                    clean_cap, drive.max_linear_decel, 0.1
                ),
                "mutant_stopping_distance_m": stopping_distance(
                    mutant_cap, drive.max_linear_decel, 0.1
                ),
                "clean_level": v4._last_v4_speed_safety["level"],
            }
        )
    near, far = rows
    clean_pass = (
        near["clean_v4_cap_mps"] == 0.0
        and far["clean_stopping_distance_m"] + v4.config.v4_braking_margin
        <= far["surface_clearance_m"] + 1e-9
        and v4._v4_drive_limits()["max_linear_decel"] == drive.max_linear_decel
    )
    mutant_pass = (
        near["mutant_v3_cap_mps"] == 0.0
        and far["mutant_stopping_distance_m"] + v4.config.v4_braking_margin
        <= far["surface_clearance_m"] + 1e-9
        and v3.config.max_linear_decel == drive.max_linear_decel
    )
    return {
        "fault": "centre_distance_threshold_and_wrong_braking",
        "check": "production hybrid v4 surface-clearance cap, stopping_distance, and bound drive-limit invariant",
        "fixture": "radius-1.0 robot, radius-0.4 pedestrian, surface clearance 0.05 and 2.05 m, drive braking 1.0 m/s^2",
        "clean": {
            "variant": v4.config.planner_variant,
            "planner_decel_mps2": v4._v4_drive_limits()["max_linear_decel"],
            "pass": clean_pass,
        },
        "mutant": {
            "variant": v3.config.planner_variant,
            "planner_decel_mps2": v3.config.max_linear_decel,
            "pass": mutant_pass,
        },
        "samples": rows,
        "detected": bool(clean_pass and not mutant_pass),
        "boundary": "Component-level speed-cap check; no hybrid episode or doorway collision outcome is inferred.",
    }


def _trace(planner: str, *, flip_goal: bool = False) -> dict:
    position = np.array([0.0, 0.0])
    heading = 0.0
    speed = 0.0
    actual_goal = np.array([8.0, 0.0])
    dt = 0.1
    distance_travelled = 0.0
    adapter = (
        SocialForcePlannerAdapter(
            SocNavPlannerConfig(
                social_force_planner_version=SOCIAL_FORCE_PLANNER_RESOLUTION_INDEPENDENT_V2
            )
        )
        if planner == "social_force"
        else None
    )
    for index in range(120):
        # The mutation affects only the social-force planner's perceived goal.
        perceived_goal = position - (actual_goal - position) if flip_goal else actual_goal
        observation = {
            "robot": {
                "position": position.copy(),
                "heading": np.array([heading]),
                "speed": np.array([speed, 0.0]),
                "radius": np.array([1.0]),
            },
            "goal": {"current": perceived_goal, "next": perceived_goal + 1.0},
            "pedestrians": {
                "positions": np.zeros((0, 2)),
                "velocities": np.zeros((0, 2)),
                "count": np.array([0.0]),
            },
            "sim": {"timestep": np.array([dt])},
        }
        command_speed, angular = (
            adapter.plan(observation) if adapter else _goal_policy(observation, max_speed=1.0)
        )
        heading += float(angular) * dt
        speed = float(command_speed)
        move = speed * dt * np.array([math.cos(heading), math.sin(heading)])
        position += move
        distance_travelled += float(np.linalg.norm(move))
        if np.linalg.norm(actual_goal - position) < 0.4:
            break
    return {
        "steps": index + 1,
        "final_position_xy": position.tolist(),
        "path_length_m": distance_travelled,
        "initial_goal_distance_m": 8.0,
        "final_goal_distance_m": float(np.linalg.norm(actual_goal - position)),
        "route_complete": bool(np.linalg.norm(actual_goal - position) < 0.4),
    }


def _row(planner: str, result: dict) -> dict:
    success = result["route_complete"]
    return {
        "episode_id": f"vv5-goal-flip:111:{planner}",
        "scenario_id": "vv5_goal_flip_empty",
        "seed": 111,
        "algo": planner,
        "steps": result["steps"],
        "status": "success" if success else "failure",
        "outcome": {
            "route_complete": success,
            "collision_event": False,
            "timeout_event": not success,
        },
        "integrity": {"effective_view": {"observation_ped_count": 0}},
        "event_ledger": {"exact_events": {"invalid_run": False}},
        "metrics": {"success": float(success), "socnavbench_path_length": result["path_length_m"]},
    }


def _goal_case() -> dict:
    baseline = _trace("goal")
    clean_candidate = _trace("social_force")
    mutant_candidate = _trace("social_force", flip_goal=True)
    settings = {
        "require_preflight": False,
        "baseline_planner": "goal",
        "pedestrian_aware_planners": ["social_force"],
        "min_planners_per_cell": 2,
        "min_paired_cells": 1,
        "min_success_rate_gap": 0.2,
    }
    source = {"release_id": "synthetic-vv5-goal-flip", "planner_ids": ["goal", "social_force"]}
    clean_report = analyze_release_rows(
        [_row("goal", baseline), _row("social_force", clean_candidate)],
        config=settings,
        source=source,
    )
    mutant_report = analyze_release_rows(
        [_row("goal", baseline), _row("social_force", mutant_candidate)],
        config=settings,
        source=source,
    )
    clean_findings = sorted(item["detector_id"] for item in clean_report["findings"])
    mutant_findings = sorted(item["detector_id"] for item in mutant_report["findings"])
    clean_pass = not clean_report["gate"]["blocked"] and not clean_findings
    mutant_pass = not mutant_report["gate"]["blocked"]
    return {
        "fault": "planner_goal_direction_flipped",
        "check": "production social_force planner in a 12 s point-robot _trace plus paired pedestrian-free baseline regression anomaly detector",
        "fixture": "actual goal (8,0), no pedestrians/obstacles, seed identity 111; only social_force perceived goal is reflected across current robot position",
        "baseline_goal": baseline,
        "clean": {
            "_trace": clean_candidate,
            "gate_blocked": clean_report["gate"]["blocked"],
            "finding_ids": clean_findings,
            "pass": clean_pass,
        },
        "mutant": {
            "_trace": mutant_candidate,
            "gate_blocked": mutant_report["gate"]["blocked"],
            "finding_ids": mutant_findings,
            "pass": mutant_pass,
        },
        "detected": bool(
            clean_pass
            and not mutant_pass
            and "pedestrian_free_baseline_regression" in mutant_findings
        ),
        "boundary": "Short kinematic control _trace, not simulator/drive dynamics or a release episode. The diagnostic uses one paired seed with min_paired_cells=1; the nominal gate requires its pinned matrix.",
    }


def main() -> None:
    """Run the selected bounded diagnostic and write a checksummed report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [_hybrid_case(), _goal_case()]
    source_paths = [
        V3,
        V4,
        ROOT / "robot_sf/planner/hybrid_rule_local_planner.py",
        ROOT / "robot_sf/planner/socnav_social_force.py",
        ROOT / "robot_sf/benchmark/map_runner/map_runner.py",
        ROOT / "robot_sf/analysis_workbench/release_row_anomalies.py",
        ROOT / "robot_sf/robot/differential_drive.py",
    ]
    report = {
        "schema_version": "issue_9735_remaining_faults_diagnostic.v1",
        "evidence_tier": "diagnostic_fixture_only",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script_sha256": _sha(Path(__file__)),
        "source_sha256": {str(path.relative_to(ROOT)): _sha(path) for path in source_paths},
        "cases": cases,
        "sensitivity": {
            "detected": sum(bool(case["detected"]) for case in cases),
            "injected": len(cases),
        },
        "release_admitted": False,
    }
    output = output_dir / "remaining_report.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha(output),
                "sensitivity": report["sensitivity"],
                "cases": cases,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
