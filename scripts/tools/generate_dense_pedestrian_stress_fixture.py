#!/usr/bin/env python3
"""Generate a multi-pedestrian dense-stress trace fixture for issue #2765.

Creates a deterministic ``simulation_trace_export.v1`` trace with three
pedestrians converging within 2 m of the robot at overlapping time windows.
This is diagnostic stress evidence only, not a full planner benchmark.

Usage::

    uv run python scripts/tools/generate_dense_pedestrian_stress_fixture.py
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1"
SOURCES_DIR = FIXTURE_DIR / "sources"

DT_S = 0.1
TOTAL_STEPS = 20

PEDESTRIANS: list[dict[str, Any]] = [
    {
        "id": 27650,
        "label": "ped_a_crossing_from_above",
        "start_pos": [1.5, 1.0],
        "velocity": [0.0, -0.8],
        "description": "Crosses the robot path from the north side at close range",
    },
    {
        "id": 27651,
        "label": "ped_b_crossing_from_below",
        "start_pos": [1.8, -0.8],
        "velocity": [0.0, 0.7],
        "description": "Crosses the robot path from the south side at close range",
    },
    {
        "id": 27652,
        "label": "ped_c_approaching_head_on",
        "start_pos": [3.0, 0.0],
        "velocity": [-0.3, 0.0],
        "description": "Approaches the robot head-on at close range",
    },
]

ROBOT_START = [0.0, 0.0]
ROBOT_VELOCITY = [1.0, 0.0]


def _position_at(start: list[float], velocity: list[float], step: int) -> list[float]:
    """Compute position at a given step using constant velocity."""
    return [
        round(start[0] + velocity[0] * step * DT_S, 4),
        round(start[1] + velocity[1] * step * DT_S, 4),
    ]


def _distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _build_dense_stress_trace() -> dict[str, Any]:
    """Build the multi-pedestrian dense stress trace."""
    frames: list[dict[str, Any]] = []

    for step in range(TOTAL_STEPS):
        t = step * DT_S
        robot_pos = _position_at(ROBOT_START, ROBOT_VELOCITY, step)
        robot_vel = list(ROBOT_VELOCITY)

        ped_positions = [p["start_pos"] for p in PEDESTRIANS]
        ped_velocities = [p["velocity"] for p in PEDESTRIANS]

        pedestrians = []
        observed_pedestrians = []
        for ped_cfg, start, vel in zip(PEDESTRIANS, ped_positions, ped_velocities, strict=True):
            pos = _position_at(start, vel, step)
            velocity = [vel[0], vel[1]]
            pedestrians.append(
                {
                    "id": ped_cfg["id"],
                    "position": pos,
                    "velocity": velocity,
                }
            )
            observed_pedestrians.append(
                {
                    "id": ped_cfg["id"],
                    "position": list(pos),
                    "velocity": list(velocity),
                }
            )

        distances = [_distance(robot_pos, p["position"]) for p in pedestrians]
        min_dist = min(distances)

        stop_feasible = min_dist > 0.25
        yield_feasible = min_dist > 0.15

        if min_dist > 1.5:
            event = "approach_dense"
            v_lin = round(ROBOT_VELOCITY[0] * 0.8, 2)
        elif min_dist > 0.8:
            event = "dense_proximity"
            v_lin = round(ROBOT_VELOCITY[0] * 0.4, 2)
        elif min_dist > 0.3:
            event = "dense_critical"
            v_lin = round(ROBOT_VELOCITY[0] * 0.15, 2)
        else:
            event = "dense_stop"
            v_lin = 0.0

        frames.append(
            {
                "step": step,
                "time_s": round(t, 4),
                "robot": {
                    "position": robot_pos,
                    "heading": 0.0,
                    "velocity": robot_vel,
                },
                "pedestrians": pedestrians,
                "observed_pedestrians": observed_pedestrians,
                "occlusion_status": {
                    "ped_a": "visible",
                    "ped_b": "visible",
                    "ped_c": "visible",
                },
                "first_visible": step == 0,
                "conflict_timing": {
                    "time_to_conflict_s": round(max(0.0, min_dist / ROBOT_VELOCITY[0]), 4),
                    "stop_feasible": stop_feasible,
                    "yield_feasible": yield_feasible,
                },
                "planner": {
                    "selected_action": {"linear_velocity": v_lin, "angular_velocity": 0.0},
                    "event": event,
                },
            }
        )

    first_closest_step = None
    for frame in frames:
        robot_pos = frame["robot"]["position"]
        dists = [_distance(robot_pos, p["position"]) for p in frame["pedestrians"]]
        if min(dists) < 2.0 and first_closest_step is None:
            first_closest_step = frame["step"]

    return {
        "schema_version": "simulation_trace_export.v1",
        "trace_id": "issue_2765_dense_pedestrian_stress_seed2765_ep0000",
        "source": {
            "scenario_id": "dense_pedestrian_stress",
            "seed": 2765,
            "planner_id": "hybrid_rule_v0_minimal",
            "episode_id": "dense_pedestrian_stress_ep0000",
            "generated_by": "deterministic fixture generator for issue #2765",
        },
        "evidence_boundary": "smoke_diagnostic_only_not_benchmark_evidence",
        "coordinate_frame": "world",
        "units": {
            "position": "m",
            "heading": "rad",
            "time": "s",
            "velocity": "m/s",
        },
        "dense_stress_metadata": {
            "issue": 2765,
            "pedestrian_count": len(PEDESTRIANS),
            "pedestrians": [
                {
                    "id": p["id"],
                    "label": p["label"],
                    "description": p["description"],
                    "start_pos": p["start_pos"],
                    "velocity": p["velocity"],
                }
                for p in PEDESTRIANS
            ],
            "robot_start": ROBOT_START,
            "robot_velocity": ROBOT_VELOCITY,
            "dt_s": DT_S,
            "total_steps": TOTAL_STEPS,
            "first_closest_step": first_closest_step,
            "expected_failure_mode": "forecast_ambiguity_in_dense_scene",
            "evidence_boundary_note": "smoke_diagnostic_only_not_benchmark_evidence",
        },
        "frames": frames,
    }


def write_fixture() -> pathlib.Path:
    """Write the dense stress trace fixture and metadata to disk."""
    trace = _build_dense_stress_trace()

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    trace_path = FIXTURE_DIR / "dense_pedestrian_stress_episode_0000.json"
    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump(trace, fh, indent=2)

    meta = {
        "algorithm": "hand_authored_deterministic_fixture",
        "boundary": "smoke_diagnostic_only_not_benchmark_evidence",
        "episode_id": trace["source"]["episode_id"],
        "issue": 2765,
        "scenario": trace["source"]["scenario_id"],
        "seed": trace["source"]["seed"],
        "source_kind": "simulation_trace_export.v1 fixture",
        "expected_failure_mode": trace["dense_stress_metadata"]["expected_failure_mode"],
        "pedestrian_count": trace["dense_stress_metadata"]["pedestrian_count"],
        "evidence_boundary": trace["dense_stress_metadata"]["evidence_boundary_note"],
        "notes": [
            "Deterministic multi-pedestrian dense-stress fixture for issue #2765.",
            f"Expected failure mode: {trace['dense_stress_metadata']['expected_failure_mode']}.",
            "Three pedestrians converge within 2 m of the robot at overlapping time windows.",
            "This fixture is not paper-facing benchmark evidence.",
        ],
    }
    meta_path = SOURCES_DIR / "issue_2765_dense_pedestrian_stress_fixture_2765_ep0000.meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return trace_path


if __name__ == "__main__":
    path = write_fixture()
    print(f"Wrote {path}")
