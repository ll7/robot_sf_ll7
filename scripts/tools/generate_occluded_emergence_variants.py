"""Generate simulated occluded-emergence variant trace fixtures.

Produces five deterministic variant traces that explore different emergence
configurations beyond the single hand-authored issue #2756 fixture.

Variants:
  - left_close: pedestrian emerges from left at close distance
  - right_close: pedestrian emerges from right at close distance
  - late_visibility: pedestrian becomes visible late with minimal reaction time
  - slow_pedestrian: slow-moving pedestrian crossing path
  - fast_pedestrian: fast-moving pedestrian crossing path

All fixtures are diagnostic/stress evidence only.  They do not replace
the #2756 hand-authored fixture or the #2777 live replay.

Usage:
  uv run python scripts/tools/generate_occluded_emergence_variants.py
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1"
SOURCES_DIR = FIXTURE_DIR / "sources"

DT_S = 0.1
TOTAL_STEPS = 16


def _occluder_bounds(side: str) -> dict[str, float]:
    """Return occluder bounding box for the given emergence side."""
    if side == "left":
        return {"x_min": 0.8, "x_max": 1.05, "y_min": -0.8, "y_max": 0.8}
    if side == "right":
        return {"x_min": 1.5, "x_max": 1.75, "y_min": -0.8, "y_max": 0.8}
    return {"x_min": 1.0, "x_max": 1.25, "y_min": -0.8, "y_max": 0.8}


def _build_variant(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Build a single variant trace dict."""
    first_visible_step: int = cfg["first_visible_step"]
    ped_speed: float = cfg["ped_speed"]
    ped_initial_y: float = cfg["ped_initial_y"]
    ped_x: float = cfg["ped_x"]
    robot_v0: float = cfg["robot_v0"]
    side: str = cfg.get("side", "center")
    conflict_y: float = cfg.get("conflict_y", 0.0)

    frames: list[dict[str, Any]] = []
    for step in range(TOTAL_STEPS):
        t = step * DT_S
        robot_x = robot_v0 * t * 0.6
        robot_pos = [round(robot_x, 4), 0.0]

        ped_y = round(ped_initial_y + ped_speed * t, 4)
        ped_pos = [ped_x, ped_y]
        ped_vel = [0.0, ped_speed]

        observed: list[dict[str, Any]] = []
        occlusion_status = "occluded"
        first_visible = False
        if step >= first_visible_step:
            observed = [{"id": cfg["ped_id"], "position": list(ped_pos), "velocity": list(ped_vel)}]
            occlusion_status = "visible"
            first_visible = step == first_visible_step

        signed_dist_to_conflict = conflict_y - ped_y
        time_to_conflict = signed_dist_to_conflict / ped_speed if ped_speed > 0 else 999.0
        previous_time_to_conflict = 999.0
        if ped_speed > 0 and step > 0:
            previous_ped_y = ped_initial_y + ped_speed * ((step - 1) * DT_S)
            previous_time_to_conflict = (conflict_y - previous_ped_y) / ped_speed
        crosses_conflict = previous_time_to_conflict > 0.0 >= time_to_conflict
        # Simplified feasibility thresholds
        stop_feasible = time_to_conflict > cfg.get("stop_horizon_s", 0.2)
        yield_feasible = time_to_conflict > cfg.get("yield_horizon_s", 0.1)

        if step < first_visible_step:
            event = "approach_occluder"
            v_lin = round(robot_v0 * 0.8, 2)
        elif crosses_conflict:
            event = "conflict_time"
            v_lin = 0.0
        elif step == first_visible_step:
            event = "first_visible"
            v_lin = round(robot_v0 * 0.5, 2)
        elif stop_feasible:
            event = "yield_start"
            v_lin = round(robot_v0 * 0.3, 2)
        elif step > cfg.get("yield_feasible_before_step", 12):
            event = "resume"
            v_lin = round(robot_v0 * 0.4, 2)
        else:
            event = "yield"
            v_lin = round(robot_v0 * 0.15, 2)

        frames.append(
            {
                "step": step,
                "time_s": round(t, 4),
                "robot": {"position": robot_pos, "heading": 0.0, "velocity": [robot_v0, 0.0]},
                "pedestrians": [
                    {"id": cfg["ped_id"], "position": list(ped_pos), "velocity": list(ped_vel)}
                ],
                "observed_pedestrians": observed,
                "occlusion_status": {"emerging_ped": occlusion_status},
                "first_visible": first_visible,
                "conflict_timing": {
                    "time_to_conflict_s": round(max(0.0, time_to_conflict), 4),
                    "stop_feasible": stop_feasible,
                    "yield_feasible": yield_feasible,
                },
                "planner": {
                    "selected_action": {"linear_velocity": v_lin, "angular_velocity": 0.0},
                    "event": event,
                },
            }
        )

    conflict_time_s = (
        round(abs(cfg["ped_initial_y"] - conflict_y) / ped_speed, 4) if ped_speed > 0 else 999.0
    )

    return {
        "schema_version": "simulation_trace_export.v1",
        "trace_id": cfg["trace_id"],
        "source": {
            "scenario_id": cfg["scenario_id"],
            "seed": cfg["seed"],
            "planner_id": "hybrid_rule_v0_minimal",
            "episode_id": cfg["episode_id"],
            "generated_by": f"deterministic variant generator for issue #2780 ({name})",
        },
        "evidence_boundary": "smoke_diagnostic_only_not_benchmark_evidence",
        "coordinate_frame": "world",
        "units": {"position": "m", "heading": "rad", "time": "s", "velocity": "m/s"},
        "occlusion": {
            "occluder_id": cfg.get("occluder_id", "static_wall_behind_corner"),
            "occluder_bounds": _occluder_bounds(side),
            "first_visible_step": first_visible_step,
            "conflict_zone_center": [cfg["ped_x"], conflict_y],
            "conflict_time_s": conflict_time_s,
            "stop_feasible_before_step": cfg.get("stop_feasible_before_step", 10),
            "yield_feasible_before_step": cfg.get("yield_feasible_before_step", 12),
        },
        "variant": {
            "name": name,
            "parent_issue": 2756,
            "issue": 2780,
            "expected_failure_mode": cfg["expected_failure_mode"],
            "variation_dimensions": cfg["variation_dimensions"],
            "safety_relevant_under_live_replay": cfg.get("safety_relevant", False),
            "evidence_boundary_note": cfg.get(
                "evidence_boundary_note",
                "diagnostic_stress_only_not_benchmark_evidence",
            ),
        },
        "frames": frames,
    }


VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "occluded_emergence_left_close": {
        "trace_id": "issue_2780_occluded_emergence_left_close_seed111_ep0000",
        "scenario_id": "issue_2780_occluded_emergence_left_close",
        "episode_id": "issue_2780_occluded_emergence_left_close_ep0000",
        "seed": 111,
        "ped_id": 27801,
        "ped_x": 0.9,
        "ped_initial_y": -1.0,
        "ped_speed": 1.2,
        "robot_v0": 1.0,
        "side": "left",
        "conflict_y": 0.0,
        "first_visible_step": 4,
        "stop_feasible_before_step": 8,
        "yield_feasible_before_step": 10,
        "stop_horizon_s": 0.25,
        "yield_horizon_s": 0.15,
        "expected_failure_mode": "late_detection",
        "variation_dimensions": ["emergence_side:left", "first_visible_step:4", "ped_speed:1.2"],
        "safety_relevant": True,
        "evidence_boundary_note": "diagnostic_stress_only_not_benchmark_evidence",
    },
    "occluded_emergence_right_close": {
        "trace_id": "issue_2780_occluded_emergence_right_close_seed111_ep0000",
        "scenario_id": "issue_2780_occluded_emergence_right_close",
        "episode_id": "issue_2780_occluded_emergence_right_close_ep0000",
        "seed": 111,
        "ped_id": 27802,
        "ped_x": 1.6,
        "ped_initial_y": -1.0,
        "ped_speed": 1.2,
        "robot_v0": 1.0,
        "side": "right",
        "conflict_y": 0.0,
        "first_visible_step": 4,
        "stop_feasible_before_step": 8,
        "yield_feasible_before_step": 10,
        "stop_horizon_s": 0.25,
        "yield_horizon_s": 0.15,
        "expected_failure_mode": "wrong_source_selection",
        "variation_dimensions": ["emergence_side:right", "first_visible_step:4", "ped_speed:1.2"],
        "safety_relevant": True,
        "evidence_boundary_note": "diagnostic_stress_only_not_benchmark_evidence",
    },
    "occluded_emergence_late_visibility": {
        "trace_id": "issue_2780_occluded_emergence_late_visibility_seed111_ep0000",
        "scenario_id": "issue_2780_occluded_emergence_late_visibility",
        "episode_id": "issue_2780_occluded_emergence_late_visibility_ep0000",
        "seed": 111,
        "ped_id": 27803,
        "ped_x": 1.2,
        "ped_initial_y": -0.8,
        "ped_speed": 1.0,
        "robot_v0": 1.0,
        "side": "center",
        "conflict_y": 0.0,
        "first_visible_step": 8,
        "stop_feasible_before_step": 10,
        "yield_feasible_before_step": 12,
        "stop_horizon_s": 0.2,
        "yield_horizon_s": 0.1,
        "expected_failure_mode": "insufficient_braking_distance",
        "variation_dimensions": ["first_visible_step:8", "ped_speed:1.0", "robot_v0:1.0"],
        "safety_relevant": True,
        "evidence_boundary_note": "diagnostic_stress_only_not_benchmark_evidence",
    },
    "occluded_emergence_slow_pedestrian": {
        "trace_id": "issue_2780_occluded_emergence_slow_pedestrian_seed111_ep0000",
        "scenario_id": "issue_2780_occluded_emergence_slow_pedestrian",
        "episode_id": "issue_2780_occluded_emergence_slow_pedestrian_ep0000",
        "seed": 111,
        "ped_id": 27804,
        "ped_x": 1.2,
        "ped_initial_y": -0.6,
        "ped_speed": 0.4,
        "robot_v0": 0.8,
        "side": "center",
        "conflict_y": 0.0,
        "first_visible_step": 5,
        "stop_feasible_before_step": 7,
        "yield_feasible_before_step": 9,
        "stop_horizon_s": 0.3,
        "yield_horizon_s": 0.2,
        "expected_failure_mode": "unnecessary_stop",
        "variation_dimensions": ["ped_speed:0.4", "robot_v0:0.8", "first_visible_step:5"],
        "safety_relevant": False,
        "evidence_boundary_note": "diagnostic_stress_only_not_benchmark_evidence",
    },
    "occluded_emergence_fast_pedestrian": {
        "trace_id": "issue_2780_occluded_emergence_fast_pedestrian_seed111_ep0000",
        "scenario_id": "issue_2780_occluded_emergence_fast_pedestrian",
        "episode_id": "issue_2780_occluded_emergence_fast_pedestrian_ep0000",
        "seed": 111,
        "ped_id": 27805,
        "ped_x": 1.2,
        "ped_initial_y": -1.2,
        "ped_speed": 2.0,
        "robot_v0": 1.2,
        "side": "center",
        "conflict_y": 0.0,
        "first_visible_step": 3,
        "stop_feasible_before_step": 5,
        "yield_feasible_before_step": 7,
        "stop_horizon_s": 0.15,
        "yield_horizon_s": 0.1,
        "expected_failure_mode": "forecast_miss",
        "variation_dimensions": ["ped_speed:2.0", "robot_v0:1.2", "first_visible_step:3"],
        "safety_relevant": True,
        "evidence_boundary_note": "diagnostic_stress_only_not_benchmark_evidence",
    },
}


def generate_all() -> dict[str, dict[str, Any]]:
    """Generate all variant traces and return them keyed by name."""
    return {name: _build_variant(name, cfg) for name, cfg in VARIANT_CONFIGS.items()}


def write_variants() -> list[pathlib.Path]:
    """Write all variant traces and metadata to disk."""
    variants = generate_all()
    written: list[pathlib.Path] = []
    for name, trace in variants.items():
        trace_path = (
            FIXTURE_DIR
            / f"occluded_emergence_{name.replace('occluded_emergence_', '')}_episode_0000.json"
        )
        with open(trace_path, "w") as fh:
            json.dump(trace, fh, indent=2)
        written.append(trace_path)

        meta = {
            "algorithm": "hand_authored_deterministic_fixture",
            "boundary": "smoke_diagnostic_only_not_benchmark_evidence",
            "episode_id": trace["source"]["episode_id"],
            "issue": 2780,
            "parent_issue": 2756,
            "scenario": trace["source"]["scenario_id"],
            "seed": trace["source"]["seed"],
            "source_kind": "simulation_trace_export.v1 fixture",
            "variant_name": name,
            "expected_failure_mode": trace["variant"]["expected_failure_mode"],
            "variation_dimensions": trace["variant"]["variation_dimensions"],
            "safety_relevant_under_live_replay": trace["variant"][
                "safety_relevant_under_live_replay"
            ],
            "evidence_boundary": trace["variant"]["evidence_boundary_note"],
            "notes": [
                "Variant of the issue #2756 occluded-emergence fixture for issue #2780.",
                f"Expected failure mode: {trace['variant']['expected_failure_mode']}.",
                f"Variation dimensions: {', '.join(trace['variant']['variation_dimensions'])}.",
                "This fixture is not paper-facing benchmark evidence.",
            ],
        }
        meta_path = SOURCES_DIR / f"issue_2780_{name}_fixture_111_ep0000.meta.json"
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        written.append(meta_path)
    return written


if __name__ == "__main__":
    paths = write_variants()
    for p in paths:
        print(p)
