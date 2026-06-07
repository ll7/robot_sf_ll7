"""Tests for trace-level failure predicate extraction."""

from __future__ import annotations

from typing import Any

from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExport,
    simulation_trace_export_from_dict,
)
from robot_sf.analysis_workbench.trace_failure_predicates import (
    TRACE_FAILURE_PREDICATE_SCHEMA_VERSION,
    extract_trace_failure_predicates,
)


def _trace(frames: list[dict[str, Any]]) -> SimulationTraceExport:
    """Build a schema-valid synthetic trace export."""
    return simulation_trace_export_from_dict(
        {
            "schema_version": "simulation_trace_export.v1",
            "trace_id": "crossing-proxy-orca-111",
            "source": {
                "scenario_id": "crossing_proxy",
                "seed": 111,
                "planner_id": "orca",
                "episode_id": "episode-001",
                "generated_by": "test",
            },
            "evidence_boundary": "analysis_workbench_only",
            "coordinate_frame": "world",
            "units": {
                "position": "m",
                "heading": "rad",
                "time": "s",
                "velocity": "m/s",
            },
            "frames": frames,
        }
    )


def _frame(
    step: int,
    *,
    robot_x: float,
    angular_velocity: float,
    linear_velocity: float = 0.0,
    pedestrian_x: float = 2.0,
    planner: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one compact frame with an optional planner metadata overlay."""
    planner_payload: dict[str, Any] = {
        "selected_action": {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
        }
    }
    planner_payload.update(planner or {})
    return {
        "step": step,
        "time_s": float(step),
        "robot": {
            "position": [robot_x, 0.0],
            "heading": 0.0,
            "velocity": [0.0, 0.0],
        },
        "pedestrians": [
            {
                "id": "ped-1",
                "position": [pedestrian_x, 0.0],
                "velocity": [0.0, 0.0],
            }
        ],
        "planner": planner_payload,
    }


def test_extracts_valid_dissertation_failure_predicates() -> None:
    """A compact trace fixture should emit explicit valid predicate records."""
    trace = _trace(
        [
            _frame(0, robot_x=0.0, angular_velocity=0.7),
            _frame(1, robot_x=0.01, angular_velocity=-0.7, pedestrian_x=0.3),
            _frame(2, robot_x=0.02, angular_velocity=0.8),
            _frame(3, robot_x=0.02, angular_velocity=-0.8),
            _frame(4, robot_x=0.02, angular_velocity=0.7, planner={"event": "timeout"}),
        ]
    )

    payload = extract_trace_failure_predicates(trace, scenario_family="crossing")

    assert payload["schema_version"] == TRACE_FAILURE_PREDICATE_SCHEMA_VERSION
    predicates = {row["predicate_id"]: row for row in payload["predicates"]}
    assert {
        "clearance_critical_interaction",
        "oscillatory_local_control",
        "zero_motion_timeout_behavior",
    }.issubset(predicates)
    assert predicates["clearance_critical_interaction"]["validity_status"] == "valid"
    assert predicates["clearance_critical_interaction"]["involved_actors"] == ["robot", "ped-1"]
    assert predicates["oscillatory_local_control"]["evidence_fields"]["angular_sign_changes"] >= 3
    assert predicates["zero_motion_timeout_behavior"]["severity"] == "high"
    assert (
        payload["summary"]["by_scenario_family"]["crossing"]["orca"]["valid"][
            "zero_motion_timeout_behavior"
        ]
        == 1
    )


def test_occlusion_near_miss_fails_closed_when_visibility_fields_missing() -> None:
    """Near-miss geometry alone should not infer occlusion-triggered failure."""
    trace = _trace([_frame(0, robot_x=0.0, angular_velocity=0.0, pedestrian_x=0.25)])

    payload = extract_trace_failure_predicates(trace, scenario_family="crossing")

    occlusion_rows = [
        row
        for row in payload["predicates"]
        if row["predicate_id"] == "occlusion_triggered_near_miss"
    ]
    assert len(occlusion_rows) == 1
    row = occlusion_rows[0]
    assert row["validity_status"] == "not_available"
    assert row["severity"] == "not_available"
    assert row["evidence_fields"]["missing_fields"] == ["planner.occlusion_or_visibility"]
    assert (
        payload["summary"]["by_scenario_family"]["crossing"]["orca"]["not_available"][
            "occlusion_triggered_near_miss"
        ]
        == 1
    )


def test_event_and_action_predicates_cover_remaining_contract_ids() -> None:
    """Bottleneck and late-evasive predicates should have smoke semantics too."""
    trace = _trace(
        [
            _frame(0, robot_x=0.0, linear_velocity=0.8, angular_velocity=0.0),
            _frame(
                1,
                robot_x=0.1,
                linear_velocity=0.8,
                angular_velocity=0.0,
                pedestrian_x=0.35,
            ),
            _frame(
                2,
                robot_x=0.11,
                linear_velocity=0.2,
                angular_velocity=1.2,
                pedestrian_x=0.35,
            ),
            _frame(
                3,
                robot_x=0.11,
                angular_velocity=0.0,
                pedestrian_x=0.35,
                planner={"event": "bottleneck_deadlock"},
            ),
        ]
    )

    payload = extract_trace_failure_predicates(trace, scenario_family="bottleneck")

    predicates = {row["predicate_id"]: row for row in payload["predicates"]}
    assert predicates["late_evasive_reaction"]["validity_status"] == "valid"
    assert predicates["late_evasive_reaction"]["evidence_fields"]["reaction_delay_steps"] == 1
    assert predicates["bottleneck_deadlock"]["validity_status"] == "valid"
    assert predicates["bottleneck_deadlock"]["evidence_fields"]["event"] == "bottleneck_deadlock"
