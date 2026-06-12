"""Tests for trace-level failure predicate extraction."""

from __future__ import annotations

from typing import Any

from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExport,
    simulation_trace_export_from_dict,
)
from robot_sf.analysis_workbench.trace_failure_predicates import (
    TRACE_FAILURE_PREDICATE_SCHEMA_VERSION,
    aggregate_trace_failure_predicate_tables,
    extract_trace_failure_predicates,
    render_trace_failure_predicate_markdown,
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


def _clean_trace(
    planner: dict[str, Any] | None = None,
    *,
    scenario_id: str = "open_field",
    seed: int = 999,
    planner_id: str = "dwa",
) -> SimulationTraceExport:
    """Build a trace with no close pedestrians, producing zero predicates."""
    default_planner: dict[str, Any] = {
        "selected_action": {"linear_velocity": 0.5, "angular_velocity": 0.0}
    }
    default_planner.update(planner or {})
    return simulation_trace_export_from_dict(
        {
            "schema_version": "simulation_trace_export.v1",
            "trace_id": f"clean-{scenario_id}-{planner_id}-{seed}",
            "source": {
                "scenario_id": scenario_id,
                "seed": seed,
                "planner_id": planner_id,
                "episode_id": "episode-clean",
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
            "frames": [
                {
                    "step": 0,
                    "time_s": 0.0,
                    "robot": {"position": [0.0, 0.0], "heading": 0.0, "velocity": [0.0, 0.0]},
                    "pedestrians": [
                        {"id": "ped-distant", "position": [10.0, 10.0], "velocity": [0.0, 0.0]}
                    ],
                    "planner": default_planner,
                },
                {
                    "step": 1,
                    "time_s": 1.0,
                    "robot": {"position": [0.5, 0.0], "heading": 0.0, "velocity": [0.5, 0.0]},
                    "pedestrians": [
                        {"id": "ped-distant", "position": [10.5, 10.0], "velocity": [0.0, 0.0]}
                    ],
                    "planner": default_planner,
                },
            ],
        }
    )


def test_zero_predicate_group_preserved_in_aggregate() -> None:
    """A trace with no observed predicates should emit a no_predicate_observed row."""
    clean = _clean_trace()
    payload = aggregate_trace_failure_predicate_tables([clean], scenario_family="open_field")
    zero_rows = [r for r in payload["rows"] if r["predicate_id"] == "no_predicate_observed"]
    assert len(zero_rows) == 1
    row = zero_rows[0]
    assert row["scenario_family"] == "open_field"
    assert row["planner_id"] == "dwa"
    assert row["seed"] == 999
    assert row["trace_source_ids"] == ["clean-open_field-dwa-999"]
    assert row["predicate_count"] == 0
    assert row["trace_denominator"] == 1
    assert row["predicate_rate_per_trace"] == 0.0
    assert payload["zero_predicate_groups"] == [
        {
            "scenario_family": "open_field",
            "planner_id": "dwa",
            "seed": 999,
            "trace_source_id": "clean-open_field-dwa-999",
            "trace_denominator": 1,
            "valid_predicate_count": 0,
            "not_available_predicate_count": 0,
            "zero_group_reason": "no_predicate_observed",
        }
    ]


def test_zero_predicate_group_rendered_in_markdown() -> None:
    """Markdown output should expose zero-predicate trace groups separately."""
    clean = _clean_trace()
    payload = aggregate_trace_failure_predicate_tables([clean], scenario_family="open_field")
    md = render_trace_failure_predicate_markdown(payload)
    assert "Trace Groups With Zero Valid Predicates" in md
    assert "open_field" in md
    assert "dwa" in md
    assert "clean-open_field-dwa-999" in md


def test_mixed_observed_and_zero_groups() -> None:
    """Observed and zero-predicate groups should coexist in the same aggregate table."""
    crossing = _trace(
        [
            _frame(0, robot_x=0.0, angular_velocity=0.7),
            _frame(1, robot_x=0.01, angular_velocity=-0.7, pedestrian_x=0.3),
        ]
    )
    clean = _clean_trace(seed=555)
    payload = aggregate_trace_failure_predicate_tables([crossing, clean], scenario_family=None)
    observed = [r for r in payload["rows"] if r["predicate_id"] != "no_predicate_observed"]
    zeros = [r for r in payload["rows"] if r["predicate_id"] == "no_predicate_observed"]
    assert len(observed) > 0
    assert len(zeros) == 1
    assert zeros[0]["seed"] == 555
    assert zeros[0]["trace_denominator"] == 1
    zero_groups = payload["zero_predicate_groups"]
    assert [row["seed"] for row in zero_groups] == [555]


def test_absent_evidence_fields_not_available() -> None:
    """A trace with near-miss geometry but no planner evidence should yield not_available."""
    trace = _trace([_frame(0, robot_x=0.0, angular_velocity=0.0, pedestrian_x=0.25)])
    payload = extract_trace_failure_predicates(trace, scenario_family="crossing")
    occlusion = next(
        r for r in payload["predicates"] if r["predicate_id"] == "occlusion_triggered_near_miss"
    )
    assert occlusion["validity_status"] == "not_available"
    assert "missing_fields" in occlusion["evidence_fields"]
    assert occlusion["evidence_fields"]["missing_fields"] == ["planner.occlusion_or_visibility"]


def test_not_available_row_preserved_separately_from_true_zero() -> None:
    """not_available rows must not be conflated with true zero observed predicate groups."""
    trace = _trace([_frame(0, robot_x=0.0, angular_velocity=0.0, pedestrian_x=0.45)])
    payload = aggregate_trace_failure_predicate_tables([trace], scenario_family="crossing")
    not_avail = [r for r in payload["rows"] if r["validity_status"] == "not_available"]
    true_zero = [r for r in payload["rows"] if r["predicate_id"] == "no_predicate_observed"]
    assert len(not_avail) >= 1
    assert all(r["predicate_count"] >= 1 for r in not_avail)
    assert all(r["predicate_count"] == 0 for r in true_zero)
    assert payload["zero_predicate_groups"] == [
        {
            "scenario_family": "crossing",
            "planner_id": "orca",
            "seed": 111,
            "trace_source_id": "crossing-proxy-orca-111",
            "trace_denominator": 1,
            "valid_predicate_count": 0,
            "not_available_predicate_count": 1,
            "zero_group_reason": "valid_predicate_count_zero",
        }
    ]
