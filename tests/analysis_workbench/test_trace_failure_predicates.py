"""Tests for trace failure predicate extraction and aggregate table rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from robot_sf.analysis_workbench.simulation_trace_export import (
    load_simulation_trace_export,
    simulation_trace_export_from_dict,
)
from robot_sf.analysis_workbench.trace_failure_predicates import (
    aggregate_trace_failure_predicate_tables,
    build_trace_failure_predicate_definitions,
    build_trace_predicate_denominator_health_report,
    extract_trace_failure_predicates,
    render_trace_failure_predicate_markdown,
)
from scripts.tools.build_trace_failure_predicate_tables import write_trace_failure_predicate_tables

TRACE_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
    / "minimal_trace.json"
)
TRACE_PREDICATE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "robot_sf"
    / "analysis_workbench"
    / "schemas"
    / "trace_failure_predicates.v1.json"
)


def _frame(
    step: int,
    time_s: float,
    *,
    robot: dict[str, object] | None = None,
    pedestrians: list[dict[str, object]] | None = None,
    action: tuple[float, float] = (0.2, 0.0),
    planner_extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one synthetic frame payload."""
    robot_payload = robot or {
        "position": [0.0, 0.0],
        "heading": 0.0,
        "velocity": [0.0, 0.0],
    }
    planner_payload = {
        "selected_action": {
            "linear_velocity": action[0],
            "angular_velocity": action[1],
        },
        "event": "step",
    }
    if planner_extra:
        planner_payload.update(planner_extra)
    return {
        "step": step,
        "time_s": time_s,
        "robot": robot_payload,
        "pedestrians": pedestrians or [],
        "planner": planner_payload,
    }


def _build_trace(
    *,
    trace_id: str,
    scenario_id: str,
    seed: int,
    planner_id: str,
    frames: list[dict[str, object]],
):
    payload = load_simulation_trace_export(TRACE_FIXTURE_PATH).to_dict()
    payload["trace_id"] = trace_id
    payload["source"]["scenario_id"] = scenario_id
    payload["source"]["seed"] = seed
    payload["source"]["planner_id"] = planner_id
    payload["source"]["episode_id"] = f"{trace_id}-episode"
    payload["frames"] = frames
    return simulation_trace_export_from_dict(payload, source=TRACE_FIXTURE_PATH)


def _aggregate_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    return list(payload.get("rows", []))


def _find_row(
    rows: list[dict[str, object]],
    *,
    predicate_id: str,
) -> dict[str, object] | None:
    return next((row for row in rows if row.get("predicate_id") == predicate_id), None)


def _assert_predicate_schema_valid(payload: dict[str, object]) -> None:
    """Assert that a trace-predicate payload validates against its public schema."""
    schema = json.loads(TRACE_PREDICATE_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    assert not errors, [error.message for error in errors]


def test_aggregate_preserves_occlusion_not_available_row() -> None:
    """Missing occlusion metadata should emit and preserve `not_available` rows."""
    trace = _build_trace(
        trace_id="trace-missing-occlusion",
        scenario_id="scenario-occlusion",
        seed=11,
        planner_id="planner-a",
        frames=[
            _frame(
                0,
                0.0,
                robot={"position": [0.0, 0.0], "heading": 0.0, "velocity": [0.0, 0.0]},
                action=(0.1, 0.0),
                pedestrians=[{"id": "ped_1", "position": [0.45, 0.0], "velocity": [0.0, 0.0]}],
            )
        ],
    )
    payload = aggregate_trace_failure_predicate_tables([trace])
    row = _find_row(
        _aggregate_rows(payload),
        predicate_id="occlusion_triggered_near_miss",
    )

    assert row is not None
    assert row["validity_status"] == "not_available"
    assert row["severity"] == "not_available"
    assert row["predicate_count"] == 1
    assert row["trace_denominator"] == 1
    assert row["predicate_rate_per_trace"] == 1.0
    assert payload["summary"]["input_trace_count"] == 1


def test_aggregate_includes_zero_motion_timeout_predicate() -> None:
    """A timeout frame plus stationary motion should emit zero-motion predicate."""
    trace = _build_trace(
        trace_id="trace-zero-motion",
        scenario_id="scenario-zero-motion",
        seed=12,
        planner_id="planner-b",
        frames=[
            _frame(
                0,
                0.0,
                robot={"position": [0.0, 0.0], "heading": 0.0, "velocity": [0.0, 0.0]},
                action=(0.0, 0.0),
            ),
            _frame(
                1,
                0.1,
                robot={"position": [0.01, 0.0], "heading": 0.0, "velocity": [0.0, 0.0]},
                action=(0.0, 0.0),
                planner_extra={"event": "timeout"},
            ),
        ],
    )
    row = _find_row(
        _aggregate_rows(aggregate_trace_failure_predicate_tables([trace])),
        predicate_id="zero_motion_timeout_behavior",
    )

    assert row is not None
    assert row["validity_status"] == "valid"
    assert row["predicate_count"] == 1


def test_aggregate_includes_oscillatory_control_predicate() -> None:
    """Alternating angular signs should emit oscillatory-local-control."""
    trace = _build_trace(
        trace_id="trace-oscillation",
        scenario_id="scenario-oscillation",
        seed=13,
        planner_id="planner-c",
        frames=[
            _frame(0, 0.0, action=(0.2, 0.2)),
            _frame(1, 0.1, action=(0.2, -0.2)),
            _frame(2, 0.2, action=(0.2, 0.2)),
            _frame(3, 0.3, action=(0.2, -0.2)),
        ],
    )
    row = _find_row(
        _aggregate_rows(aggregate_trace_failure_predicate_tables([trace])),
        predicate_id="oscillatory_local_control",
    )

    assert row is not None
    assert row["validity_status"] == "valid"
    assert row["severity"] == "medium"


def test_aggregate_includes_bottleneck_deadlock_predicate() -> None:
    """Planner deadlock events should be preserved as aggregate predicate rows."""
    trace = _build_trace(
        trace_id="trace-bottleneck",
        scenario_id="scenario-bottleneck",
        seed=14,
        planner_id="planner-d",
        frames=[
            _frame(
                0,
                0.0,
                planner_extra={"event": "bottleneck_deadlock"},
            )
        ],
    )
    row = _find_row(
        _aggregate_rows(aggregate_trace_failure_predicate_tables([trace])),
        predicate_id="bottleneck_deadlock",
    )

    assert row is not None
    assert row["predicate_count"] == 1
    assert row["validity_status"] == "valid"


def test_aggregate_includes_clearance_critical_interaction_predicate() -> None:
    """Near-clearance observations should emit clearance-critical interaction predicates."""
    trace = _build_trace(
        trace_id="trace-clearance",
        scenario_id="scenario-clearance",
        seed=15,
        planner_id="planner-e",
        frames=[
            _frame(
                0,
                0.0,
                pedestrians=[{"id": "ped_1", "position": [0.1, 0.1], "velocity": [0.0, 0.0]}],
            )
        ],
    )
    row = _find_row(
        _aggregate_rows(aggregate_trace_failure_predicate_tables([trace])),
        predicate_id="clearance_critical_interaction",
    )

    assert row is not None
    assert row["validity_status"] == "valid"
    assert row["severity"] == "high"


def test_aggregate_includes_collision_and_low_progress_predicates() -> None:
    """Collision and low-progress rows should be reusable predicate-table records."""
    collision_trace = _build_trace(
        trace_id="trace-collision",
        scenario_id="scenario-collision",
        seed=21,
        planner_id="planner-k",
        frames=[
            _frame(
                0,
                0.0,
                robot={
                    "position": [0.0, 0.0],
                    "heading": 0.0,
                    "velocity": [0.0, 0.0],
                    "radius": 0.18,
                },
                pedestrians=[
                    {
                        "id": "ped_1",
                        "position": [0.3, 0.0],
                        "velocity": [0.0, 0.0],
                        "radius": 0.14,
                    }
                ],
            )
        ],
    )
    low_progress_trace = _build_trace(
        trace_id="trace-low-progress",
        scenario_id="scenario-low-progress",
        seed=22,
        planner_id="planner-l",
        frames=[
            _frame(
                0,
                0.0,
                robot={"position": [0.0, 0.0], "heading": 0.0, "velocity": [0.0, 0.0]},
                action=(0.1, 0.0),
            ),
            _frame(
                1,
                1.0,
                robot={"position": [0.12, 0.0], "heading": 0.0, "velocity": [0.0, 0.0]},
                action=(0.0, 0.0),
                planner_extra={"event": "timeout"},
            ),
        ],
    )

    collision_row = _find_row(
        _aggregate_rows(aggregate_trace_failure_predicate_tables([collision_trace])),
        predicate_id="collision",
    )
    low_progress_row = _find_row(
        _aggregate_rows(aggregate_trace_failure_predicate_tables([low_progress_trace])),
        predicate_id="low_progress",
    )

    assert collision_row is not None
    assert collision_row["validity_status"] == "valid"
    assert collision_row["severity"] == "critical"
    assert low_progress_row is not None
    assert low_progress_row["validity_status"] == "valid"
    assert low_progress_row["severity"] == "medium"


def test_predicate_definitions_document_inputs_thresholds_and_denominators() -> None:
    """Reusable predicate definitions should be machine-readable and complete."""
    definitions = {
        definition["predicate_id"]: definition
        for definition in build_trace_failure_predicate_definitions()
    }

    assert "collision" in definitions
    assert definitions["collision"]["units"]["distance"] == "m"
    assert definitions["collision"]["thresholds"]["missing_radius_near_distance_m"] == 0.2
    assert "denominator" in definitions["collision"]["denominator_semantics"]
    assert "low_progress" in definitions
    assert (
        definitions["low_progress"]["thresholds"]["low_progress_displacement_threshold_m"] == 0.25
    )


def test_collision_uses_radius_overlap_not_fixed_center_distance() -> None:
    """Collision should follow benchmark footprint-overlap semantics."""
    trace = _build_trace(
        trace_id="trace-radius-collision",
        scenario_id="scenario-radius-collision",
        seed=25,
        planner_id="planner-radius",
        frames=[
            _frame(
                0,
                0.0,
                robot={
                    "position": [0.0, 0.0],
                    "heading": 0.0,
                    "velocity": [0.0, 0.0],
                    "radius": 0.18,
                },
                pedestrians=[
                    {
                        "id": "ped_1",
                        "position": [0.3, 0.0],
                        "velocity": [0.0, 0.0],
                        "radius": 0.14,
                    }
                ],
            )
        ],
    )

    predicates = extract_trace_failure_predicates(trace)["predicates"]
    collision = next(row for row in predicates if row["predicate_id"] == "collision")

    assert collision["validity_status"] == "valid"
    assert collision["evidence_fields"]["distance_m"] == 0.3
    assert collision["evidence_fields"]["collision_distance_m"] == pytest.approx(0.32)


def test_collision_checks_all_pedestrians_not_only_nearest_center() -> None:
    """A larger non-nearest pedestrian can overlap even when the nearest does not."""
    trace = _build_trace(
        trace_id="trace-non-nearest-radius-collision",
        scenario_id="scenario-radius-collision",
        seed=28,
        planner_id="planner-radius",
        frames=[
            _frame(
                0,
                0.0,
                robot={
                    "position": [0.0, 0.0],
                    "heading": 0.0,
                    "velocity": [0.0, 0.0],
                    "radius": 0.1,
                },
                pedestrians=[
                    {
                        "id": "ped_small_nearest",
                        "position": [0.22, 0.0],
                        "velocity": [0.0, 0.0],
                        "radius": 0.05,
                    },
                    {
                        "id": "ped_large_farther",
                        "position": [0.3, 0.0],
                        "velocity": [0.0, 0.0],
                        "radius": 0.22,
                    },
                ],
            )
        ],
    )

    collisions = [
        row
        for row in extract_trace_failure_predicates(trace)["predicates"]
        if row["predicate_id"] == "collision" and row["validity_status"] == "valid"
    ]

    assert [row["involved_actors"][1] for row in collisions] == ["ped_large_farther"]
    assert collisions[0]["evidence_fields"]["collision_distance_m"] == pytest.approx(0.32)


def test_collision_near_contact_without_radii_is_not_available() -> None:
    """Near-contact geometry without radii should not become a collision claim."""
    trace = _build_trace(
        trace_id="trace-missing-radius-collision",
        scenario_id="scenario-missing-radius-collision",
        seed=26,
        planner_id="planner-radius",
        frames=[
            _frame(
                0,
                0.0,
                pedestrians=[{"id": "ped_1", "position": [0.1, 0.0], "velocity": [0.0, 0.0]}],
            )
        ],
    )

    predicates = extract_trace_failure_predicates(trace)["predicates"]
    collision = next(row for row in predicates if row["predicate_id"] == "collision")

    assert collision["validity_status"] == "not_available"
    assert collision["severity"] == "not_available"
    assert collision["evidence_fields"]["missing_fields"] == [
        "robot.radius",
        "pedestrians.radius",
    ]


def test_mixed_valid_and_failed_traces_report_pipeline_failure_denominator() -> None:
    """Mixed valid and failed trace bundles should fail closed at the aggregate denominator."""
    valid_trace = _build_trace(
        trace_id="trace-valid-with-failed-peer",
        scenario_id="scenario-mixed-denominator",
        seed=27,
        planner_id="planner-mixed",
        frames=[_frame(0, 0.0)],
    )

    payload = aggregate_trace_failure_predicate_tables(
        [valid_trace], failed_trace_ids=["bad-trace.json"]
    )
    health = build_trace_predicate_denominator_health_report(payload)

    assert payload["denominator_status"] == "pipeline_failure"
    assert health["summary"]["denominator_status"] == "pipeline_failure"


def test_zero_trace_bundle_reports_unavailable_denominator() -> None:
    """Zero valid input traces should be explicit, not a silent zero denominator."""
    payload = aggregate_trace_failure_predicate_tables([])
    health = build_trace_predicate_denominator_health_report(payload)

    assert payload["denominator_status"] == "not_available"
    assert payload["summary"]["input_trace_count"] == 0
    assert payload["rows"] == []
    assert health["summary"]["denominator_status"] == "not_available"
    assert all(report["total_denominator_count"] == 0 for report in health["predicates"].values())


def test_matrix_occlusion_logical_field_accepts_visibility_alias() -> None:
    """Matrix-required logical occlusion evidence should accept supported trace aliases."""
    trace = _build_trace(
        trace_id="trace-occlusion-alias",
        scenario_id="scenario-occlusion-alias",
        seed=24,
        planner_id="planner-n",
        frames=[
            _frame(
                0,
                0.0,
                pedestrians=[{"id": "ped_1", "position": [0.2, 0.0], "velocity": [0.0, 0.0]}],
                planner_extra={"visibility": {"line_of_sight": False}},
            )
        ],
    )
    matrix = {
        "schema_version": "trace_predicate_matrix.v1",
        "name": "occlusion_alias_matrix",
        "status": "proposed",
        "matrix": {
            "required_trace_fields_by_predicate": {
                "occlusion_triggered_near_miss": ["planner.occlusion_or_visibility"]
            }
        },
    }

    payload = aggregate_trace_failure_predicate_tables([trace], matrix=matrix)
    occlusion_row = _find_row(
        _aggregate_rows(payload),
        predicate_id="occlusion_triggered_near_miss",
    )

    assert occlusion_row is not None
    assert occlusion_row["validity_status"] == "valid"


def test_trace_failure_predicate_payloads_validate_public_schema() -> None:
    """Single-trace and aggregate predicate outputs should validate against the public schema."""
    trace = _build_trace(
        trace_id="trace-schema",
        scenario_id="scenario-schema",
        seed=23,
        planner_id="planner-m",
        frames=[
            _frame(
                0,
                0.0,
                pedestrians=[{"id": "ped_1", "position": [0.1, 0.0], "velocity": [0.0, 0.0]}],
            )
        ],
    )

    _assert_predicate_schema_valid(extract_trace_failure_predicates(trace))
    _assert_predicate_schema_valid(aggregate_trace_failure_predicate_tables([trace]))


def test_denominator_and_markdown_render_include_rate_fields() -> None:
    """Aggregate payload and markdown output should expose denominator and rate columns."""
    traces = [
        _build_trace(
            trace_id=f"trace-denom-{index}",
            scenario_id="scenario-denom",
            seed=16,
            planner_id="planner-f",
            frames=[
                _frame(
                    0,
                    0.0,
                    action=(0.1, 0.0),
                    pedestrians=[{"id": "ped_1", "position": [0.45, 0.0], "velocity": [0.0, 0.0]}],
                )
            ],
        )
        for index in range(2)
    ]
    payload = aggregate_trace_failure_predicate_tables(traces)
    row = _find_row(
        _aggregate_rows(payload),
        predicate_id="occlusion_triggered_near_miss",
    )

    assert row is not None
    assert row["trace_source_ids"] == ["trace-denom-0", "trace-denom-1"]
    assert row["trace_denominator"] == 2
    assert row["predicate_count"] == 2
    assert row["predicate_rate_per_trace"] == 1.0

    markdown = render_trace_failure_predicate_markdown(payload)
    assert "trace_denominator" in markdown
    assert "predicate_rate_per_trace" in markdown
    assert "scenario-denom" in markdown


def test_scenario_family_override_normalizes_group_without_rewriting_source() -> None:
    """Scenario-family overrides should label groups without hiding the source scenario."""
    trace = _build_trace(
        trace_id="trace-family-override",
        scenario_id="classic_bottleneck_medium",
        seed=17,
        planner_id="planner-g",
        frames=[
            _frame(
                0,
                0.0,
                planner_extra={"event": "bottleneck_deadlock"},
            )
        ],
    )

    single_trace_payload = extract_trace_failure_predicates(
        trace,
        scenario_family="bottleneck",
    )
    aggregate_payload = aggregate_trace_failure_predicate_tables(
        [trace],
        scenario_family="bottleneck",
    )

    assert single_trace_payload["source"]["scenario_id"] == "classic_bottleneck_medium"
    assert single_trace_payload["predicates"][0]["scenario_family"] == "bottleneck"
    row = _find_row(
        _aggregate_rows(aggregate_payload),
        predicate_id="bottleneck_deadlock",
    )
    assert row is not None
    assert row["scenario_family"] == "bottleneck"
    assert aggregate_payload["summary"]["input_trace_count"] == 1


def test_writer_persists_json_and_markdown_outputs(tmp_path: Path) -> None:
    """The CLI writer should persist denominator-aware JSON and diagnostic Markdown."""
    trace = _build_trace(
        trace_id="trace-writer",
        scenario_id="scenario-writer",
        seed=18,
        planner_id="planner-h",
        frames=[
            _frame(
                0,
                0.0,
                planner_extra={"event": "bottleneck_deadlock"},
            )
        ],
    )
    trace_path = tmp_path / "trace.json"
    json_output = tmp_path / "tables.json"
    markdown_output = tmp_path / "tables.md"
    trace_path.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")

    write_trace_failure_predicate_tables(
        traces=[trace_path],
        output_json=json_output,
        output_markdown=markdown_output,
    )

    assert '"trace_denominator"' in json_output.read_text(encoding="utf-8")
    markdown = markdown_output.read_text(encoding="utf-8")
    assert "diagnostic-only" in markdown
    assert "trace_denominator" in markdown


def test_markdown_renderer_skips_invalid_rows_and_preserves_falsy_values() -> None:
    """Renderer should ignore malformed rows without replacing valid falsy identifiers."""
    markdown = render_trace_failure_predicate_markdown(
        {
            "schema_version": "trace_failure_predicates.v1",
            "table_kind": "aggregate_by_predicate_group",
            "summary": {"input_trace_count": 1},
            "rows": [
                None,
                "bad-row",
                {
                    "scenario_family": 0,
                    "planner_id": "",
                    "seed": 0,
                    "predicate_id": "zero_like",
                    "validity_status": "valid",
                    "severity": "low",
                    "predicate_count": 0,
                    "trace_denominator": 1,
                    "predicate_rate_per_trace": 0.0,
                },
            ],
        }
    )

    assert "| 0 |  | 0 |  | zero_like | valid | low | 0 | 1 | 0.0000 |" in markdown
    assert "bad-row" not in markdown


def _build_malformed_trace_file(tmp_path: Path, filename: str) -> Path:
    """Build a malformed trace file for testing pipeline failures."""
    malformed_trace_path = tmp_path / filename
    malformed_trace_path.write_text("this is not valid json {", encoding="utf-8")
    return malformed_trace_path


def test_pipeline_failure_handling(tmp_path: Path) -> None:
    """Pipeline failures should be captured in denominator health reports."""
    from robot_sf.analysis_workbench.trace_failure_predicates import (
        TRACE_FAILURE_PREDICATE_IDS,
        VALIDITY_STATUS_PIPELINE_FAILURE,
    )

    valid_trace = _build_trace(
        trace_id="trace-valid",
        scenario_id="scenario-valid",
        seed=1,
        planner_id="planner-A",
        frames=[_frame(0, 0.0)],
    )
    valid_trace_path = tmp_path / "valid_trace.json"
    valid_trace_path.write_text(json.dumps(valid_trace.to_dict()), encoding="utf-8")

    malformed_trace_path = _build_malformed_trace_file(tmp_path, "malformed_trace.json")

    json_output = tmp_path / "tables.json"
    markdown_output = tmp_path / "tables.md"
    denominator_health_json_output = tmp_path / "denominator_health.json"

    json_path, md_path, dh_json_path = write_trace_failure_predicate_tables(
        traces=[valid_trace_path, malformed_trace_path],
        output_json=json_output,
        output_markdown=markdown_output,
        output_denominator_health_json=denominator_health_json_output,
    )

    assert json_path == json_output
    assert md_path == markdown_output
    assert dh_json_path == denominator_health_json_output

    dh_report = json.loads(dh_json_path.read_text(encoding="utf-8"))
    assert dh_report["summary"]["total_traces_processed"] == 2
    assert dh_report["summary"]["failed_trace_count"] == 1

    for pred_id in TRACE_FAILURE_PREDICATE_IDS:
        pred_summary = dh_report["predicates"][pred_id]
        assert pred_summary["total_denominator_count"] == 2
        assert pred_summary["pipeline_failure_count"] == 1
        assert pred_summary["pipeline_failure_ratio"] == 0.5

    markdown = md_path.read_text(encoding="utf-8")
    assert "Traces with Pipeline Failures: 1" in markdown
    assert f"`{VALIDITY_STATUS_PIPELINE_FAILURE}`" in markdown
    assert "claim-ineligible" in markdown


def test_predicate_denominator_health_in_payload() -> None:
    """The aggregate payload should include correct predicate denominator health."""
    from robot_sf.analysis_workbench.trace_failure_predicates import (
        VALIDITY_STATUS_NO_PREDICATE_OBSERVED,
        VALIDITY_STATUS_PIPELINE_FAILURE,
        VALIDITY_STATUS_UNAVAILABLE_DATA,
        VALIDITY_STATUS_VALID,
    )

    trace1 = _build_trace(
        trace_id="t1",
        scenario_id="s1",
        seed=1,
        planner_id="p1",
        frames=[
            _frame(
                0,
                0.0,
                planner_extra={"visibility": {"line_of_sight": True}},
                pedestrians=[{"id": "ped_1", "position": [0.1, 0.1], "velocity": [0.0, 0.0]}],
            )
        ],
    )
    trace2 = _build_trace(
        trace_id="t2",
        scenario_id="s1",
        seed=2,
        planner_id="p1",
        frames=[
            _frame(
                0,
                0.0,
                pedestrians=[{"id": "ped_1", "position": [0.45, 0.0], "velocity": [0.0, 0.0]}],
            )
        ],
    )
    trace3 = _build_trace(
        trace_id="t3",
        scenario_id="s1",
        seed=3,
        planner_id="p1",
        frames=[
            _frame(
                0,
                0.0,
                pedestrians=[{"id": "ped_1", "position": [5.0, 5.0], "velocity": [0.0, 0.0]}],
            )
        ],
    )

    payload = aggregate_trace_failure_predicate_tables([trace1, trace2, trace3])

    dh = payload["predicate_denominator_health"]

    cci_counts = dh["clearance_critical_interaction"]
    assert cci_counts[VALIDITY_STATUS_VALID] == 1
    assert cci_counts[VALIDITY_STATUS_UNAVAILABLE_DATA] == 0
    assert cci_counts[VALIDITY_STATUS_NO_PREDICATE_OBSERVED] == 2

    otnm_counts = dh["occlusion_triggered_near_miss"]
    assert otnm_counts[VALIDITY_STATUS_VALID] == 0
    assert otnm_counts[VALIDITY_STATUS_UNAVAILABLE_DATA] == 1
    assert otnm_counts[VALIDITY_STATUS_NO_PREDICATE_OBSERVED] == 2

    zmtb_counts = dh["zero_motion_timeout_behavior"]
    assert zmtb_counts[VALIDITY_STATUS_VALID] == 0
    assert zmtb_counts[VALIDITY_STATUS_UNAVAILABLE_DATA] == 0
    assert zmtb_counts[VALIDITY_STATUS_NO_PREDICATE_OBSERVED] == 3

    for pred_id in dh:
        assert VALIDITY_STATUS_PIPELINE_FAILURE in dh[pred_id]


def test_denominator_health_counts_trace_groups_not_predicate_rows() -> None:
    """Multiple rows for one predicate family should not inflate denominator health."""
    from robot_sf.analysis_workbench.trace_failure_predicates import VALIDITY_STATUS_VALID

    trace = _build_trace(
        trace_id="trace-repeated-clearance",
        scenario_id="scenario-repeated-clearance",
        seed=20,
        planner_id="planner-j",
        frames=[
            _frame(
                0,
                0.0,
                planner_extra={"visibility": {"line_of_sight": True}},
                pedestrians=[{"id": "ped_1", "position": [0.1, 0.1], "velocity": [0.0, 0.0]}],
            ),
            _frame(
                1,
                0.1,
                planner_extra={"visibility": {"line_of_sight": True}},
                pedestrians=[{"id": "ped_1", "position": [0.2, 0.1], "velocity": [0.0, 0.0]}],
            ),
        ],
    )

    payload = aggregate_trace_failure_predicate_tables([trace])
    row = _find_row(
        _aggregate_rows(payload),
        predicate_id="clearance_critical_interaction",
    )
    assert row is not None
    assert row["predicate_count"] == 2
    assert (
        payload["predicate_denominator_health"]["clearance_critical_interaction"][
            VALIDITY_STATUS_VALID
        ]
        == 1
    )


def test_denominator_health_json_report() -> None:
    """The denominator health JSON report should be correctly structured and populated."""
    trace1 = _build_trace(
        trace_id="t1",
        scenario_id="s1",
        seed=1,
        planner_id="p1",
        frames=[
            _frame(
                0,
                0.0,
                planner_extra={"visibility": {"line_of_sight": True}},
                pedestrians=[{"id": "ped_1", "position": [0.1, 0.1], "velocity": [0.0, 0.0]}],
            )
        ],
    )
    trace2 = _build_trace(
        trace_id="t2",
        scenario_id="s1",
        seed=2,
        planner_id="p1",
        frames=[
            _frame(
                0,
                0.0,
                pedestrians=[{"id": "ped_1", "position": [0.45, 0.0], "velocity": [0.0, 0.0]}],
            )
        ],
    )
    trace3 = _build_trace(
        trace_id="t3",
        scenario_id="s1",
        seed=3,
        planner_id="p1",
        frames=[
            _frame(
                0,
                0.0,
                pedestrians=[{"id": "ped_1", "position": [5.0, 5.0], "velocity": [0.0, 0.0]}],
            )
        ],
    )

    payload = aggregate_trace_failure_predicate_tables([trace1, trace2, trace3])
    dh_report = build_trace_predicate_denominator_health_report(payload)

    assert dh_report["schema_version"] == "trace_predicate_denominator_health.v1"
    assert dh_report["summary"]["total_traces_processed"] == 3
    assert dh_report["summary"]["failed_trace_count"] == 0

    cci_summary = dh_report["predicates"]["clearance_critical_interaction"]
    assert cci_summary["valid_count"] == 1
    assert cci_summary["unavailable_data_count"] == 0
    assert cci_summary["no_predicate_observed_count"] == 2
    assert cci_summary["total_denominator_count"] == 3
    assert cci_summary["unavailable_data_ratio"] == 0.0
    assert cci_summary["pipeline_failure_count"] == 0

    otnm_summary = dh_report["predicates"]["occlusion_triggered_near_miss"]
    assert otnm_summary["valid_count"] == 0
    assert otnm_summary["unavailable_data_count"] == 1
    assert otnm_summary["no_predicate_observed_count"] == 2
    assert otnm_summary["total_denominator_count"] == 3
    assert otnm_summary["unavailable_data_ratio"] == 1 / 3
    assert otnm_summary["pipeline_failure_count"] == 0
    assert otnm_summary["claim_eligibility"] == "claim-ineligible"


def test_absent_expected_slices_are_rendered_as_caveats() -> None:
    """Markdown should identify absent matrix slices separately from zero observed predicates."""
    trace = _build_trace(
        trace_id="open-field-dwa-999",
        scenario_id="open_field",
        seed=999,
        planner_id="dwa",
        frames=[_frame(0, 0.0, planner_extra={"event": "step"})],
    )
    matrix = {
        "schema_version": "trace_predicate_matrix.v1",
        "name": "missing_slice_matrix",
        "status": "rate_interpretable",
        "matrix": {
            "scenario_families": ["open_field"],
            "planners": ["dwa"],
            "seeds": [999, 1000],
        },
    }

    payload = aggregate_trace_failure_predicate_tables([trace], matrix=matrix)
    markdown = render_trace_failure_predicate_markdown(payload)

    assert "Expected Matrix Slices Absent From Inputs" in markdown
    assert "absent_expected_slice" in markdown
    assert "open_field" in markdown
    assert "1000" in markdown
    assert "expected matrix slice is absent" in markdown


def test_writer_persists_denominator_health_json_when_requested(tmp_path: Path) -> None:
    """The writer should persist compact denominator health JSON when requested."""
    trace = _build_trace(
        trace_id="trace-health-writer",
        scenario_id="scenario-health-writer",
        seed=19,
        planner_id="planner-i",
        frames=[
            _frame(
                0,
                0.0,
                pedestrians=[{"id": "ped_1", "position": [0.1, 0.1], "velocity": [0.0, 0.0]}],
            )
        ],
    )
    trace_path = tmp_path / "trace.json"
    json_output = tmp_path / "tables.json"
    markdown_output = tmp_path / "tables.md"
    health_output = tmp_path / "denominator_health.json"
    trace_path.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")

    paths = write_trace_failure_predicate_tables(
        traces=[trace_path],
        output_json=json_output,
        output_markdown=markdown_output,
        output_denominator_health_json=health_output,
    )

    assert paths == (json_output, markdown_output, health_output)
    health = json.loads(health_output.read_text(encoding="utf-8"))
    assert health["schema_version"] == "trace_predicate_denominator_health.v1"
    assert health["summary"]["predicate_family_count"] == 8


def test_markdown_report_with_denominator_health(tmp_path: Path) -> None:
    """Markdown report should include denominator health section and caveats."""
    from robot_sf.analysis_workbench.trace_failure_predicates import (
        VALIDITY_STATUS_NO_PREDICATE_OBSERVED,
        VALIDITY_STATUS_PIPELINE_FAILURE,
        VALIDITY_STATUS_UNAVAILABLE_DATA,
        VALIDITY_STATUS_VALID,
    )

    trace1 = _build_trace(
        trace_id="t1",
        scenario_id="s1",
        seed=1,
        planner_id="p1",
        frames=[
            _frame(
                0,
                0.0,
                pedestrians=[{"id": "ped_1", "position": [0.1, 0.1], "velocity": [0.0, 0.0]}],
            )
        ],
    )
    trace2 = _build_trace(
        trace_id="t2",
        scenario_id="s1",
        seed=2,
        planner_id="p1",
        frames=[
            _frame(
                0,
                0.0,
                pedestrians=[{"id": "ped_1", "position": [0.45, 0.0], "velocity": [0.0, 0.0]}],
            )
        ],
    )
    trace3 = _build_trace(
        trace_id="t3",
        scenario_id="s1",
        seed=3,
        planner_id="p1",
        frames=[
            _frame(
                0,
                0.0,
                pedestrians=[{"id": "ped_1", "position": [5.0, 5.0], "velocity": [0.0, 0.0]}],
            )
        ],
    )
    malformed_trace_path = _build_malformed_trace_file(tmp_path, "malformed_trace_t4.json")

    valid_trace_path1 = tmp_path / "valid_trace_t1.json"
    valid_trace_path1.write_text(json.dumps(trace1.to_dict()), encoding="utf-8")
    valid_trace_path2 = tmp_path / "valid_trace_t2.json"
    valid_trace_path2.write_text(json.dumps(trace2.to_dict()), encoding="utf-8")
    valid_trace_path3 = tmp_path / "valid_trace_t3.json"
    valid_trace_path3.write_text(json.dumps(trace3.to_dict()), encoding="utf-8")
    json_output = tmp_path / "tables.json"
    markdown_output = tmp_path / "tables.md"
    denominator_health_json_output = tmp_path / "denominator_health.json"

    write_trace_failure_predicate_tables(
        traces=[valid_trace_path1, valid_trace_path2, valid_trace_path3, malformed_trace_path],
        output_json=json_output,
        output_markdown=markdown_output,
        output_denominator_health_json=denominator_health_json_output,
    )

    markdown_content = markdown_output.read_text(encoding="utf-8")

    assert "## Predicate Denominator Health Report" in markdown_content
    assert "### Missingness Categories" in markdown_content
    assert f"- `{VALIDITY_STATUS_VALID}`: predicate evaluated" in markdown_content
    assert (
        f"- `{VALIDITY_STATUS_UNAVAILABLE_DATA}`: required trace fields were missing"
        in markdown_content
    )
    assert (
        f"- `{VALIDITY_STATUS_NO_PREDICATE_OBSERVED}`: predicate was not emitted"
        in markdown_content
    )
    assert (
        f"- `{VALIDITY_STATUS_PIPELINE_FAILURE}`: the trace could not be loaded" in markdown_content
    )
    assert "### Predicate-specific Denominator Health" in markdown_content

    assert "clearance_critical_interaction" in markdown_content
    assert "occlusion_triggered_near_miss" in markdown_content
    assert "zero_motion_timeout_behavior" in markdown_content
    assert "claim-ineligible" in markdown_content
    assert "pipeline failed; do not make predicate-family claims" in markdown_content
