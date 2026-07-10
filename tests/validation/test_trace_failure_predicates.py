"""Tests for trace-level failure predicate extraction."""

from __future__ import annotations

from typing import Any

from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExport,
    simulation_trace_export_from_dict,
)
from robot_sf.analysis_workbench.trace_failure_predicates import (
    MATRIX_CLAIM_INELIGIBLE,
    MATRIX_STATUS_DIAGNOSTIC_ONLY,
    MATRIX_STATUS_PROPOSED,
    MATRIX_STATUS_RATE_INTERPRETABLE,
    TRACE_FAILURE_PREDICATE_SCHEMA_VERSION,
    aggregate_trace_failure_predicate_tables,
    build_trace_predicate_denominator_health_report,
    extract_trace_failure_predicates,
    load_trace_predicate_matrix,
    matrix_required_fields_for_predicate,
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
    # Seconds-valued companion to reaction_delay_steps (issue #5000): frames carry time_s, so a
    # reacted late-evasive event always exposes a latency in seconds, not only a step count.
    assert predicates["late_evasive_reaction"]["evidence_fields"]["response_latency_s"] == 1.0
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


def _minimal_matrix() -> dict[str, Any]:
    """Return a minimal trace-predicate benchmark matrix for tests."""
    return {
        "schema_version": "trace_predicate_matrix.v1",
        "name": "test_matrix",
        "status": MATRIX_STATUS_PROPOSED,
        "claim_boundary": "Test matrix; not rate evidence.",
        "matrix": {
            "scenario_families": ["crossing", "bottleneck", "open_field"],
            "planners": ["orca", "dwa"],
            "seeds": [111, 999],
            "required_trace_fields_by_predicate": {
                "bottleneck_deadlock": ["planner.event"],
                "zero_motion_timeout_behavior": ["planner.event"],
            },
        },
    }


def _matrix_for_slices(
    *,
    scenario_families: list[str],
    planners: list[str],
    seeds: list[int],
    status: str = MATRIX_STATUS_RATE_INTERPRETABLE,
) -> dict[str, Any]:
    """Return a matrix with an explicit expected scenario/planner/seed denominator."""
    matrix = _minimal_matrix()
    matrix["status"] = status
    matrix["matrix"]["scenario_families"] = scenario_families
    matrix["matrix"]["planners"] = planners
    matrix["matrix"]["seeds"] = seeds
    return matrix


def test_load_trace_predicate_matrix_rejects_bad_schema(tmp_path: Any) -> None:
    """Loading a matrix with the wrong schema version should fail closed."""
    import yaml

    matrix_path = tmp_path / "bad_matrix.yaml"
    matrix_path.write_text(
        yaml.dump({"schema_version": "trace_predicate_matrix.v0"}),
        encoding="utf-8",
    )
    try:
        load_trace_predicate_matrix(matrix_path)
        raise AssertionError("Expected ValueError for unsupported schema version")
    except ValueError as exc:
        assert "Unsupported trace predicate matrix schema version" in str(exc)


def test_matrix_required_fields_lookup() -> None:
    """Required fields for a predicate should come from the matrix."""
    matrix = _minimal_matrix()
    assert matrix_required_fields_for_predicate(matrix, "bottleneck_deadlock") == ["planner.event"]
    assert matrix_required_fields_for_predicate(matrix, "occlusion_triggered_near_miss") == []


def test_missing_matrix_marks_diagnostic_only() -> None:
    """Without a matrix, aggregate payloads must be diagnostic-only and claim-ineligible."""
    clean = _clean_trace()
    payload = aggregate_trace_failure_predicate_tables([clean], scenario_family="open_field")
    assert payload["matrix_metadata"]["status"] == MATRIX_STATUS_DIAGNOSTIC_ONLY
    assert payload["matrix_metadata"]["rate_interpretation"] == "not_allowed"
    health = build_trace_predicate_denominator_health_report(payload)
    assert all(
        report["claim_eligibility"] == MATRIX_CLAIM_INELIGIBLE
        for report in health["predicates"].values()
    )


def test_missing_matrix_required_fields_fail_closed() -> None:
    """A trace missing matrix-required fields should emit not_available rows."""
    clean = _clean_trace()
    matrix = _minimal_matrix()
    payload = aggregate_trace_failure_predicate_tables(
        [clean], scenario_family="open_field", matrix=matrix
    )
    assert payload["matrix_metadata"]["status"] == MATRIX_STATUS_PROPOSED
    assert payload["matrix_metadata"]["rate_interpretation"] == (
        "not_allowed_until_matrix_is_promoted"
    )
    not_available_rows = [r for r in payload["rows"] if r["validity_status"] == "not_available"]
    affected_predicates = {r["predicate_id"] for r in not_available_rows}
    assert "bottleneck_deadlock" in affected_predicates
    assert "zero_motion_timeout_behavior" in affected_predicates
    for row in not_available_rows:
        assert row["predicate_count"] >= 1
        assert row["severity"] == "not_available"


def test_matrix_claim_eligibility_requires_compliance() -> None:
    """Even with a matrix, missing required fields make the report claim-ineligible."""
    clean = _clean_trace()
    matrix = _minimal_matrix()
    payload = aggregate_trace_failure_predicate_tables(
        [clean], scenario_family="open_field", matrix=matrix
    )
    health = build_trace_predicate_denominator_health_report(payload)
    deadlock_report = health["predicates"]["bottleneck_deadlock"]
    assert deadlock_report["unavailable_data_count"] >= 1
    assert deadlock_report["claim_eligibility"] == MATRIX_CLAIM_INELIGIBLE


def test_complete_expected_matrix_slices_have_no_absent_slice_health() -> None:
    """Loaded traces covering every expected matrix slice should not add absent-slice health."""
    trace = _clean_trace(
        planner={"event": "step"},
        scenario_id="open_field",
        planner_id="dwa",
        seed=999,
    )
    matrix = _matrix_for_slices(
        scenario_families=["open_field"],
        planners=["dwa"],
        seeds=[999],
    )

    payload = aggregate_trace_failure_predicate_tables([trace], matrix=matrix)
    health = build_trace_predicate_denominator_health_report(payload)

    assert payload["absent_expected_slices"] == []
    assert payload["summary"]["absent_expected_slice_count"] == 0
    assert all(
        report["absent_expected_slice_count"] == 0 for report in health["predicates"].values()
    )


def test_missing_expected_matrix_slice_fails_closed() -> None:
    """A predeclared scenario/planner/seed slice absent from inputs should block claims."""
    trace = _clean_trace(
        planner={"event": "step"},
        scenario_id="open_field",
        planner_id="dwa",
        seed=999,
    )
    matrix = _matrix_for_slices(
        scenario_families=["open_field"],
        planners=["dwa"],
        seeds=[999, 1000],
    )

    payload = aggregate_trace_failure_predicate_tables([trace], matrix=matrix)
    health = build_trace_predicate_denominator_health_report(payload)

    assert payload["denominator_status"] == "not_available"
    assert payload["summary"]["absent_expected_slice_count"] == 1
    assert payload["absent_expected_slices"] == [
        {
            "scenario_family": "open_field",
            "planner_id": "dwa",
            "seed": 1000,
            "status": "absent_expected_slice",
        }
    ]
    for report in health["predicates"].values():
        assert report["absent_expected_slice_count"] == 1
        assert report["claim_eligibility"] == MATRIX_CLAIM_INELIGIBLE
        assert "expected matrix slice is absent" in report["allowed_wording"]


def test_failed_expected_matrix_slice_not_reported_absent() -> None:
    """Failed trace IDs that identify an expected slice should not become absent-slice zeros."""
    trace = _clean_trace(
        planner={"event": "step"},
        scenario_id="open_field",
        planner_id="dwa",
        seed=999,
    )
    matrix = _matrix_for_slices(
        scenario_families=["open_field"],
        planners=["dwa"],
        seeds=[999, 1000, 1001],
    )

    payload = aggregate_trace_failure_predicate_tables(
        [trace],
        matrix=matrix,
        failed_trace_ids=["open_field-dwa-1000.json"],
    )
    health = build_trace_predicate_denominator_health_report(payload)

    assert payload["denominator_status"] == "pipeline_failure"
    assert payload["summary"]["absent_expected_slice_count"] == 1
    assert payload["absent_expected_slices"] == [
        {
            "scenario_family": "open_field",
            "planner_id": "dwa",
            "seed": 1001,
            "status": "absent_expected_slice",
        }
    ]
    for report in health["predicates"].values():
        assert report["pipeline_failure_count"] == 1
        assert report["absent_expected_slice_count"] == 1
        assert report["claim_eligibility"] == MATRIX_CLAIM_INELIGIBLE


def test_structured_failed_slice_metadata_preferred_over_filename() -> None:
    """Structured failed slices should satisfy their slice without loose filename matching."""
    trace = _clean_trace(
        planner={"event": "step"},
        scenario_id="open_field",
        planner_id="dwa",
        seed=999,
    )
    matrix = _matrix_for_slices(
        scenario_families=["open_field"],
        planners=["dwa"],
        seeds=[999, 1000, 1001],
    )

    payload = aggregate_trace_failure_predicate_tables(
        [trace],
        matrix=matrix,
        failed_trace_ids=["open_field-dwa-1000.json"],
        failed_trace_slices=[
            {
                "scenario_family": "open_field",
                "scenario_id": "open_field",
                "planner_id": "dwa",
                "seed": 1001,
                "source_path": "open_field-dwa-1000.json",
            }
        ],
    )

    assert payload["summary"]["absent_expected_slice_count"] == 1
    assert payload["absent_expected_slices"] == [
        {
            "scenario_family": "open_field",
            "planner_id": "dwa",
            "seed": 1000,
            "status": "absent_expected_slice",
        }
    ]
    assert payload["failed_trace_slices"] == [
        {
            "scenario_family": "open_field",
            "scenario_id": "open_field",
            "planner_id": "dwa",
            "seed": 1001,
            "source_path": "open_field-dwa-1000.json",
        }
    ]


def test_structured_failed_slice_metadata_combines_uncovered_failed_ids() -> None:
    """Structured metadata should not hide unrelated failed trace IDs."""
    trace = _clean_trace(
        planner={"event": "step"},
        scenario_id="open_field",
        planner_id="dwa",
        seed=999,
    )
    matrix = _matrix_for_slices(
        scenario_families=["open_field"],
        planners=["dwa"],
        seeds=[999, 1000, 1001],
    )

    payload = aggregate_trace_failure_predicate_tables(
        [trace],
        matrix=matrix,
        failed_trace_ids=["open_field-dwa-1000.json"],
        failed_trace_slices=[
            {
                "scenario_family": "open_field",
                "scenario_id": "open_field",
                "planner_id": "dwa",
                "seed": 1001,
                "source_path": "open_field-dwa-1001.json",
            }
        ],
    )

    assert payload["summary"]["failed_trace_count"] == 2
    assert payload["summary"]["absent_expected_slice_count"] == 0
    assert payload["absent_expected_slices"] == []


def test_structured_failed_slice_metadata_deduplicates_source_path_count() -> None:
    """The same failed source reported by ID and structured metadata counts once."""
    matrix = _matrix_for_slices(
        scenario_families=["open_field"],
        planners=["dwa"],
        seeds=[1000],
    )

    payload = aggregate_trace_failure_predicate_tables(
        [],
        matrix=matrix,
        failed_trace_ids=["open_field-dwa-1000.json"],
        failed_trace_slices=[
            {
                "scenario_family": "open_field",
                "scenario_id": "open_field",
                "planner_id": "dwa",
                "seed": 1000,
                "source_path": "/tmp/open_field-dwa-1000.json",
            }
        ],
    )

    assert payload["summary"]["failed_trace_count"] == 1
    assert payload["summary"]["absent_expected_slice_count"] == 0


def test_incomplete_structured_failed_slice_metadata_fails_closed() -> None:
    """Incomplete structured failed slices should not satisfy expected matrix slices."""
    matrix = _matrix_for_slices(
        scenario_families=["open_field"],
        planners=["dwa"],
        seeds=[1000],
    )

    payload = aggregate_trace_failure_predicate_tables(
        [],
        matrix=matrix,
        failed_trace_ids=["open_field-dwa-1000.json"],
        failed_trace_slices=[
            {
                "scenario_family": "open_field",
                "planner_id": "dwa",
                "source_path": "open_field-dwa-1000.json",
            }
        ],
    )

    assert payload["denominator_status"] == "pipeline_failure"
    assert payload["summary"]["failed_trace_count"] == 1
    assert payload["summary"]["absent_expected_slice_count"] == 1
    assert payload["absent_expected_slices"] == [
        {
            "scenario_family": "open_field",
            "planner_id": "dwa",
            "seed": 1000,
            "status": "absent_expected_slice",
        }
    ]
    assert payload["failed_trace_slices"] == [
        {
            "scenario_family": "open_field",
            "planner_id": "dwa",
            "source_path": "open_field-dwa-1000.json",
        }
    ]


def test_failed_trace_id_requires_ordered_expected_slice_identifier() -> None:
    """Failed trace IDs should not satisfy expected slices through loose token collisions."""
    trace = _clean_trace(
        planner={"event": "step"},
        scenario_id="open_field",
        planner_id="dwa",
        seed=999,
    )
    matrix = _matrix_for_slices(
        scenario_families=["open_field"],
        planners=["dwa"],
        seeds=[999, 1000],
    )

    payload = aggregate_trace_failure_predicate_tables(
        [trace],
        matrix=matrix,
        failed_trace_ids=["dwa-open-field-rerun-1000.json"],
    )

    assert payload["summary"]["absent_expected_slice_count"] == 1
    assert payload["absent_expected_slices"] == [
        {
            "scenario_family": "open_field",
            "planner_id": "dwa",
            "seed": 1000,
            "status": "absent_expected_slice",
        }
    ]


def test_failed_trace_id_does_not_match_seed_prefix() -> None:
    """Failed trace IDs with longer seed tokens should not satisfy shorter expected seeds."""
    matrix = _matrix_for_slices(
        scenario_families=["open_field"],
        planners=["dwa"],
        seeds=[1000],
    )

    payload = aggregate_trace_failure_predicate_tables(
        [],
        matrix=matrix,
        failed_trace_ids=["open_field-dwa-10000.json"],
    )

    assert payload["summary"]["absent_expected_slice_count"] == 1
    assert payload["absent_expected_slices"] == [
        {
            "scenario_family": "open_field",
            "planner_id": "dwa",
            "seed": 1000,
            "status": "absent_expected_slice",
        }
    ]


def test_matrix_expected_slices_ignore_none_fields() -> None:
    """Matrix None entries should be missing, not coerced into literal slice identifiers."""
    trace = _clean_trace(
        planner={"event": "step"},
        scenario_id="open_field",
        planner_id="dwa",
        seed=999,
    )
    matrix = _matrix_for_slices(
        scenario_families=["open_field"],
        planners=["dwa"],
        seeds=[999],
    )
    matrix["matrix"]["scenario_families"].append(None)
    matrix["matrix"]["planners"].append(None)

    payload = aggregate_trace_failure_predicate_tables([trace], matrix=matrix)

    assert payload["summary"]["expected_matrix_slice_count"] == 1
    assert payload["absent_expected_slices"] == []


def test_proposed_matrix_keeps_satisfied_rows_claim_ineligible() -> None:
    """A proposed matrix is a prerequisite, not evidence that rates may be interpreted."""
    trace = _clean_trace(planner={"event": "timeout"})
    matrix = _minimal_matrix()
    payload = aggregate_trace_failure_predicate_tables(
        [trace], scenario_family="open_field", matrix=matrix
    )
    health = build_trace_predicate_denominator_health_report(payload)
    assert payload["matrix_metadata"]["status"] == MATRIX_STATUS_PROPOSED
    assert all(
        report["claim_eligibility"] == MATRIX_CLAIM_INELIGIBLE
        for report in health["predicates"].values()
    )


def test_matrix_with_all_required_fields_allows_rate_interpretation() -> None:
    """A promoted matrix plus satisfied required fields can allow rate interpretation."""
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
    matrix = _minimal_matrix()
    matrix["status"] = MATRIX_STATUS_RATE_INTERPRETABLE
    payload = extract_trace_failure_predicates(trace, scenario_family="bottleneck", matrix=matrix)
    assert payload["matrix_metadata"]["status"] == MATRIX_STATUS_RATE_INTERPRETABLE
    predicates = {row["predicate_id"]: row for row in payload["predicates"]}
    assert predicates["bottleneck_deadlock"]["validity_status"] == "valid"
    assert predicates["late_evasive_reaction"]["validity_status"] == "valid"
