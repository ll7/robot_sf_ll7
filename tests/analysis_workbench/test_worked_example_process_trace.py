"""Contract tests for ``worked_example_process_trace.v1`` diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from robot_sf import analysis_workbench
from robot_sf.analysis_workbench import interaction_coordinates as coordinates
from robot_sf.analysis_workbench.interaction_coordinates import (
    WORKED_EXAMPLE_PROCESS_TRACE_SCHEMA_VERSION,
    ConflictZoneSpec,
    RouteSpec,
    WorkedExampleProcessTraceValidationError,
    build_worked_example_process_trace,
    build_worked_example_process_trace_from_export,
    load_registered_conflict_zone_spec,
    load_registered_route_spec,
    load_worked_example_process_trace_schema,
    validate_worked_example_process_trace,
)
from robot_sf.analysis_workbench.process_trace_receipt import simulation_trace_receipt_sha256
from robot_sf.analysis_workbench.simulation_trace_export import simulation_trace_export_from_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "scripts" / "analysis" / "build_worked_example_process_trace.py"
TRACE_FIXTURE_PATH = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
    / "minimal_trace.json"
)
GEOMETRY_REGISTRY_FIXTURE_PATH = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "analysis_workbench"
    / "process_trace_geometry_registry_v1"
    / "fixture_registry.json"
)
GEOMETRY_REGISTRY_ARTIFACT_REF = (
    "tests/fixtures/analysis_workbench/process_trace_geometry_registry_v1/fixture_registry.json"
)


def test_process_trace_coordinate_frames_explicit_available_and_unavailable() -> None:
    """All four frame families should expose explicit status and reasons."""

    trace = simulation_trace_export_from_dict(_trace_payload())

    unavailable = build_worked_example_process_trace_from_export(trace)
    assert unavailable["coordinate_frames"]["world"]["status"] == "available"
    assert unavailable["coordinate_frames"]["route"] == {
        "status": "unavailable",
        "reason": "registered_route_unavailable",
        "input_contract": {"status": "not_supplied"},
    }
    assert unavailable["coordinate_frames"]["conflict"] == {
        "status": "unavailable",
        "reason": "registered_conflict_zone_unavailable",
        "input_contract": {"status": "not_supplied"},
    }
    assert unavailable["coordinate_frames"]["relative_interaction"]["status"] == "available"

    available = build_worked_example_process_trace_from_export(
        trace,
        route=_registered_route("r-main"),
        conflict_zone=_registered_conflict_zone("door"),
    )

    validate_worked_example_process_trace(available)
    assert available["schema_version"] == WORKED_EXAMPLE_PROCESS_TRACE_SCHEMA_VERSION
    assert available["coordinate_frames"]["route"]["status"] == "available"
    assert available["coordinate_frames"]["conflict"]["status"] == "available"
    assert available["coordinate_frames"]["route"]["input_contract"]["status"] == "supplied"
    assert available["coordinate_frames"]["conflict"]["input_contract"]["status"] == "supplied"
    assert available["frames"][1]["route"]["s_m"] == pytest.approx(0.2)
    assert available["frames"][0]["relative_interaction"]["relative_longitudinal_m"] == (
        pytest.approx(1.0)
    )


def test_analysis_input_contract_binds_identity_and_rejects_whole_projection_transplant() -> None:
    """Same-source derived views cannot erase the exact analysis inputs under one identity."""

    trace = simulation_trace_export_from_dict(_trace_payload())
    pair_trace = simulation_trace_export_from_dict(
        _trace_payload(trace_id="pair-right", planner_id="planner-b", seed=7)
    )
    bound = build_worked_example_process_trace_from_export(
        trace,
        route=_registered_route("route-b"),
        conflict_zone=_registered_conflict_zone("zone-b"),
        focal_actor_id="ped-a",
        pair_trace=pair_trace,
        pair_comparison_grain="matched_planner_pair",
        encounter_report=_encounter_report(start_time_s=0.1, end_time_s=0.2),
        encounter_report_input_checksum="0" * 64,
    )
    alternate = build_worked_example_process_trace_from_export(trace)

    assert bound["analysis_input_contract"]["schema_version"] == (
        "worked_example_process_trace_analysis_input.v1"
    )
    assert bound["process_trace_id"].endswith(bound["analysis_input_sha256"])
    assert bound["analysis_input_contract"]["encounter_report"]["status"] == "supplied"
    assert bound["analysis_input_contract"]["pair_trace"]["status"] == "supplied"

    forged = deepcopy(alternate)
    forged["process_trace_id"] = bound["process_trace_id"]
    with pytest.raises(Exception, match="/process_trace_id"):
        validate_worked_example_process_trace(forged)

    derived_transplant = deepcopy(bound)
    for key in (
        "coordinate_frames",
        "frames",
        "diagnostics",
        "event_anchors",
        "event_anchor_hierarchy",
    ):
        derived_transplant[key] = deepcopy(alternate[key])
    with pytest.raises(Exception, match="/coordinate_frames"):
        validate_worked_example_process_trace(derived_transplant)

    pair_transplant = deepcopy(bound)
    pair_transplant["pair_compatibility"] = deepcopy(alternate["pair_compatibility"])
    with pytest.raises(Exception, match="/pair_compatibility"):
        validate_worked_example_process_trace(pair_transplant)


@pytest.mark.parametrize(
    ("path", "value", "error_path"),
    (
        (
            ["encounters", "focal", "actor_contiguity", "status"],
            "unavailable",
            "/encounters/focal/actor_contiguity/status",
        ),
        (
            ["frames", 0, "encounter_interval"],
            {"status": "in_interval", "reason": "canonical_encounter_interval"},
            "/frames/0/encounter_interval/status",
        ),
        (["frames", 0, "frame_index"], 99, "/frames/0/frame_index"),
        (["source_coordinate_frame"], "robot", "/source_coordinate_frame"),
        (["units"], {"distance": "km"}, "/units"),
        (["evidence_boundary"], "analysis_workbench_only_but_forged", "/evidence_boundary"),
    ),
)
def test_full_artifact_replay_binds_focal_interval_and_envelope(
    path: list[object],
    value: object,
    error_path: str,
) -> None:
    """Every public surface must reconstruct from source and bound analysis inputs."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()),
        encounter_report=_encounter_report(start_time_s=0.1, end_time_s=0.2),
        encounter_report_input_checksum="0" * 64,
    )
    forged = deepcopy(payload)
    _set_path(forged, path, value)

    with pytest.raises(Exception, match=error_path):
        validate_worked_example_process_trace(forged)


@pytest.mark.parametrize(
    ("path", "value", "error_path"),
    (
        (
            ["encounters", "focal", "actor_contiguity", "contiguous"],
            1,
            "/encounters/focal/actor_contiguity/contiguous",
        ),
        (
            ["pair_compatibility", "provenance_gate", "checks", "scenario_id_equal"],
            1,
            "/pair_compatibility/provenance_gate/checks/scenario_id_equal",
        ),
        (
            ["pair_compatibility", "route_spawn_separation", "initial_robot_separation_m"],
            False,
            "/pair_compatibility/route_spawn_separation/initial_robot_separation_m",
        ),
    ),
)
def test_full_artifact_replay_is_json_type_sensitive(
    path: list[object],
    value: object,
    error_path: str,
) -> None:
    """JSON booleans and numbers must never replay as interchangeable values."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()),
        pair_trace=simulation_trace_export_from_dict(
            _trace_payload(trace_id="pair-right", planner_id="planner-b")
        ),
        pair_comparison_grain="matched_planner_pair",
    )
    forged = deepcopy(payload)
    _set_path(forged, path, value)

    with pytest.raises(Exception, match=error_path):
        validate_worked_example_process_trace(forged)


def test_public_schema_types_pair_and_actor_contiguity_fields_strictly() -> None:
    """The public schema must reject bool/number substitutions without replay support."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()),
        pair_trace=simulation_trace_export_from_dict(
            _trace_payload(trace_id="pair-right", planner_id="planner-b")
        ),
        pair_comparison_grain="matched_planner_pair",
    )
    validator = Draft202012Validator(load_worked_example_process_trace_schema())
    probes = (
        (["encounters", "focal", "actor_contiguity", "contiguous"], 1),
        (["pair_compatibility", "provenance_gate", "checks", "scenario_id_equal"], 1),
        (["pair_compatibility", "route_spawn_separation", "initial_robot_separation_m"], False),
    )

    for path, value in probes:
        forged = deepcopy(payload)
        _set_path(forged, path, value)
        assert any(
            path[: len(error.absolute_path)] == list(error.absolute_path)
            for error in validator.iter_errors(forged)
        )


def test_external_expected_artifact_digest_rejects_coherent_report_downgrade() -> None:
    """Admission, not a self-authored receipt, binds a fully rewritten artifact."""

    trace = simulation_trace_export_from_dict(_trace_payload())
    admitted = build_worked_example_process_trace_from_export(
        trace,
        encounter_report=_encounter_report(start_time_s=0.1, end_time_s=0.2),
        encounter_report_input_checksum="0" * 64,
    )
    coherently_downgraded = build_worked_example_process_trace_from_export(trace)

    validate_worked_example_process_trace(
        admitted,
        expected_artifact_sha256=analysis_workbench.worked_example_process_trace_artifact_sha256(
            admitted
        ),
    )
    with pytest.raises(Exception, match="/artifact_sha256"):
        validate_worked_example_process_trace(
            coherently_downgraded,
            expected_artifact_sha256=analysis_workbench.worked_example_process_trace_artifact_sha256(
                admitted
            ),
        )


def test_public_artifact_digest_hashes_exact_official_writer_bytes() -> None:
    """External admission uses the deterministic bytes written to disk, including newline."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload())
    )
    expected_bytes = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    expected_sha256 = hashlib.sha256(expected_bytes).hexdigest()

    assert analysis_workbench.serialize_worked_example_process_trace(payload) == expected_bytes
    assert (
        analysis_workbench.worked_example_process_trace_artifact_sha256(payload) == expected_sha256
    )
    validate_worked_example_process_trace(
        payload,
        expected_artifact_sha256=expected_sha256,
    )
    with pytest.raises(Exception, match="/artifact_sha256"):
        validate_worked_example_process_trace(
            payload,
            expected_artifact_sha256=_json_digest(payload),
        )


def test_public_validation_normalizes_malformed_json_types() -> None:
    """Malformed digests, values, and non-string keys must never leak raw TypeError."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload())
    )
    with pytest.raises(WorkedExampleProcessTraceValidationError, match="/artifact_sha256"):
        validate_worked_example_process_trace(
            payload,
            expected_artifact_sha256=123,  # type: ignore[arg-type]
        )

    malformed = deepcopy(payload)
    malformed["source_trace"]["content_receipt"]["content_contract"]["frames"][0]["planner"][1] = (
        "non-string-key"
    )
    with pytest.raises(WorkedExampleProcessTraceValidationError):
        validate_worked_example_process_trace(malformed)

    trace = simulation_trace_export_from_dict(_trace_payload())
    malformed_frame = replace(
        trace.frames[0],
        planner={**trace.frames[0].planner, 1: "non-string-key"},
    )
    malformed_trace = replace(trace, frames=(malformed_frame, *trace.frames[1:]))
    with pytest.raises(WorkedExampleProcessTraceValidationError):
        build_worked_example_process_trace_from_export(malformed_trace)


def test_public_serializer_rejects_values_outside_exact_json_domain() -> None:
    """Digest helpers cannot coerce keys, containers, or nonfinite scalars into collisions."""

    malformed_payloads = (
        {1: "integer-key"},
        {"tuple": (1, 2)},
        {"nonfinite": math.nan},
    )
    for malformed in malformed_payloads:
        with pytest.raises(WorkedExampleProcessTraceValidationError):
            analysis_workbench.serialize_worked_example_process_trace(malformed)  # type: ignore[arg-type]
        with pytest.raises(WorkedExampleProcessTraceValidationError):
            analysis_workbench.worked_example_process_trace_artifact_sha256(malformed)  # type: ignore[arg-type]


def test_source_receipt_rejects_tuple_json_lookalike() -> None:
    """A malformed tuple cannot share a source receipt with its JSON-list lookalike."""

    list_payload = _trace_payload()
    list_payload["frames"][0]["planner"]["receipt_probe"] = [1, 2]
    build_worked_example_process_trace_from_export(simulation_trace_export_from_dict(list_payload))

    tuple_payload = deepcopy(list_payload)
    tuple_payload["frames"][0]["planner"]["receipt_probe"] = (1, 2)
    with pytest.raises(WorkedExampleProcessTraceValidationError):
        build_worked_example_process_trace_from_export(
            simulation_trace_export_from_dict(tuple_payload)
        )


def test_source_receipt_digest_rejects_tuple_json_lookalike() -> None:
    """The receipt digest itself must reject tuple-to-list canonicalization."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload())
    )
    malformed_receipt = deepcopy(payload["source_trace"]["content_receipt"])
    malformed_receipt["content_contract"]["frames"][0]["planner"]["receipt_probe"] = (1, 2)

    with pytest.raises(TypeError):
        simulation_trace_receipt_sha256(malformed_receipt)


def test_pair_receipt_rejects_tuple_json_lookalike() -> None:
    """A malformed pair tuple cannot share its content and analysis receipt with a list."""

    left = simulation_trace_export_from_dict(_trace_payload(trace_id="pair-tuple-left"))
    list_payload = _trace_payload(trace_id="pair-tuple-right", planner_id="planner-b")
    list_payload["frames"][0]["planner"]["receipt_probe"] = [1, 2]
    build_worked_example_process_trace_from_export(
        left,
        pair_trace=simulation_trace_export_from_dict(list_payload),
        pair_comparison_grain="matched_planner_pair",
    )

    tuple_payload = deepcopy(list_payload)
    tuple_payload["frames"][0]["planner"]["receipt_probe"] = (1, 2)
    with pytest.raises(WorkedExampleProcessTraceValidationError):
        build_worked_example_process_trace_from_export(
            left,
            pair_trace=simulation_trace_export_from_dict(tuple_payload),
            pair_comparison_grain="matched_planner_pair",
        )


def test_report_receipt_rejects_tuple_json_lookalike() -> None:
    """A malformed report tuple cannot share receipt, analysis, or writer SHA with a list."""

    trace = simulation_trace_export_from_dict(_trace_payload())
    list_report = _encounter_report(start_time_s=0.1, end_time_s=0.2)
    build_worked_example_process_trace_from_export(
        trace,
        encounter_report=list_report,
        encounter_report_input_checksum="0" * 64,
    )

    tuple_report = deepcopy(list_report)
    tuple_report["exclusions"] = ()
    with pytest.raises(WorkedExampleProcessTraceValidationError):
        build_worked_example_process_trace_from_export(
            trace,
            encounter_report=tuple_report,
            encounter_report_input_checksum="0" * 64,
        )


def test_direct_geometry_inputs_cannot_coerce_nonfinite_values_or_keys() -> None:
    """Direct RouteSpec inputs fail cleanly instead of colliding with JSON lookalikes."""

    trace = simulation_trace_export_from_dict(_trace_payload())
    actual_nan = RouteSpec("route", (0.0, 0.0), (math.nan, 0.0))
    literal_lookalike = RouteSpec(
        "route",
        (0.0, 0.0),
        ({"nonfinite_number": "nan"}, 0.0),  # type: ignore[arg-type]
    )
    integer_key = RouteSpec(
        "route",
        (0.0, 0.0),
        (1.0, 0.0),
        geometry={1: "integer-key"},  # type: ignore[dict-item]
    )

    for route in (actual_nan, literal_lookalike, integer_key):
        with pytest.raises(WorkedExampleProcessTraceValidationError):
            build_worked_example_process_trace_from_export(trace, route=route)


def test_bound_report_rejects_coherent_inner_selected_record_rehash() -> None:
    """Inner report receipts cannot replace the separately bound report input."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()),
        encounter_report=_encounter_report(start_time_s=0.1, end_time_s=0.2),
        encounter_report_input_checksum="0" * 64,
    )
    forged = deepcopy(payload)
    declared = forged["encounters"]["focal"]["declared_encounter"]
    report_input = declared["report_input_contract"]
    selected = report_input["content_contract"]["encounters"][report_input["selected_entry_index"]]
    selected["minimum_clearance_m"] = 0.123
    declared["canonical_record"]["minimum_clearance_m"] = 0.123
    report_input["selected_entry_sha256"] = _json_digest(selected)
    report_input["content_sha256"] = _json_digest(report_input["content_contract"])

    with pytest.raises(Exception, match="/encounters/focal/declared_encounter"):
        validate_worked_example_process_trace(forged)


def test_full_reconstruction_rejects_coherent_source_and_focal_rebinding() -> None:
    """A new source receipt and identity cannot bless another focal actor's derived views."""

    source_a = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(
            _trace_payload(trace_id="focal-source-a", actor_switch=True)
        ),
        focal_actor_id="ped-a",
    )
    source_b = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(
            _trace_payload(trace_id="focal-source-b", actor_switch=True)
        ),
        focal_actor_id="ped-b",
    )
    forged = deepcopy(source_b)
    forged["source_trace"] = deepcopy(source_a["source_trace"])
    forged["analysis_input_contract"]["source_trace_content_sha256"] = source_a["source_trace"][
        "content_sha256"
    ]
    forged["analysis_input_contract"]["focal_actor_id"] = "ped-a"
    analysis_digest = _json_digest(forged["analysis_input_contract"])
    forged["analysis_input_sha256"] = analysis_digest
    forged["process_trace_id"] = f"focal-source-a-process-trace-{analysis_digest}"

    with pytest.raises(Exception, match="/encounters/focal/actor_id"):
        validate_worked_example_process_trace(forged)


@pytest.mark.parametrize(
    ("mutation", "error_path"),
    (
        ("unknown_top_level", "/source_trace/content_receipt"),
        ("promoted_evidence", "/source_trace/content_receipt"),
        ("unknown_units", "/source_trace/content_receipt"),
    ),
)
def test_embedded_source_receipt_is_exact_simulation_trace_export_contract(
    mutation: str,
    error_path: str,
) -> None:
    """The durable source receipt cannot broaden the admitted export schema."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(nonfinite_heading_step=1))
    )
    forged = deepcopy(payload)
    receipt = forged["source_trace"]["content_receipt"]
    contract = receipt["content_contract"]
    if mutation == "unknown_top_level":
        contract["promoted_claim"] = True
    elif mutation == "promoted_evidence":
        contract["evidence_boundary"] = "benchmark_evidence"
    else:
        contract["units"]["acceleration"] = "m/s^2"
    source_digest = _json_digest(receipt)
    forged["source_trace"]["content_sha256"] = source_digest
    forged["analysis_input_contract"]["source_trace_content_sha256"] = source_digest
    analysis_digest = _json_digest(forged["analysis_input_contract"])
    forged["analysis_input_sha256"] = analysis_digest
    forged["process_trace_id"] = f"{contract['trace_id']}-process-trace-{analysis_digest}"

    with pytest.raises(Exception, match=error_path):
        validate_worked_example_process_trace(forged)


def test_process_trace_actor_consistency_and_actor_switch_reporting() -> None:
    """The focal encounter should not switch when global nearest actor changes."""

    trace = simulation_trace_export_from_dict(_trace_payload(actor_switch=True))

    payload = build_worked_example_process_trace_from_export(trace)

    assert payload["encounters"]["focal"]["actor_id"] == "ped-a"
    assert payload["encounters"]["focal"]["encounter_id"] == "ped-a:encounter-0001"
    assert [
        frame["relative_interaction"]["actor_id"]
        for frame in payload["frames"]
        if frame["relative_interaction"]["status"] == "available"
    ] == ["ped-a", "ped-a", "ped-a", "ped-a"]
    switches = payload["encounters"]["actor_switch_events"]
    assert switches == [
        {
            "event_type": "global_minimum_actor_switch",
            "step": 2,
            "time_s": 0.2,
            "previous_actor_id": "ped-a",
            "new_actor_id": "ped-b",
            "status": "available",
        }
    ]


def test_process_trace_surface_clearance_and_fail_closed_closest_approach() -> None:
    """Surface clearance must stay distinct and velocity degeneracy must be unavailable."""

    trace = simulation_trace_export_from_dict(_trace_payload(static_relative_velocity=True))

    payload = build_worked_example_process_trace_from_export(trace)

    rel = payload["frames"][0]["relative_interaction"]
    assert rel["center_distance_m"] == pytest.approx(1.0)
    assert rel["proxy_surface_clearance_m"] == pytest.approx(0.5)
    assert rel["geometric_body_clearance_status"] == "unavailable"
    assert rel["closest_approach"] == {
        "status": "unavailable",
        "reason": "degenerate_relative_velocity",
    }


def test_process_trace_event_anchors_are_versioned_and_profiled() -> None:
    """Semantic anchors should report detector version, source fields, and status."""

    trace = simulation_trace_export_from_dict(_trace_payload(turn_and_decelerate=True))

    payload = build_worked_example_process_trace_from_export(trace)
    events = {event["event_type"]: event for event in payload["event_anchors"]}

    assert events["minimum_clearance"]["status"] == "available"
    assert events["minimum_clearance"]["detector_profile_version"] == (
        "worked_example_event_detectors.v1"
    )
    assert events["minimum_clearance"]["visual_anchor_eligibility"]["eligible"] is True
    assert events["first_material_deceleration"]["step"] == 1
    assert events["first_material_turn_response"]["step"] == 2
    assert events["conflict_zone_entry"]["status"] == "unavailable"
    assert events["first_safety_predicate_breach"]["status"] == "available"
    assert "trace_end_boundary" not in events
    assert events["terminal_event"]["status"] == "unavailable"
    assert events["terminal_event"]["event_relative_time"]["status"] == "unavailable"
    assert events["terminal_event"]["visual_anchor_eligibility"] == {
        "eligible": False,
        "reason": "event_unavailable",
    }
    assert payload["event_anchor_hierarchy"]["selected_anchor"]["event_type"] in {
        "exact_collision_event",
        "minimum_clearance",
        "first_safety_predicate_breach",
    }
    assert payload["frames"][0]["event_alignment"]["status"] == "available"
    assert payload["frames"][0]["event_alignment"]["tau_s"] == pytest.approx(
        -payload["event_anchor_hierarchy"]["anchor_time_s"]
    )
    assert payload["diagnostics"]["stall"]["profile_version"] == "worked_example_phase_profile.v1"
    assert payload["diagnostics"]["reversal_counts"]["profile_version"] == (
        "worked_example_reversal_profile.v1"
    )


def test_pair_compatibility_rejects_divergence_without_shared_prefix() -> None:
    """Doorway-style matched starts with early divergence must prohibit difference curves."""

    left = simulation_trace_export_from_dict(_trace_payload(trace_id="doorway-seed-113"))
    right = simulation_trace_export_from_dict(
        _trace_payload(trace_id="doorway-seed-114", diverge_after_start=True, seed=114)
    )

    payload = build_worked_example_process_trace_from_export(
        left,
        pair_trace=right,
        pair_comparison_grain="matched_realization_pair",
    )
    pair = payload["pair_compatibility"]

    assert pair["initial_state_equivalence"]["equivalent"] is True
    assert pair["shared_prefix"]["shared_prefix"] is False
    assert pair["divergence_interpretation"] == {
        "allowed": False,
        "reason": "no_shared_prefix_reject_divergence_output",
    }


def test_pair_compatibility_matched_start_common_anchors_without_duration_normalization() -> None:
    """Matched starts can share anchors without forcing episodes onto a common duration."""

    left = simulation_trace_export_from_dict(_trace_payload(trace_id="double-bottleneck-goal"))
    right_payload = _trace_payload(trace_id="double-bottleneck-ppo", seed=118)
    right_payload["frames"].append(
        {
            "step": 4,
            "time_s": 0.4,
            "robot": {
                "position": [0.8, 0.0],
                "heading": 0.0,
                "velocity": [1.0, 0.0],
                "radius": 0.25,
            },
            "pedestrians": [
                {"id": "ped-a", "position": [1.4, 0.0], "velocity": [0.0, 0.0], "radius": 0.25}
            ],
            "planner": {
                "selected_action": {"linear_velocity": 0.5, "angular_velocity": 0.0},
                "encounter": {"actor_id": "ped-a", "encounter_id": "ped-a:encounter-0001"},
                "event": "step",
                "run_config": {
                    "map_id": "fixture-map",
                    "horizon": 4,
                    "config_digest": "a" * 64,
                    "time_step_s": 0.1,
                },
            },
        }
    )
    right = simulation_trace_export_from_dict(right_payload)

    payload = build_worked_example_process_trace_from_export(
        left,
        pair_trace=right,
        pair_comparison_grain="matched_realization_pair",
    )
    pair = payload["pair_compatibility"]

    assert pair["initial_state_equivalence"]["equivalent"] is True
    assert pair["shared_prefix"]["shared_prefix"] is True
    assert any(
        anchor["event_type"] == "minimum_clearance" for anchor in pair["valid_common_event_anchors"]
    )
    assert pair["duration_normalization"] == {"applied": False}


def test_pair_grain_fail_closed_requires_declared_relationship_and_metadata() -> None:
    """Pair gates should require grain-specific relationships and metadata."""

    left = simulation_trace_export_from_dict(_trace_payload(planner_id="planner-a", seed=7))
    right = simulation_trace_export_from_dict(
        _trace_payload(planner_id="planner-b", seed=7, config_digest="b" * 64)
    )

    planner_pair = build_worked_example_process_trace_from_export(
        left,
        pair_trace=right,
        pair_comparison_grain="matched_planner_pair",
    )["pair_compatibility"]
    assert planner_pair["status"] == "available"
    assert planner_pair["provenance_gate"]["checks"]["seed_equal"] is True
    assert planner_pair["provenance_gate"]["checks"]["planner_id_different"] is True
    assert planner_pair["provenance_gate"]["checks"]["config_digest_equal"] is False

    wrong_grain = build_worked_example_process_trace_from_export(
        left,
        pair_trace=right,
        pair_comparison_grain="matched_realization_pair",
    )["pair_compatibility"]
    assert wrong_grain["status"] == "incompatible"
    assert wrong_grain["provenance_gate"]["checks"]["planner_id_equal"] is False

    missing_meta_payload = _trace_payload(include_run_config=False, planner_id="planner-a", seed=7)
    missing_meta = simulation_trace_export_from_dict(missing_meta_payload)
    missing_meta_pair = simulation_trace_export_from_dict(
        _trace_payload(include_run_config=False, planner_id="planner-b", seed=7)
    )
    pair = build_worked_example_process_trace_from_export(
        missing_meta,
        pair_trace=missing_meta_pair,
        pair_comparison_grain="matched_planner_pair",
    )["pair_compatibility"]
    assert pair["status"] == "incompatible"
    assert pair["provenance_gate"]["checks"]["map_id_present"] is False

    realization_config_mismatch = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(planner_id="ppo", seed=7)),
        pair_trace=simulation_trace_export_from_dict(
            _trace_payload(planner_id="ppo", seed=8, config_digest="b" * 64)
        ),
        pair_comparison_grain="matched_realization_pair",
    )["pair_compatibility"]
    assert realization_config_mismatch["status"] == "incompatible"
    assert realization_config_mismatch["provenance_gate"]["checks"]["config_digest_equal"] is False


def test_process_trace_cli_is_deterministic(tmp_path: Path) -> None:
    """Repeated CLI generation should produce byte-identical JSON."""

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    command = [
        sys.executable,
        str(CLI_PATH),
        "--input",
        str(TRACE_FIXTURE_PATH),
        "--geometry-registry",
        str(GEOMETRY_REGISTRY_FIXTURE_PATH),
        "--route-entry-id",
        "fixture-route",
        "--conflict-zone-entry-id",
        "fixture-zone",
        "--out",
    ]

    for output in (first, second):
        result = subprocess.run(
            [*command, str(output)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    assert first.read_bytes() == second.read_bytes()
    writer_sha256 = hashlib.sha256(second.read_bytes()).hexdigest()
    assert result.stdout.strip() == f"wrote {second} sha256={writer_sha256}"
    validate_worked_example_process_trace(
        json.loads(first.read_text(encoding="utf-8")),
        expected_artifact_sha256=writer_sha256,
    )


def test_process_trace_cli_resolves_canonical_geometry_owners(tmp_path: Path) -> None:
    """A repeatable logical-ref resolver verifies route and conflict owners without path leaks."""

    registry = json.loads(GEOMETRY_REGISTRY_FIXTURE_PATH.read_text(encoding="utf-8"))
    route_entry = next(
        entry for entry in registry["routes"] if entry["entry_id"] == "fixture-route"
    )
    conflict_entry = next(
        entry for entry in registry["conflict_zones"] if entry["entry_id"] == "fixture-zone"
    )
    owner_ref = "owners/cli-fixture-map.json"
    route_selector = {"map_id": "fixture-map", "route_id": "fixture-route"}
    conflict_selector = {"map_id": "fixture-map", "zone_id": "fixture-zone"}
    owner_path = tmp_path / "owner.json"
    owner_path.write_text(
        json.dumps(
            {
                "schema_version": "process_trace_geometry_owner.v1",
                "geometry_bindings": [
                    {"selector": route_selector, "geometry": route_entry["geometry"]},
                    {"selector": conflict_selector, "geometry": conflict_entry["geometry"]},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    owner_digest = hashlib.sha256(owner_path.read_bytes()).hexdigest()
    route_entry["upstream_binding"] = {
        "kind": "canonical_source",
        "source_artifact_ref": owner_ref,
        "source_content_sha256": owner_digest,
        "selector": route_selector,
    }
    conflict_entry["upstream_binding"] = {
        "kind": "canonical_source",
        "source_artifact_ref": owner_ref,
        "source_content_sha256": owner_digest,
        "selector": conflict_selector,
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    output_path = tmp_path / "process-trace.json"

    result = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--input",
            str(TRACE_FIXTURE_PATH),
            "--geometry-registry",
            str(registry_path),
            "--route-entry-id",
            "fixture-route",
            "--conflict-zone-entry-id",
            "fixture-zone",
            "--geometry-owner",
            f"{owner_ref}={owner_path}",
            "--out",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    for key in ("route", "conflict"):
        assert payload["coordinate_frames"][key]["status"] == "available"
        assert (
            payload["coordinate_frames"][key]["registry"]["owner_validation"]["status"]
            == "verified"
        )
    assert str(owner_path) not in json.dumps(payload, sort_keys=True)


@pytest.mark.parametrize(
    "owner_args",
    (
        ["missing-equals"],
        ["owners/map.json=/tmp/one.json", "owners/map.json=/tmp/two.json"],
    ),
)
def test_process_trace_cli_rejects_malformed_or_duplicate_owner_resolvers(
    tmp_path: Path,
    owner_args: list[str],
) -> None:
    """Each logical owner ref must have exactly one well-formed private path mapping."""

    command = [
        sys.executable,
        str(CLI_PATH),
        "--input",
        str(TRACE_FIXTURE_PATH),
        "--out",
        str(tmp_path / "output.json"),
    ]
    for owner_arg in owner_args:
        command.extend(("--geometry-owner", owner_arg))
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--geometry-owner" in result.stderr


def test_process_trace_cli_requires_pair_grain_for_pair_input(tmp_path: Path) -> None:
    """Pair inputs must declare their comparison grain explicitly."""

    result = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--input",
            str(TRACE_FIXTURE_PATH),
            "--pair-input",
            str(TRACE_FIXTURE_PATH),
            "--out",
            str(tmp_path / "out.json"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--pair-comparison-grain is required" in result.stderr


def test_process_trace_rejects_mismatched_requested_actor() -> None:
    """Requested actor IDs must not contradict declared encounter bindings."""

    trace = simulation_trace_export_from_dict(_trace_payload())

    payload = build_worked_example_process_trace_from_export(trace, focal_actor_id="ped-missing")

    assert payload["coordinate_frames"]["relative_interaction"] == {
        "status": "unavailable",
        "reason": "requested_focal_actor_missing",
    }
    assert payload["encounters"]["focal"]["reason"] == "requested_focal_actor_missing"


def test_process_trace_preserves_robot_frame_without_world_claim() -> None:
    """Robot-frame source traces must not be labeled as world-frame diagnostics."""

    trace = simulation_trace_export_from_dict(_trace_payload(coordinate_frame="robot"))

    payload = build_worked_example_process_trace_from_export(trace)

    assert payload["source_coordinate_frame"] == "robot"
    assert payload["coordinate_frames"]["world"]["status"] == "unavailable"
    assert payload["frames"][0]["world"]["status"] == "unavailable"
    assert payload["frames"][0]["source_coordinates"]["coordinate_frame"] == "robot"


def test_robot_frame_route_and_conflict_contracts_are_explicitly_unavailable() -> None:
    """Valid world geometry cannot make robot-frame source coordinates projectable."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(coordinate_frame="robot")),
        route=_registered_route("route-b"),
        conflict_zone=_registered_conflict_zone("zone-b"),
    )

    assert {
        key: payload["coordinate_frames"]["route"][key]
        for key in ("status", "reason", "source_coordinate_frame")
    } == {
        "status": "unavailable",
        "reason": "source_coordinate_frame_not_world",
        "source_coordinate_frame": "robot",
    }
    assert payload["coordinate_frames"]["route"]["input_contract"]["status"] == "supplied"
    assert {
        key: payload["coordinate_frames"]["conflict"][key]
        for key in ("status", "reason", "source_coordinate_frame")
    } == {
        "status": "unavailable",
        "reason": "source_coordinate_frame_not_world",
        "source_coordinate_frame": "robot",
    }
    assert payload["coordinate_frames"]["conflict"]["input_contract"]["status"] == "supplied"
    assert {frame["route"]["status"] for frame in payload["frames"]} == {"unavailable"}
    assert {frame["conflict"]["status"] for frame in payload["frames"]} == {"unavailable"}
    validate_worked_example_process_trace(payload)


def test_invalid_conflict_zone_is_unavailable_everywhere() -> None:
    """Invalid conflict geometry should not leak per-frame occupancy values."""

    trace = simulation_trace_export_from_dict(_trace_payload())

    payload = build_worked_example_process_trace_from_export(
        trace,
        conflict_zone=ConflictZoneSpec("bad-zone", (0.0, 0.0), -1.0, "bad-zone.v1", "1" * 64),
    )

    assert payload["coordinate_frames"]["conflict"]["reason"] == "registered_conflict_zone_invalid"
    assert {frame["conflict"]["status"] for frame in payload["frames"]} == {"unavailable"}
    assert payload["diagnostics"]["conflict_zone_occupancy"]["robot_duration_s"] is None


def test_exact_collision_uses_telemetry_and_proxy_overlap_is_separate() -> None:
    """Proxy overlap must not masquerade as exact collision telemetry."""

    trace = simulation_trace_export_from_dict(_trace_payload(proxy_overlap=True))

    payload = build_worked_example_process_trace_from_export(trace)
    events = {event["event_type"]: event for event in payload["event_anchors"]}

    assert events["exact_collision_event"]["status"] == "unavailable"
    assert events["proxy_overlap_event"]["status"] == "available"


def test_pair_compatibility_allows_sensitivity_anchors_without_divergence() -> None:
    """Doorway start/spawn sensitivity can align anchors but not emit divergence curves."""

    left = simulation_trace_export_from_dict(_trace_payload(trace_id="doorway-seed-113"))
    right = simulation_trace_export_from_dict(
        _trace_payload(trace_id="doorway-seed-114", actor_start_offset=0.2, seed=114)
    )

    payload = build_worked_example_process_trace_from_export(
        left,
        pair_trace=right,
        pair_comparison_grain="matched_realization_pair",
    )
    pair = payload["pair_compatibility"]

    assert pair["status"] == "available"
    assert pair["initial_state_equivalence"]["equivalent"] is False
    assert pair["shared_prefix"]["shared_prefix"] is False
    assert (
        pair["divergence_interpretation"]["reason"] == "no_shared_prefix_reject_divergence_output"
    )


def test_schema_rejects_metrics_snqi_and_unknown_event_fields() -> None:
    """The schema should reject benchmark metric replacements and arbitrary event objects."""

    trace = simulation_trace_export_from_dict(_trace_payload())
    payload = build_worked_example_process_trace_from_export(trace)
    payload["metrics"] = {}

    with pytest.raises(Exception, match="Additional properties"):
        validate_worked_example_process_trace(payload)

    payload.pop("metrics")
    payload["event_anchors"][0]["extra"] = True
    with pytest.raises(Exception, match="/event_anchors/0"):
        validate_worked_example_process_trace(payload)


def test_process_trace_binds_canonical_near_miss_encounter_interval() -> None:
    """Canonical near_miss_encounter.v1 reports should define actor and interval binding."""

    trace = simulation_trace_export_from_dict(_trace_payload())

    payload = build_worked_example_process_trace_from_export(
        trace,
        encounter_report=_encounter_report(start_time_s=0.1, end_time_s=0.2),
        encounter_report_input_checksum="0" * 64,
    )

    focal = payload["encounters"]["focal"]
    assert focal["source"] == "canonical_near_miss_encounter_report"
    assert focal["declared_encounter"]["canonical_record"]["minimum_clearance_m"] == pytest.approx(
        0.5
    )
    report_input = focal["declared_encounter"]["report_input_contract"]
    assert report_input["schema_version"] == "near_miss_encounter_report_input.v1"
    assert report_input["content_contract"] == _encounter_report(start_time_s=0.1, end_time_s=0.2)
    assert report_input["content_sha256"] == _json_digest(report_input["content_contract"])
    assert report_input["selected_entry_sha256"] == _json_digest(
        focal["declared_encounter"]["canonical_record"]
    )
    assert [frame["relative_interaction"]["status"] for frame in payload["frames"]] == [
        "unavailable",
        "available",
        "available",
        "unavailable",
    ]
    assert payload["diagnostics"]["coverage"]["relative_interaction"]["status"] == "complete"
    assert payload["diagnostics"]["coverage"]["relative_interaction"]["frame_count"] == 2
    assert payload["diagnostics"]["threshold_exposure"]["duration_s"] == pytest.approx(0.0)


def test_canonical_encounter_selector_searches_all_report_encounters() -> None:
    """Actor and encounter selectors must not be constrained by the earliest report record."""

    trace = simulation_trace_export_from_dict(_trace_payload(actor_switch=True))
    report = _encounter_report(start_time_s=0.0, end_time_s=0.1)
    second = deepcopy(report["encounters"][0])
    second.update(
        {
            "encounter_id": "ped-b:encounter-0002",
            "actor_id": "ped-b",
            "start_time_s": 0.2,
            "end_time_s": 0.3,
            "duration_s": 0.1,
            "valid_exposure_duration_s": 0.1,
        }
    )
    report["encounters"].append(second)
    report["denominator"].update(
        {
            "actor_count": 2,
            "encounter_count": 2,
            "valid_exposure_duration_s": 0.2,
        }
    )

    by_actor = build_worked_example_process_trace_from_export(
        trace,
        focal_actor_id="ped-b",
        encounter_report=report,
        encounter_report_input_checksum="0" * 64,
    )
    by_encounter = build_worked_example_process_trace_from_export(
        trace,
        focal_encounter_id="ped-b:encounter-0002",
        encounter_report=report,
        encounter_report_input_checksum="0" * 64,
    )

    for payload in (by_actor, by_encounter):
        focal = payload["encounters"]["focal"]
        assert focal["status"] == "available"
        assert focal["actor_id"] == "ped-b"
        assert focal["encounter_id"] == "ped-b:encounter-0002"
        assert focal["declared_encounter"]["canonical_record"]["actor_id"] == "ped-b"
        validate_worked_example_process_trace(payload)


def test_canonical_encounter_ids_are_globally_unique_before_selection() -> None:
    """Duplicate canonical IDs anywhere in a report make focal selection unavailable."""

    report = _encounter_report()
    first_duplicate = deepcopy(report["encounters"][0])
    first_duplicate.update(
        {
            "encounter_id": "ped-b:encounter-0002",
            "actor_id": "ped-b",
            "start_time_s": 0.1,
            "end_time_s": 0.2,
        }
    )
    second_duplicate = deepcopy(first_duplicate)
    second_duplicate.update({"start_time_s": 0.2, "end_time_s": 0.3})
    report["encounters"].extend((first_duplicate, second_duplicate))
    report["denominator"]["actor_count"] = 2
    report["denominator"]["encounter_count"] = 3

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(actor_switch=True)),
        focal_actor_id="ped-a",
        encounter_report=report,
        encounter_report_input_checksum="0" * 64,
    )

    assert payload["encounters"]["focal"]["status"] == "unavailable"
    assert payload["encounters"]["focal"]["reason"] == "canonical_encounter_id_not_unique"


def test_canonical_encounter_id_cannot_name_another_actor() -> None:
    """Actor-prefixed canonical IDs must agree with the record actor binding."""

    report = _encounter_report()
    report["encounters"][0]["encounter_id"] = "ped-b:encounter-0001"
    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(actor_switch=True)),
        focal_encounter_id="ped-b:encounter-0001",
        encounter_report=report,
        encounter_report_input_checksum="0" * 64,
    )

    assert payload["encounters"]["focal"]["status"] == "unavailable"
    assert payload["encounters"]["focal"]["reason"] == "canonical_encounter_id_actor_mismatch"


def test_planner_encounter_identity_supports_actor_ids_containing_colons() -> None:
    """Actor prefixes remain exact when the actor ID itself contains the delimiter."""

    actor_id = "group:ped-a"
    encounter_id = f"{actor_id}:encounter-0001"
    trace_payload = _trace_payload(trace_id="colon-actor-planner-hint")
    for frame in trace_payload["frames"]:
        frame["pedestrians"][0]["id"] = actor_id
        frame["planner"]["encounter"] = {
            "actor_id": actor_id,
            "encounter_id": encounter_id,
        }

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(trace_payload)
    )

    assert payload["encounters"]["focal"]["status"] == "available"
    assert payload["encounters"]["focal"]["actor_id"] == actor_id
    assert payload["encounters"]["focal"]["encounter_id"] == encounter_id
    validate_worked_example_process_trace(payload)


def test_canonical_encounter_identity_supports_actor_ids_containing_colons() -> None:
    """Canonical report binding uses the complete actor ID as its encounter prefix."""

    actor_id = "group:ped-a"
    encounter_id = f"{actor_id}:encounter-0001"
    trace_payload = _trace_payload(trace_id="colon-actor-canonical-report")
    for frame in trace_payload["frames"]:
        frame["pedestrians"][0]["id"] = actor_id
    report = _encounter_report(start_time_s=0.1, end_time_s=0.2)
    report["encounters"][0]["actor_id"] = actor_id
    report["encounters"][0]["encounter_id"] = encounter_id

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(trace_payload),
        encounter_report=report,
        encounter_report_input_checksum="0" * 64,
    )

    assert payload["encounters"]["focal"]["status"] == "available"
    assert payload["encounters"]["focal"]["actor_id"] == actor_id
    assert payload["encounters"]["focal"]["encounter_id"] == encounter_id
    validate_worked_example_process_trace(payload)


@pytest.mark.parametrize(
    ("mode", "expected_reason"),
    (
        ("cross_frame_actor", "planner_encounter_actor_hint_ambiguous"),
        ("same_frame_list_actor", "planner_encounter_actor_hint_ambiguous"),
        ("cross_frame_encounter", "planner_encounter_id_hint_ambiguous"),
        ("single_hint_actor_mismatch", "planner_encounter_id_actor_mismatch"),
    ),
)
def test_planner_encounter_hints_fail_closed_on_any_ambiguity(
    mode: str,
    expected_reason: str,
) -> None:
    """All frames and list entries contribute to one fail-closed hint decision."""

    trace_payload = _trace_payload(actor_switch=True)
    if mode == "cross_frame_actor":
        trace_payload["frames"][1]["planner"]["encounter"] = {
            "actor_id": "ped-b",
            "encounter_id": "ped-b:encounter-0001",
        }
    elif mode == "same_frame_list_actor":
        for frame in trace_payload["frames"]:
            frame["planner"].pop("encounter")
        trace_payload["frames"][0]["planner"]["encounters"] = [
            {"actor_id": "ped-a", "encounter_id": "ped-a:encounter-0001"},
            {"actor_id": "ped-b", "encounter_id": "ped-b:encounter-0001"},
        ]
    elif mode == "cross_frame_encounter":
        trace_payload["frames"][1]["planner"]["encounter"] = {
            "actor_id": "ped-a",
            "encounter_id": "ped-a:encounter-0002",
        }
    else:
        for frame in trace_payload["frames"]:
            frame["planner"]["encounter"] = {
                "actor_id": "ped-a",
                "encounter_id": "ped-b:encounter-0001",
            }

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(trace_payload)
    )

    assert payload["encounters"]["focal"]["status"] == "unavailable"
    assert payload["encounters"]["focal"]["reason"] == expected_reason


def test_unrelated_canonical_encounter_checksum_abstains() -> None:
    """Canonical encounter reports must bind to the input trace checksum."""

    trace = simulation_trace_export_from_dict(_trace_payload())

    payload = build_worked_example_process_trace_from_export(
        trace,
        encounter_report=_encounter_report(),
        encounter_report_input_checksum="f" * 64,
    )

    focal = payload["encounters"]["focal"]
    assert focal["status"] == "unavailable"
    assert focal["reason"] == "canonical_encounter_input_checksum_mismatch"


def test_stale_canonical_encounter_checksum_digest_abstains() -> None:
    """Canonical encounter reports must not accept stale checksum digests."""

    report = _encounter_report()
    report["provenance"]["input_checksum_digest"] = "1" * 64

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()),
        encounter_report=report,
        encounter_report_input_checksum="0" * 64,
    )

    focal = payload["encounters"]["focal"]
    assert focal["status"] == "unavailable"
    assert focal["reason"] == "canonical_encounter_input_checksum_digest_mismatch"


def test_canonical_collision_shapes_and_timing_are_respected() -> None:
    """Only canonical collision shapes should drive exact collision anchors."""

    boolean_payload = _trace_payload(collision_mode="outcome_boolean")
    boolean = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(boolean_payload)
    )
    boolean_event = {event["event_type"]: event for event in boolean["event_anchors"]}[
        "exact_collision_event"
    ]
    assert boolean_event["status"] == "unavailable"
    assert boolean_event["reason"] == "collision_observed_time_unavailable"
    assert boolean_event["collision_observed"] is True

    typed_payload = _trace_payload(collision_mode="ledger_typed")
    typed = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(typed_payload)
    )
    typed_event = {event["event_type"]: event for event in typed["event_anchors"]}[
        "exact_collision_event"
    ]
    assert typed_event["status"] == "available"
    assert typed_event["time_s"] == pytest.approx(0.15)
    assert typed_event["collision_partner_id"] == "ped-a"
    assert typed_event["collision_partner_type"] == "pedestrian"
    assert typed_event["actor_id"] == "ped-a"
    assert typed_event["focal_binding"] == {
        "status": "available",
        "reason": "collision_partner_matches_focal_actor",
        "actor_id": "ped-a",
    }

    non_focal_payload = _trace_payload(collision_mode="ledger_ped_b")
    non_focal = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(non_focal_payload)
    )
    non_focal_event = {event["event_type"]: event for event in non_focal["event_anchors"]}[
        "exact_collision_event"
    ]
    assert non_focal_event["status"] == "available"
    assert non_focal_event["actor_id"] is None
    assert non_focal_event["collision_partner_id"] == "ped-b"
    assert non_focal_event["collision_partner_type"] == "pedestrian"
    assert non_focal_event["focal_binding"] == {
        "status": "unavailable",
        "reason": "collision_partner_not_focal_actor",
    }

    static_payload = _trace_payload(collision_mode="static_geometry_collision")
    static = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(static_payload)
    )
    static_event = {event["event_type"]: event for event in static["event_anchors"]}[
        "exact_collision_event"
    ]
    assert static_event["status"] == "available"
    assert static_event["time_s"] == pytest.approx(0.15)
    assert static_event["actor_id"] is None
    assert static_event["collision_partner_id"] is None
    assert static_event["collision_partner_type"] == "static_geometry"
    assert static_event["focal_binding"] == {
        "status": "unavailable",
        "reason": "collision_partner_not_focal_actor",
    }

    invented_payload = _trace_payload(collision_mode="invented_events")
    invented = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(invented_payload)
    )
    invented_event = {event["event_type"]: event for event in invented["event_anchors"]}[
        "exact_collision_event"
    ]
    assert invented_event["status"] == "unavailable"
    assert invented_event["reason"] == "required_signal_unavailable"

    legacy_payload = _trace_payload(collision_mode="legacy_collision_time_s")
    legacy = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(legacy_payload)
    )
    legacy_event = {event["event_type"]: event for event in legacy["event_anchors"]}[
        "exact_collision_event"
    ]
    assert legacy_event["status"] == "unavailable"

    time_only_payload = _trace_payload(collision_mode="time_only_collision_record")
    time_only = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(time_only_payload)
    )
    assert {event["event_type"]: event for event in time_only["event_anchors"]}[
        "exact_collision_event"
    ]["status"] == "unavailable"

    outside = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(collision_mode="ledger_collision_late")),
        encounter_report=_encounter_report(start_time_s=0.0, end_time_s=0.2),
        encounter_report_input_checksum="0" * 64,
    )
    outside_event = {event["event_type"]: event for event in outside["event_anchors"]}[
        "exact_collision_event"
    ]
    assert outside_event["status"] == "unavailable"
    assert outside_event["reason"] == "collision_time_outside_trace_sample_bounds"

    zero_sample_interval = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(collision_mode="ledger_typed")),
        encounter_report=_encounter_report(start_time_s=10.0, end_time_s=11.0),
        encounter_report_input_checksum="0" * 64,
    )
    zero_events = {event["event_type"]: event for event in zero_sample_interval["event_anchors"]}
    assert zero_sample_interval["diagnostics"]["coverage"]["frame_count"] == 0
    assert zero_events["exact_collision_event"]["status"] == "available"
    assert zero_events["exact_collision_event"]["actor_id"] is None
    assert zero_events["exact_collision_event"]["focal_binding"] == {
        "status": "unavailable",
        "reason": "collision_time_outside_encounter_interval",
    }
    assert zero_sample_interval["event_anchor_hierarchy"]["selected_anchor"]["event_type"] == (
        "exact_collision_event"
    )


def test_pair_collision_identity_includes_truthful_episode_partner() -> None:
    """Common exact-collision anchors require the same canonical partner identity."""

    left_payload = _trace_payload(
        trace_id="collision-left",
        planner_id="planner-a",
        seed=7,
        collision_mode="static_geometry_collision",
    )
    right_payload = _trace_payload(
        trace_id="collision-right",
        planner_id="planner-b",
        seed=7,
        collision_mode="static_geometry_collision",
    )
    matching = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(left_payload),
        pair_trace=simulation_trace_export_from_dict(right_payload),
        pair_comparison_grain="matched_planner_pair",
    )["pair_compatibility"]
    common_collision = next(
        anchor
        for anchor in matching["valid_common_event_anchors"]
        if anchor["event_type"] == "exact_collision_event"
    )
    assert common_collision["collision_partner_type"] == "static_geometry"
    assert common_collision["collision_partner_id"] is None

    right_payload["frames"][1]["planner"]["event_ledger"]["collision_events"][0][
        "collision_partner_type"
    ] = "boundary"
    mismatched = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(left_payload),
        pair_trace=simulation_trace_export_from_dict(right_payload),
        pair_comparison_grain="matched_planner_pair",
    )["pair_compatibility"]
    assert not any(
        anchor["event_type"] == "exact_collision_event"
        for anchor in mismatched["valid_common_event_anchors"]
    )


def test_pair_receipt_replay_uses_registered_route_and_conflict_geometry() -> None:
    """Right-event replay must use the same registered coordinate contracts as construction."""

    left_payload = _trace_payload(trace_id="geometry-pair-left", planner_id="planner-a", seed=7)
    right_payload = _trace_payload(trace_id="geometry-pair-right", planner_id="planner-b", seed=7)
    left_payload["frames"][-1]["robot"]["position"] = [1.0, 0.0]
    right_payload["frames"][-1]["robot"]["position"] = [1.0, 0.0]
    left = simulation_trace_export_from_dict(left_payload)
    right = simulation_trace_export_from_dict(right_payload)
    payload = build_worked_example_process_trace_from_export(
        left,
        route=_registered_route("r-main"),
        conflict_zone=_registered_conflict_zone("door"),
        pair_trace=right,
        pair_comparison_grain="matched_planner_pair",
    )

    assert any(
        event["event_type"] == "conflict_zone_entry"
        for event in payload["pair_compatibility"]["right_event_anchors"]
    )
    validate_worked_example_process_trace(payload)


def test_terminal_event_has_no_timed_contract() -> None:
    """Terminal fallback must stay unavailable until a canonical timed contract exists."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload())
    )
    terminal_event = {event["event_type"]: event for event in payload["event_anchors"]}[
        "terminal_event"
    ]
    assert terminal_event["status"] == "unavailable"
    assert terminal_event["source_fields"] == ["terminal_event_contract_unavailable"]


def test_partial_radius_and_nonfinite_heading_fail_closed() -> None:
    """Subset minima and anchors must abstain when telemetry is partial or nonfinite."""

    missing_radius_payload = _trace_payload(missing_actor_radius_step=1)
    missing_radius = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(missing_radius_payload)
    )
    events = {event["event_type"]: event for event in missing_radius["event_anchors"]}
    assert missing_radius["diagnostics"]["minimum_proxy_surface_clearance_m"] is None
    assert events["minimum_clearance"]["status"] == "unavailable"

    bad_heading_payload = _trace_payload(nonfinite_heading_step=1)
    bad_heading = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(bad_heading_payload)
    )
    assert bad_heading["frames"][1]["world"]["robot"]["heading"] is None
    assert bad_heading["frames"][1]["relative_interaction"] == {
        "status": "unavailable",
        "reason": "missing_or_nonfinite_robot_heading",
    }


def test_threshold_breach_uses_declared_diagnostic_threshold() -> None:
    """Safety-predicate breach should trigger below 0.4 m, before proxy overlap."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload())
    )
    events = {event["event_type"]: event for event in payload["event_anchors"]}

    assert events["first_safety_predicate_breach"]["status"] == "available"
    assert events["proxy_overlap_event"]["status"] == "not_observed"
    assert payload["event_anchor_hierarchy"]["fallback_order"] == [
        "exact_collision_event",
        "minimum_clearance",
        "first_safety_predicate_breach",
        "sustained_stall_onset",
        "terminal_event",
    ]


def test_nonfinite_command_and_missing_stall_speed_fail_closed() -> None:
    """NaN commands and missing speed should propagate as unavailable, not numeric evidence."""

    bad_command = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(nan_command_step=1))
    )
    assert bad_command["frames"][1]["commands"] == {
        "status": "unavailable",
        "reason": "selected_action_nonfinite",
    }

    missing_speed = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(
            _trace_payload(stall_pattern=[0.0, 0.0, 0.0, 1.0], missing_velocity_step=1)
        )
    )
    stall = missing_speed["diagnostics"]["stall"]
    assert stall["status"] == "unavailable"
    assert stall["sustained_stall_duration_s"] is None
    assert stall["sustained_stall_onset_step"] is None
    assert {event["event_type"]: event for event in missing_speed["event_anchors"]}[
        "sustained_stall_onset"
    ]["status"] == "unavailable"

    missing_recovery = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(
            _trace_payload(stall_pattern=[0.0, 0.0, 0.0, 0.0, 1.0], missing_velocity_step=3)
        )
    )
    recovery_events = {event["event_type"]: event for event in missing_recovery["event_anchors"]}
    assert missing_recovery["diagnostics"]["stall"]["status"] == "unavailable"
    assert recovery_events["sustained_stall_onset"]["status"] == "unavailable"
    assert recovery_events["recovery_onset"]["status"] == "unavailable"

    command_gap = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(
            _trace_payload(turn_and_decelerate=True, nan_command_step=1)
        )
    )
    assert {event["event_type"]: event for event in command_gap["event_anchors"]}[
        "first_material_deceleration"
    ]["status"] == "unavailable"

    relative_gap = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(nonfinite_heading_step=1))
    )
    relative_events = {event["event_type"]: event for event in relative_gap["event_anchors"]}
    assert relative_events["minimum_clearance"]["status"] == "unavailable"
    assert relative_events["first_safety_predicate_breach"]["status"] == "unavailable"


def test_registry_checksum_must_bind_geometry() -> None:
    """Registry checksums must be SHA-256 hex and match the declared geometry."""

    invalid = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()),
        route=RouteSpec("r-main", (0.0, 0.0), (10.0, 0.0), "route-fixture.v1", "bad"),
    )
    assert invalid["coordinate_frames"]["route"]["reason"] == "registered_route_checksum_invalid"

    mismatched = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()),
        route=RouteSpec("r-main", (0.0, 0.0), (10.0, 0.0), "route-fixture.v1", "0" * 64),
    )
    assert mismatched["coordinate_frames"]["route"]["reason"] == (
        "registered_route_checksum_geometry_mismatch"
    )


def test_external_geometry_registry_receipts_fail_closed(tmp_path: Path) -> None:
    """Raw-file, unique-entry, and replayed receipt bindings must be external evidence."""

    registry_path = tmp_path / "registry.json"
    registry_path.write_bytes(GEOMETRY_REGISTRY_FIXTURE_PATH.read_bytes())
    route = load_registered_route_spec(registry_path, "route-b")
    conflict = load_registered_conflict_zone_spec(registry_path, "zone-b")
    trace = simulation_trace_export_from_dict(_trace_payload())

    available = build_worked_example_process_trace_from_export(
        trace, route=route, conflict_zone=conflict
    )
    route_receipt = available["coordinate_frames"]["route"]["registry"]
    assert route_receipt["artifact_ref"] == GEOMETRY_REGISTRY_ARTIFACT_REF
    assert str(tmp_path) not in json.dumps(available, sort_keys=True)
    assert route_receipt["coordinate_frame"] == "world"
    assert route_receipt["geometry_kind"] == "line_segment"
    assert route_receipt["resolved_geometry"] == available["coordinate_frames"]["route"]["geometry"]
    assert route_receipt["upstream_binding"] == {
        "kind": "fixture_only",
        "fixture_id": "issue-6790-route-b",
    }
    assert route_receipt["owner_validation"] == {
        "status": "fixture_only",
        "reason": "fixture_binding_not_canonical_owner",
    }

    forged = deepcopy(available)
    forged["coordinate_frames"]["route"]["registry"]["content_sha256"] = "f" * 64
    for frame in forged["frames"]:
        frame["route"]["registry"]["content_sha256"] = "f" * 64
    with pytest.raises(Exception, match="external geometry registry receipt"):
        validate_worked_example_process_trace(forged)

    downgraded = deepcopy(available)
    downgraded["coordinate_frames"]["route"]["status"] = "unavailable"
    downgraded["coordinate_frames"]["route"]["reason"] = "registered_route_unavailable"
    for frame in downgraded["frames"]:
        frame["route"] = {"status": "unavailable", "reason": "registered_route_unavailable"}
    _rebuild_dependent_process_views(downgraded, trace)
    with pytest.raises(Exception, match="external geometry registry receipt"):
        validate_worked_example_process_trace(downgraded)

    registry_path.unlink()
    with pytest.raises(Exception, match="external geometry registry receipt"):
        validate_worked_example_process_trace(
            available,
            geometry_registry_paths={GEOMETRY_REGISTRY_ARTIFACT_REF: registry_path},
        )
    registry_path.write_bytes(GEOMETRY_REGISTRY_FIXTURE_PATH.read_bytes())
    registry_path.write_text(registry_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    tampered = build_worked_example_process_trace_from_export(
        trace, route=route, conflict_zone=conflict
    )
    assert tampered["coordinate_frames"]["route"]["reason"] == (
        "registered_route_registry_content_mismatch"
    )
    assert tampered["coordinate_frames"]["conflict"]["reason"] == (
        "registered_conflict_zone_registry_content_mismatch"
    )


def test_geometry_registry_receipts_are_checkout_portable(tmp_path: Path) -> None:
    """Stable receipts stay identical while replay resolves each checkout-local artifact."""

    portable_registry = json.loads(GEOMETRY_REGISTRY_FIXTURE_PATH.read_text(encoding="utf-8"))
    owner_ref = "maps/registry.json"
    route_selector = {"map_id": "fixture-map", "spawn_id": 1, "goal_id": 2}
    conflict_selector = {"map_id": "fixture-map", "zone_id": "fixture-zone"}
    owner_bytes = json.dumps(
        {
            "schema_version": "process_trace_geometry_owner.v1",
            "geometry_bindings": [
                {
                    "selector": route_selector,
                    "geometry": next(
                        entry
                        for entry in portable_registry["routes"]
                        if entry["entry_id"] == "fixture-route"
                    )["geometry"],
                },
                {
                    "selector": conflict_selector,
                    "geometry": next(
                        entry
                        for entry in portable_registry["conflict_zones"]
                        if entry["entry_id"] == "fixture-zone"
                    )["geometry"],
                },
            ],
        },
        sort_keys=True,
    ).encode("utf-8")
    canonical_binding = {
        "kind": "canonical_source",
        "source_artifact_ref": owner_ref,
        "source_content_sha256": hashlib.sha256(owner_bytes).hexdigest(),
        "selector": route_selector,
    }
    next(entry for entry in portable_registry["routes"] if entry["entry_id"] == "fixture-route")[
        "upstream_binding"
    ] = canonical_binding
    next(
        entry
        for entry in portable_registry["conflict_zones"]
        if entry["entry_id"] == "fixture-zone"
    )["upstream_binding"] = {
        **canonical_binding,
        "selector": conflict_selector,
    }
    registry_bytes = (json.dumps(portable_registry, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    payloads: list[dict[str, object]] = []
    registries: list[Path] = []
    owners: list[Path] = []
    for checkout_name in ("checkout-a", "checkout-b"):
        checkout = tmp_path / checkout_name
        trace_path = checkout / "inputs" / "trace.json"
        registry_path = checkout / "geometry" / "registry.json"
        owner_path = checkout / owner_ref
        trace_path.parent.mkdir(parents=True)
        registry_path.parent.mkdir(parents=True)
        owner_path.parent.mkdir(parents=True)
        trace_path.write_bytes(TRACE_FIXTURE_PATH.read_bytes())
        registry_path.write_bytes(registry_bytes)
        owner_path.write_bytes(owner_bytes)
        registries.append(registry_path)
        owners.append(owner_path)
        payload = build_worked_example_process_trace(
            trace_path,
            route=load_registered_route_spec(
                registry_path,
                "fixture-route",
                geometry_owner_paths={owner_ref: owner_path},
            ),
            conflict_zone=load_registered_conflict_zone_spec(
                registry_path,
                "fixture-zone",
                geometry_owner_paths={owner_ref: owner_path},
            ),
        )
        validate_worked_example_process_trace(
            payload,
            geometry_registry_paths={
                GEOMETRY_REGISTRY_ARTIFACT_REF: registry_path,
                owner_ref: owner_path,
            },
        )
        payloads.append(payload)

    assert payloads[0] == payloads[1]
    assert _json_digest(payloads[0]) == _json_digest(payloads[1])
    assert payloads[0]["coordinate_frames"]["route"]["registry"]["upstream_binding"] == (
        canonical_binding
    )
    serialized = json.dumps(payloads[0], sort_keys=True)
    assert str(tmp_path / "checkout-a") not in serialized
    assert str(tmp_path / "checkout-b") not in serialized

    for registry_path, owner_path, payload in zip(registries, owners, payloads, strict=True):
        pristine = registry_path.read_bytes()
        registry_path.write_bytes(pristine + b"\n")
        with pytest.raises(Exception, match="external geometry registry receipt"):
            validate_worked_example_process_trace(
                payload,
                geometry_registry_paths={
                    GEOMETRY_REGISTRY_ARTIFACT_REF: registry_path,
                    owner_ref: owner_path,
                },
            )
        registry_path.write_bytes(pristine)
        validate_worked_example_process_trace(
            payload,
            geometry_registry_paths={
                GEOMETRY_REGISTRY_ARTIFACT_REF: registry_path,
                owner_ref: owner_path,
            },
        )


def test_geometry_owner_schema_is_public_and_strict() -> None:
    """The canonical owner envelope, selectors, and geometry are a public strict contract."""

    schema = coordinates.load_process_trace_geometry_owner_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    valid = {
        "schema_version": "process_trace_geometry_owner.v1",
        "geometry_bindings": [
            {
                "selector": {"map_id": "fixture-map", "spawn_id": 1},
                "geometry": {
                    "type": "line_segment",
                    "start": [0.0, 0.0],
                    "end": [1.0, 0.0],
                },
            }
        ],
    }
    assert not list(validator.iter_errors(valid))

    malformed_selector = deepcopy(valid)
    malformed_selector["geometry_bindings"][0]["selector"] = {"map_id": ["not-scalar"]}
    assert list(validator.iter_errors(malformed_selector))
    malformed_geometry = deepcopy(valid)
    malformed_geometry["geometry_bindings"][0]["geometry"]["start"] = [0.0]
    assert list(validator.iter_errors(malformed_geometry))
    extra_envelope = {**valid, "private_path": "/tmp/owner.json"}
    assert list(validator.iter_errors(extra_envelope))


@pytest.mark.parametrize(
    ("invalid_binding", "expected_reason"),
    (
        (
            {
                "selector": {"unrelated": math.nan},
                "geometry": {
                    "type": "line_segment",
                    "start": [0.0, 0.0],
                    "end": [1.0, 0.0],
                },
            },
            "registered_route_owner_artifact_invalid_json",
        ),
        (
            {
                "selector": {"unrelated": ["not-scalar"]},
                "geometry": {
                    "type": "line_segment",
                    "start": [0.0, 0.0],
                    "end": [1.0, 0.0],
                },
            },
            "registered_route_owner_selector_invalid",
        ),
        (
            {
                "selector": {"unrelated": "binding"},
                "geometry": {
                    "type": "line_segment",
                    "start": [0.0],
                    "end": [1.0, 0.0],
                },
            },
            "registered_route_owner_geometry_invalid",
        ),
    ),
)
def test_geometry_owner_validates_entire_envelope_before_selector_scan(
    tmp_path: Path,
    invalid_binding: dict[str, object],
    expected_reason: str,
) -> None:
    """An unrelated malformed binding cannot be skipped in favor of one valid match."""

    registry = json.loads(GEOMETRY_REGISTRY_FIXTURE_PATH.read_text(encoding="utf-8"))
    route_entry = next(entry for entry in registry["routes"] if entry["entry_id"] == "r-main")
    selector = {"map_id": "fixture-map", "spawn_id": 1, "goal_id": 2}
    owner_ref = "owners/strict-owner.json"
    owner_path = tmp_path / "strict-owner.json"
    owner_path.write_text(
        json.dumps(
            {
                "schema_version": "process_trace_geometry_owner.v1",
                "geometry_bindings": [
                    invalid_binding,
                    {"selector": selector, "geometry": route_entry["geometry"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    route_entry["upstream_binding"] = {
        "kind": "canonical_source",
        "source_artifact_ref": owner_ref,
        "source_content_sha256": hashlib.sha256(owner_path.read_bytes()).hexdigest(),
        "selector": selector,
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    route = load_registered_route_spec(
        registry_path,
        "r-main",
        geometry_owner_paths={owner_ref: owner_path},
    )

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()),
        route=route,
    )

    assert payload["coordinate_frames"]["route"]["status"] == "unavailable"
    assert payload["coordinate_frames"]["route"]["reason"] == expected_reason
    with pytest.raises(WorkedExampleProcessTraceValidationError):
        coordinates.load_process_trace_geometry_owner(owner_path)


@pytest.mark.parametrize("overflow_token", ["1e400", "1e999"])
def test_geometry_owner_rejects_numeric_overflow_anywhere(
    tmp_path: Path,
    overflow_token: str,
) -> None:
    """An unrelated overflowing JSON number cannot be ignored while one owner matches."""

    registry = json.loads(GEOMETRY_REGISTRY_FIXTURE_PATH.read_text(encoding="utf-8"))
    route_entry = next(entry for entry in registry["routes"] if entry["entry_id"] == "r-main")
    selector = {"map_id": "fixture-map", "spawn_id": 1, "goal_id": 2}
    owner_ref = "owners/overflow-owner.json"
    owner_path = tmp_path / "overflow-owner.json"
    owner_payload = {
        "schema_version": "process_trace_geometry_owner.v1",
        "geometry_bindings": [
            {
                "selector": {"unrelated_overflow": 0.0},
                "geometry": {
                    "type": "line_segment",
                    "start": [0.0, 0.0],
                    "end": [1.0, 0.0],
                },
            },
            {"selector": selector, "geometry": route_entry["geometry"]},
        ],
    }
    owner_text = json.dumps(owner_payload).replace(
        '"unrelated_overflow": 0.0',
        f'"unrelated_overflow": {overflow_token}',
    )
    owner_path.write_text(owner_text, encoding="utf-8")
    route_entry["upstream_binding"] = {
        "kind": "canonical_source",
        "source_artifact_ref": owner_ref,
        "source_content_sha256": hashlib.sha256(owner_path.read_bytes()).hexdigest(),
        "selector": selector,
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    route = load_registered_route_spec(
        registry_path,
        "r-main",
        geometry_owner_paths={owner_ref: owner_path},
    )

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()),
        route=route,
    )

    assert payload["coordinate_frames"]["route"] == {
        "status": "unavailable",
        "reason": "registered_route_owner_artifact_invalid_json",
        "input_contract": payload["coordinate_frames"]["route"]["input_contract"],
    }
    with pytest.raises(WorkedExampleProcessTraceValidationError):
        coordinates.load_process_trace_geometry_owner(owner_path)


def test_canonical_geometry_owner_must_resolve_digest_and_exact_selector_geometry(
    tmp_path: Path,
) -> None:
    """An adapter registry cannot invent availability without its canonical owner."""

    owner_ref = "owners/fixture-map.json"
    owner_path = tmp_path / "fixture-map.json"
    route_geometry = {"type": "line_segment", "start": [0.0, 0.0], "end": [10.0, 0.0]}
    selector = {"map_id": "fixture-map", "spawn_id": 1, "goal_id": 2}

    def registry_with_owner_digest(owner_digest: str) -> Path:
        registry = json.loads(GEOMETRY_REGISTRY_FIXTURE_PATH.read_text(encoding="utf-8"))
        registry["routes"][0]["upstream_binding"] = {
            "kind": "canonical_source",
            "source_artifact_ref": owner_ref,
            "source_content_sha256": owner_digest,
            "selector": selector,
        }
        registry_path = tmp_path / f"registry-{owner_digest[:8]}.json"
        registry_path.write_text(json.dumps(registry), encoding="utf-8")
        return registry_path

    missing_path = tmp_path / "missing-owner.json"
    missing_route = load_registered_route_spec(
        registry_with_owner_digest("0" * 64),
        "r-main",
        geometry_owner_paths={owner_ref: missing_path},
    )
    missing = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()), route=missing_route
    )
    assert missing["coordinate_frames"]["route"]["reason"] == (
        "registered_route_owner_artifact_missing"
    )

    owner_path.write_text(json.dumps({"map_id": "fixture-map"}), encoding="utf-8")
    fabricated_digest = hashlib.sha256(owner_path.read_bytes()).hexdigest()
    fabricated_route = load_registered_route_spec(
        registry_with_owner_digest(fabricated_digest),
        "r-main",
        geometry_owner_paths={owner_ref: owner_path},
    )
    fabricated = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()), route=fabricated_route
    )
    assert fabricated["coordinate_frames"]["route"]["reason"] == (
        "registered_route_owner_artifact_schema_invalid"
    )

    owner_path.write_text(
        json.dumps(
            {
                "schema_version": "process_trace_geometry_owner.v1",
                "geometry_bindings": [{"selector": selector, "geometry": route_geometry}],
            }
        ),
        encoding="utf-8",
    )
    verified_digest = hashlib.sha256(owner_path.read_bytes()).hexdigest()
    verified_route = load_registered_route_spec(
        registry_with_owner_digest(verified_digest),
        "r-main",
        geometry_owner_paths={owner_ref: owner_path},
    )
    verified = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()), route=verified_route
    )
    assert verified["coordinate_frames"]["route"]["status"] == "available"
    assert verified["coordinate_frames"]["route"]["registry"]["owner_validation"] == {
        "status": "verified",
        "source_artifact_ref": owner_ref,
        "source_content_sha256": verified_digest,
        "selector": selector,
        "geometry_sha256": _json_digest(route_geometry),
    }

    owner_path.write_text("{}", encoding="utf-8")
    with pytest.raises(Exception, match="canonical owner artifact"):
        validate_worked_example_process_trace(
            verified,
            geometry_registry_paths={
                GEOMETRY_REGISTRY_ARTIFACT_REF: Path(verified_route.registry_path),
                owner_ref: owner_path,
            },
        )


@pytest.mark.parametrize(
    "source_artifact_ref",
    [
        "/tmp/maps/registry.yaml",
        "file:///tmp/maps/registry.yaml",
        "maps\\registry.yaml",
        "~/maps/registry.yaml",
        "maps/../registry.yaml",
        "maps/./registry.yaml",
        "C:/maps/registry.yaml",
        "c:/maps/registry.yaml",
        "C:\\maps\\registry.yaml",
        "C:registry.json",
        "C:owners/map.json",
        "c:owners/map.json",
    ],
)
def test_canonical_upstream_binding_rejects_machine_local_or_unstable_references(
    tmp_path: Path,
    source_artifact_ref: str,
) -> None:
    """Canonical source bindings must not reintroduce checkout-local path identity."""

    registry = json.loads(GEOMETRY_REGISTRY_FIXTURE_PATH.read_text(encoding="utf-8"))
    registry["routes"][0]["upstream_binding"] = {
        "kind": "canonical_source",
        "source_artifact_ref": source_artifact_ref,
        "source_content_sha256": "2" * 64,
        "selector": {"map_id": "fixture-map", "spawn_id": 1, "goal_id": 2},
    }
    registry_path = tmp_path / "invalid-upstream-ref.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(Exception, match="invalid process-trace geometry registry envelope"):
        load_registered_route_spec(registry_path, "r-main")


@pytest.mark.parametrize("artifact_ref", ["C:registry.json", "C:owners/map.json"])
def test_geometry_registry_rejects_windows_drive_relative_artifact_refs(
    tmp_path: Path,
    artifact_ref: str,
) -> None:
    """The registry's own public identity cannot use Windows drive-relative syntax."""

    registry = json.loads(GEOMETRY_REGISTRY_FIXTURE_PATH.read_text(encoding="utf-8"))
    registry["artifact_ref"] = artifact_ref
    registry_path = tmp_path / "drive-relative-registry-ref.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(Exception, match="invalid process-trace geometry registry envelope"):
        load_registered_route_spec(registry_path, "r-main")
    replayed = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()),
        route=replace(_registered_route("r-main"), registry_artifact_ref=artifact_ref),
    )
    assert replayed["coordinate_frames"]["route"]["status"] == "unavailable"
    assert replayed["coordinate_frames"]["route"]["reason"] == (
        "registered_route_registry_receipt_invalid"
    )


def test_geometry_registry_missing_and_duplicate_entries_fail_closed(tmp_path: Path) -> None:
    """An entry ID must resolve exactly once in the bound raw registry artifact."""

    trace = simulation_trace_export_from_dict(_trace_payload())
    route = _registered_route("route-b")
    missing = build_worked_example_process_trace_from_export(
        trace, route=replace(route, registry_entry_id="missing-route")
    )
    assert missing["coordinate_frames"]["route"]["reason"] == (
        "registered_route_registry_entry_missing"
    )
    conflict = _registered_conflict_zone("zone-b")
    missing_conflict = build_worked_example_process_trace_from_export(
        trace, conflict_zone=replace(conflict, registry_entry_id="missing-zone")
    )
    assert missing_conflict["coordinate_frames"]["conflict"]["reason"] == (
        "registered_conflict_zone_registry_entry_missing"
    )

    registry_payload = json.loads(GEOMETRY_REGISTRY_FIXTURE_PATH.read_text(encoding="utf-8"))
    duplicate = deepcopy(registry_payload["routes"][1])
    registry_payload["routes"].append(duplicate)
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        json.dumps(registry_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    duplicate_route = replace(
        route,
        registry_path=str(duplicate_path.resolve()),
        registry_content_sha256=hashlib.sha256(duplicate_path.read_bytes()).hexdigest(),
    )
    result = build_worked_example_process_trace_from_export(trace, route=duplicate_route)
    assert result["coordinate_frames"]["route"]["reason"] == (
        "registered_route_registry_entry_ambiguous"
    )

    non_world_payload = deepcopy(registry_payload)
    non_world_payload["routes"].pop()
    non_world_payload["coordinate_frame"] = "robot"
    non_world_path = tmp_path / "non-world.json"
    non_world_path.write_text(json.dumps(non_world_payload), encoding="utf-8")
    with pytest.raises(Exception, match="invalid process-trace geometry registry envelope"):
        load_registered_route_spec(non_world_path, "route-b")


@pytest.mark.parametrize("poison", ["NaN", "Infinity", "1e400", "duplicate_key"])
def test_geometry_registry_load_and_replay_reject_entire_document_poison(
    tmp_path: Path,
    poison: str,
) -> None:
    """Unselected poison must fail both initial registry loading and later receipt replay."""

    registry = json.loads(GEOMETRY_REGISTRY_FIXTURE_PATH.read_text(encoding="utf-8"))
    marker = 12345.6789
    unrelated_circle = next(
        entry for entry in registry["conflict_zones"] if entry["geometry"]["type"] == "circle"
    )
    unrelated_circle["geometry"]["center"][0] = marker
    raw_text = json.dumps(registry, sort_keys=True)
    if poison == "duplicate_key":
        registry_id_field = f'"registry_id": "{registry["registry_id"]}"'
        raw_text = raw_text.replace(
            registry_id_field,
            f"{registry_id_field}, {registry_id_field}",
            1,
        )
    else:
        raw_text = raw_text.replace(str(marker), poison, 1)
    registry_path = tmp_path / f"poison-{poison}.json"
    registry_path.write_text(raw_text, encoding="utf-8")

    with pytest.raises(WorkedExampleProcessTraceValidationError):
        load_registered_route_spec(registry_path, "r-main")

    pristine_route = load_registered_route_spec(GEOMETRY_REGISTRY_FIXTURE_PATH, "r-main")
    replay_route = replace(
        pristine_route,
        registry_path=str(registry_path),
        registry_content_sha256=hashlib.sha256(registry_path.read_bytes()).hexdigest(),
    )
    replayed = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()),
        route=replay_route,
    )
    assert replayed["coordinate_frames"]["route"]["status"] == "unavailable"
    assert replayed["coordinate_frames"]["route"]["reason"] == ("registered_route_registry_invalid")


@pytest.mark.parametrize("entry_id", ["point-owner", "polygon-owner"])
def test_non_circular_conflict_owner_geometry_is_explicitly_unavailable(entry_id: str) -> None:
    """Known point/polygon owner geometry must abstain until its projection is versioned."""

    result = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()),
        conflict_zone=_registered_conflict_zone(entry_id),
    )

    assert {key: result["coordinate_frames"]["conflict"][key] for key in ("status", "reason")} == {
        "status": "unavailable",
        "reason": f"registered_conflict_zone_{entry_id.removesuffix('-owner')}_projection_unavailable",
    }
    assert result["coordinate_frames"]["conflict"]["input_contract"]["status"] == "supplied"


def test_registered_polyline_projection_and_ambiguity_contract() -> None:
    """Ordered polylines use cumulative arclength and abstain on ties or branching geometry."""

    trace_payload = _trace_payload()
    positions = [[0.2, 0.0], [0.8, 0.0], [1.0, 0.4], [1.0, 1.0]]
    for frame, position in zip(trace_payload["frames"], positions, strict=True):
        frame["robot"]["position"] = position
    polyline = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(trace_payload),
        route=_registered_route("polyline-turn"),
    )
    assert polyline["coordinate_frames"]["route"]["geometry"] == {
        "type": "polyline",
        "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]],
    }
    assert polyline["frames"][2]["route"]["s_m"] == pytest.approx(1.4)
    assert polyline["frames"][2]["route"]["n_m"] == pytest.approx(0.0)
    assert polyline["frames"][2]["route"]["progress_rate_mps"] == pytest.approx(0.0)
    validate_worked_example_process_trace(polyline)

    for entry_id in ("branching", "self-intersecting", "adjacent-backtracking"):
        rejected = build_worked_example_process_trace_from_export(
            simulation_trace_export_from_dict(_trace_payload()),
            route=_registered_route(entry_id),
        )
        assert rejected["coordinate_frames"]["route"]["reason"] == (
            "registered_route_branching_or_ambiguous_geometry"
        )

    ambiguous_payload = _trace_payload()
    for frame in ambiguous_payload["frames"]:
        frame["robot"]["position"] = [1.0, 1.0]
    ambiguous = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(ambiguous_payload),
        route=_registered_route("ambiguous-u"),
    )
    assert {frame["route"]["reason"] for frame in ambiguous["frames"]} == {
        "ambiguous_route_projection"
    }
    validate_worked_example_process_trace(ambiguous)


def test_stall_duration_requires_qualifying_contiguous_run() -> None:
    """Separated one-frame stalls should not accumulate into sustained stall duration."""

    payload = _trace_payload(stall_pattern=[0.0, 1.0, 0.0, 1.0])
    trace = simulation_trace_export_from_dict(payload)

    result = build_worked_example_process_trace_from_export(
        trace,
        route=_registered_route("r-main"),
    )

    stall = result["diagnostics"]["stall"]
    assert stall["sustained_stall_duration_s"] == 0.0
    assert stall["sustained_stall_onset_step"] is None
    assert stall["speed_coverage"]["status"] == "complete"


def test_zero_frame_canonical_interval_marks_derived_diagnostics_unavailable() -> None:
    """A canonical interval with no sampled frames must not emit zero-valued diagnostics."""

    trace_payload = _trace_payload()
    encounter_report = _encounter_report(start_time_s=10.0, end_time_s=11.0)
    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(trace_payload),
        encounter_report=encounter_report,
        encounter_report_input_checksum="0" * 64,
    )

    assert payload["diagnostics"]["coverage"]["frame_count"] == 0
    assert payload["diagnostics"]["coverage"]["relative_interaction"]["status"] == "unavailable"
    assert payload["diagnostics"]["threshold_exposure"]["status"] == "unavailable"
    assert payload["diagnostics"]["threshold_exposure"]["duration_s"] is None
    assert payload["diagnostics"]["stall"]["status"] == "unavailable"
    assert payload["diagnostics"]["stall"]["speed_coverage"]["status"] == "unavailable"
    assert payload["diagnostics"]["reversal_counts"]["status"] == "unavailable"
    assert payload["diagnostics"]["reversal_counts"]["heading_reversal_count"] is None


def test_reversal_counts_require_complete_heading_and_route_velocity() -> None:
    """Reversal diagnostics should abstain when required route or heading signals are absent."""

    missing_heading = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(nonfinite_heading_step=1))
    )
    assert missing_heading["diagnostics"]["reversal_counts"] == {
        "profile_version": "worked_example_reversal_profile.v1",
        "direction_semantics": "robot_heading_and_velocity_projection",
        "status": "unavailable",
        "reason": "missing_robot_heading",
        "heading_reversal_count": None,
        "velocity_reversal_count": None,
    }

    missing_route_velocity = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(missing_velocity_step=1)),
        route=_registered_route("r-main"),
    )
    assert missing_route_velocity["diagnostics"]["reversal_counts"]["status"] == "unavailable"
    assert missing_route_velocity["diagnostics"]["reversal_counts"]["reason"] == (
        "missing_robot_velocity"
    )


def test_collision_scan_preserves_earliest_canonical_episode_collision() -> None:
    """Focal binding is metadata and must not replace an earlier episode collision."""

    trace_payload = _trace_payload(collision_mode="ledger_unrelated_then_focal")
    encounter_report = _encounter_report(start_time_s=0.10, end_time_s=0.17)
    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(trace_payload),
        encounter_report=encounter_report,
        encounter_report_input_checksum="0" * 64,
    )
    collision = next(
        event
        for event in payload["event_anchors"]
        if event["event_type"] == "exact_collision_event"
    )

    assert collision["status"] == "available"
    assert collision["collision_partner_id"] == "ped-b"
    assert collision["actor_id"] is None
    assert collision["focal_binding"] == {
        "status": "unavailable",
        "reason": "collision_partner_not_focal_actor",
    }
    assert collision["time_s"] == pytest.approx(0.12)
    assert collision["step"] == 2


def test_collision_step_is_anchored_from_time_not_ledger_container_frame() -> None:
    """A delayed ledger receipt cannot move the sampled frame anchoring an exact event."""

    trace_payload = _trace_payload()
    trace_payload["frames"][3]["planner"]["event_ledger"] = {
        "schema_version": "EpisodeEventLedger.v2",
        "collision_events": [
            {
                "collision_partner_type": "pedestrian",
                "collision_partner_id": "ped-a",
                "collision_time": 0.15,
                "relative_speed_at_contact": 1.0,
                "clearance_series_source": "simulator.contact",
                "exact_event_source": "simulator.collision",
            }
        ],
    }
    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(trace_payload)
    )
    collision = next(
        event
        for event in payload["event_anchors"]
        if event["event_type"] == "exact_collision_event"
    )

    assert collision["time_s"] == pytest.approx(0.15)
    assert collision["step"] == 2


def test_collision_timestamp_rejects_boolean_as_numeric_zero() -> None:
    """A JSON boolean cannot become the exact episode collision at time zero."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(collision_mode="ledger_boolean_time"))
    )
    collision = next(
        event
        for event in payload["event_anchors"]
        if event["event_type"] == "exact_collision_event"
    )

    assert collision["status"] == "unavailable"
    assert collision["reason"] == "invalid_collision_event_record_shape"
    assert "time_s" not in collision


def test_route_frames_project_only_selected_focal_actor() -> None:
    """Public route frames should carry selected focal projection, not contextual actors."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(actor_switch=True)),
        route=_registered_route("r-main"),
    )
    route = payload["frames"][0]["route"]

    assert route["focal_actor_status"] == "available"
    assert route["focal_actor_s_m"] == pytest.approx(1.0)
    assert route["focal_actor_n_m"] == pytest.approx(0.0)
    assert "contextual_actors" not in route


def test_source_and_pair_provenance_bind_trace_content_and_time_step() -> None:
    """Process and pair records should expose deterministic raw-content receipts."""

    left_payload = _trace_payload(trace_id="pair-left", planner_id="planner-a", seed=7)
    right_payload = _trace_payload(trace_id="pair-right", planner_id="planner-b", seed=7)
    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(left_payload),
        pair_trace=simulation_trace_export_from_dict(right_payload),
        pair_comparison_grain="matched_planner_pair",
    )

    assert payload["source_trace"]["content_sha256"] == _canonical_trace_checksum(left_payload)
    assert payload["source_trace"]["run_config_contract"]["status"] == "available"
    assert payload["source_trace"]["run_config_contract"]["time_step_s"] == pytest.approx(0.1)
    assert payload["pair_compatibility"]["provenance_gate"]["left_content_sha256"] == (
        _canonical_trace_checksum(left_payload)
    )
    assert payload["pair_compatibility"]["provenance_gate"]["right_content_sha256"] == (
        _canonical_trace_checksum(right_payload)
    )

    missing_step = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(time_step_s=None))
    )
    assert missing_step["source_trace"]["run_config_contract"] == {
        "status": "unavailable",
        "reason": "run_config_time_step_unavailable",
    }


def test_semantic_validator_rejects_forged_event_pair_and_source_records() -> None:
    """Validator should reject forged inventory, terminal, pair anchor, and source receipts."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(
            _trace_payload(trace_id="pair-left", planner_id="a", seed=7)
        ),
        pair_trace=simulation_trace_export_from_dict(
            _trace_payload(trace_id="pair-right", planner_id="b", seed=7)
        ),
        pair_comparison_grain="matched_planner_pair",
    )

    probes = [
        (["event_anchors", 0, "event_type"], "terminal_event", "/event_anchors"),
        (["event_anchors", 9, "status"], "available", "/event_anchors/9/status"),
        (["event_anchors", 9, "time_s"], 0.4, "/event_anchors/9/time_s"),
        (
            ["event_anchors", 4, "collision_partner_id"],
            "ped-b",
            "/event_anchors/4/collision_partner_id",
        ),
        (["source_trace", "content_sha256"], "0" * 63, "/source_trace/content_sha256"),
        (
            ["pair_compatibility", "provenance_gate", "left_content_sha256"],
            "0" * 63,
            "/pair_compatibility/provenance_gate/left_content_sha256",
        ),
    ]
    for target_path, value, expected_path in probes:
        forged = deepcopy(payload)
        _set_path(forged, target_path, value)
        with pytest.raises(Exception, match=expected_path):
            validate_worked_example_process_trace(forged)

    forged_anchor = deepcopy(payload)
    if not forged_anchor["pair_compatibility"]["valid_common_event_anchors"]:
        pytest.skip("fixture did not produce common anchors")
    forged_anchor["pair_compatibility"]["valid_common_event_anchors"][0]["left_event_id"] = (
        "missing-event"
    )
    with pytest.raises(Exception, match="/pair_compatibility/valid_common_event_anchors/0"):
        validate_worked_example_process_trace(forged_anchor)


def test_semantic_validator_rejects_coherently_moved_event_and_frame_records() -> None:
    """Event and frame records must replay from public source-coordinate frames."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(turn_and_decelerate=True)),
        route=_registered_route("r-main"),
        conflict_zone=_registered_conflict_zone("door"),
    )

    moved = deepcopy(payload)
    minimum = moved["event_anchors"][0]
    minimum["event_id"] = "step-0002-minimum-clearance"
    minimum["time_s"] = moved["frames"][2]["time_s"]
    minimum["step"] = moved["frames"][2]["step"]
    minimum["event_relative_time"] = {
        "status": "available",
        "anchor_time_s": moved["frames"][2]["time_s"],
        "tau_s": 0.0,
    }
    with pytest.raises(Exception, match="/event_anchors/0"):
        validate_worked_example_process_trace(moved)

    probes = [
        (["frames", 0, "world", "robot", "position"], [99.0, 0.0], "/frames/0/world/robot"),
        (["frames", 0, "route", "s_m"], 99.0, "/frames/0/route/s_m"),
        (
            ["frames", 0, "route", "focal_actor_s_m"],
            99.0,
            "/frames/0/route/focal_actor_s_m",
        ),
        (
            ["frames", 0, "conflict", "robot_signed_distance_to_zone_m"],
            99.0,
            "/frames/0/conflict/robot_signed_distance_to_zone_m",
        ),
        (
            [
                "frames",
                0,
                "relative_interaction",
                "closest_approach",
                "center_distance_at_closest_approach_m",
            ],
            99.0,
            "/frames/0/relative_interaction/closest_approach",
        ),
    ]
    for target_path, value, expected_path in probes:
        forged = deepcopy(payload)
        _set_path(forged, target_path, value)
        with pytest.raises(Exception, match=expected_path):
            validate_worked_example_process_trace(forged)


def test_top_level_route_and_conflict_contracts_bind_frame_geometry() -> None:
    """Top-level route/conflict declarations must match every available frame record."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()),
        route=_registered_route("route-b"),
        conflict_zone=_registered_conflict_zone("zone-b"),
    )

    route_forged = deepcopy(payload)
    route_forged["coordinate_frames"]["route"] = {
        "status": "available",
        "reason": "registered_straight_route",
        "route_id": "route-a",
        "provenance_id": "route-a.v1",
        "registry_checksum": _route_checksum((0.0, 0.0), (5.0, 0.0)),
        "coordinate_frame": "world",
        "geometry": {"type": "line_segment", "start": [0.0, 0.0], "end": [5.0, 0.0]},
    }
    with pytest.raises(Exception, match="/frames/0/route/route_id"):
        validate_worked_example_process_trace(route_forged)

    conflict_forged = deepcopy(payload)
    conflict_forged["coordinate_frames"]["conflict"] = {
        "status": "available",
        "reason": "registered_circular_conflict_zone",
        "zone_id": "zone-a",
        "provenance_id": "zone-a.v1",
        "registry_checksum": _zone_checksum((2.0, 0.0), 0.25),
        "coordinate_frame": "world",
        "geometry": {"type": "circle", "center": [2.0, 0.0], "radius_m": 0.25},
    }
    with pytest.raises(Exception, match="/frames/0/conflict/zone_id"):
        validate_worked_example_process_trace(conflict_forged)


def test_availability_contracts_reject_bidirectional_status_forgery() -> None:
    """Top-level and frame availability status must stay mutually consistent."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()),
        route=_registered_route("route-b"),
        conflict_zone=_registered_conflict_zone("zone-b"),
    )

    route_top_unavailable = deepcopy(payload)
    route_top_unavailable["coordinate_frames"]["route"] = {
        "status": "unavailable",
        "reason": "registered_route_unavailable",
    }
    with pytest.raises(Exception, match="/coordinate_frames/route/status"):
        validate_worked_example_process_trace(route_top_unavailable)

    route_frame_unavailable = deepcopy(payload)
    route_frame_unavailable["frames"][0]["route"] = {
        "status": "unavailable",
        "reason": "registered_route_unavailable",
    }
    with pytest.raises(Exception, match="/frames/0/route/status"):
        validate_worked_example_process_trace(route_frame_unavailable)

    relative_top_unavailable = deepcopy(payload)
    relative_top_unavailable["coordinate_frames"]["relative_interaction"] = {
        "status": "unavailable",
        "reason": "requested_focal_actor_missing",
    }
    with pytest.raises(Exception, match="/coordinate_frames/relative_interaction/status"):
        validate_worked_example_process_trace(relative_top_unavailable)


def test_geometry_input_contract_rejects_coherent_projection_downgrade() -> None:
    """Derived route/conflict unavailability cannot erase supplied registry inputs."""

    trace = simulation_trace_export_from_dict(_trace_payload())
    payload = build_worked_example_process_trace_from_export(
        trace,
        route=_registered_route("route-b"),
        conflict_zone=_registered_conflict_zone("zone-b"),
    )

    for key, reason in (
        ("route", "registered_route_unavailable"),
        ("conflict", "registered_conflict_zone_unavailable"),
    ):
        forged = deepcopy(payload)
        input_contract = deepcopy(forged["coordinate_frames"][key]["input_contract"])
        forged["coordinate_frames"][key] = {
            "status": "unavailable",
            "reason": reason,
            "input_contract": input_contract,
        }
        for frame in forged["frames"]:
            frame[key] = {"status": "unavailable", "reason": reason}
        _rebuild_dependent_process_views(forged, trace)

        with pytest.raises(Exception, match=rf"/coordinate_frames/{key}"):
            validate_worked_example_process_trace(forged)

    missing_receipt = deepcopy(payload)
    missing_receipt["coordinate_frames"]["route"].pop("input_contract")
    with pytest.raises(Exception, match="/coordinate_frames/route/input_contract"):
        validate_worked_example_process_trace(missing_receipt)


def test_world_and_pair_right_input_contracts_replay_canonical_inputs() -> None:
    """World claims and pair-right geometry inputs must remain source-bound."""

    robot_trace = simulation_trace_export_from_dict(_trace_payload(coordinate_frame="robot"))
    robot_payload = build_worked_example_process_trace_from_export(robot_trace)
    robot_payload["coordinate_frames"]["world"] = {
        "status": "available",
        "reason": "source_trace_world_frame",
        "source_coordinate_frame": "world",
    }
    with pytest.raises(Exception, match="/coordinate_frames/world"):
        validate_worked_example_process_trace(robot_payload)

    left = simulation_trace_export_from_dict(
        _trace_payload(trace_id="pair-input-left", planner_id="planner-a", seed=7)
    )
    right = simulation_trace_export_from_dict(
        _trace_payload(trace_id="pair-input-right", planner_id="planner-b", seed=7)
    )
    paired = build_worked_example_process_trace_from_export(
        left,
        pair_trace=right,
        pair_comparison_grain="matched_planner_pair",
        route=_registered_route("route-b"),
        conflict_zone=_registered_conflict_zone("zone-b"),
    )
    assert paired["pair_compatibility"]["right_coordinate_input_contract"] == {
        "route": paired["coordinate_frames"]["route"]["input_contract"],
        "conflict": paired["coordinate_frames"]["conflict"]["input_contract"],
    }
    paired["pair_compatibility"]["right_coordinate_input_contract"]["route"] = {
        "status": "not_supplied"
    }
    with pytest.raises(Exception, match="/pair_compatibility/right_coordinate_input_contract"):
        validate_worked_example_process_trace(paired)


def test_robot_frame_still_verifies_supplied_registry_receipts() -> None:
    """Projection unavailability must not suppress external input-receipt validation."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(coordinate_frame="robot")),
        route=_registered_route("route-b"),
        conflict_zone=_registered_conflict_zone("zone-b"),
    )
    assert payload["coordinate_frames"]["route"]["status"] == "unavailable"
    assert payload["coordinate_frames"]["conflict"]["status"] == "unavailable"

    for key in ("route", "conflict"):
        forged = deepcopy(payload)
        forged["coordinate_frames"][key]["input_contract"]["registry"]["content_sha256"] = "f" * 64
        forged["pair_compatibility"]["right_coordinate_input_contract"][key] = deepcopy(
            forged["coordinate_frames"][key]["input_contract"]
        )
        with pytest.raises(Exception, match="external geometry registry receipt"):
            validate_worked_example_process_trace(forged)

        erased = deepcopy(payload)
        erased["coordinate_frames"][key]["input_contract"]["registry"]["artifact_ref"] = None
        erased["pair_compatibility"]["right_coordinate_input_contract"][key] = deepcopy(
            erased["coordinate_frames"][key]["input_contract"]
        )
        with pytest.raises(Exception, match="required for supplied registry input"):
            validate_worked_example_process_trace(erased)


def test_frame_availability_replays_source_content_after_dependent_views_rebuild() -> None:
    """Allowlisted reasons cannot downgrade source-available coordinate projections."""

    trace = simulation_trace_export_from_dict(_trace_payload())
    payload = build_worked_example_process_trace_from_export(
        trace,
        route=_registered_route("route-b"),
        conflict_zone=_registered_conflict_zone("zone-b"),
    )

    for frame_key, reason in (
        ("world", "missing_robot_position"),
        ("route", "missing_robot_position"),
        ("conflict", "missing_robot_position"),
        ("relative_interaction", "missing_or_nonfinite_robot_heading"),
    ):
        forged = deepcopy(payload)
        forged["frames"][1][frame_key] = {"status": "unavailable", "reason": reason}
        if frame_key == "world":
            forged["frames"][1][frame_key].update(
                {
                    "robot": deepcopy(payload["frames"][1]["world"]["robot"]),
                    "focal_actor": deepcopy(payload["frames"][1]["world"]["focal_actor"]),
                }
            )
        _rebuild_dependent_process_views(forged, trace)

        with pytest.raises(Exception, match=rf"/frames/1/{frame_key}/status"):
            validate_worked_example_process_trace(forged)


def test_frame_availability_replay_preserves_canonical_missing_and_outside_frames() -> None:
    """Source missingness and canonical encounter bounds still produce valid unavailability."""

    trace_payload = _trace_payload(nonfinite_heading_step=1)
    trace_payload["frames"][2]["robot"]["position"] = [float("nan"), 0.0]
    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(trace_payload),
        route=_registered_route("route-b"),
        conflict_zone=_registered_conflict_zone("zone-b"),
        encounter_report=_encounter_report(start_time_s=0.1, end_time_s=0.2),
        encounter_report_input_checksum="0" * 64,
    )

    assert payload["frames"][0]["encounter_interval"]["status"] == "outside_interval"
    assert payload["frames"][0]["relative_interaction"]["status"] == "unavailable"
    assert payload["frames"][1]["relative_interaction"] == {
        "status": "unavailable",
        "reason": "missing_or_nonfinite_robot_heading",
    }
    assert payload["frames"][2]["world"]["status"] == "unavailable"
    assert payload["frames"][2]["route"]["status"] == "unavailable"
    assert payload["frames"][2]["conflict"]["status"] == "unavailable"
    assert payload["frames"][2]["relative_interaction"]["status"] == "unavailable"
    validate_worked_example_process_trace(payload)


def test_focal_actor_identity_binds_to_source_coordinates() -> None:
    """Coordinated focal/event/relative renames must not override source actor identity."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(collision_mode="ledger_typed")),
        encounter_report=_encounter_report(start_time_s=0.1, end_time_s=0.2),
        encounter_report_input_checksum="0" * 64,
    )
    assert payload["frames"][1]["source_coordinates"]["focal_actor_id"] == "ped-a"

    forged = deepcopy(payload)
    forged["encounters"]["focal"]["actor_id"] = "ghost"
    forged["encounters"]["focal"]["declared_encounter"]["actor_id"] = "ghost"
    forged["encounters"]["focal"]["declared_encounter"]["canonical_record"]["actor_id"] = "ghost"
    for frame in forged["frames"]:
        relative = frame["relative_interaction"]
        if relative["status"] == "available":
            relative["actor_id"] = "ghost"
    for event in forged["event_anchors"]:
        if event.get("actor_id") == "ped-a":
            event["actor_id"] = "ghost"
        if event.get("collision_partner_id") == "ped-a":
            event["collision_partner_id"] = "ghost"

    with pytest.raises(Exception, match="/frames/1/relative_interaction/actor_id"):
        validate_worked_example_process_trace(forged)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("actor_id", "ghost"),
        ("encounter_id", "ghost:encounter-9999"),
        ("minimum_clearance_m", -999.0),
    ),
)
def test_canonical_encounter_record_replays_report_receipt(field: str, value: object) -> None:
    """Canonical focal records must replay the schema-valid report entry exactly."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()),
        encounter_report=_encounter_report(start_time_s=0.1, end_time_s=0.2),
        encounter_report_input_checksum="0" * 64,
    )
    payload["encounters"]["focal"]["declared_encounter"]["canonical_record"][field] = value

    with pytest.raises(Exception, match="/encounters/focal/declared_encounter"):
        validate_worked_example_process_trace(payload)


def test_canonical_encounter_receipt_cannot_be_downgraded_to_planner_hint() -> None:
    """A canonical focal source keeps its report receipt mandatory after mutation."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()),
        encounter_report=_encounter_report(start_time_s=0.1, end_time_s=0.2),
        encounter_report_input_checksum="0" * 64,
    )
    declared = payload["encounters"]["focal"]["declared_encounter"]
    declared["schema_version"] = "planner_actor_hint.v1"
    declared.pop("report_input_contract")

    with pytest.raises(Exception, match="canonical report contract"):
        validate_worked_example_process_trace(payload)


def test_canonical_encounter_coherent_digest_rewrite_still_obeys_source_schema() -> None:
    """Rehashing an invalid selected record cannot bypass the canonical #6709 schema."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload()),
        encounter_report=_encounter_report(start_time_s=0.1, end_time_s=0.2),
        encounter_report_input_checksum="0" * 64,
    )
    declared = payload["encounters"]["focal"]["declared_encounter"]
    report_input = declared["report_input_contract"]
    selected = report_input["content_contract"]["encounters"][report_input["selected_entry_index"]]
    declared["canonical_record"]["minimum_clearance_m"] = -999.0
    selected["minimum_clearance_m"] = -999.0
    report_input["selected_entry_sha256"] = _json_digest(selected)
    report_input["content_sha256"] = _json_digest(report_input["content_contract"])

    with pytest.raises(Exception, match="minimum_clearance_m"):
        validate_worked_example_process_trace(payload)


def test_profiles_and_claim_boundary_are_exact_versioned_contracts() -> None:
    """Threshold NaN and causal text cannot masquerade as diagnostic policy."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload())
    )
    nonfinite = deepcopy(payload)
    nonfinite["profiles"]["threshold_profile"]["proxy_surface_clearance_threshold_m"] = float("nan")
    with pytest.raises(Exception, match="/profiles"):
        validate_worked_example_process_trace(nonfinite)

    causal = deepcopy(payload)
    causal["claim_boundary"] = "Paper-grade causal proof that planner choice caused safety."
    with pytest.raises(Exception, match="/claim_boundary"):
        validate_worked_example_process_trace(causal)


def test_duplicate_actor_ids_fail_before_focal_lookup_on_left_and_pair_right() -> None:
    """Per-frame actor identity must be unique in both embedded source contracts."""

    duplicate_payload = _trace_payload()
    duplicate = deepcopy(duplicate_payload["frames"][0]["pedestrians"][0])
    duplicate["position"] = [99.0, 99.0]
    duplicate_payload["frames"][0]["pedestrians"].append(duplicate)
    duplicate_trace = simulation_trace_export_from_dict(duplicate_payload)

    with pytest.raises(Exception, match="duplicate pedestrian id 'ped-a'"):
        build_worked_example_process_trace_from_export(duplicate_trace)

    valid_left = simulation_trace_export_from_dict(
        _trace_payload(trace_id="valid-left", planner_id="planner-a", seed=7)
    )
    with pytest.raises(Exception, match="pair trace.*duplicate pedestrian id 'ped-a'"):
        build_worked_example_process_trace_from_export(
            valid_left,
            pair_trace=duplicate_trace,
            pair_comparison_grain="matched_planner_pair",
        )


def test_global_minimum_and_switches_replay_from_source_actor_inventory() -> None:
    """Global minima cannot be replaced by an invented self-consistent actor series."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(actor_switch=True))
    )
    assert payload["frames"][0]["source_coordinates"]["contextual_actors"][0]["actor_id"] == "ped-a"

    forged = deepcopy(payload)
    for index, frame in enumerate(forged["frames"]):
        frame["global_minimum_actor"] = {
            "status": "available",
            "actor_id": "ghost",
            "center_distance_m": 0.01 + index,
        }
    forged["encounters"]["global_minimum_over_all_actors"] = {
        "status": "available",
        "reason": "nearest_actor_by_center_distance",
        "series": [
            {
                "step": frame["step"],
                "time_s": frame["time_s"],
                "actor_id": "ghost",
                "center_distance_m": frame["global_minimum_actor"]["center_distance_m"],
            }
            for frame in forged["frames"]
        ],
    }
    forged["encounters"]["actor_switch_events"] = []

    with pytest.raises(Exception, match="/frames/0/global_minimum_actor"):
        validate_worked_example_process_trace(forged)


def test_derived_source_coordinates_and_commands_replay_from_content_receipt() -> None:
    """Swapping a valid source-A receipt into source-B derived frames must reject."""

    source_a = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(
            _trace_payload(trace_id="source-a", actor_start_offset=0.4)
        )
    )
    source_b = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(trace_id="source-b"))
    )

    forged = deepcopy(source_b)
    forged["source_trace"] = deepcopy(source_a["source_trace"])
    with pytest.raises(Exception, match="/frames/0/source_coordinates"):
        validate_worked_example_process_trace(forged)

    forged_command = deepcopy(source_b)
    forged_command["source_trace"]["content_receipt"]["content_contract"]["frames"][0]["planner"][
        "selected_action"
    ]["linear_velocity"] = 0.123
    forged_command["source_trace"]["content_sha256"] = _digest_contract(
        forged_command["source_trace"]["content_receipt"]
    )
    with pytest.raises(Exception, match="/frames/0/commands"):
        validate_worked_example_process_trace(forged_command)


def test_pair_receipts_resolve_right_events_and_bind_content_sha() -> None:
    """Pair common anchors and content hashes should bind to real left/right receipts."""

    left_payload = _trace_payload(trace_id="pair-left", planner_id="planner-a", seed=7)
    right_payload = _trace_payload(trace_id="pair-right", planner_id="planner-b", seed=7)
    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(left_payload),
        pair_trace=simulation_trace_export_from_dict(right_payload),
        pair_comparison_grain="matched_planner_pair",
    )
    assert payload["pair_compatibility"]["valid_common_event_anchors"]

    forged_left_sha = deepcopy(payload)
    forged_left_sha["pair_compatibility"]["provenance_gate"]["left_content_sha256"] = "0" * 64
    with pytest.raises(Exception, match="/pair_compatibility/provenance_gate/left_content_sha256"):
        validate_worked_example_process_trace(forged_left_sha)

    forged_right_sha = deepcopy(payload)
    forged_right_sha["pair_compatibility"]["provenance_gate"]["right_content_sha256"] = "0" * 64
    with pytest.raises(Exception, match="/pair_compatibility/provenance_gate/right_content_sha256"):
        validate_worked_example_process_trace(forged_right_sha)

    forged_right_anchor = deepcopy(payload)
    forged_right_anchor["pair_compatibility"]["valid_common_event_anchors"][0]["right_event_id"] = (
        "missing-right-event"
    )
    with pytest.raises(Exception, match="/pair_compatibility/valid_common_event_anchors/0"):
        validate_worked_example_process_trace(forged_right_anchor)


def test_pair_compatibility_replays_status_provenance_and_seed_checks() -> None:
    """Coherent summary flips cannot make a seed-mismatched planner pair compatible."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(
            _trace_payload(trace_id="pair-left", planner_id="planner-a", seed=7)
        ),
        pair_trace=simulation_trace_export_from_dict(
            _trace_payload(trace_id="pair-right", planner_id="planner-b", seed=8)
        ),
        pair_comparison_grain="matched_planner_pair",
    )
    pair = payload["pair_compatibility"]
    assert pair["status"] == "incompatible"
    assert pair["provenance_gate"]["compatible"] is False

    forged = deepcopy(payload)
    forged_pair = forged["pair_compatibility"]
    forged_pair["status"] = "available"
    forged_pair["provenance_gate"]["compatible"] = True
    forged_pair["provenance_gate"]["checks"]["seed_equal"] = True
    forged_pair["provenance_gate"]["checks"]["seed_different"] = False

    with pytest.raises(Exception, match="/pair_compatibility/status"):
        validate_worked_example_process_trace(forged)


def test_pair_compatibility_replays_shared_prefix_and_divergence_eligibility() -> None:
    """An incompatible pair cannot forge a shared prefix to enable divergence output."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(
            _trace_payload(trace_id="pair-left", planner_id="planner-a", seed=7)
        ),
        pair_trace=simulation_trace_export_from_dict(
            _trace_payload(
                trace_id="pair-right",
                planner_id="planner-b",
                seed=8,
                diverge_after_start=True,
            )
        ),
        pair_comparison_grain="matched_planner_pair",
    )
    pair = payload["pair_compatibility"]
    assert pair["status"] == "incompatible"
    assert pair["shared_prefix"]["shared_prefix"] is False

    forged = deepcopy(payload)
    forged["pair_compatibility"]["shared_prefix"]["shared_prefix"] = True
    forged["pair_compatibility"]["divergence_interpretation"] = {
        "allowed": True,
        "reason": "shared_prefix_available",
    }
    with pytest.raises(Exception, match="/pair_compatibility/shared_prefix/shared_prefix"):
        validate_worked_example_process_trace(forged)


def test_matched_planner_pair_requires_equal_replayed_time_step_contracts() -> None:
    """Different declared time steps make an otherwise matched planner pair incompatible."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(
            _trace_payload(
                trace_id="pair-left",
                planner_id="planner-a",
                seed=7,
                time_step_s=0.1,
            )
        ),
        pair_trace=simulation_trace_export_from_dict(
            _trace_payload(
                trace_id="pair-right",
                planner_id="planner-b",
                seed=7,
                time_step_s=0.2,
            )
        ),
        pair_comparison_grain="matched_planner_pair",
    )
    pair = payload["pair_compatibility"]

    assert pair["status"] == "incompatible"
    assert pair["provenance_gate"]["checks"]["time_step_s_equal"] is False
    assert pair["provenance_gate"]["time_step_contracts"]["left"]["time_step_s"] == 0.1
    assert pair["provenance_gate"]["time_step_contracts"]["right"]["time_step_s"] == 0.2
    validate_worked_example_process_trace(payload)


def test_source_and_pair_hashes_recompute_from_canonical_content_receipts() -> None:
    """Coherent hash rewrites should fail against embedded canonical trace contracts."""

    left_payload = _trace_payload(trace_id="pair-left", planner_id="planner-a", seed=7)
    right_payload = _trace_payload(trace_id="pair-right", planner_id="planner-b", seed=7)
    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(left_payload),
        pair_trace=simulation_trace_export_from_dict(right_payload),
        pair_comparison_grain="matched_planner_pair",
    )

    assert payload["source_trace"]["content_receipt"]["content_contract"] == (
        _canonical_trace_contract(left_payload)
    )
    assert payload["pair_compatibility"]["right_source_trace"]["content_receipt"][
        "content_contract"
    ] == (_canonical_trace_contract(right_payload))

    forged_left = deepcopy(payload)
    forged_left["source_trace"]["content_sha256"] = "0" * 64
    forged_left["pair_compatibility"]["provenance_gate"]["left_content_sha256"] = "0" * 64
    with pytest.raises(Exception, match="/source_trace/content_sha256"):
        validate_worked_example_process_trace(forged_left)

    forged_right = deepcopy(payload)
    forged_right["pair_compatibility"]["right_source_trace"]["content_sha256"] = "0" * 64
    forged_right["pair_compatibility"]["provenance_gate"]["right_content_sha256"] = "0" * 64
    with pytest.raises(Exception, match="/pair_compatibility/right_source_trace/content_sha256"):
        validate_worked_example_process_trace(forged_right)


def test_common_anchors_require_derived_right_receipt_ids() -> None:
    """Right receipt IDs cannot be internally renamed with matching common anchors."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(
            _trace_payload(trace_id="pair-left", planner_id="planner-a", seed=7)
        ),
        pair_trace=simulation_trace_export_from_dict(
            _trace_payload(trace_id="pair-right", planner_id="planner-b", seed=7)
        ),
        pair_comparison_grain="matched_planner_pair",
    )
    assert payload["pair_compatibility"]["valid_common_event_anchors"]

    forged = deepcopy(payload)
    first_anchor = forged["pair_compatibility"]["valid_common_event_anchors"][0]
    old_right_id = first_anchor["right_event_id"]
    new_right_id = f"renamed-{old_right_id}"
    first_anchor["right_event_id"] = new_right_id
    for receipt in forged["pair_compatibility"]["right_event_anchors"]:
        if receipt["event_id"] == old_right_id:
            receipt["event_id"] = new_right_id
            break

    with pytest.raises(Exception, match="/pair_compatibility/right_event_anchors"):
        validate_worked_example_process_trace(forged)


def test_right_event_receipts_replay_from_right_content_receipt() -> None:
    """Right receipts cannot be coherently moved by changing step/time/id and common anchor."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(
            _trace_payload(trace_id="pair-left", planner_id="planner-a", seed=7)
        ),
        pair_trace=simulation_trace_export_from_dict(
            _trace_payload(trace_id="pair-right", planner_id="planner-b", seed=7)
        ),
        pair_comparison_grain="matched_planner_pair",
    )
    assert payload["pair_compatibility"]["valid_common_event_anchors"]
    forged = deepcopy(payload)
    anchor = forged["pair_compatibility"]["valid_common_event_anchors"][0]
    receipt = next(
        item
        for item in forged["pair_compatibility"]["right_event_anchors"]
        if item["event_id"] == anchor["right_event_id"]
    )
    receipt["step"] = receipt["step"] + 1
    receipt["time_s"] = receipt["time_s"] + 0.1
    receipt["event_id"] = f"step-{receipt['step']:04d}-{receipt['event_type'].replace('_', '-')}"
    receipt["event_relative_time"] = {
        "status": "available",
        "anchor_time_s": receipt["time_s"],
        "tau_s": 0.0,
    }
    anchor["right_event_id"] = receipt["event_id"]

    with pytest.raises(Exception, match="/pair_compatibility/right_event_anchors"):
        validate_worked_example_process_trace(forged)


def test_exact_collision_event_replay_rejects_coherent_hierarchy_and_status_forgery() -> None:
    """Exact collision anchors must replay even when dependent hierarchy/frame fields are aligned."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(collision_mode="ledger_typed")),
        encounter_report=_encounter_report(start_time_s=0.1, end_time_s=0.2),
        encounter_report_input_checksum="0" * 64,
    )
    collision = payload["event_anchors"][4]
    assert collision["event_type"] == "exact_collision_event"
    assert collision["status"] == "available"

    forged = deepcopy(payload)
    forged_collision = forged["event_anchors"][4]
    forged_collision["time_s"] = 0.2
    forged_collision["event_relative_time"] = {
        "status": "available",
        "anchor_time_s": 0.2,
        "tau_s": 0.0,
    }
    forged["event_anchor_hierarchy"]["available_anchors"][0]["time_s"] = 0.2
    forged["event_anchor_hierarchy"]["selected_anchor"]["time_s"] = 0.2
    forged["event_anchor_hierarchy"]["anchor_time_s"] = 0.2
    for frame in forged["frames"]:
        frame["event_alignment"]["anchor_time_s"] = 0.2
        frame["event_alignment"]["tau_s"] = frame["time_s"] - 0.2
    with pytest.raises(Exception, match="/event_anchors/4"):
        validate_worked_example_process_trace(forged)

    unavailable = deepcopy(payload)
    unavailable["event_anchors"][4] = {
        "event_id": "exact_collision_event-unavailable",
        "event_type": "exact_collision_event",
        "detector_profile_version": "worked_example_event_detectors.v1",
        "status": "unavailable",
        "confidence": "not_available",
        "actor_id": "ped-a",
        "zone_id": None,
        "reason": "attacker_selected_reason",
        "source_fields": [
            "planner.outcome.collision_event",
            "planner.event_ledger.collision_events",
        ],
        "event_relative_time": {
            "status": "unavailable",
            "reason": "event_unavailable",
        },
        "visual_anchor_eligibility": {
            "eligible": False,
            "reason": "event_unavailable",
        },
    }
    with pytest.raises(Exception, match="/event_anchors/4"):
        validate_worked_example_process_trace(unavailable)

    promoted = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload())
    )
    promoted["event_anchors"][4] = {
        **collision,
        "time_s": promoted["frames"][1]["time_s"],
        "step": promoted["frames"][1]["step"],
    }
    with pytest.raises(Exception, match="/event_anchors/4"):
        validate_worked_example_process_trace(promoted)


def test_collision_time_must_be_within_declared_and_sampled_bounds() -> None:
    """Canonical collision anchors must not pin out-of-sample times to edge frames."""

    for mode in ("ledger_collision_before_trace", "ledger_collision_after_trace"):
        payload = build_worked_example_process_trace_from_export(
            simulation_trace_export_from_dict(_trace_payload(collision_mode=mode))
        )
        collision = next(
            event
            for event in payload["event_anchors"]
            if event["event_type"] == "exact_collision_event"
        )
        assert collision["status"] == "unavailable"
        assert collision["reason"] == "collision_time_outside_trace_sample_bounds"


def test_run_config_contract_scans_all_frames_and_rejects_bool_or_inconsistent_time_step() -> None:
    """Configured time_step_s must be a consistent non-bool contract across declarations."""

    bool_step = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(time_step_s=True))
    )
    assert bool_step["source_trace"]["run_config_contract"] == {
        "status": "unavailable",
        "reason": "run_config_time_step_unavailable",
    }

    missing_later = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(missing_later_time_step=True))
    )
    assert missing_later["source_trace"]["run_config_contract"] == {
        "status": "unavailable",
        "reason": "run_config_time_step_unavailable",
    }

    inconsistent = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(inconsistent_time_step=True))
    )
    assert inconsistent["source_trace"]["run_config_contract"] == {
        "status": "unavailable",
        "reason": "run_config_time_step_inconsistent",
    }

    missing_later_run_config = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(missing_later_run_config=True))
    )
    assert missing_later_run_config["source_trace"]["run_config_contract"] == {
        "status": "unavailable",
        "reason": "run_config_unavailable",
    }


def test_run_config_contract_rejects_declared_time_step_that_disagrees_with_samples() -> None:
    """The declared step is preserved but cannot override sampled step/time increments."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(time_step_s=0.2))
    )

    assert payload["source_trace"]["run_config_contract"] == {
        "status": "unavailable",
        "reason": "run_config_time_step_trace_mismatch",
        "time_step_s": 0.2,
        "observed_time_step_s": 0.1,
        "config_digest": "a" * 64,
        "source": "planner.run_config",
    }
    validate_worked_example_process_trace(payload)


def test_run_config_contract_replays_from_embedded_source_content() -> None:
    """Run-config status/time/reason cannot be changed independently from source content."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload())
    )
    forged_time = deepcopy(payload)
    forged_time["source_trace"]["run_config_contract"]["time_step_s"] = 0.2
    with pytest.raises(Exception, match="/source_trace/run_config_contract"):
        validate_worked_example_process_trace(forged_time)

    forged_status = deepcopy(payload)
    forged_status["source_trace"]["run_config_contract"] = {
        "status": "unavailable",
        "reason": "attacker_reason",
    }
    with pytest.raises(Exception, match="/source_trace/run_config_contract"):
        validate_worked_example_process_trace(forged_status)


def test_embedded_content_receipts_are_strict_json_for_nonfinite_source_values() -> None:
    """Builder payloads with admitted nonfinite source values must still serialize strictly."""

    for trace_payload in (
        _trace_payload(nonfinite_heading_step=1),
        _trace_payload(nan_command_step=1),
        _trace_payload(missing_actor_radius_step=1),
    ):
        payload = build_worked_example_process_trace_from_export(
            simulation_trace_export_from_dict(trace_payload)
        )
        json.dumps(payload, allow_nan=False, sort_keys=True)


def test_nonfinite_receipt_is_injective_against_literal_planner_objects() -> None:
    """An actual NaN and a literal lookalike planner object need distinct identities."""

    literal_payload = _trace_payload(trace_id="literal-nonfinite-object")
    literal_payload["frames"][0]["planner"]["receipt_probe"] = {"nonfinite_number": "nan"}
    nan_payload = deepcopy(literal_payload)
    nan_payload["frames"][0]["planner"]["receipt_probe"] = math.nan

    literal = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(literal_payload)
    )
    actual_nan = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(nan_payload)
    )

    assert literal["source_trace"]["content_sha256"] != actual_nan["source_trace"]["content_sha256"]
    assert literal["analysis_input_sha256"] != actual_nan["analysis_input_sha256"]
    assert literal["process_trace_id"] != actual_nan["process_trace_id"]
    assert coordinates.worked_example_process_trace_artifact_sha256(
        literal
    ) != coordinates.worked_example_process_trace_artifact_sha256(actual_nan)

    left = simulation_trace_export_from_dict(_trace_payload(trace_id="pair-left"))
    literal_pair = build_worked_example_process_trace_from_export(
        left,
        pair_trace=simulation_trace_export_from_dict(literal_payload),
        pair_comparison_grain="matched_planner_pair",
    )
    nan_pair = build_worked_example_process_trace_from_export(
        left,
        pair_trace=simulation_trace_export_from_dict(nan_payload),
        pair_comparison_grain="matched_planner_pair",
    )
    assert (
        literal_pair["pair_compatibility"]["right_source_trace"]["content_sha256"]
        != nan_pair["pair_compatibility"]["right_source_trace"]["content_sha256"]
    )
    assert (
        literal_pair["analysis_input_contract"]["pair_trace"]["content_sha256"]
        != nan_pair["analysis_input_contract"]["pair_trace"]["content_sha256"]
    )


def test_nonfinite_receipt_uses_canonical_paths_and_rejects_ambiguous_ledgers() -> None:
    """Only sorted, unique, non-overlapping RFC6901 paths may restore null targets."""

    trace_payload = _trace_payload()
    trace_payload["frames"][0]["planner"]["receipt~/probe"] = math.nan
    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(trace_payload)
    )
    receipt = payload["source_trace"]["content_receipt"]
    assert receipt["nonfinite_numbers"] == [
        {
            "path": "/frames/0/planner/receipt~0~1probe",
            "value": "nan",
        }
    ]

    mutations = []
    malformed_escape = deepcopy(payload)
    malformed_escape["source_trace"]["content_receipt"]["nonfinite_numbers"][0]["path"] = (
        "/frames/0/planner/receipt~2probe"
    )
    mutations.append(malformed_escape)

    duplicate = deepcopy(payload)
    duplicate["source_trace"]["content_receipt"]["nonfinite_numbers"].append(
        deepcopy(duplicate["source_trace"]["content_receipt"]["nonfinite_numbers"][0])
    )
    mutations.append(duplicate)

    prefix_conflict = deepcopy(payload)
    prefix_conflict["source_trace"]["content_receipt"]["nonfinite_numbers"].insert(
        0,
        {"path": "/frames/0/planner", "value": "nan"},
    )
    mutations.append(prefix_conflict)

    non_null_target = deepcopy(payload)
    non_null_target["source_trace"]["content_receipt"]["content_contract"]["frames"][0]["planner"][
        "receipt~/probe"
    ] = 0.0
    mutations.append(non_null_target)

    wrong_container = deepcopy(payload)
    wrong_container["source_trace"]["content_receipt"]["nonfinite_numbers"][0]["path"] = (
        "/frames/not-an-index/planner/receipt~0~1probe"
    )
    mutations.append(wrong_container)

    for forged in mutations:
        forged["source_trace"]["content_sha256"] = _json_digest(
            forged["source_trace"]["content_receipt"]
        )
        with pytest.raises(Exception, match="/source_trace/content_receipt"):
            validate_worked_example_process_trace(forged)


def test_semantic_validator_rejects_available_event_without_coordinates() -> None:
    """Available semantic anchors require step/time and non-empty records."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload())
    )
    payload["event_anchors"][0].pop("time_s")
    payload["frames"][0]["relative_interaction"] = {}
    payload["event_anchor_hierarchy"]["selected_anchor"] = {"event_type": "banana"}
    payload["pair_compatibility"]["comparison_grain"]["grain_id"] = "banana"
    with pytest.raises(Exception, match="/event_anchors/0/time_s"):
        validate_worked_example_process_trace(payload)


def test_semantic_validator_rejects_bogus_nested_records_and_hierarchy_selection() -> None:
    """Nested schema and semantic checks should reject bogus replacement records."""

    base_payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(collision_mode="ledger_typed"))
    )
    route_payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(collision_mode="ledger_typed")),
        route=_registered_route("r-main"),
        conflict_zone=_registered_conflict_zone("door"),
    )

    probes: list[tuple[str, list[object], object]] = [
        ("/frames/0/source_coordinates/robot", ["frames", 0, "source_coordinates", "robot"], {}),
        (
            "/frames/0/source_coordinates/focal_actor",
            ["frames", 0, "source_coordinates", "focal_actor"],
            {"position": ["banana"], "heading": 0.0, "velocity": [0.0, 0.0], "radius_m": 0.25},
        ),
        ("/frames/0/world/robot", ["frames", 0, "world", "robot"], {"position": [0.0, 0.0]}),
        ("/frames/0/world/focal_actor", ["frames", 0, "world", "focal_actor"], {"banana": True}),
        ("/frames/0/route", ["frames", 0, "route"], {"status": "available", "route_id": "r"}),
        ("/frames/0/conflict", ["frames", 0, "conflict"], {"status": "available", "zone_id": "z"}),
        (
            "/frames/0/relative_interaction",
            ["frames", 0, "relative_interaction"],
            {"status": "available", "actor_id": "ped-a"},
        ),
        (
            "/frames/0/relative_interaction/closest_approach",
            ["frames", 0, "relative_interaction", "closest_approach"],
            {"status": "available", "time_to_closest_approach_s": "banana"},
        ),
        (
            "/frames/0/relative_interaction/actor_id",
            ["frames", 0, "relative_interaction", "actor_id"],
            "ped-b",
        ),
        (
            "/frames/0/relative_interaction/proxy_surface_clearance_m",
            ["frames", 0, "relative_interaction", "proxy_surface_clearance_m"],
            None,
        ),
        (
            "/frames/0/relative_interaction/proxy_surface_clearance_m",
            ["frames", 0, "relative_interaction", "proxy_surface_clearance_status"],
            "unavailable",
        ),
        (
            "/frames/0/route/geometry_checksum",
            ["frames", 0, "route", "geometry", "start"],
            [99.0, 0.0],
        ),
        (
            "/frames/0/conflict/geometry_checksum",
            ["frames", 0, "conflict", "geometry", "center"],
            [99.0, 0.0],
        ),
        (
            "/frames/0/commands",
            ["frames", 0, "commands"],
            {"status": "available", "commanded": "banana"},
        ),
        (
            "/frames/0/commands/executed",
            ["frames", 0, "commands"],
            {
                "status": "available",
                "commanded": {"linear_velocity": 1.0},
                "executed": {"linear_velocity": ["banana"]},
                "executed_status": "available",
            },
        ),
        (
            "/diagnostics/route_progress",
            ["diagnostics", "route_progress"],
            {"status": "available", "start_s_m": "banana", "end_s_m": 1.0, "delta_s_m": 1.0},
        ),
        ("/diagnostics/stall", ["diagnostics", "stall"], {"status": "available"}),
        (
            "/diagnostics/stall/status",
            ["diagnostics", "stall"],
            {
                **route_payload["diagnostics"]["stall"],
                "status": "unavailable",
                "sustained_stall_duration_s": 1.0,
            },
        ),
        (
            "/diagnostics/reversal_counts",
            ["diagnostics", "reversal_counts"],
            {
                "profile_version": "v",
                "direction_semantics": "v",
                "heading_reversal_count": -1,
                "velocity_reversal_count": 0,
            },
        ),
        ("/diagnostics/coverage", ["diagnostics", "coverage"], {"frame_count": 0}),
        (
            "/diagnostics/coverage/relative_interaction/missing_frame_count",
            ["diagnostics", "coverage", "relative_interaction", "missing_frame_count"],
            99,
        ),
        (
            "/diagnostics/threshold_exposure/duration_s",
            ["diagnostics", "threshold_exposure", "duration_s"],
            "banana",
        ),
        (
            "/diagnostics/threshold_exposure/status",
            ["diagnostics", "threshold_exposure"],
            {
                **route_payload["diagnostics"]["threshold_exposure"],
                "status": "unavailable",
                "duration_s": 1.0,
            },
        ),
        (
            "/encounters/global_minimum_over_all_actors/series/0",
            ["encounters", "global_minimum_over_all_actors", "series", 0],
            {"step": 0, "time_s": 0.0, "actor_id": 7, "center_distance_m": 1.0},
        ),
    ]
    for expected_path, target_path, value in probes:
        payload = deepcopy(route_payload)
        _set_path(payload, target_path, value)
        with pytest.raises(Exception, match=expected_path):
            validate_worked_example_process_trace(payload)

    payload = deepcopy(base_payload)
    payload["event_anchor_hierarchy"]["available_anchors"][0]["event_type"] = "banana"
    with pytest.raises(Exception, match="/event_anchor_hierarchy/available_anchors"):
        validate_worked_example_process_trace(payload)

    payload = deepcopy(base_payload)
    payload["event_anchor_hierarchy"]["available_anchors"].pop(0)
    with pytest.raises(Exception, match="/event_anchor_hierarchy/available_anchors"):
        validate_worked_example_process_trace(payload)

    payload = deepcopy(base_payload)
    anchors = payload["event_anchor_hierarchy"]["available_anchors"]
    assert len(anchors) >= 2
    payload["event_anchor_hierarchy"]["selected_anchor"] = anchors[-1]
    payload["event_anchor_hierarchy"]["anchor_time_s"] = anchors[-1]["time_s"]
    with pytest.raises(Exception, match="/event_anchor_hierarchy/selected_anchor"):
        validate_worked_example_process_trace(payload)

    payload = deepcopy(base_payload)
    payload["frames"][0]["event_alignment"]["anchor_event_id"] = "forged-anchor"
    with pytest.raises(Exception, match="/frames/0/event_alignment/anchor_event_id"):
        validate_worked_example_process_trace(payload)

    payload = deepcopy(base_payload)
    payload["frames"][0]["event_alignment"]["anchor_event_type"] = "minimum_clearance"
    with pytest.raises(Exception, match="/frames/0/event_alignment/anchor_event_type"):
        validate_worked_example_process_trace(payload)

    payload = deepcopy(base_payload)
    payload["frames"][0]["event_alignment"]["anchor_time_s"] = 123.0
    payload["frames"][0]["event_alignment"]["tau_s"] = payload["frames"][0]["time_s"] - 123.0
    with pytest.raises(Exception, match="/frames/0/event_alignment/anchor_time_s"):
        validate_worked_example_process_trace(payload)

    payload = deepcopy(base_payload)
    wrong_anchor = payload["event_anchor_hierarchy"]["available_anchors"][-1]
    payload["event_anchor_hierarchy"]["selected_anchor"] = wrong_anchor
    payload["event_anchor_hierarchy"]["anchor_time_s"] = wrong_anchor["time_s"]
    for frame in payload["frames"]:
        if frame["event_alignment"]["status"] == "available":
            frame["event_alignment"]["anchor_event_id"] = wrong_anchor["event_id"]
            frame["event_alignment"]["anchor_event_type"] = wrong_anchor["event_type"]
            frame["event_alignment"]["anchor_time_s"] = wrong_anchor["time_s"]
            frame["event_alignment"]["tau_s"] = frame["time_s"] - wrong_anchor["time_s"]
    with pytest.raises(Exception, match="/event_anchor_hierarchy/selected_anchor"):
        validate_worked_example_process_trace(payload)


def test_matched_planner_empty_actor_realizations_remain_compatible() -> None:
    """Empty actor sets are equal state, not actor divergence."""

    left = simulation_trace_export_from_dict(
        _trace_payload(no_pedestrians=True, planner_id="planner-a", seed=7)
    )
    right = simulation_trace_export_from_dict(
        _trace_payload(no_pedestrians=True, planner_id="planner-b", seed=7, config_digest="b" * 64)
    )

    pair = build_worked_example_process_trace_from_export(
        left,
        pair_trace=right,
        pair_comparison_grain="matched_planner_pair",
    )["pair_compatibility"]

    assert pair["status"] == "available"
    assert pair["initial_state_equivalence"]["equivalent"] is True
    assert pair["initial_state_equivalence"]["actor_id_sets_equal"] is True
    assert pair["initial_state_equivalence"]["max_actor_position_delta_m"] is None


def test_matched_realization_nonfinite_heading_cannot_enable_divergence() -> None:
    """Comparison missingness must fail shared-prefix eligibility closed."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(
            _trace_payload(
                trace_id="pair-left",
                planner_id="ppo",
                seed=7,
                nonfinite_heading_step=1,
            )
        ),
        pair_trace=simulation_trace_export_from_dict(
            _trace_payload(
                trace_id="pair-right",
                planner_id="ppo",
                seed=8,
                nonfinite_heading_step=1,
            )
        ),
        pair_comparison_grain="matched_realization_pair",
    )
    pair = payload["pair_compatibility"]

    assert pair["shared_prefix"]["shared_prefix"] is False
    assert pair["shared_prefix"]["reason"] == "missing_robot_heading"
    assert pair["divergence_interpretation"]["allowed"] is False
    validate_worked_example_process_trace(payload)


def _set_path(payload: dict[str, object], path: list[object], value: object) -> None:
    cursor: object = payload
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    last = path[-1]
    if isinstance(last, int):
        cursor[last] = value  # type: ignore[index]
    else:
        cursor[last] = value  # type: ignore[index]


def _rebuild_dependent_process_views(
    payload: dict[str, object],
    trace: object,
) -> None:
    """Rebuild every mutable frame-derived view used by semantic validation."""

    focal = payload["encounters"]["focal"]  # type: ignore[index]
    frames = payload["frames"]  # type: ignore[assignment]
    event_frames = coordinates._diagnostic_frames(frames)  # type: ignore[arg-type]
    events = coordinates._event_anchors(
        trace,
        frames=event_frames,
        episode_frames=frames,
        focal_actor_id=focal.get("actor_id"),
        focal_interval=coordinates._focal_interval_bounds(focal),
    )
    hierarchy = coordinates._event_anchor_hierarchy(events)
    payload["frames"] = coordinates._frames_with_event_alignment(frames, hierarchy)  # type: ignore[arg-type]
    payload["event_anchors"] = events
    payload["event_anchor_hierarchy"] = hierarchy
    payload["diagnostics"] = coordinates._diagnostics(
        event_frames,
        route_available=any(frame["route"].get("status") == "available" for frame in event_frames),
    )
    focal["actor_contiguity"] = coordinates._actor_contiguity(
        event_frames,
        focal.get("actor_id"),
        declared=focal.get("declared_encounter"),
    )


def _canonical_trace_checksum(payload: dict[str, object]) -> str:
    return _digest_contract(
        {
            "schema_version": "simulation_trace_export_receipt.v1",
            "content_contract": _canonical_trace_contract(payload),
            "nonfinite_numbers": [],
        }
    )


def _digest_contract(contract: object) -> str:
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonical_trace_contract(payload: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": payload["schema_version"],
        "trace_id": payload["trace_id"],
        "source": payload["source"],
        "evidence_boundary": payload["evidence_boundary"],
        "coordinate_frame": payload["coordinate_frame"],
        "units": payload["units"],
        "frames": [
            {
                "step": frame["step"],
                "time_s": frame["time_s"],
                "robot": frame["robot"],
                "pedestrians": list(frame["pedestrians"]),
                "planner": frame["planner"],
            }
            for frame in payload["frames"]  # type: ignore[index]
        ],
    }


def _trace_payload(  # noqa: C901, PLR0912, PLR0913
    *,
    trace_id: str = "process-trace-fixture",
    seed: int = 113,
    actor_switch: bool = False,
    static_relative_velocity: bool = False,
    turn_and_decelerate: bool = False,
    diverge_after_start: bool = False,
    coordinate_frame: str = "world",
    proxy_overlap: bool = False,
    actor_start_offset: float = 0.0,
    planner_id: str = "ppo",
    include_run_config: bool = True,
    config_digest: str = "a" * 64,
    time_step_s: object = 0.1,
    missing_later_time_step: bool = False,
    inconsistent_time_step: bool = False,
    missing_later_run_config: bool = False,
    collision_mode: str | None = None,
    missing_actor_radius_step: int | None = None,
    nonfinite_heading_step: int | None = None,
    stall_pattern: list[float] | None = None,
    nan_command_step: int | None = None,
    missing_velocity_step: int | None = None,
    no_pedestrians: bool = False,
) -> dict[str, object]:
    robot_vel = [0.0, 0.0] if static_relative_velocity else [1.0, 0.0]
    ped_vel = [0.0, 0.0]
    actions = [
        {"linear_velocity": 1.0, "angular_velocity": 0.0},
        {"linear_velocity": 0.7 if turn_and_decelerate else 1.0, "angular_velocity": 0.0},
        {"linear_velocity": 0.7, "angular_velocity": 0.4 if turn_and_decelerate else 0.0},
        {"linear_velocity": 0.7, "angular_velocity": 0.0},
    ]
    positions = [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.0]]
    if stall_pattern is not None and len(stall_pattern) > len(positions):
        for step in range(len(positions), len(stall_pattern)):
            positions.append([0.2 * step, 0.0])
            actions.append({"linear_velocity": 0.7, "angular_velocity": 0.0})
    if diverge_after_start:
        positions[1:] = [[0.2, 0.1], [0.4, 0.2], [0.6, 0.3]]
    frames = []
    for step, position in enumerate(positions):
        actor = {
            "id": "ped-a",
            "position": [
                (0.2 if proxy_overlap else 1.0) + actor_start_offset + 0.1 * step,
                0.0,
            ],
            "velocity": ped_vel,
            "radius": 0.25,
        }
        if missing_actor_radius_step == step:
            actor.pop("radius")
        peds = [] if no_pedestrians else [actor]
        if actor_switch:
            peds.append(
                {
                    "id": "ped-b",
                    "position": [3.0 if step < 2 else 0.45, 0.0],
                    "velocity": [0.0, 0.0],
                    "radius": 0.25,
                }
            )
        if collision_mode == "ledger_ped_b":
            peds.append(
                {
                    "id": "ped-b",
                    "position": [0.45, 0.0],
                    "velocity": [0.0, 0.0],
                    "radius": 0.25,
                }
            )
        encounter = {
            "actor_id": "ped-a",
            "encounter_id": "ped-a:encounter-0001",
        }
        planner = {
            "selected_action": actions[step],
            "encounter": encounter,
            "event": "step",
        }
        if nan_command_step == step:
            planner["selected_action"] = {"linear_velocity": float("nan"), "angular_velocity": 0.0}
        if include_run_config and not (missing_later_run_config and step > 0):
            planner["run_config"] = {
                "map_id": "fixture-map",
                "horizon": 4,
                "config_digest": config_digest,
            }
            if not (missing_later_time_step and step > 0) and time_step_s is not None:
                planner["run_config"]["time_step_s"] = (
                    0.2 if inconsistent_time_step and step > 0 else time_step_s
                )
        if collision_mode == "outcome_boolean" and step == 1:
            planner["outcome"] = {"collision_event": True}
        if collision_mode == "ledger_typed" and step == 1:
            planner["event_ledger"] = {
                "schema_version": "EpisodeEventLedger.v2",
                "collision_events": [
                    {
                        "collision_partner_type": "pedestrian",
                        "collision_partner_id": "ped-a",
                        "collision_time": 0.15,
                        "relative_speed_at_contact": 1.0,
                        "clearance_series_source": "simulator.contact",
                        "exact_event_source": "simulator.collision",
                    }
                ],
            }
        if collision_mode == "ledger_ped_b" and step == 1:
            planner["event_ledger"] = {
                "schema_version": "EpisodeEventLedger.v2",
                "collision_events": [
                    {
                        "collision_partner_type": "pedestrian",
                        "collision_partner_id": "ped-b",
                        "collision_time": 0.15,
                        "relative_speed_at_contact": 1.0,
                        "clearance_series_source": "simulator.contact",
                        "exact_event_source": "simulator.collision",
                    }
                ],
            }
        if collision_mode == "ledger_unrelated_then_focal" and step == 1:
            planner["event_ledger"] = {
                "schema_version": "EpisodeEventLedger.v2",
                "collision_events": [
                    {
                        "collision_partner_type": "pedestrian",
                        "collision_partner_id": "ped-b",
                        "collision_time": 0.12,
                        "relative_speed_at_contact": 1.0,
                        "clearance_series_source": "simulator.contact",
                        "exact_event_source": "simulator.collision",
                    },
                    {
                        "collision_partner_type": "pedestrian",
                        "collision_partner_id": "ped-a",
                        "collision_time": 0.15,
                        "relative_speed_at_contact": 1.0,
                        "clearance_series_source": "simulator.contact",
                        "exact_event_source": "simulator.collision",
                    },
                ],
            }
        if collision_mode == "ledger_collision_late" and step == 3:
            planner["event_ledger"] = {
                "schema_version": "EpisodeEventLedger.v2",
                "collision_events": [
                    {
                        "collision_partner_type": "pedestrian",
                        "collision_partner_id": "ped-a",
                        "collision_time": 0.35,
                        "relative_speed_at_contact": 1.0,
                        "clearance_series_source": "simulator.contact",
                        "exact_event_source": "simulator.collision",
                    }
                ],
            }
        if collision_mode == "ledger_collision_before_trace" and step == 1:
            planner["event_ledger"] = {
                "schema_version": "EpisodeEventLedger.v2",
                "collision_events": [
                    {
                        "collision_partner_type": "pedestrian",
                        "collision_partner_id": "ped-a",
                        "collision_time": -1.0,
                        "relative_speed_at_contact": 1.0,
                        "clearance_series_source": "simulator.contact",
                        "exact_event_source": "simulator.collision",
                    }
                ],
            }
        if collision_mode == "ledger_collision_after_trace" and step == 1:
            planner["event_ledger"] = {
                "schema_version": "EpisodeEventLedger.v2",
                "collision_events": [
                    {
                        "collision_partner_type": "pedestrian",
                        "collision_partner_id": "ped-a",
                        "collision_time": 10.0,
                        "relative_speed_at_contact": 1.0,
                        "clearance_series_source": "simulator.contact",
                        "exact_event_source": "simulator.collision",
                    }
                ],
            }
        if collision_mode == "static_geometry_collision" and step == 1:
            planner["event_ledger"] = {
                "schema_version": "EpisodeEventLedger.v2",
                "collision_events": [
                    {
                        "collision_partner_type": "static_geometry",
                        "collision_partner_id": None,
                        "collision_time": 0.15,
                        "relative_speed_at_contact": None,
                        "clearance_series_source": "simulator.contact",
                        "exact_event_source": "simulator.collision",
                    }
                ],
            }
        if collision_mode == "time_only_collision_record" and step == 1:
            planner["event_ledger"] = {
                "schema_version": "EpisodeEventLedger.v2",
                "collision_events": [{"collision_time": 0.15}],
            }
        if collision_mode == "ledger_boolean_time" and step == 1:
            planner["event_ledger"] = {
                "schema_version": "EpisodeEventLedger.v2",
                "collision_events": [
                    {
                        "collision_partner_type": "pedestrian",
                        "collision_partner_id": "ped-a",
                        "collision_time": False,
                        "relative_speed_at_contact": 1.0,
                        "clearance_series_source": "simulator.contact",
                        "exact_event_source": "simulator.collision",
                    }
                ],
            }
        if collision_mode == "legacy_collision_time_s" and step == 1:
            planner["event_ledger"] = {
                "schema_version": "EpisodeEventLedger.v2",
                "collision_events": [{"collision_time_s": 0.15}],
            }
        if collision_mode == "invented_events" and step == 1:
            planner["event_ledger"] = {"events": [{"event_type": "collision", "time_s": 0.15}]}
        velocity = [float(stall_pattern[step]), 0.0] if stall_pattern is not None else robot_vel
        robot = {
            "position": position,
            "heading": float("nan") if nonfinite_heading_step == step else 0.0,
            "velocity": velocity,
            "radius": 0.25,
        }
        if missing_velocity_step == step:
            robot["velocity"] = [float("nan"), 0.0]
        frames.append(
            {
                "step": step,
                "time_s": step * 0.1,
                "robot": robot,
                "pedestrians": peds,
                "planner": planner,
            }
        )
    return {
        "schema_version": "simulation_trace_export.v1",
        "trace_id": trace_id,
        "source": {
            "scenario_id": "narrow_doorway",
            "seed": seed,
            "planner_id": planner_id,
            "episode_id": f"{trace_id}-episode",
            "generated_by": "unit-test fixture",
        },
        "evidence_boundary": "analysis_workbench_only",
        "coordinate_frame": coordinate_frame,
        "units": {"position": "m", "heading": "rad", "time": "s", "velocity": "m/s"},
        "frames": frames,
    }


def _encounter_report(*, start_time_s: float = 0.0, end_time_s: float = 0.3) -> dict[str, object]:
    return {
        "schema_version": "near_miss_encounter.v1",
        "status": "complete",
        "evidence_status": "diagnostic-only",
        "claim_boundary": "diagnostic encounter grouping only",
        "profile": {
            "schema_version": "NearMissEncounterProfile.v1",
            "profile_id": "unit-test-profile",
            "qualification_rule": "distance_or_ttc",
            "continuity_gap_s": 0.2,
            "distance_threshold_m": 1.0,
            "ttc_threshold_s": 2.0,
            "units": {"distance": "m", "time": "s", "speed": "m/s"},
        },
        "units": {
            "time": "s",
            "distance": "m",
            "speed": "m/s",
            "encounter_duration": "s",
            "valid_exposure_duration": "s",
        },
        "denominator": {
            "sample_unit": "trace_sample",
            "encounter_unit": "encounter",
            "input_sample_count": 4,
            "actor_count": 1,
            "qualifying_sample_count": 2,
            "encounter_count": 1,
            "valid_exposure_duration_s": end_time_s - start_time_s,
        },
        "encounters": [
            {
                "schema_version": "near_miss_encounter.v1",
                "encounter_id": "ped-a:encounter-0001",
                "actor_id": "ped-a",
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "duration_s": end_time_s - start_time_s,
                "minimum_clearance_m": 0.5,
                "minimum_ttc_s": 0.4,
                "maximum_closing_speed_mps": 1.0,
                "minimum_pet_s": None,
                "sample_count": 2,
                "valid_exposure_duration_s": end_time_s - start_time_s,
                "termination_reason": "trace_end",
                "contact_terminated": False,
                "contact_status": "not-observed",
                "contact_time_s": None,
                "unavailable_fields": [],
                "evidence_status": "diagnostic-only",
            }
        ],
        "exclusions": [],
        "missingness": {"field_counts": {}, "sample_exclusion_counts": {}},
        "provenance": {
            "source_commit": "0" * 40,
            "release_id": "unit-test",
            "bundle_id": "unit-test",
            "input_checksums": {"trace": "0" * 64},
            "input_checksum_digest": _input_checksum_digest({"trace": "0" * 64}),
        },
    }


def _route_checksum(start: tuple[float, float], end: tuple[float, float]) -> str:
    return _json_digest({"type": "line_segment", "start": list(start), "end": list(end)})


def _registered_route(entry_id: str) -> RouteSpec:
    return load_registered_route_spec(GEOMETRY_REGISTRY_FIXTURE_PATH, entry_id)


def _registered_conflict_zone(entry_id: str) -> ConflictZoneSpec:
    return load_registered_conflict_zone_spec(GEOMETRY_REGISTRY_FIXTURE_PATH, entry_id)


def _zone_checksum(center: tuple[float, float], radius_m: float) -> str:
    return _json_digest({"type": "circle", "center": list(center), "radius_m": radius_m})


def _input_checksum_digest(checksums: dict[str, str]) -> str:
    return _json_digest(checksums)


def _json_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
