"""Contract tests for ``worked_example_process_trace.v1`` diagnostics."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.interaction_coordinates import (
    WORKED_EXAMPLE_PROCESS_TRACE_SCHEMA_VERSION,
    ConflictZoneSpec,
    RouteSpec,
    build_worked_example_process_trace_from_export,
    validate_worked_example_process_trace,
)
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


def test_process_trace_coordinate_frames_explicit_available_and_unavailable() -> None:
    """All four frame families should expose explicit status and reasons."""

    trace = simulation_trace_export_from_dict(_trace_payload())

    unavailable = build_worked_example_process_trace_from_export(trace)
    assert unavailable["coordinate_frames"]["world"]["status"] == "available"
    assert unavailable["coordinate_frames"]["route"] == {
        "status": "unavailable",
        "reason": "registered_route_unavailable",
    }
    assert unavailable["coordinate_frames"]["conflict"] == {
        "status": "unavailable",
        "reason": "registered_conflict_zone_unavailable",
    }
    assert unavailable["coordinate_frames"]["relative_interaction"]["status"] == "available"

    available = build_worked_example_process_trace_from_export(
        trace,
        route=RouteSpec(
            "r-main",
            (0.0, 0.0),
            (10.0, 0.0),
            "route-fixture.v1",
            _route_checksum((0.0, 0.0), (10.0, 0.0)),
        ),
        conflict_zone=ConflictZoneSpec(
            "door",
            (1.0, 0.0),
            0.25,
            "zone-fixture.v1",
            _zone_checksum((1.0, 0.0), 0.25),
        ),
    )

    validate_worked_example_process_trace(available)
    assert available["schema_version"] == WORKED_EXAMPLE_PROCESS_TRACE_SCHEMA_VERSION
    assert available["coordinate_frames"]["route"]["status"] == "available"
    assert available["coordinate_frames"]["conflict"]["status"] == "available"
    assert available["frames"][1]["route"]["s_m"] == pytest.approx(0.2)
    assert available["frames"][0]["relative_interaction"]["relative_longitudinal_m"] == (
        pytest.approx(1.0)
    )


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
        "--route-id",
        "fixture-route",
        "--route-provenance-id",
        "fixture-route.v1",
        "--route-registry-checksum",
        _route_checksum((0.0, 0.0), (2.0, 0.0)),
        "--route-start",
        "0",
        "0",
        "--route-end",
        "2",
        "0",
        "--conflict-zone-id",
        "fixture-zone",
        "--conflict-provenance-id",
        "fixture-zone.v1",
        "--conflict-registry-checksum",
        _zone_checksum((1.0, 0.0), 0.25),
        "--conflict-center",
        "1",
        "0",
        "--conflict-radius-m",
        "0.25",
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
    validate_worked_example_process_trace(json.loads(first.read_text(encoding="utf-8")))


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
    assert [frame["relative_interaction"]["status"] for frame in payload["frames"]] == [
        "unavailable",
        "available",
        "available",
        "unavailable",
    ]
    assert payload["diagnostics"]["coverage"]["relative_interaction"]["status"] == "complete"
    assert payload["diagnostics"]["coverage"]["relative_interaction"]["frame_count"] == 2
    assert payload["diagnostics"]["threshold_exposure"]["duration_s"] == pytest.approx(0.0)


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

    non_focal_payload = _trace_payload(collision_mode="ledger_ped_b")
    non_focal = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(non_focal_payload)
    )
    non_focal_event = {event["event_type"]: event for event in non_focal["event_anchors"]}[
        "exact_collision_event"
    ]
    assert non_focal_event["status"] == "unavailable"
    assert non_focal_event["reason"] == "collision_not_bound_to_focal_encounter"

    static_payload = _trace_payload(collision_mode="static_geometry_collision")
    static = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(static_payload)
    )
    static_event = {event["event_type"]: event for event in static["event_anchors"]}[
        "exact_collision_event"
    ]
    assert static_event["status"] == "unavailable"
    assert static_event["reason"] == "collision_not_bound_to_focal_encounter"

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
    assert outside_event["reason"] == "collision_time_outside_encounter_interval"

    zero_sample_interval = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(collision_mode="ledger_typed")),
        encounter_report=_encounter_report(start_time_s=10.0, end_time_s=11.0),
        encounter_report_input_checksum="0" * 64,
    )
    zero_events = {event["event_type"]: event for event in zero_sample_interval["event_anchors"]}
    assert zero_sample_interval["diagnostics"]["coverage"]["frame_count"] == 0
    assert zero_events["exact_collision_event"]["status"] == "unavailable"
    assert (
        zero_events["exact_collision_event"]["reason"]
        == "collision_time_outside_encounter_interval"
    )
    assert zero_sample_interval["event_anchor_hierarchy"]["status"] == "unavailable"


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


def test_stall_duration_requires_qualifying_contiguous_run() -> None:
    """Separated one-frame stalls should not accumulate into sustained stall duration."""

    payload = _trace_payload(stall_pattern=[0.0, 1.0, 0.0, 1.0])
    trace = simulation_trace_export_from_dict(payload)

    result = build_worked_example_process_trace_from_export(
        trace,
        route=RouteSpec(
            "r-main",
            (0.0, 0.0),
            (10.0, 0.0),
            "route-fixture.v1",
            _route_checksum((0.0, 0.0), (10.0, 0.0)),
        ),
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
        route=RouteSpec(
            "r-main",
            (0.0, 0.0),
            (10.0, 0.0),
            "route-fixture.v1",
            _route_checksum((0.0, 0.0), (10.0, 0.0)),
        ),
    )
    assert missing_route_velocity["diagnostics"]["reversal_counts"]["status"] == "unavailable"
    assert missing_route_velocity["diagnostics"]["reversal_counts"]["reason"] == (
        "missing_robot_velocity"
    )


def test_focal_collision_scan_uses_declared_interval_not_sample_span() -> None:
    """Later focal collisions inside declared bounds should beat earlier unrelated collisions."""

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
    assert collision["collision_partner_id"] == "ped-a"
    assert collision["time_s"] == pytest.approx(0.15)
    assert collision["step"] == 1


def test_route_frames_project_only_selected_focal_actor() -> None:
    """Public route frames should carry selected focal projection, not contextual actors."""

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(actor_switch=True)),
        route=RouteSpec(
            "r-main",
            (0.0, 0.0),
            (10.0, 0.0),
            "route-fixture.v1",
            _route_checksum((0.0, 0.0), (10.0, 0.0)),
        ),
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
        route=RouteSpec(
            "r-main",
            (0.0, 0.0),
            (10.0, 0.0),
            "route-fixture.v1",
            _route_checksum((0.0, 0.0), (10.0, 0.0)),
        ),
        conflict_zone=ConflictZoneSpec(
            "door",
            (1.0, 0.0),
            0.25,
            "zone-fixture.v1",
            _zone_checksum((1.0, 0.0), 0.25),
        ),
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
        route=RouteSpec(
            "route-b",
            (0.0, 0.0),
            (10.0, 0.0),
            "route-b.v1",
            _route_checksum((0.0, 0.0), (10.0, 0.0)),
        ),
        conflict_zone=ConflictZoneSpec(
            "zone-b",
            (1.0, 0.0),
            0.25,
            "zone-b.v1",
            _zone_checksum((1.0, 0.0), 0.25),
        ),
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


def test_source_and_pair_hashes_recompute_from_canonical_content_receipts() -> None:
    """Coherent hash rewrites should fail against embedded canonical trace contracts."""

    left_payload = _trace_payload(trace_id="pair-left", planner_id="planner-a", seed=7)
    right_payload = _trace_payload(trace_id="pair-right", planner_id="planner-b", seed=7)
    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(left_payload),
        pair_trace=simulation_trace_export_from_dict(right_payload),
        pair_comparison_grain="matched_planner_pair",
    )

    assert payload["source_trace"]["content_contract"] == _canonical_trace_contract(left_payload)
    assert payload["pair_compatibility"]["right_source_trace"]["content_contract"] == (
        _canonical_trace_contract(right_payload)
    )

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
        route=RouteSpec(
            "r-main",
            (0.0, 0.0),
            (10.0, 0.0),
            "route-fixture.v1",
            _route_checksum((0.0, 0.0), (10.0, 0.0)),
        ),
        conflict_zone=ConflictZoneSpec(
            "door",
            (1.0, 0.0),
            0.25,
            "zone-fixture.v1",
            _zone_checksum((1.0, 0.0), 0.25),
        ),
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


def _set_path(payload: dict[str, object], path: list[object], value: object) -> None:
    cursor: object = payload
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    last = path[-1]
    if isinstance(last, int):
        cursor[last] = value  # type: ignore[index]
    else:
        cursor[last] = value  # type: ignore[index]


def _canonical_trace_checksum(payload: dict[str, object]) -> str:
    contract = _canonical_trace_contract(payload)
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


def _zone_checksum(center: tuple[float, float], radius_m: float) -> str:
    return _json_digest({"type": "circle", "center": list(center), "radius_m": radius_m})


def _input_checksum_digest(checksums: dict[str, str]) -> str:
    return _json_digest(checksums)


def _json_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
