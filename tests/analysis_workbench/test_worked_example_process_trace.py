"""Contract tests for ``worked_example_process_trace.v1`` diagnostics."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
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


def test_terminal_event_requires_typed_provenance() -> None:
    """Terminal fallback must come from typed terminal evidence, not the last frame."""

    no_terminal = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload())
    )
    no_terminal_event = {event["event_type"]: event for event in no_terminal["event_anchors"]}[
        "terminal_event"
    ]
    assert no_terminal_event["status"] == "unavailable"

    typed_terminal = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(terminal_mode="ledger_typed"))
    )
    terminal_event = {event["event_type"]: event for event in typed_terminal["event_anchors"]}[
        "terminal_event"
    ]
    assert terminal_event["status"] == "available"
    assert terminal_event["time_s"] == pytest.approx(0.3)
    assert terminal_event["terminal_reason"] == "trace_end"

    outside = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(terminal_mode="ledger_late")),
        encounter_report=_encounter_report(start_time_s=0.0, end_time_s=0.2),
        encounter_report_input_checksum="0" * 64,
    )
    outside_terminal = {event["event_type"]: event for event in outside["event_anchors"]}[
        "terminal_event"
    ]
    assert outside_terminal["status"] == "unavailable"
    assert outside_terminal["reason"] == "terminal_time_outside_encounter_interval"


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

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(terminal_mode="ledger_typed"))
    )
    payload["frames"][0]["commands"] = {"status": "available", "commanded": "banana"}
    with pytest.raises(Exception, match="/frames/0/commands"):
        validate_worked_example_process_trace(payload)

    payload = build_worked_example_process_trace_from_export(
        simulation_trace_export_from_dict(_trace_payload(terminal_mode="ledger_typed"))
    )
    anchors = payload["event_anchor_hierarchy"]["available_anchors"]
    assert len(anchors) >= 2
    payload["event_anchor_hierarchy"]["selected_anchor"] = anchors[-1]
    payload["event_anchor_hierarchy"]["anchor_time_s"] = anchors[-1]["time_s"]
    with pytest.raises(Exception, match="/event_anchor_hierarchy/selected_anchor"):
        validate_worked_example_process_trace(payload)


def _trace_payload(  # noqa: C901, PLR0913
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
    collision_mode: str | None = None,
    terminal_mode: str | None = None,
    missing_actor_radius_step: int | None = None,
    nonfinite_heading_step: int | None = None,
    stall_pattern: list[float] | None = None,
    nan_command_step: int | None = None,
    missing_velocity_step: int | None = None,
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
        peds = [actor]
        if actor_switch:
            peds.append(
                {
                    "id": "ped-b",
                    "position": [3.0 if step < 2 else 0.45, 0.0],
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
        if include_run_config:
            planner["run_config"] = {
                "map_id": "fixture-map",
                "horizon": 4,
                "config_digest": config_digest,
            }
        if collision_mode == "outcome_boolean" and step == 1:
            planner["outcome"] = {"collision_event": True}
        if collision_mode == "ledger_typed" and step == 1:
            planner["event_ledger"] = {
                "schema_version": "EpisodeEventLedger.v2",
                "collision_events": [
                    {
                        "collision_time": 0.15,
                        "actor_id": "robot",
                        "collision_partner_id": "ped-a",
                        "collision_partner_type": "pedestrian",
                    }
                ],
            }
        if collision_mode == "ledger_collision_late" and step == 3:
            planner["event_ledger"] = {
                "schema_version": "EpisodeEventLedger.v2",
                "collision_events": [
                    {
                        "collision_time": 0.35,
                        "actor_id": "robot",
                        "collision_partner_id": "ped-a",
                        "collision_partner_type": "pedestrian",
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
        if terminal_mode == "ledger_typed" and step == 3:
            planner["event_ledger"] = {
                **planner.get("event_ledger", {"schema_version": "EpisodeEventLedger.v2"}),
                "terminal_events": [{"terminal_time": 0.3, "terminal_reason": "trace_end"}],
            }
        if terminal_mode == "ledger_late" and step == 3:
            planner["event_ledger"] = {
                **planner.get("event_ledger", {"schema_version": "EpisodeEventLedger.v2"}),
                "terminal_events": [{"terminal_time": 0.35, "terminal_reason": "trace_end"}],
            }
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
