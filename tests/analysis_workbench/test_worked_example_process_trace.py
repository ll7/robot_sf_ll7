"""Contract tests for ``worked_example_process_trace.v1`` diagnostics."""

from __future__ import annotations

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
        route=RouteSpec("r-main", (0.0, 0.0), (10.0, 0.0), "route-fixture.v1"),
        conflict_zone=ConflictZoneSpec("door", (1.0, 0.0), 0.25, "zone-fixture.v1"),
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

    payload = build_worked_example_process_trace_from_export(left, pair_trace=right)
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

    payload = build_worked_example_process_trace_from_export(left, pair_trace=right)
    pair = payload["pair_compatibility"]

    assert pair["initial_state_equivalence"]["equivalent"] is True
    assert pair["shared_prefix"]["shared_prefix"] is True
    assert "minimum_clearance" in pair["valid_common_event_anchors"]
    assert pair["duration_normalization"] == {"applied": False}


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
        conflict_zone=ConflictZoneSpec("bad-zone", (0.0, 0.0), -1.0, "bad-zone.v1"),
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


def test_pair_compatibility_gates_actor_start_mismatch() -> None:
    """Known-bad doorway-style actor/spawn mismatch should be incompatible."""

    left = simulation_trace_export_from_dict(_trace_payload(trace_id="doorway-seed-113"))
    right = simulation_trace_export_from_dict(
        _trace_payload(trace_id="doorway-seed-114", actor_start_offset=0.2, seed=114)
    )

    payload = build_worked_example_process_trace_from_export(left, pair_trace=right)
    pair = payload["pair_compatibility"]

    assert pair["status"] == "incompatible"
    assert pair["initial_state_equivalence"]["equivalent"] is False
    assert pair["divergence_interpretation"]["reason"] == "scenario_or_initial_state_incompatible"


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


def _trace_payload(  # noqa: PLR0913
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
        peds = [
            {
                "id": "ped-a",
                "position": [
                    (0.2 if proxy_overlap else 1.0) + actor_start_offset + 0.1 * step,
                    0.0,
                ],
                "velocity": ped_vel,
                "radius": 0.25,
            }
        ]
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
            "profile_version": "near_miss_encounter.v1",
            "start_step": 0,
            "end_step": 3,
            "start_time_s": 0.0,
            "end_time_s": 0.3,
            "available_duration_s": 0.3,
            "min_clearance_m": -0.3 if proxy_overlap else 0.5,
            "min_ttc_s": 0.4,
            "min_pet_s": None,
            "contact": False,
        }
        frames.append(
            {
                "step": step,
                "time_s": step * 0.1,
                "robot": {
                    "position": position,
                    "heading": 0.0,
                    "velocity": robot_vel,
                    "radius": 0.25,
                },
                "pedestrians": peds,
                "planner": {
                    "selected_action": actions[step],
                    "encounter": encounter,
                    "event": "step",
                },
            }
        )
    return {
        "schema_version": "simulation_trace_export.v1",
        "trace_id": trace_id,
        "source": {
            "scenario_id": "narrow_doorway",
            "seed": seed,
            "planner_id": "ppo",
            "episode_id": f"{trace_id}-episode",
            "generated_by": "unit-test fixture",
        },
        "evidence_boundary": "analysis_workbench_only",
        "coordinate_frame": coordinate_frame,
        "units": {"position": "m", "heading": "rad", "time": "s", "velocity": "m/s"},
        "frames": frames,
    }
