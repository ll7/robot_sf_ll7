"""Focused regressions for the shared-helper delegation in PR #9233.

Every changed ``point_distance`` call site in
``robot_sf/analysis_workbench/event_alignment.py`` and
``robot_sf/analysis_workbench/interaction_coordinates.py`` is exercised here so
the exact-head changed-line coverage gate observes the new lines.
"""

from __future__ import annotations

from robot_sf.analysis_workbench import event_alignment, interaction_coordinates
from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExport,
    SimulationTraceFrame,
    SimulationTraceSource,
)


def _robot(position: tuple[float, float], velocity: tuple[float, float]) -> dict:
    return {"position": list(position), "velocity": list(velocity), "heading": 0.0}


def _pedestrian(ped_id: str, position: tuple[float, float], velocity: tuple[float, float]) -> dict:
    return {
        "id": ped_id,
        "position": list(position),
        "velocity": list(velocity),
    }


def _trace(
    robot_pos: tuple[float, float],
    robot_vel: tuple[float, float],
    pedestrians: list[dict],
) -> SimulationTraceExport:
    return SimulationTraceExport(
        schema_version="simulation_trace_export.v1",
        trace_id="trace",
        source=SimulationTraceSource(
            scenario_id="s",
            seed=1,
            planner_id="p",
            episode_id="e",
            generated_by="test",
        ),
        evidence_boundary="diagnostic",
        coordinate_frame="world",
        units={},
        frames=[
            SimulationTraceFrame(
                step=0,
                time_s=0.0,
                robot=_robot(robot_pos, robot_vel),
                pedestrians=pedestrians,
                planner={},
            )
        ],
    )


def test_initial_equivalence_reports_position_and_velocity_deltas() -> None:
    left = _trace((0.0, 0.0), (1.0, 0.0), [_pedestrian("a", (2.0, 0.0), (0.0, 1.0))])
    right = _trace((3.0, 4.0), (1.0, 0.0), [_pedestrian("a", (5.0, 0.0), (0.0, 1.0))])
    result = event_alignment._initial_equivalence(
        left, right, position_tolerance_m=100.0, heading_tolerance_rad=3.2
    )
    assert result["robot_position_delta_m"] == 5.0
    assert result["robot_velocity_delta_mps"] == 0.0
    assert result["actor_position_delta_m"]["a"] == 3.0
    assert result["actor_velocity_delta_mps"]["a"] == 0.0


def test_full_state_equal_rejects_diverged_robot_position() -> None:
    left = _trace((0.0, 0.0), (1.0, 0.0), [])
    right = _trace((3.0, 4.0), (1.0, 0.0), [])
    equal, reason = event_alignment._full_state_equal(
        left.frames[0],
        right.frames[0],
        position_tolerance_m=1.0,
        heading_tolerance_rad=3.2,
    )
    assert (equal, reason) == (False, "robot_position_diverged")


def test_full_state_equal_rejects_diverged_actor_position() -> None:
    left = _trace((0.0, 0.0), (1.0, 0.0), [_pedestrian("a", (2.0, 0.0), (0.0, 1.0))])
    right = _trace((0.0, 0.0), (1.0, 0.0), [_pedestrian("a", (9.0, 0.0), (0.0, 1.0))])
    equal, reason = event_alignment._full_state_equal(
        left.frames[0],
        right.frames[0],
        position_tolerance_m=1.0,
        heading_tolerance_rad=3.2,
    )
    assert (equal, reason) == (False, "actor_position_diverged")


def test_global_minimum_actor_selects_nearest() -> None:
    frame = {
        "source_coordinates": {
            "robot": {"position": [0.0, 0.0]},
            "contextual_actors": [
                {"actor_id": "far", "position": [3.0, 4.0]},
                {"actor_id": "near", "position": [1.0, 0.0]},
            ],
        }
    }
    result = interaction_coordinates._global_minimum_actor_from_source(frame)
    assert result["status"] == "available"
    assert result["actor_id"] == "near"
    assert result["center_distance_m"] == 1.0


def test_route_geometry_rejects_degenerate_segment() -> None:
    geometry = {"type": "line_segment", "start": [0.0, 0.0], "end": [0.0, 0.0]}
    assert (
        interaction_coordinates._route_geometry_unavailable_reason(geometry)
        == "registered_route_degenerate"
    )


def test_nearest_source_actor_selects_nearest() -> None:
    frame = {
        "source_coordinates": {
            "robot": {"position": [0.0, 0.0]},
            "contextual_actors": [
                {"actor_id": "far", "position": [3.0, 4.0]},
                {"actor_id": "near", "position": [0.0, 2.0]},
            ],
        }
    }
    (nearest,) = [
        c for c in [interaction_coordinates._nearest_source_actor(frame)] if c is not None
    ]
    assert nearest["actor_id"] == "near"
    assert nearest["center_distance_m"] == 2.0


def test_nearest_actor_selects_nearest_pedestrian() -> None:
    frame = SimulationTraceFrame(
        step=0,
        time_s=0.0,
        robot=_robot((0.0, 0.0), (0.0, 0.0)),
        pedestrians=[
            _pedestrian("far", (3.0, 4.0), (0.0, 0.0)),
            _pedestrian("near", (1.0, 0.0), (0.0, 0.0)),
        ],
        planner={},
    )
    result = interaction_coordinates._nearest_actor(frame, robot_pos=(0.0, 0.0))
    assert result["status"] == "available"
    assert result["actor_id"] == "near"
    assert result["center_distance_m"] == 1.0
