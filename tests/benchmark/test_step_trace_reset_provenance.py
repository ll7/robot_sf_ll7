"""Contract tests for reset-time and per-step trace provenance (#9262)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from robot_sf.benchmark.map_runner.map_runner_episode import (
    _build_reset_provenance,
    _surface_clearances_m,
    _trace_surface_radii_m,
)
from robot_sf.benchmark.map_runner.map_runner_trace import _trace_pedestrians


def _reset_kwargs(**overrides: object) -> dict[str, object]:
    """Return a colliding reset fixture; callers override individual inputs."""
    base: dict[str, object] = {
        "initial_robot_pos": np.array([0.0, 0.0]),
        "initial_robot_heading": 0.0,
        "initial_robot_velocity": np.array([1.0, 0.0]),
        "initial_ped_positions": np.array([[0.5, 0.0], [5.0, 5.0]]),
        "initial_ped_velocities": None,
        "initial_ped_headings": np.array([0.0, 1.0]),
        "trace_actor_ids": ["p0", "p1"],
        "robot_radius": 0.5,
        "ped_radius": 0.4,
        "scenario": {},
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


def test_reset_provenance_records_collision_at_reset() -> None:
    """A sub-radii reset separation must report negative clearance and collision."""
    reset = _build_reset_provenance(**_reset_kwargs())

    assert reset["min_surface_clearance_m"] == -0.4
    assert reset["collision_at_reset"] is True
    assert reset["pedestrians"][0]["surface_clearance_m"] == -0.4
    assert reset["pedestrians"][0]["heading"] == 0.0
    assert reset["pedestrians"][0]["actor_id"] == "p0"
    assert reset["robot"]["position"] == [0.0, 0.0]


def test_reset_provenance_clean_separation_reports_no_collision() -> None:
    """A clear reset must report positive clearance and no collision."""
    reset = _build_reset_provenance(**_reset_kwargs(initial_ped_positions=np.array([[5.0, 5.0]])))

    assert reset["collision_at_reset"] is False
    assert reset["min_surface_clearance_m"] is not None
    assert reset["min_surface_clearance_m"] > 0.0


def test_reset_provenance_without_pedestrians_is_explicit() -> None:
    """An empty reset population reports no collision with an explicit null minimum."""
    reset = _build_reset_provenance(
        **_reset_kwargs(
            initial_ped_positions=np.empty((0, 2)),
            initial_ped_headings=np.empty((0,)),
            trace_actor_ids=[],
        )
    )

    assert reset["pedestrians"] == []
    assert reset["min_surface_clearance_m"] is None
    assert reset["collision_at_reset"] is False


def test_reset_provenance_marks_unavailable_edges_explicitly() -> None:
    """Missing headings and unretained spawn/route edges must stay explicit."""
    reset = _build_reset_provenance(
        **_reset_kwargs(initial_ped_headings=None, scenario={"unrelated": 1})
    )

    assert reset["pedestrians"][0]["heading"] is None
    assert reset["heading_source"].startswith("unavailable:")
    assert reset["spawn"]["status"] == "unavailable"
    assert reset["spawn"]["reason"] == "spawn_sampler_decision_not_retained"
    assert reset["spawn"]["owner"].endswith("populate_simulation")
    assert reset["routes"]["status"] == "unavailable"
    assert reset["routes"]["reason"] == "route_objects_not_retained_per_episode"
    # The whole block must remain digest-addressable for diagnostic bundles.
    json.dumps(reset)


def test_reset_provenance_echoes_scenario_spawn_keys_verbatim() -> None:
    """Scenario-carried spawn/route keys are echoed without interpretation."""
    reset = _build_reset_provenance(
        **_reset_kwargs(scenario={"spawn_config": "dense", "routes": 3, "other": [1]})
    )

    assert reset["spawn"]["scenario_echo"] == {"spawn_config": "dense", "routes": 3}


def test_per_step_frames_carry_heading_and_clearance() -> None:
    """Pedestrian frames attach simulator headings and surface clearance verbatim."""
    frames = _trace_pedestrians(
        np.array([[1.0, 0.0], [0.0, 2.0]]),
        None,
        0.1,
        None,
        None,
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        ["a", "b"],
        headings=np.array([0.5, float("nan")]),
        surface_clearances=np.array([0.1, 1.1]),
    )

    assert frames[0]["heading"] == 0.5
    assert frames[0]["surface_clearance_m"] == 0.1
    assert frames[1]["heading"] is None
    assert frames[1]["surface_clearance_m"] == 1.1


def test_surface_clearance_uses_canonical_radii() -> None:
    """Clearance is center distance minus robot and pedestrian radii."""
    clearance = _surface_clearances_m(
        np.array([0.0, 0.0]),
        np.array([[3.0, 4.0], [0.5, 0.0]]),
        robot_radius=0.5,
        ped_radius=0.4,
    )

    assert clearance[0] == 4.1
    assert clearance[1] == -0.4


def test_trace_radii_fall_back_to_canonical_defaults() -> None:
    """Missing radius configuration falls back to the documented defaults."""
    robot_radius, ped_radius = _trace_surface_radii_m(SimpleNamespace(sim_config=SimpleNamespace()))

    assert (robot_radius, ped_radius) == (1.0, 0.4)
