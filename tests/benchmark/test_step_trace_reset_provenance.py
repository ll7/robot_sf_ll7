"""Contract tests for reset-time and per-step trace provenance (#9262)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from robot_sf.benchmark.map_runner.map_runner_episode import (
    _build_reset_provenance,
    _finalize_trace_metadata,
    _read_sampler_capture,
    _read_simulator_ped_headings,
    _step_build_simulation_trace,
    _StepLoopState,
    _surface_clearances_m,
    _trace_surface_radii_m,
)
from robot_sf.benchmark.map_runner.map_runner_trace import (
    _optional_trace_float,
    _trace_pedestrians,
)
from robot_sf.ped_npc.spawn_capture import SpawnSamplerCapture


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


def test_trace_radii_rejects_malformed_values() -> None:
    """Non-numeric or negative radii fall back instead of poisoning clearance math."""
    config = SimpleNamespace(
        robot_config=SimpleNamespace(radius="wide"),
        sim_config=SimpleNamespace(time_per_step_in_secs=0.1, ped_radius=-2.0),
    )

    assert _trace_surface_radii_m(config) == (1.0, 0.4)


def test_read_simulator_ped_headings_round_trips_rows() -> None:
    """Simulator-tracked headings are recorded verbatim when well-formed."""
    simulator = SimpleNamespace(
        _initial_ped_headings=[0.25, -1.0],
        ped_headings=[0.5, 0.0],
    )

    initial = _read_simulator_ped_headings(simulator, 2, initial=True)
    current = _read_simulator_ped_headings(simulator, 2, initial=False)

    assert initial is not None and current is not None
    assert list(initial) == [0.25, -1.0]
    assert list(current) == [0.5, 0.0]


def test_read_simulator_ped_headings_fails_closed() -> None:
    """Missing, short, or non-finite heading state stays unavailable, never guessed."""
    assert _read_simulator_ped_headings(SimpleNamespace(), 1, initial=True) is None
    assert (
        _read_simulator_ped_headings(SimpleNamespace(_initial_ped_headings=[0.0]), 2, initial=True)
        is None
    )
    assert (
        _read_simulator_ped_headings(SimpleNamespace(ped_headings=[float("nan")]), 1, initial=False)
        is None
    )


def test_optional_trace_float_rejects_non_numeric() -> None:
    """Non-numeric frame values become explicit nulls, never exceptions."""
    assert _optional_trace_float("far") is None
    assert _optional_trace_float(None) is None
    assert _optional_trace_float(float("inf")) is None
    assert _optional_trace_float(0.5) == 0.5


def test_reset_provenance_skips_nonfinite_ped_rows() -> None:
    """NaN placeholder rows never leak into world geometry or clearance math."""
    reset = _build_reset_provenance(
        **_reset_kwargs(
            initial_ped_positions=np.array([[float("nan"), 0.0], [5.0, 5.0]]),
            trace_actor_ids=["ghost", "p1"],
        )
    )

    assert [frame["id"] for frame in reset["pedestrians"]] == ["p1"]
    assert reset["collision_at_reset"] is False


def test_reset_provenance_handles_malformed_robot_pose() -> None:
    """An unparseable reset pose yields explicit nulls, never a guessed ledger."""
    reset = _build_reset_provenance(**_reset_kwargs(initial_robot_pos="nowhere"))

    assert reset["robot"]["position"] is None
    assert reset["min_surface_clearance_m"] is None
    assert reset["collision_at_reset"] is None


def test_reset_provenance_without_scenario_dict() -> None:
    """A missing scenario record keeps an empty echo instead of failing."""
    reset = _build_reset_provenance(**_reset_kwargs(scenario=None))

    assert reset["spawn"]["scenario_echo"] == {}
    assert reset["collision_at_reset"] is True


def test_reset_provenance_records_velocities_when_available() -> None:
    """Supplied reset velocities are recorded verbatim alongside poses."""
    reset = _build_reset_provenance(
        **_reset_kwargs(
            initial_ped_positions=np.array([[5.0, 5.0]]),
            initial_ped_velocities=np.array([[0.0, 1.0]]),
            initial_ped_headings=np.array([0.0]),
            trace_actor_ids=["p1"],
        )
    )

    assert reset["pedestrians"][0]["velocity"] == [0.0, 1.0]
    assert reset["collision_at_reset"] is False


def _step_harness() -> tuple[_StepLoopState, SimpleNamespace, SimpleNamespace]:
    """Build a synthetic step-loop harness for one annotated pedestrian."""
    state = _StepLoopState(obs={})
    state.previous_trace_robot_pos = np.zeros(2, dtype=float)
    state.previous_trace_ped_pos = None
    slc = SimpleNamespace(
        record_simulation_step_trace=True,
        record_forces=False,
        config=SimpleNamespace(sim_config=SimpleNamespace(time_per_step_in_secs=0.1)),
        single_pedestrian_intent_metadata=None,
        single_pedestrian_vru_metadata=None,
    )
    sim = SimpleNamespace(
        robot_pos=np.array([0.0, 0.0], dtype=float),
        peds=np.array([[1.0, 0.0]], dtype=float),
        forces_arr=None,
        heading=0.0,
        ped_headings=np.array([0.5], dtype=float),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={},
        selected_action_payload={},
        applied_environment_action_payload={},
        actuation_step=None,
        step_visible=None,
        step_confidence=None,
        step_visibility_status="not_available",
        step_visibility_reason=None,
    )
    return state, slc, sim


def test_step_builder_annotates_heading_and_clearance() -> None:
    """One step records the simulator heading and canonical surface clearance."""
    state, slc, sim = _step_harness()

    _step_build_simulation_trace(state, slc, step_idx=0, sim=sim)

    frame = state.simulation_step_trace[0]["pedestrians"][0]
    assert frame["heading"] == 0.5
    assert frame["surface_clearance_m"] == -0.4


def test_finalize_attaches_reset_block_with_schema_version() -> None:
    """Finalization keeps the v1 schema key and attaches the reset ledger."""
    algo_meta: dict[str, object] = {}
    config = SimpleNamespace(sim_config=SimpleNamespace(time_per_step_in_secs=0.1))

    _finalize_trace_metadata(
        algo_meta,  # type: ignore[arg-type]
        config=config,
        initial_goal_distance=10.0,
        planner_decision_trace=[],
        simulation_step_trace=[],
        record_planner_decision_trace=False,
        record_simulation_step_trace=True,
        record_forces=False,
        scenario={},
        ped_pos_arr=np.empty((0, 2)),
        ped_forces_arr=np.empty((0, 2)),
        robot_pos_arr=np.empty((0, 2)),
        robot_config=None,
        initial_robot_pos=np.array([0.0, 0.0]),
        initial_robot_heading=0.0,
        initial_ped_positions=np.array([[1.0, 0.0]]),
        initial_robot_velocity=None,
        initial_ped_velocities=None,
        initial_ped_headings=np.array([0.0]),
        trace_actor_ids=["p0"],
        horizon_val=400,
        termination_reason="max_steps",
        safety_events=[],
    )

    trace = algo_meta["simulation_step_trace"]
    assert trace["schema_version"] == "simulation-step-trace.v1"
    assert trace["reset"]["collision_at_reset"] is True
    assert trace["reset"]["min_surface_clearance_m"] == -0.4


def test_reset_provenance_rejects_short_velocity_rows() -> None:
    """A short velocity row must not reach positional indexing (issue #9268)."""
    reset = _build_reset_provenance(
        **_reset_kwargs(
            initial_ped_positions=np.array([[5.0, 5.0]]),
            initial_ped_velocities=np.array([[1.0]]),
            initial_ped_headings=np.array([0.0]),
            trace_actor_ids=["p1"],
        )
    )

    assert reset["pedestrians"][0]["velocity"] is None
    assert reset["pedestrians"][0]["heading"] == 0.0


def test_reset_provenance_rejects_nonscalar_heading_entries() -> None:
    """A non-scalar heading entry must not raise on float conversion (issue #9268)."""
    reset = _build_reset_provenance(
        **_reset_kwargs(
            initial_ped_positions=np.array([[5.0, 5.0]]),
            initial_ped_velocities=np.array([[0.0, 1.0]]),
            initial_ped_headings=np.array([[0.0, 1.0]]),
            trace_actor_ids=["p1"],
        )
    )

    assert reset["pedestrians"][0]["velocity"] == [0.0, 1.0]
    assert reset["pedestrians"][0]["heading"] is None


def test_reset_provenance_rejects_nonfinite_velocity_rows() -> None:
    """An infinite velocity row must record null instead of leaking (issue #9268)."""
    reset = _build_reset_provenance(
        **_reset_kwargs(
            initial_ped_positions=np.array([[5.0, 5.0]]),
            initial_ped_velocities=np.array([[float("inf"), 0.0]]),
            trace_actor_ids=["p1"],
        )
    )

    assert reset["pedestrians"][0]["velocity"] is None


def test_reset_scenario_echo_nulls_nonfinite_floats() -> None:
    """Non-finite scenario echoes become null for strict-JSON safety (issue #9268)."""
    reset = _build_reset_provenance(
        **_reset_kwargs(
            scenario={"spawn_config": "dense", "routes": float("inf"), "waypoints": float("nan")}
        )
    )

    assert reset["spawn"]["scenario_echo"] == {
        "spawn_config": "dense",
        "routes": None,
        "waypoints": None,
    }
    json.dumps(reset, allow_nan=False)


def test_reset_provenance_block_is_strict_json_serializable() -> None:
    """The whole reset block must survive allow_nan=False (issue #9268)."""
    reset = _build_reset_provenance(
        **_reset_kwargs(
            initial_ped_positions=np.array([[5.0, 5.0]]),
            initial_ped_velocities=np.array([[0.0, 1.0]]),
            initial_ped_headings=np.array([0.0]),
            trace_actor_ids=["p1"],
            scenario={"spawn_config": "dense", "horizon": 600.0},
        )
    )

    json.dumps(reset, allow_nan=False)


def test_reset_provenance_rejects_ragged_velocity_rows() -> None:
    """Ragged velocity rows must not reach asarray crashes (issue #9268)."""
    velocities = np.empty(1, dtype=object)
    velocities[0] = np.array([0.0])
    reset = _build_reset_provenance(
        **_reset_kwargs(
            initial_ped_positions=np.array([[5.0, 5.0]]),
            initial_ped_velocities=velocities,
            trace_actor_ids=["p1"],
        )
    )

    assert reset["pedestrians"][0]["velocity"] is None


def test_reset_provenance_rejects_string_velocity_rows() -> None:
    """Non-numeric velocity rows must record null (issue #9268)."""
    velocities = np.empty(1, dtype=object)
    velocities[0] = "fast"
    reset = _build_reset_provenance(
        **_reset_kwargs(
            initial_ped_positions=np.array([[5.0, 5.0]]),
            initial_ped_velocities=velocities,
            trace_actor_ids=["p1"],
        )
    )

    assert reset["pedestrians"][0]["velocity"] is None


def test_reset_provenance_accepts_single_element_heading_array() -> None:
    """A single-element heading array is an unambiguous scalar (issue #9268)."""
    reset = _build_reset_provenance(
        **_reset_kwargs(
            initial_ped_positions=np.array([[5.0, 5.0]]),
            initial_ped_headings=np.array([np.array([0.5])], dtype=object),
            trace_actor_ids=["p1"],
        )
    )

    assert reset["pedestrians"][0]["heading"] == 0.5


def test_reset_provenance_rejects_nonfinite_scalar_heading() -> None:
    """An infinite scalar heading must record null (issue #9268)."""
    reset = _build_reset_provenance(
        **_reset_kwargs(
            initial_ped_positions=np.array([[5.0, 5.0]]),
            initial_ped_headings=[float("inf")],
            trace_actor_ids=["p1"],
        )
    )

    assert reset["pedestrians"][0]["heading"] is None


def _capture_mapping() -> dict[str, object]:
    """Return a fixed sampler-capture mapping shaped like the producer output."""
    return {
        "route_anchor_attempts": 7,
        "route_anchor_failures": 1,
        "obstacle_rejections": 12,
        "separation_rejections": 3,
        "accepted_samples": 42,
        "assigned_routes": [
            {
                "group_index": 0,
                "spawn_id": 2,
                "goal_id": 5,
                "source_path_id": "route-a",
                "source_label": "",
                "initial_section": 1,
                "ped_offset": 0,
                "waypoint_count": 2,
                "waypoints": [[0.0, 0.0], [3.0, 4.0]],
            }
        ],
    }


def test_reset_provenance_exposes_sampler_capture_edges() -> None:
    """A capture record flips the spawn/routes edges to available (issue #9312)."""
    reset = _build_reset_provenance(
        **_reset_kwargs(sampler_capture=_capture_mapping())  # type: ignore[arg-type]
    )

    assert reset["spawn"]["status"] == "available"
    assert reset["spawn"]["owner"].endswith("populate_simulation")
    assert reset["spawn"]["route_anchor_attempts"] == 7
    assert reset["spawn"]["route_anchor_failures"] == 1
    assert reset["spawn"]["obstacle_rejections"] == 12
    assert reset["spawn"]["separation_rejections"] == 3
    assert reset["spawn"]["accepted_samples"] == 42
    assert reset["routes"]["status"] == "available"
    assert reset["routes"]["assigned_routes"][0]["waypoints"] == [[0.0, 0.0], [3.0, 4.0]]
    json.dumps(reset)


def test_reset_provenance_malformed_capture_fails_closed() -> None:
    """Malformed capture content must degrade to unavailable, never crash."""
    reset = _build_reset_provenance(
        **_reset_kwargs(  # type: ignore[arg-type]
            sampler_capture={"route_anchor_attempts": "many", "assigned_routes": "nope"}
        )
    )

    assert reset["spawn"]["status"] == "unavailable"
    assert reset["spawn"]["reason"] == "spawn_sampler_decision_not_retained"
    assert reset["routes"]["status"] == "unavailable"
    assert reset["routes"]["reason"] == "route_objects_not_retained_per_episode"
    json.dumps(reset)


def test_read_sampler_capture_returns_mapping_only_for_well_formed_records() -> None:
    """The trace-side reader accepts mappings and rejects everything else."""
    capture = SpawnSamplerCapture(route_anchor_attempts=4, accepted_samples=9)
    env = SimpleNamespace(simulator=SimpleNamespace(sampler_capture=capture))
    assert _read_sampler_capture(env) == capture.to_mapping()

    assert _read_sampler_capture(SimpleNamespace()) is None
    assert _read_sampler_capture(SimpleNamespace(simulator=SimpleNamespace())) is None
    broken = SimpleNamespace(
        simulator=SimpleNamespace(sampler_capture=SimpleNamespace(to_mapping=None))
    )
    assert _read_sampler_capture(broken) is None
