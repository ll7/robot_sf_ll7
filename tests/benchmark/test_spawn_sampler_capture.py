"""Contract tests for the opt-in spawn-sampler capture (issue #9312).

The capture must record real sampler decisions per episode, must expose the
per-episode route/zone assignments at trace finalization, and must leave both
the spawn path and the default reset provenance untouched when disabled.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon as _ShapelyPolygon
from shapely.prepared import prep

from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.benchmark.map_runner.map_runner_episode import (
    _build_reset_provenance,
    _finalize_trace_metadata,
    _init_step_loop_state,
)
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import GlobalRoute
from robot_sf.ped_npc.ped_population import (
    PedSpawnConfig,
    _populate_scattered_background,
    _sample_scatter_point,
    populate_ped_routes,
)
from robot_sf.ped_npc.spawn_capture import (
    SCHEMA_VERSION,
    SpawnSamplerCapture,
    reset_block_payload,
)

_MAP_ID = "uni_campus_big"
_ZONE = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))


def _route() -> GlobalRoute:
    return GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(0.0, 0.0), (10.0, 0.0)],
        spawn_zone=_ZONE,
        goal_zone=_ZONE,
    )


def _seeded_route_config() -> PedSpawnConfig:
    return PedSpawnConfig(
        peds_per_area_m2=0.5,
        max_group_members=3,
        group_member_probs=[0.5, 0.3, 0.2],
        route_spawn_seed=7,
        route_spawn_distribution="spread",
        route_spawn_jitter_frac=0.1,
    )


def _reset_kwargs(**overrides: object) -> dict[str, object]:
    """Return a minimal reset fixture; callers override individual inputs."""
    base: dict[str, object] = {
        "initial_robot_pos": np.array([0.0, 0.0]),
        "initial_robot_heading": 0.0,
        "initial_robot_velocity": None,
        "initial_ped_positions": np.array([[5.0, 5.0]]),
        "initial_ped_velocities": None,
        "initial_ped_headings": np.array([0.0]),
        "trace_actor_ids": ["p0"],
        "robot_radius": 0.5,
        "ped_radius": 0.4,
        "scenario": {},
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


def test_capture_serialization_is_deterministic_and_json_safe() -> None:
    """Records render sorted by pedestrian id with strict-JSON-safe values."""
    capture = SpawnSamplerCapture()
    capture.record_attempts(5, source="route", attempts=3, obstacle_rejections=1)
    capture.record_spawn_point(5, source="route", point=(1.0, 2.0))
    capture.record_route_assignment(5, source="route", route_index=1, waypoints=[(0.0, 0.0)])
    capture.record_attempts(1, source="synthesized", attempts=2, separation_rejections=1)
    capture.record_spawn_point(1, source="synthesized", point=(0.5, 0.5))

    payload = capture.to_dict()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert [record["ped_id"] for record in payload["pedestrians"]] == [1, 5]
    assert payload["counts"]["pedestrians"] == 2
    assert payload["counts"]["route_assignments"] == 1
    assert payload["counts"]["attempts"] == 5
    assert payload["pedestrians"][1]["route_waypoints"] == [[0.0, 0.0]]
    # Re-rendering the same capture must be byte-identical (deterministic order).
    assert json.dumps(payload, allow_nan=False, sort_keys=True) == json.dumps(
        capture.to_dict(), allow_nan=False, sort_keys=True
    )


def test_capture_marks_uninstrumented_counters_explicitly() -> None:
    """A recorded point without sampler counters never reports a false zero."""
    capture = SpawnSamplerCapture()
    capture.record_spawn_point(0, source="crowded_zone", point=(1.0, 1.0))
    capture.record_zone_assignment(0, zone_id=2)

    rendered = capture.to_dict()["pedestrians"][0]

    assert rendered["counters_recorded"] is False
    assert rendered["zone_index"] == 2


def test_capture_skips_non_finite_values() -> None:
    """Malformed points and waypoints are dropped instead of breaking strict JSON."""
    capture = SpawnSamplerCapture()
    capture.record_spawn_point(0, source="route", point=(float("nan"), 1.0))
    capture.record_route_assignment(
        0, source="route", route_index=0, waypoints=[(0.0, float("inf")), (1.0, 1.0)]
    )

    payload = capture.to_dict()
    assert payload["pedestrians"][0]["spawn_point"] is None
    assert payload["pedestrians"][0]["route_waypoints"] == [[1.0, 1.0]]
    json.dumps(payload, allow_nan=False)


def test_capture_rekeys_spawner_local_ids_onto_merged_rows() -> None:
    """Route and single records shift onto their merged-state row indices.

    Each spawner numbers its pedestrians from zero, so crowd and route records with
    the same local id must stay distinct until the merge offsets are applied.
    """
    capture = SpawnSamplerCapture()
    capture.record_spawn_point(0, source="crowded_zone", point=(0.0, 0.0))
    capture.record_spawn_point(0, source="route", point=(1.0, 0.0))
    capture.record_spawn_point(0, source="single", point=(2.0, 0.0))

    assert sorted(capture.records) == [("crowded_zone", 0), ("route", 0), ("single", 0)]
    capture.apply_merged_offsets(route_offset=4, single_offset=9)

    assert capture.records[("route", 0)].ped_id == 4
    assert capture.records[("single", 0)].ped_id == 9
    assert capture.records[("crowded_zone", 0)].ped_id == 0
    payload = capture.to_dict()
    assert payload["counts"]["pedestrians"] == 3
    assert payload["counts"]["offsets_applied"] is True
    assert payload["route_offset"] == 4
    assert payload["single_offset"] == 9
    assert [record["ped_id"] for record in payload["pedestrians"]] == [0, 4, 9]


def test_route_spawn_capture_records_attempts_and_assigned_waypoints() -> None:
    """A seeded route spawn records attempts, the accepted point, and the route polyline."""
    capture = SpawnSamplerCapture()
    states, _groups, _assignments, _sections = populate_ped_routes(
        _seeded_route_config(), [_route()], capture=capture
    )

    payload = capture.to_dict()
    assert payload["counts"]["pedestrians"] == states.shape[0]
    assert payload["counts"]["route_assignments"] == states.shape[0]
    assert payload["counts"]["attempts"] >= states.shape[0]
    first = payload["pedestrians"][0]
    assert first["counters_recorded"] is True
    assert first["source"] == "route"
    assert first["route_waypoints"] == [[0.0, 0.0], [10.0, 0.0]]
    # The captured point is the state row the simulation actually uses.
    assert first["spawn_point"] == pytest.approx(states[first["ped_id"], 0:2].tolist())


def test_route_spawn_capture_leaves_default_path_byte_identical() -> None:
    """Passing no capture must not change the produced states."""
    config = _seeded_route_config()
    captured_states, _, _, _ = populate_ped_routes(
        config, [_route()], capture=SpawnSamplerCapture()
    )
    default_states, _, _, _ = populate_ped_routes(config, [_route()])

    assert np.array_equal(captured_states, default_states)


class _SequenceRng:
    """Deterministic stand-in for the generator's ``uniform`` draw only."""

    def __init__(self, draws: list[tuple[float, float]]) -> None:
        self._draws = list(draws)

    def uniform(self, low: float, high: float, size: int) -> np.ndarray:
        return np.asarray(self._draws.pop(0), dtype=float) * (high - low) + low


def test_scatter_sampler_records_rejections_and_accepted_point() -> None:
    """Obstacle and separation rejections are counted before the accepted candidate."""
    # Triangle with a=(0,0), b=(4,0), c=(0,4): draw (rw, rh) maps to (4-4rw-4rh, 4rh).
    zone = ((0.0, 0.0), (4.0, 0.0), (0.0, 4.0))
    exclusion = [prep(_ShapelyPolygon([(2.0, 0.0), (4.0, 0.0), (2.0, 4.0), (4.0, 4.0)]))]
    rng = _SequenceRng(
        [
            (0.125, 0.125),  # (3.0, 0.5): inside the exclusion polygon
            (0.125, 0.125),  # (3.0, 0.5): inside the exclusion polygon
            (0.0, 0.875),  # (0.5, 3.5): clear of geometry, but inside 2*ped_radius
            (0.875, 0.0),  # (0.5, 0.0): accepted
        ]
    )
    capture = SpawnSamplerCapture()

    point = _sample_scatter_point(
        zone,
        rng,
        exclusion,
        [(0.5, 3.5)],
        ped_radius=0.4,
        capture=capture,
        ped_id=2,
    )

    rendered = capture.to_dict()["pedestrians"][0]
    assert point == pytest.approx((0.5, 0.0))
    assert rendered["ped_id"] == 2
    assert rendered["counters_recorded"] is True
    assert rendered["attempts"] == 4
    assert rendered["obstacle_rejections"] == 2
    assert rendered["separation_checks"] == 1
    assert rendered["separation_rejections"] == 1
    assert rendered["spawn_point"] == pytest.approx([0.5, 0.0])


def test_synthesized_background_capture_covers_every_pedestrian() -> None:
    """Forced geometry-less backgrounds record each scattered pedestrian and its zone."""
    capture = SpawnSamplerCapture()
    config = PedSpawnConfig(
        peds_per_area_m2=0.5,
        max_group_members=3,
        group_member_probs=[0.5, 0.3, 0.2],
        route_spawn_seed=11,
    )

    states, _groups, zone_assignments, _zones, _rng = _populate_scattered_background(
        config, 6, (0.0, 20.0, 0.0, 20.0), [], 0.4, capture=capture
    )

    payload = capture.to_dict()
    assert payload["counts"]["by_source"] == {"synthesized": states.shape[0]}
    for record in payload["pedestrians"]:
        assert record["zone_index"] == zone_assignments[record["ped_id"]]
        assert record["spawn_point"] == pytest.approx(states[record["ped_id"], 0:2].tolist())


def test_reset_block_reports_captured_spawn_and_route_edges() -> None:
    """With a capture the reset edges report captured payloads instead of unavailable ones."""
    capture = SpawnSamplerCapture()
    capture.record_spawn_point(0, source="route", point=(1.0, 1.0))
    capture.record_route_assignment(
        0, source="route", route_index=3, waypoints=[(0.0, 0.0), (1.0, 1.0)]
    )
    payload = reset_block_payload(capture)
    assert payload is not None

    reset = _build_reset_provenance(**_reset_kwargs(spawn_capture=payload))

    assert reset["spawn"]["status"] == "captured"
    assert reset["spawn"]["counts"]["pedestrians"] == 1
    assert reset["spawn"]["schema_version"] == SCHEMA_VERSION
    assert reset["routes"]["status"] == "captured"
    assert reset["routes"]["assignments"] == [
        {
            "ped_id": 0,
            "route_index": 3,
            "route_waypoints": [[0.0, 0.0], [1.0, 1.0]],
        }
    ]
    assert reset["routes"]["zone_assignments"] == []


def test_empty_capture_falls_back_to_unavailable_reset_edges() -> None:
    """An opt-in capture that recorded nothing must not claim a captured edge."""
    assert reset_block_payload(None) is None
    assert reset_block_payload(SpawnSamplerCapture()) is None

    reset = _build_reset_provenance(**_reset_kwargs(spawn_capture=None))

    assert reset["spawn"]["status"] == "unavailable"
    assert reset["routes"]["status"] == "unavailable"


def test_build_env_config_maps_scenario_opt_in() -> None:
    """The scenario flag enables the capture on the env simulation settings."""
    scenario = {
        "name": "spawn_capture_flag",
        "map_id": _MAP_ID,
        "simulation_config": {"max_episode_steps": 1},
        "robot_config": {"type": "differential_drive"},
        "record_spawn_sampler_capture": True,
    }

    config = build_env_config(scenario, scenario_path=Path("."))
    assert config.sim_config.record_spawn_sampler_capture is True

    default_config = build_env_config(
        {key: value for key, value in scenario.items() if key != "record_spawn_sampler_capture"},
        scenario_path=Path("."),
    )
    assert default_config.sim_config.record_spawn_sampler_capture is False


def test_simulator_without_opt_in_keeps_no_capture() -> None:
    """The default env construction retains no capture object."""
    env = make_robot_env(config=RobotSimulationConfig(map_id=_MAP_ID), seed=0)
    try:
        assert env.simulator.spawn_capture is None
    finally:
        env.close()


def test_env_capture_reaches_trace_reset_block() -> None:
    """End-to-end: opt-in env capture populates the step-trace reset provenance."""
    config = RobotSimulationConfig(map_id=_MAP_ID)
    config.sim_config.record_spawn_sampler_capture = True
    env = make_robot_env(config=config, seed=0)
    try:
        obs, _ = env.reset(seed=0)
        state = _init_step_loop_state(obs=obs, env=env, config=config, hybrid_source_field=None)
        payload = state.spawn_capture
        assert payload is not None
        ped_count = int(np.asarray(env.simulator.ped_pos).reshape(-1, 2).shape[0])
        assert payload["counts"]["pedestrians"] == ped_count
        assert payload["counts"]["route_assignments"] > 0

        algo_meta: dict[str, object] = {}
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
            initial_robot_pos=state.initial_robot_pos,
            initial_robot_heading=state.initial_robot_heading,
            initial_ped_positions=state.initial_ped_positions,
            initial_robot_velocity=state.initial_robot_velocity,
            initial_ped_velocities=state.initial_ped_velocities,
            initial_ped_headings=state.initial_ped_headings,
            trace_actor_ids=state.trace_actor_ids,
            horizon_val=10,
            termination_reason="max_steps",
            safety_events=[],
            spawn_capture=state.spawn_capture,
        )

        reset = algo_meta["simulation_step_trace"]["reset"]  # type: ignore[index]
        assert reset["spawn"]["status"] == "captured"
        assert reset["spawn"]["counts"]["pedestrians"] == ped_count
        assert reset["routes"]["status"] == "captured"
        assert len(reset["routes"]["assignments"]) == payload["counts"]["route_assignments"]
    finally:
        env.close()
