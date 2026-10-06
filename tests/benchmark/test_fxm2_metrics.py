"""FXM2 independent value regressions; fixtures never step a simulator."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark import metrics as m
from robot_sf.benchmark import path_utils
from robot_sf.benchmark.map_runner.map_runner_episode import _init_step_loop_state
from robot_sf.benchmark.metric_definitions import metric_schema_version


def episode(positions, *, reached=None, goal=(1000, 0)):
    pos = np.asarray(positions, dtype=float)
    return m.EpisodeData(
        pos,
        np.zeros_like(pos),
        np.zeros_like(pos),
        np.zeros((len(pos), 0, 2)),
        np.zeros((len(pos), 0, 2)),
        np.asarray(goal, dtype=float),
        0.1,
        reached,
    )


@pytest.mark.parametrize("index", [0, 1])
def test_real_merging_timeout_keeps_route_progress(index):
    fixture = Path(__file__).parents[1] / "fixtures/benchmark/fxm2_merging_goal_dev1003_1004.json"
    raw = json.loads(fixture.read_text())[index]
    data = episode(raw["positions"], goal=raw["goal"])
    data.initial_robot_pos = np.asarray(raw["reset"])
    data.route_waypoints = np.vstack([data.initial_robot_pos, raw["waypoints"]])
    # Independent displacement lower bound: all 1.4s windows round to at least 0.66m/s.
    speeds = np.linalg.norm(np.diff(data.robot_pos, axis=0), axis=1) / data.dt
    assert min(np.convolve(speeds, np.ones(14) / 14, mode="valid")) >= 0.655
    result = m.compute_deadlock_stall(data)
    assert result["deadlock"] is False
    assert result["stall_window_count"] == 0


def test_frozen_route_and_goal_zone_survive_navigation_mutation():
    nav = SimpleNamespace(
        waypoints=[(0, 0), (5, 0), (5, 5)],
        goal_zone=[(4, 4), (6, 4), (6, 6)],
        completion_policy="goal_zone_entry_v1",
    )
    sim = SimpleNamespace(robot_pos=np.array([[0, 0]]), ped_pos=np.zeros((0, 2)), robot_navs=[nav])
    state = _init_step_loop_state(
        obs={},
        env=SimpleNamespace(simulator=sim),
        config=SimpleNamespace(),
        hybrid_source_field=None,
    )
    assert getattr(state, "route_waypoints", None) is not None
    nav.waypoints[:] = [(99, 99)]
    nav.goal_zone[:] = [(99, 99)] * 3
    assert state.route_waypoints.tolist() == [[0, 0], [5, 0], [5, 5]]
    assert state.goal_zone.tolist() == [[4, 4], [6, 4], [6, 6]]


def test_failure_efficiency_is_undefined():
    data = episode([[1, 0], [2, 0]])
    data.initial_robot_pos = np.zeros(2)
    assert math.isnan(m.path_efficiency(data, 4.0))


def test_success_efficiency_is_unclipped_and_reference_violation_flagged():
    data = episode([[1, 0], [2, 0]], reached=1)
    data.initial_robot_pos = np.zeros(2)
    assert m.path_efficiency(data, 4.0) == pytest.approx(2.0)
    values = m.compute_all_metrics(data, horizon=10, shortest_path_len=4.0)
    assert values["path_efficiency_reference_violation"] is True


@pytest.mark.parametrize("nested", [False, True])
def test_unmarked_v2_only_diagnostics_are_refused(nested):
    raw = {"deadlock_stall": {"schema_version": "deadlock-stall.v2"}}
    if nested:
        raw = {"metrics": raw}
    with pytest.raises(ValueError, match="missing metric_schema_version"):
        metric_schema_version(raw)


def test_fidelity_snqi_projection_preserves_marker():
    from scripts.benchmark.run_fidelity_sensitivity_campaign import _snqi_input_metrics

    assert (
        _snqi_input_metrics({"metric_schema_version": "robot-sf-metrics.v2", "success": 1.0})[
            "metric_schema_version"
        ]
        == "robot-sf-metrics.v2"
    )


def test_latency_snqi_projection_preserves_marker():
    from robot_sf.benchmark.latency.control_action_latency_snqi import (
        _snqi_metrics,
        derive_inputs_from_raw_rows,
    )

    row = {
        "axis": "control_action_latency",
        "metric_schema_version": "robot-sf-metrics.v2",
        "metrics": {
            "metric_schema_version": "robot-sf-metrics.v2",
            "time_to_goal_norm": 0.5,
            "near_miss_rate": 0.0,
            "comfort_exposure_mean": 0.0,
        },
        "seed": 1003,
        "planner": "goal",
        "planner_group": "goal",
        "scenario_id": "dev",
        "success": True,
        "collision": False,
        "steps": 40,
        "execution_mode": "native",
        "availability_status": "available",
        "action_latency": {"effective_steps": 0, "effective_ms": 0.0},
    }
    entry = derive_inputs_from_raw_rows([row])[0]
    assert _snqi_metrics(entry)["metric_schema_version"] == "robot-sf-metrics.v2"


def test_goal_zone_shortest_path_is_to_continuous_polygon():
    from robot_sf.nav.map_config import MapDefinition
    from robot_sf.nav.obstacle import Obstacle

    md = MapDefinition(
        12,
        10,
        [Obstacle([(4, 4), (6, 4), (6, 6), (4, 6)])],
        [((1, 1), (2, 1), (2, 2))],
        [],
        [((8, 3), (10, 3), (10, 7))],
        [(0, 12, 0, 0), (0, 12, 10, 10), (0, 0, 0, 10), (12, 12, 0, 10)],
        [],
        [],
        [],
        [],
    )
    reference = getattr(path_utils, "compute_completion_reference_length", None)
    if reference is None:
        # Before implementation, the producer always uses the point reference.
        actual = path_utils.compute_shortest_path_length(
            md, np.array([1.0, 5.0]), np.array([10.0, 5.0])
        )
    else:
        actual = reference(
            md,
            np.array([1.0, 5.0]),
            np.array([10.0, 5.0]),
            goal_zone=np.array([(8, 3), (10, 3), (10, 7)]),
            completion_policy="goal_zone_entry_v1",
            scenario_id="dev",
            seed=1003,
        )
    # Bend at (4, +/-1), then (6, +/-1), enter zone at (8, +/-1).
    assert actual == pytest.approx(math.sqrt(10) + 4)


def test_success_only_efficiency_survives_json_as_null():
    data = episode([[1, 0], [2, 0]])
    data.initial_robot_pos = np.zeros(2)
    values = m.compute_all_metrics(data, horizon=10, shortest_path_len=4)
    assert math.isnan(values["path_efficiency"])
    serialized = m.post_process_metrics(values, snqi_weights=None, snqi_baseline=None)
    assert "path_efficiency" in serialized and serialized["path_efficiency"] is None
    json.dumps(serialized, allow_nan=False)


def test_trace_cohort_cannot_mix_reference_versions_even_with_same_metric_marker():
    from robot_sf.benchmark.metric_definitions import require_uniform_metric_schema

    rows = [
        {
            "metric_schema_version": "robot-sf-metrics.v2",
            "algorithm_metadata": {
                "simulation_step_trace": {"schema_version": f"simulation-step-trace.v{v}"}
            },
        }
        for v in (1, 2)
    ]
    with pytest.raises(ValueError, match="trace_schema_version_mismatch"):
        require_uniform_metric_schema(rows)


def test_completion_with_collision_has_no_path_efficiency():
    data = episode([[1, 0], [2, 0]], reached=1)
    data.collision_event = True
    assert math.isnan(m.path_efficiency(data, 1.0))
    assert math.isnan(
        m.compute_all_metrics(data, horizon=10, shortest_path_len=1)["path_efficiency"]
    )


def test_map_producer_uses_same_zone_reference_for_efficiency_and_ideal_time():
    import inspect

    from robot_sf.benchmark.map_runner.map_runner_episode import _compute_post_loop_metrics
    from robot_sf.nav.map_config import MapDefinition

    md = MapDefinition(
        12,
        10,
        [],
        [((1, 1), (2, 1), (2, 2))],
        [],
        [((8, 3), (10, 3), (10, 7))],
        [(0, 12, 0, 0), (0, 12, 10, 10), (0, 0, 0, 10), (12, 12, 0, 10)],
        [],
        [],
        [],
        [],
    )
    kwargs = dict(  # noqa: C408 - named independent producer fixture
        robot_positions=[np.array([9.0, 5.0])],
        initial_robot_pos=np.array([1.0, 5.0]),
        robot_headings=[0.0],
        ped_positions=[np.zeros((0, 2))],
        ped_forces=[np.zeros((0, 2))],
        visibility_trace=[],
        track_confidence_trace=[],
        visibility_evidence_statuses=[],
        visibility_evidence_reasons=[],
        reached_goal_step=0,
        collision_seen=False,
        ped_collision_seen=False,
        obstacle_collision_seen=False,
        robot_collision_seen=False,
        map_def=md,
        goal_vec=np.array([10.0, 5.0]),
        scenario={"name": "dev"},
        config=SimpleNamespace(
            sim_config=SimpleNamespace(time_per_step_in_secs=0.1),
            robot_config=SimpleNamespace(max_linear_speed=1.0),
        ),
        horizon_val=10,
        record_forces=False,
        experimental_ped_impact=False,
        ped_impact_radius_m=5.0,
        ped_impact_window_steps=15,
        completion_policy="goal_zone_entry_v1",
        goal_zone=np.array([(8, 3), (10, 3), (10, 7)]),
        seed=1003,
    )
    # The previous producer lacks completion-policy inputs; run its old binding
    # too, so red evidence is the wrong numeric reference rather than TypeError.
    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in inspect.signature(_compute_post_loop_metrics).parameters
    }
    result = _compute_post_loop_metrics(**kwargs)
    assert result.shortest_path == pytest.approx(7.0)
    assert result.metrics_raw["path_efficiency"] == pytest.approx(7 / 8)
    assert result.metrics_raw["time_to_goal_ideal_ratio"] == pytest.approx(0.1 / 7)
