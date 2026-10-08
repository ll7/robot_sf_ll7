"""Analytic footprint regressions; base uses the historical scalar when the block is absent."""

import inspect
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark import metrics as m
from robot_sf.benchmark.critical_intervals import (
    _compute_interval_metrics_in_window,
    extract_critical_intervals,
    report_to_dict,
    summarize_interval_metrics,
)
from robot_sf.benchmark.map_runner import map_runner_episode as producer
from robot_sf.benchmark.metric_definitions import require_uniform_metric_schema
from robot_sf.benchmark.near_miss_ttc import compute_ttc_near_miss_diagnostic
from robot_sf.benchmark.path_utils import compute_completion_reference_length
from robot_sf.benchmark.runner import _scenario_ped_radius_m, _scenario_robot_radius_m
from robot_sf.benchmark.trace_scene_figure import EpisodeTrace, _draw_timeline
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle

MARKER = {"footprint_metric_schema_version": "robot-sf-footprint.v1"}


def episode(*, distance=1.7, velocity=(2, 0)):
    """Return a stationary pedestrian and moving robot snapshot pair."""
    data = m.EpisodeData(
        np.array([[0.0, 0.75], [0.0, 0.75]]),
        np.tile(velocity, (2, 1)),
        np.zeros((2, 2)),
        np.tile([[[distance, 0.75]]], (2, 1, 1)),
        np.zeros((2, 1, 2)),
        np.array([10.0, 0.75]),
        0.1,
        episode_metadata=MARKER,
    )
    return data


def block(data):
    """Exercise the public dispatcher, using base scalars for a meaningful red receipt."""
    values = m.compute_all_metrics(data, horizon=10, shortest_path_len=10)
    return values.get("footprint_metrics", values)


def test_sparse_wall_segment_contact_between_endpoints():
    data = episode()
    data.obstacles = np.array([[-5.0, 0], [5.0, 0]])
    data.obstacle_segments = np.array([[[-5.0, 0], [5.0, 0]]])
    assert block(data)["wall_collisions"] == 2


def test_agent_contact_uses_explicit_combined_radii():
    data = episode()
    data.other_agents_pos = np.tile([[[1.5, 0.75]]], (2, 1, 1))
    data.other_agents_radii = np.array([1.0])
    assert block(data)["agent_collisions"] == 2


def test_head_on_ttc_is_time_to_disc_contact():
    assert block(episode())["time_to_collision_min"] == pytest.approx(0.15)


@pytest.mark.parametrize("velocity", [(1, 2), (-2, 0)])
def test_ttc_excludes_off_axis_miss_and_receding_pair(velocity):
    assert math.isnan(block(episode(distance=3, velocity=velocity))["time_to_collision_min"])


def test_stationary_overlap_ttc_is_zero():
    assert block(episode(distance=1, velocity=(0, 0)))["time_to_collision_min"] == 0


def test_personal_space_uses_surface_gap():
    data = episode()
    # Existing public scalar protects the old centre definition independently.
    assert m.space_compliance(data) == 0
    assert block(data).get("space_compliance", m.space_compliance(data)) == 1


def test_opted_in_near_miss_ttc_diagnostic_uses_contact_geometry():
    result = compute_ttc_near_miss_diagnostic(episode(), t_thr=0.5)
    assert result["near_miss_ttc__min_ttc_s"] == pytest.approx(0.15)
    assert result["near_miss_ttc__count"] == 1


def map_with_barrier(gap=False):
    """Return a thin barrier with either a detour or a robot-inaccessible opening."""
    obstacles = [Obstacle([(4, 0), (6, 0), (6, 4), (4, 4)])]
    if gap:
        obstacles += [Obstacle([(4, 5), (6, 5), (6, 10), (4, 10)])]
    return MapDefinition(
        12,
        10,
        obstacles,
        [((1, 4), (2, 4), (2, 6))],
        [],
        [((8, 3), (10, 3), (10, 7))],
        [(0, 12, 0, 0), (0, 12, 10, 10), (0, 0, 0, 10), (12, 12, 0, 10)],
        [],
        [],
        [],
        [],
    )


def radius_reference(md, start):
    """Run the existing reference API on both versions; old binding omits radius."""
    kwargs = {
        "completion_policy": "goal_zone_entry_v1",
        "goal_zone": np.array([(8, 3), (10, 3), (10, 7)]),
        "robot_radius": 1.0,
    }
    kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in inspect.signature(compute_completion_reference_length).parameters
    }
    return compute_completion_reference_length(md, np.array(start), np.array([10.0, 5.0]), **kwargs)


def test_reference_does_not_pass_through_subdiameter_opening():
    assert math.isnan(radius_reference(map_with_barrier(gap=True), [1.0, 4.5]))


def test_reference_inflates_obstacle_and_separates_radius_cache_key():
    md = map_with_barrier()
    old = compute_completion_reference_length(
        md,
        np.array([1.0, 3.5]),
        np.array([10.0, 5.0]),
        completion_policy="goal_zone_entry_v1",
        goal_zone=np.array([(8, 3), (10, 3), (10, 7)]),
    )
    new = radius_reference(md, [1.0, 3.5])
    # Any admissible path crosses x=4 at y>=5; this hand-derived lower bound
    # proves that a zero-inflation reference cannot satisfy the new contract.
    assert new >= math.hypot(3, 1.5) + 4
    assert new > old + 0.1
    assert radius_reference(md, [1.0, 3.5]) == new


def test_map_producer_adds_separate_inflated_reference_and_preserves_v2():
    md = map_with_barrier(gap=True)
    result = producer._compute_post_loop_metrics(
        robot_positions=[np.array([0.5, 4.5])],
        initial_robot_pos=np.array([1.0, 4.5]),
        robot_headings=[0.0],
        ped_positions=[np.empty((0, 2))],
        ped_forces=[],
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
        goal_vec=np.array([10.0, 4.5]),
        scenario={"name": "dev", "metadata": MARKER},
        config=SimpleNamespace(
            sim_config=SimpleNamespace(time_per_step_in_secs=0.1, ped_radius=0.4),
            robot_config=SimpleNamespace(radius=1.0, max_linear_speed=2.0),
        ),
        horizon_val=10,
        record_forces=False,
        experimental_ped_impact=False,
        ped_impact_radius_m=2.0,
        ped_impact_window_steps=5,
        completion_policy="goal_zone_entry_v1",
        goal_zone=np.array([(8, 3), (10, 3), (10, 7)]),
        seed=1001,
    )
    assert result.shortest_path == pytest.approx(7)
    current = result.metrics_raw.get(
        "footprint_metrics", {"shortest_path_len": result.shortest_path}
    )
    assert math.isnan(current["shortest_path_len"])
    assert current.get("wall_collisions", result.metrics_raw["wall_collisions"]) == 1
    assert result.metrics_raw["metric_schema_version"] == "robot-sf-metrics.v2"


def test_force_inputs_pair_with_pre_integration_positions():
    data = episode(distance=5)
    data.ped_forces[:] = [10, 0]
    data.robot_force_samples = [
        {
            "peds_pos": [[1.7, 0.75]],
            "robot_pos": [0.0, 0.75],
            "total_forces": [[10, 0]],
            "components": [],
            "forces": [[0, 0]],
        }
        for _ in range(2)
    ]
    result = block(data)
    # The old post-step geometry sees a 3.6m gap, and loses both near-robot events.
    legacy_events = float(
        np.count_nonzero(
            (np.linalg.norm(data.peds_pos - data.robot_pos[:, None], axis=-1) - 1.4 < 0.5)
            & (np.linalg.norm(data.ped_forces, axis=-1) > 2)
        )
    )
    assert result.get("force_exceed_near_robot", legacy_events) == 2
    assert result["comfort_exposure"] == 1


def test_runner_records_total_force_input_pairs(monkeypatch):
    state = producer._StepLoopState(obs={})
    if hasattr(state, "force_input_robot_position"):
        state.force_input_robot_position = np.array([0.0, 0.75])
    sim = SimpleNamespace(
        robot_pos=np.array([[0.2, 0.75]]),
        ped_pos=np.array([[5.0, 0.75]]),
        last_ped_forces=np.array([[10.0, 0]]),
        last_robot_ped_forces=np.array([[1.0, 0]]),
        last_robot_force_inputs={"peds_pos": [[1.7, 0.75]], "components": []},
    )
    monkeypatch.setattr(
        producer, "_visibility_evidence_for_step", lambda **_: (None, None, "unavailable", None)
    )
    monkeypatch.setattr(producer, "_read_simulator_ped_headings", lambda *_args, **_kwargs: None)
    producer._step_snapshot_and_record(
        state,
        SimpleNamespace(record_forces=True, footprint_metrics=True, config=SimpleNamespace()),
        env=SimpleNamespace(simulator=sim),
        obs={},
    )
    sample = state.robot_force_samples[0]
    assert sample.get("total_forces") == [[10.0, 0]]
    assert sample["peds_pos"] == [[1.7, 0.75]]
    assert sample["robot_pos"] == [0.0, 0.75]
    assert state.ped_positions[0].tolist() == [[5.0, 0.75]]


def test_interval_gap_and_near_miss_use_declared_radii():
    trace = {
        **MARKER,
        "robot_radius_m": 1.0,
        "ped_radius_m": 0.4,
        "robot_pos": [[0.0, 0.0], [0.0, 0.0]],
        "peds_pos": [[[1.7, 0.0]], [[1.3, 0.0]]],
        "dt": 0.1,
    }
    result = _compute_interval_metrics_in_window(trace, start=0)
    assert result["min_clearance_m"] == pytest.approx(-0.1)
    assert result["near_miss_count"] == 1
    assert result["collision_flag"] is True


def test_interval_anchor_detects_gap_without_centre_overlap():
    trace = {
        **MARKER,
        "robot_radius_m": 1.0,
        "ped_radius_m": 0.4,
        "robot_pos": [[0.0, 0.0], [0.0, 0.0]],
        "peds_pos": [[[1.7, 0.0]], [[2.0, 0.0]]],
        "dt": 0.1,
    }
    config = {
        "schema_version": "critical-intervals.v1",
        "critical_intervals": {
            "collision_or_near_miss": {"enabled": True, "before_s": 0, "after_s": 0.1}
        },
    }
    intervals = extract_critical_intervals(trace, config)
    assert intervals[0].status == "available"
    assert intervals[0].anchor_step == 0


def test_figure_envelope_comes_from_physical_radii():
    from matplotlib import pyplot as plt

    trace = EpisodeTrace(
        metadata={
            **MARKER,
            "robot_radius_m": 0.6,
            "ped_radius_m": 0.2,
            "summary": {"global_min_distance_step": 0},
        },
        steps=(0, 1),
        time_s=(0.0, 0.1),
        robot_xy=((0.0, 0.0), (0.0, 0.0)),
        robot_heading_rad=(0.0, 0.0),
        executed_speed_m_s=(0.0, 0.0),
        min_robot_ped_distance_m=(1.0, 1.0),
        nearest_pedestrian_id=("a", "a"),
        pedestrian_tracks={},
    )
    fig, ax = plt.subplots()
    try:
        handles = _draw_timeline(ax, trace, (), collision_envelope_m=1.4, comfort_distance_m=1.2)
        assert handles[0].get_label() == "collision envelope (0.8 m)"
        assert any(np.allclose(line.get_ydata(), [0.8, 0.8]) for line in ax.lines)
    finally:
        plt.close(fig)


def test_opted_in_runner_defaults_match_physical_settings():
    scenario = {"metadata": MARKER}
    assert _scenario_robot_radius_m(scenario) == 1.0
    assert _scenario_ped_radius_m(scenario) == 0.4
    assert _scenario_robot_radius_m({}) == 0.3
    assert _scenario_ped_radius_m({}) == 0.35


def test_legacy_dispatcher_stays_unchanged():
    old = episode()
    old.episode_metadata = None
    new = replace(old, episode_metadata=MARKER)
    legacy = m.compute_all_metrics(old, horizon=10, shortest_path_len=10)
    opted = m.compute_all_metrics(new, horizon=10, shortest_path_len=10)
    for key in legacy.keys() - {"_episode_metadata"}:
        assert repr(legacy[key]) == repr(opted[key])
    assert "footprint_metrics" not in legacy


def test_interval_contact_ttc_uses_same_disc_geometry():
    trace = {
        **MARKER,
        "robot_radius_m": 1.0,
        "ped_radius_m": 0.4,
        "robot_pos": [[0.0, 0.0], [0.0, 0.0]],
        "robot_vel": [[2.0, 0.0], [2.0, 0.0]],
        "peds_pos": [[[1.7, 0.0]], [[1.7, 0.0]]],
        "ped_vel": [[[0.0, 0.0]], [[0.0, 0.0]]],
        "dt": 0.1,
    }
    assert _compute_interval_metrics_in_window(trace, start=0)["min_ttc_s"] == pytest.approx(0.15)


def test_sweep_effectively_enables_footprint_definition(tmp_path):
    from scripts.validation.run_empty_world_sweep import build_derived_inputs

    kwargs = {
        "seeds": [1001],
        "arms": ["goal"],
        "scenarios_filter": None,
        "workers": 4,
        "out_dir": tmp_path,
        "step_trace": False,
        "footprint_metrics": True,
    }
    kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in inspect.signature(build_derived_inputs).parameters
    }
    _, scenarios = build_derived_inputs("main", **kwargs)
    assert all(
        scenario["metadata"].get("footprint_metric_schema_version") == "robot-sf-footprint.v1"
        for scenario in scenarios
    )


def test_interval_report_declares_changed_ttc_convention():
    trace = {
        **MARKER,
        "robot_radius_m": 1.0,
        "ped_radius_m": 0.4,
        "robot_pos": [[0.0, 0.0], [0.0, 0.0]],
        "peds_pos": [[[1.7, 0.0]], [[1.7, 0.0]]],
        "dt": 0.1,
    }
    report = report_to_dict(summarize_interval_metrics(trace, []))
    assert report["ttc_convention"] == "disc_contact_seconds.v1"
    assert report["footprint_metric_schema_version"] == "robot-sf-footprint.v1"


def test_physical_start_inside_polygon_approximation_band_stays_reachable():
    md = map_with_barrier()
    # Actual wall gap 1.003m exceeds the 1m radius; the circumscribed corner
    # approximation must not reject this physically valid start.
    start = [2.997, 2.0]
    assert math.isfinite(radius_reference(md, start))


def test_reference_rejects_contact_only_aperture():
    md = map_with_barrier(gap=True)
    md.obstacles[1] = Obstacle([(4, 6), (6, 6), (6, 10), (4, 10)])
    # Door width 2m equals the diameter: traversing it requires wall contact.
    assert math.isnan(radius_reference(md, [1.0, 5.0]))


def test_aggregation_refuses_mixed_opt_in_definitions():
    rows = [{"metric_schema_version": "robot-sf-metrics.v2", "metrics": {}} for _ in range(2)]
    rows[1]["metrics"]["footprint_metrics"] = {"schema_version": "robot-sf-footprint.v1"}
    with pytest.raises(ValueError, match="footprint metric definitions"):
        require_uniform_metric_schema(rows)
