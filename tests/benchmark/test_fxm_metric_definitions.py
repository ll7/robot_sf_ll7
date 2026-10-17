"""Independent oracles for shared metric defects reported in issue #10007."""

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark import metrics as m
from robot_sf.benchmark.map_runner.map_runner_episode import _init_step_loop_state


def episode(positions, *, dt=0.1, reached=None, goal=(10.0, 0.0), peds=None):
    pos = np.asarray(positions, dtype=float)
    pedestrians = np.zeros((len(pos), 0, 2)) if peds is None else np.asarray(peds)
    return m.EpisodeData(
        pos,
        np.zeros_like(pos),
        np.zeros_like(pos),
        pedestrians,
        np.zeros_like(pedestrians),
        np.asarray(goal),
        dt,
        reached,
    )


def test_final_route_goal_is_captured_at_reset():
    """The handoff waypoint cannot define the metric or trace progress target."""
    sim = SimpleNamespace(
        robot_pos=np.array([[0.0, 0.0]]),
        ped_pos=np.zeros((0, 2)),
        goal_pos=[(2.0, 0.0)],
        robot_navs=[SimpleNamespace(waypoints=[(2.0, 0.0), (5.0, 1.0), (12.0, 3.0)])],
    )
    state = _init_step_loop_state(
        obs={},
        env=SimpleNamespace(simulator=sim),
        config=SimpleNamespace(),
        hybrid_source_field=None,
    )
    assert state.goal_vec.tolist() == [12.0, 3.0]
    assert state.initial_goal_distance == pytest.approx(np.sqrt(153))
    sim.robot_navs[0].waypoints[-1] = (99.0, 99.0)
    assert state.goal_vec.tolist() == [12.0, 3.0]


def test_slow_steady_approach_is_not_a_deadlock():
    """0.4m/s gains 0.56m over 15 samples, exceeding the 0.05m window threshold."""
    data = episode([[t * 0.04, 0.0] for t in range(40)])
    assert m.compute_deadlock_stall(data)["deadlock"] is False
    assert m.compute_deadlock_stall(data)["stall_window_count"] == 0


def test_collision_episode_is_not_a_deadlock():
    data = episode([[0.0, 0.0]] * 40, peds=np.zeros((40, 1, 2)))
    assert m.compute_deadlock_stall(data)["deadlock"] is False


def test_terminal_only_window_does_not_count():
    data = episode([[-0.1, 0.0]] + [[0.0, 0.0]] * 15)
    assert m.compute_deadlock_stall(data)["stall_window_count"] == 0
    data = episode([[0.0, 0.0]] * 16)
    assert m.compute_deadlock_stall(data)["stall_window_count"] == 1


@pytest.mark.parametrize("dt, expected", [(0.1, 20.0), (0.2, 10.0)])
def test_jerk_has_physical_time_units(dt, expected):
    data = episode([[0.0, 0.0]] * 4, dt=dt)
    data.robot_acc = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [100.0, 0.0]])
    assert m.jerk_mean(data) == pytest.approx(expected)


def test_first_completed_step_has_positive_elapsed_goal_time():
    data = episode([[1.0, 0.0]], dt=0.1, reached=0)
    assert m.time_to_goal(data) == pytest.approx(0.1)
    assert m.time_to_goal_norm(data, 10) == pytest.approx(0.1)
    assert m.time_to_goal_norm_success_only(data, 10) == pytest.approx(0.1)
    assert m.compute_all_metrics(data, horizon=10, shortest_path_len=1.0)[
        "time_to_goal_norm"
    ] == pytest.approx(0.1)


def test_first_segment_is_included_without_adding_safety_samples():
    data = episode([[3.0, 4.0], [6.0, 4.0]], goal=(6.0, 4.0))
    # Reset-to-first sample is 5m; next segment 3m. Post-step safety stays length2.
    data.initial_robot_pos = np.array([0.0, 0.0])
    assert m.path_length(data) == pytest.approx(8.0)
    assert m.socnavbench_path_length_ratio(data) == pytest.approx((8.0 + 1e-5) / np.sqrt(52))
    assert data.robot_pos.shape == (2, 2)


@pytest.mark.parametrize("score", ["legacy", "v0", "v1"])
def test_old_anchors_cannot_normalize_new_metric_definitions(score):
    from robot_sf.benchmark.snqi.compute import compute_snqi_v0, compute_snqi_v1

    values = {"metric_schema_version": "robot-sf-metrics.v2", "jerk_mean": 2.0, "success": 0.0}
    anchors = {
        name: {"med": 0.0, "p95": 1.0}
        for name in (
            "time_to_goal_norm",
            "collisions",
            "near_misses",
            "comfort_exposure",
            "force_exceed_events",
            "jerk_mean",
        )
    }
    scoring = {
        "legacy": lambda: m.snqi(values, {}, baseline_stats=anchors),
        "v0": lambda: compute_snqi_v0(values, {}, anchors),
        "v1": lambda: compute_snqi_v1(values, {}, anchors),
    }
    with pytest.raises(ValueError, match="incompatible metric definitions"):
        scoring[score]()


def test_aggregation_rejects_mixed_metric_meanings():
    import copy
    import json
    from pathlib import Path

    from robot_sf.benchmark.aggregate import compute_aggregates

    fixture = (
        Path(__file__).resolve().parents[1] / "fixtures/benchmark/golden/aggregate_episodes.jsonl"
    )
    row = json.loads(fixture.read_text().splitlines()[0])
    rows = [row, copy.deepcopy(row)]
    rows[1]["metric_schema_version"] = "robot-sf-metrics.v2"
    rows[1]["metrics"]["metric_schema_version"] = "robot-sf-metrics.v2"
    with pytest.raises(ValueError, match="incompatible metric definitions"):
        compute_aggregates(rows)


def test_calibration_cannot_pool_definitions_or_reuse_legacy_fixed_anchors():
    from robot_sf.benchmark.snqi.calibration import normalization_anchor_variants

    rows = [{"metrics": {"jerk_mean": 0.2, "metric_schema_version": "robot-sf-metrics.v2"}}]
    with pytest.raises(ValueError, match="incompatible metric definitions"):
        normalization_anchor_variants(rows, {"jerk_mean": {"med": 0.0, "p95": 1.0}})
    rows.append({"metrics": {"jerk_mean": 0.02}})
    with pytest.raises(ValueError, match="incompatible metric definitions"):
        normalization_anchor_variants(rows, {})


def test_real_dev_episode_reset_segment_and_goal_reference():
    """Captured simulator coordinates, with an independent math.dist path oracle."""
    import json
    from pathlib import Path

    fixture_path = (
        Path(__file__).resolve().parents[1] / "fixtures/benchmark/fxm_merging_orca_dev1001.json"
    )
    raw = json.loads(fixture_path.read_text())
    data = episode(raw["positions"], goal=raw["terminal_goal"])
    data.initial_robot_pos = np.array(raw["start"])
    assert m.path_length(data) == pytest.approx(raw["expected_path_m"])
    assert m.socnavbench_path_length_ratio(data) == pytest.approx(raw["expected_path_ratio"])
    sim = SimpleNamespace(
        robot_pos=np.array([raw["start"]]),
        ped_pos=np.zeros((0, 2)),
        goal_pos=[raw["reset_waypoint"]],
        robot_navs=[SimpleNamespace(waypoints=[raw["reset_waypoint"], raw["terminal_goal"]])],
    )
    state = _init_step_loop_state(
        obs={},
        env=SimpleNamespace(simulator=sim),
        config=SimpleNamespace(),
        hybrid_source_field=None,
    )
    assert state.goal_vec.tolist() == raw["terminal_goal"]


def test_synthetic_reset_sample_keeps_one_action_elapsed_time():
    """The synthetic producer indexes completion in reset-inclusive samples."""
    from robot_sf.benchmark.runner import _build_episode_data

    data = _build_episode_data(
        [np.array([0.0, 0.0]), np.array([1.0, 0.0])],
        [np.zeros(2), np.array([10.0, 0.0])],
        [np.zeros(2), np.zeros(2)],
        [np.zeros((0, 2)), np.zeros((0, 2))],
        [np.zeros((0, 2)), np.zeros((0, 2))],
        obstacles=None,
        goal=np.array([1.0, 0.0]),
        dt=0.1,
        reached_goal_step=1,
    )
    assert m.time_to_goal(data) == pytest.approx(0.1)
    assert m.time_to_goal_norm(data, 10) == pytest.approx(0.1)
    assert m.path_length(data) == pytest.approx(1.0)
    assert m.path_efficiency(data, 1.0) == pytest.approx(1.0)


def test_classic_producer_keeps_corrected_elapsed_time():
    """Its scalar override must agree with the common elapsed-time definition."""
    from robot_sf.benchmark.full_classic.orchestrator import _compute_episode_metrics

    pos = np.array([[1.0, 0.0], [2.0, 0.0]])
    values = _compute_episode_metrics(
        SimpleNamespace(job_id="dev-1001", scenario_id="dev", seed=1001),
        SimpleNamespace(),
        SimpleNamespace(),
        robot_pos=pos,
        robot_vel=np.zeros_like(pos),
        robot_acc=np.zeros_like(pos),
        ped_pos=np.zeros((2, 0, 2)),
        ped_forces=np.zeros((2, 0, 2)),
        dt=0.1,
        reached_goal_step=1,
        goal=np.array([2.0, 0.0]),
        horizon=10,
        robot_radius=0.3,
        ped_radius=0.3,
    )
    assert values["time_to_goal"] == pytest.approx(0.2)
