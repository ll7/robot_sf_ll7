"""Hand-calculated controls for diagnostic geometry, time, censoring and seed accounting."""

import copy
from types import SimpleNamespace

import pytest

from scripts.validation.analyze_hzev_dynamic_window import (
    FLOW_CLASS,
    aggregate,
    episode_metrics,
    segment_distance,
)


def trace(*, arm="stationary", positions=None, recurring=False, smoke=False, budget=400, dt=10.0):
    """Known 80-second trace: corridor occupied through 20s, clear from 30s."""
    positions = positions or [[5, 0], [5, 0], [5, 0], *([[5, 10]] * 6)]
    frames = [
        {
            "step": i - 1,
            "time_s": i * dt,
            "robot": {"position": [0, 0]},
            "pedestrians": [{"position": p}],
        }
        for i, p in enumerate(positions)
    ]
    return {
        "hzev": {
            "start": [0, 0],
            "goal": [10, 0],
            "dt_s": dt,
            "seed": 1001,
            "arm": arm,
            "scenario_id": "control",
            "steps": len(frames) - 1,
            "authored_budget_steps": budget,
            "recurring_flow": recurring,
            "respawns": [],
            "first_terminal": None,
            "initial_frame": frames[0],
            "diagnostic_trace": {"steps": frames[1:]},
            "identity": {"smoke": smoke},
        },
        "algorithm_metadata": {"simulation_step_trace": {"steps": frames[1:]}},
    }


def test_clearance_returns_first_clear_sample_after_last_reentry():
    r = trace(dt=0.1)
    metrics = episode_metrics(r)
    assert metrics["T_clear_s"] == pytest.approx(0.3)
    # 0.3 + 10/2 + 2 = 7.3 s = 73 steps.
    assert metrics["wait_then_go_steps"] == 73
    assert metrics["classification"] == "wait-exploitable at the authored budget"
    r["hzev"]["diagnostic_trace"]["steps"][4]["pedestrians"][0]["position"] = [
        5,
        0,
    ]
    assert episode_metrics(r)["T_clear_s"] == pytest.approx(0.6)


def test_finite_corridor_width_and_alternative_interpretation():
    # Endpoint distance, not distance to an infinite line: sqrt(3^2+4^2)=5.
    assert segment_distance((-3, 4), (0, 0), (10, 0)) == 5
    assert segment_distance((3, 4), (0, 0), (0, 0)) == 5
    r = trace(dt=0.1, positions=[[5, 4], *([[5, 10]] * 8)])
    m = episode_metrics(r)
    assert m["T_clear_s"] == 0.1
    assert m["T_clear_centerline_3m_observed_tail_s"] == 0.0
    r = trace(dt=0.1, positions=[[5, 4.5], *([[5, 10]] * 8)])
    assert episode_metrics(r)["T_clear_s"] == 0.1


def test_wait_budget_units_and_authored_budget():
    # 30s + 5s travel + 2s margin = 37s; dt=0.1 gives 370 steps.
    r = trace(dt=0.1)
    frames = r["hzev"]["diagnostic_trace"]["steps"]
    frames[:] = [
        {
            "step": i,
            "time_s": (i + 1) * 0.1,
            "robot": {"position": [0, 0]},
            "pedestrians": [{"position": [5, 0] if i < 299 else [5, 10]}],
        }
        for i in range(800)
    ]
    r["hzev"]["steps"] = 800
    m = episode_metrics(r)
    assert m["wait_then_go_time_s"] == 37
    assert m["wait_then_go_within_steps"] == {"400": True, "500": True, "600": True}
    r["hzev"]["authored_budget_steps"] = 360
    assert episode_metrics(r)["classification"] == "budget in the dynamic window"


def test_censoring_and_recurring_flow_are_not_finite_clearance():
    r = trace(positions=[[5, 0]] * 9, budget=4)
    m = episode_metrics(r)
    assert m["T_clear_s"] is None
    assert m["classification"] == "budget in the dynamic window"
    r = trace(recurring=True)
    m = episode_metrics(r)
    assert m["T_clear_s"] is None
    assert m["T_clear_observed_tail_s"] == 30
    assert m["classification"] == FLOW_CLASS
    r = trace(smoke=True)
    assert episode_metrics(r)["classification"].startswith("unresolved")


def test_movers_use_robot_distance_and_preserve_no_interaction():
    r = trace(arm="orca", positions=[[2, 0], [2.5, 0], *([[3, 0]] * 7)])
    r["hzev"]["first_terminal"] = {"time_s": 0, "reason": "contact"}
    m = episode_metrics(r)
    assert m["T_last_interaction_s"] == 10
    assert m["T_last_interaction_before_first_terminal_s"] == 0
    r = trace(arm="goal", positions=[[20, 0]] * 9)
    assert episode_metrics(r)["T_last_interaction_s"] is None
    assert episode_metrics(r)["interaction_observed"] is False


def test_seed_median_max_and_disagreement():
    rows = []
    for seed, clear in ((1001, 10), (1002, 20), (1003, 30), (1004, 40), (1005, 50)):
        r = trace(budget=4)
        for f in [
            r["hzev"]["initial_frame"],
            *r["hzev"]["diagnostic_trace"]["steps"],
        ]:
            f["pedestrians"][0]["position"] = [5, 0] if f["time_s"] < clear else [5, 10]
        r["hzev"]["seed"] = seed
        rows.append(episode_metrics(r))
    s = aggregate(rows)[0]
    assert s["T_clear_s"] == {"median": 30, "max": 50, "n_observed": 5, "n_unavailable": 0}
    assert s["seed_disagreement"] is True
    assert s["classification"] == "wait-exploitable at the authored budget"


@pytest.mark.parametrize("fault", ["gap", "nan", "short", "moving", "seed"])
def test_invalid_trace_fails_closed(fault):
    r = copy.deepcopy(trace())
    frames = r["hzev"]["diagnostic_trace"]["steps"]
    if fault == "gap":
        frames[0]["step"] = 5
    elif fault == "nan":
        frames[0]["pedestrians"][0]["position"][0] = float("nan")
    elif fault == "short":
        frames.pop()
    elif fault == "moving":
        frames[0]["robot"]["position"] = [1, 0]
    else:
        r["hzev"]["seed"] = 111
    with pytest.raises(ValueError):
        episode_metrics(r)


@pytest.mark.parametrize("arm", ["stationary", "orca", "goal"])
@pytest.mark.parametrize("events", ["collision_goal", "goal_collision", "simultaneous", "timeout"])
def test_ordinary_outcome_isolated_from_800_step_continuation(monkeypatch, tmp_path, arm, events):
    """Exercise acquisition with controlled events and the unchanged production validator."""
    import json

    import numpy as np

    from robot_sf.benchmark.map_runner import map_runner_episode as episode
    from robot_sf.benchmark.termination_reason import outcome_contradictions
    from scripts.validation import run_hzev_dynamic_window as diagnostic

    cfg = SimpleNamespace(
        dt=0.1,
        record_forces=False,
        scenario_matrix_path=None,
        planners=[SimpleNamespace(key=arm, algo=arm)],
    )
    monkeypatch.setattr(diagnostic, "load_packet", lambda _: (cfg, {"hzev": {}}, []))
    monkeypatch.setattr(episode, "_step_collision_events", lambda **_: [])
    monkeypatch.setattr(
        episode,
        "_init_step_loop_state",
        lambda **_: episode._StepLoopState(
            obs=None,
            initial_robot_pos=np.array([0.0, 0.0]),
            initial_ped_positions=np.array([[5.0, 0.0]]),
        ),
    )

    def controlled_episode(**kwargs):
        env = SimpleNamespace(
            simulator=SimpleNamespace(
                map_def=None,
                goal_pos=np.array([[10.0, 0.0]]),
                robot_navs=[SimpleNamespace(waypoints=[[10.0, 0.0]])],
                robots=[SimpleNamespace(config=SimpleNamespace(radius=0.3))],
                peds_behaviors=[],
            )
        )
        state = episode._init_step_loop_state(env=env)
        slc = SimpleNamespace(collision_event_context=None)
        for i in range(800):
            collision = (i == 9 and events in {"collision_goal", "simultaneous"}) or (
                i == 29 and events == "goal_collision"
            )
            goal = (i == 9 and events in {"goal_collision", "simultaneous"}) or (
                i == 29 and events == "collision_goal"
            )
            sim = SimpleNamespace(
                info={"meta": {"is_pedestrian_collision": collision, "is_route_complete": goal}},
                robot_pos=np.array([0.0, 0.0]),
                peds=np.array([[5.0, 0.0]]),
                terminated=collision or goal,
                truncated=i == 799,
            )
            state.robot_positions.append(sim.robot_pos.copy())
            state.simulation_step_trace.append(
                {
                    "step": i,
                    "time_s": (i + 1) * 0.1,
                    "robot": {"position": [0.0, 0.0]},
                    "pedestrians": [{"position": [5.0, 0.0]}],
                }
            )
            assert episode._step_collision_and_termination(state, slc, step_idx=i, sim=sim) is False
        result = episode._build_step_loop_result(state)
        outcome, _ = episode._episode_outcome(result)
        metrics = {
            "success": float(result.reached_goal_step is not None),
            "collisions": float(result.collision_seen),
        }
        contradictions = outcome_contradictions(
            termination_reason=result.termination_reason,
            outcome=outcome,
            metrics=metrics,
        )
        if contradictions:
            raise ValueError("Episode integrity contradictions: " + "; ".join(contradictions))
        return {
            "outcome": outcome,
            "termination_reason": result.termination_reason,
            "metrics": metrics,
            "steps": len(result.robot_positions),
            "algorithm_metadata": {
                "simulation_step_trace": {"steps": result.simulation_step_trace}
            },
        }

    monkeypatch.setattr(episode, "run_map_episode", controlled_episode)
    output = tmp_path / "row.json"
    scenario = {"name": "control", "metadata": {"hzev_authored_budget_steps": 400}}
    diagnostic.acquire((scenario, 1001, arm, 800, "unused", {"smoke": False}, str(output)))
    row = json.loads(output.read_text())
    assert len(row["hzev"]["diagnostic_trace"]["steps"]) == 800
    expected_steps = 800 if events == "timeout" else 10
    assert row["steps"] == expected_steps
    assert len(row["algorithm_metadata"]["simulation_step_trace"]["steps"]) == expected_steps
    assert row["outcome"]["collision_event"] is (events in {"collision_goal", "simultaneous"})
    assert row["outcome"]["route_complete"] is (events == "goal_collision")
    assert row["outcome"]["timeout_event"] is (events == "timeout")
    assert row["hzev"]["first_terminal"]["step"] == expected_steps - 1
    assert not outcome_contradictions(
        termination_reason=row["termination_reason"], outcome=row["outcome"], metrics=row["metrics"]
    )
    from scripts.validation.analyze_hzev_dynamic_window import validate_ordinary_episode

    validate_ordinary_episode(row)
    if expected_steps < 800:
        polluted = copy.deepcopy(row)
        polluted["steps"] = 800
        polluted["algorithm_metadata"]["simulation_step_trace"] = polluted["hzev"][
            "diagnostic_trace"
        ]
        with pytest.raises(ValueError, match="post-terminal continuation"):
            validate_ordinary_episode(polluted)


def test_worker_limit_respects_cpu_and_slurm_allocation(monkeypatch):
    """A 16-CPU Slurm allocation admits 15 workers even on a larger host."""
    from scripts.validation.run_hzev_dynamic_window import worker_limit

    monkeypatch.setattr("os.cpu_count", lambda: 32)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert worker_limit() == 32
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    assert worker_limit() == 16
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "0")
    with pytest.raises(ValueError, match="positive"):
        worker_limit()


def test_two_seeded_short_episodes_identical_with_one_and_three_workers(tmp_path):
    """Real simulation states, outcomes and metrics are invariant to worker count."""
    import json
    from concurrent.futures import ProcessPoolExecutor

    from scripts.validation.run_hzev_dynamic_window import CONFIG, acquire, load_packet

    _, _, scenarios = load_packet(CONFIG)
    scenario = next(s for s in scenarios if s["name"] == "classic_bottleneck_low")
    identity = {"smoke": True}
    jobs = [
        (scenario, seed, "goal", 60, str(CONFIG), identity, str(tmp_path / f"serial-{seed}.json"))
        for seed in (1001, 1002)
    ]
    for job in jobs:
        acquire(job)
    parallel = [(*job[:-1], str(tmp_path / f"parallel-{job[1]}.json")) for job in jobs]
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(acquire, parallel, chunksize=1))
    for serial, concurrent in zip(jobs, parallel, strict=True):
        a, b = (json.loads(open(job[-1]).read()) for job in (serial, concurrent))
        # Acquisition timestamps and measured execution speed are observations of
        # the host, not seeded simulation results. Everything else must match.
        for row in (a, b):
            for key in ("timestamps", "wall_time_sec", "timing"):
                row.pop(key)
        assert a == b


def test_collision_prefix_metrics_match_normal_episode(tmp_path):
    """Diagnostic continuation preserves the metrics of the ordinary collision episode."""
    import json
    import math

    from robot_sf.benchmark.map_runner import map_runner as runner
    from robot_sf.benchmark.map_runner.map_runner_episode import run_map_episode
    from scripts.validation.run_hzev_dynamic_window import (
        CONFIG,
        acquire,
        episode_scenario,
        load_packet,
    )

    cfg, _, scenarios = load_packet(CONFIG)
    scenario = next(s for s in scenarios if s["name"] == "classic_bottleneck_low")
    normal = run_map_episode(
        scenario=episode_scenario(scenario),
        seed=1002,
        horizon=60,
        dt=cfg.dt,
        record_forces=cfg.record_forces,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        scenario_path=cfg.scenario_matrix_path,
        algo_config={},
        record_simulation_step_trace=True,
        policy_builder=runner._build_policy,
    )
    output = tmp_path / "diagnostic.json"
    acquire((scenario, 1002, "goal", 60, str(CONFIG), {"smoke": True}, str(output)))
    continued = json.loads(output.read_text())

    def json_measurements(value):
        # Auxiliary infinity sentinels become annotated nulls in the diagnostic;
        # JSON also turns tuples into lists. Compare the same representation.
        if isinstance(value, dict):
            return {k: json_measurements(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [json_measurements(v) for v in value]
        return None if isinstance(value, float) and not math.isfinite(value) else value

    assert normal["steps"] == continued["steps"] == 23
    assert normal["outcome"] == continued["outcome"]
    assert json_measurements(normal["metrics"]) == continued["metrics"]
    assert (
        json_measurements(normal["algorithm_metadata"]["simulation_step_trace"]["steps"])
        == (continued["algorithm_metadata"]["simulation_step_trace"]["steps"])
    )
