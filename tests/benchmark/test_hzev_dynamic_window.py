"""Hand-calculated controls for diagnostic geometry, time, censoring and seed accounting."""

import copy

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
    r["algorithm_metadata"]["simulation_step_trace"]["steps"][4]["pedestrians"][0]["position"] = [
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
    frames = r["algorithm_metadata"]["simulation_step_trace"]["steps"]
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
            *r["algorithm_metadata"]["simulation_step_trace"]["steps"],
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
    frames = r["algorithm_metadata"]["simulation_step_trace"]["steps"]
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
