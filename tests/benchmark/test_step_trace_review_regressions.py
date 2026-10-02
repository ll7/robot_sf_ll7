"""RV4 regressions using an unchanged runtime canary and real actor-bearing matrix."""

import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from robot_sf.benchmark.map_runner.map_runner import _run_map_episode
from robot_sf.benchmark.step_trace_invariants import check_episode
from robot_sf.scenario_certification.v1 import scenario_actor_source_census
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios
from scripts.validation import generate_step_traces_dev_seeds as generator
from scripts.validation.check_step_trace_invariants import main as check_main


@pytest.fixture(scope="module")
def runtime_row():
    path = Path("configs/scenarios/canary_corridor.yaml")
    scenario = load_scenarios(path)[0]
    return _run_map_episode(
        scenario,
        1001,
        horizon=4,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        scenario_path=path,
        record_simulation_step_trace=True,
    )


@pytest.mark.parametrize(
    "defect",
    [
        "nan",
        "missing_goal",
        "missing_collision",
        "empty_steps",
        "missing_trace",
        "wrong_schema",
        "incomplete_steps",
        "missing_reset_velocity",
        "missing_ped_clearance",
    ],
)
def test_gate_refuses_unavailable_invariant_coverage(  # noqa: C901
    runtime_row, tmp_path, defect
):
    row = deepcopy(runtime_row)
    trace = row["algorithm_metadata"]["simulation_step_trace"]
    if defect == "nan":
        trace["dt"] = float("nan")
        for step in trace["steps"]:
            step["robot"]["position"] = [float("nan"), float("nan")]
            step["robot"]["heading"] = float("nan")
    elif defect == "missing_goal":
        del trace["steps"][0]["goal"]
    elif defect == "missing_collision":
        del trace["steps"][0]["collision"]
    elif defect == "empty_steps":
        trace["steps"] = []
    elif defect == "missing_trace":
        del row["algorithm_metadata"]["simulation_step_trace"]
    elif defect == "wrong_schema":
        trace["schema_version"] = "unknown"
    elif defect == "incomplete_steps":
        trace["steps"].pop()
    elif defect == "missing_reset_velocity":
        del trace["reset"]["robot"]["velocity"]
    elif defect == "missing_ped_clearance":
        trace["steps"][0]["pedestrians"] = [{"position": [100, 100]}]
    file = tmp_path / "episodes.jsonl"
    file.write_text(json.dumps(row) + "\n")
    report = tmp_path / "report.json"
    assert check_main([str(file), "--fail-on-violation", "--out-json", str(report)]) == 1
    summary = json.loads(report.read_text())
    assert summary["meta"]["coverage_complete"] is False
    assert any(not item["eligible"] for item in summary["coverage"][0]["invariants"].values())


def test_gate_refuses_empty_input(tmp_path):
    assert check_main([str(tmp_path), "--fail-on-violation"]) == 1


def test_origin_is_checked_with_nonzero_next_waypoint(runtime_row):
    row = deepcopy(runtime_row)
    trace = row["algorithm_metadata"]["simulation_step_trace"]
    step = deepcopy(trace["steps"][0])
    trace["steps"] = []
    for i in range(80):
        item = deepcopy(step)
        item["robot"]["position"] = [10 - (i + 1) * 0.05, 10 - (i + 1) * 0.05]
        item["robot"]["heading"] = -3 * math.pi / 4
        item["goal"] = {"current": [30, 10], "next": [10, 40]}
        trace["steps"].append(item)
    violations, _ = check_episode(row, enabled=["a_goal_heading"])
    assert "towards_origin_not_current" in {v.kind for v in violations}


def test_initial_acceleration_uses_known_reset_velocity(runtime_row):
    row = deepcopy(runtime_row)
    trace = row["algorithm_metadata"]["simulation_step_trace"]
    trace["reset"]["robot"].update(position=[0, 0], heading=0, velocity=[0, 0], angular_velocity=0)
    trace["steps"] = [deepcopy(trace["steps"][0])]
    trace["steps"][0]["robot"].update(position=[0.2, 0], heading=0, velocity=[2, 0])
    violations, _ = check_episode(row, enabled=["b_drive_limits"])
    accel = [v for v in violations if v.kind == "accel"]
    assert accel and accel[0].step_start == 0
    assert accel[0].detail["limit"] == 1.0  # unchanged canary resolves dataclass drive default


def test_empty_world_generator_removes_real_map_actors(monkeypatch, tmp_path):
    seen = []

    def inspect_episode(scenario, seed, **kwargs):
        assert seed == 1001
        config = build_robot_config_from_scenario(scenario, scenario_path=kwargs["scenario_path"])
        census = scenario_actor_source_census(config)
        seen.append(census)
        assert census["verified_empty"] is True
        return {"termination_reason": "max_steps", "steps": 1}

    monkeypatch.setattr(generator, "_run_map_episode", inspect_episode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--algo",
            "goal",
            "--seeds",
            "1001",
            "--scenario-ids",
            "classic_bottleneck_medium",
            "--empty-world",
            "--out",
            str(tmp_path / "trace.jsonl"),
        ],
    )
    assert generator.main() == 0
    assert seen


def test_runtime_trace_records_reset_angular_velocity(runtime_row):
    """Measured reset yaw rate makes the first acceleration step checkable."""
    from robot_sf.benchmark.step_trace_invariants import invariant_coverage

    trace = runtime_row["algorithm_metadata"]["simulation_step_trace"]
    assert trace["reset"]["robot"]["angular_velocity"] == 0.0
    assert invariant_coverage(runtime_row)["b_drive_limits"]["eligible"]


def test_initial_yaw_acceleration_is_checked(runtime_row):
    row = deepcopy(runtime_row)
    trace = row["algorithm_metadata"]["simulation_step_trace"]
    trace["steps"] = [deepcopy(trace["steps"][0])]
    trace["reset"]["robot"].update(position=[0, 0], heading=0, velocity=[0, 0])
    trace["steps"][0]["robot"].update(position=[0, 0], heading=0.2)
    violations, _ = check_episode(row, enabled=["b_drive_limits"])
    yaw_accel = [v for v in violations if v.kind == "yaw_accel"]
    assert yaw_accel and yaw_accel[0].step_start == 0
    assert yaw_accel[0].detail["limit"] == 1.0


def test_new_v2_trace_can_have_complete_coverage():
    """A real dev1001 episode has every step invariant available, including reset rates."""
    from robot_sf.benchmark.step_trace_invariants import invariant_coverage

    path = Path("configs/scenarios/canary_corridor.yaml")
    row = _run_map_episode(
        load_scenarios(path)[0],
        1001,
        horizon=40,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        scenario_path=path,
        record_simulation_step_trace=True,
    )
    coverage = invariant_coverage(row)
    assert all(c["eligible"] for c in coverage.values()), coverage


@pytest.mark.parametrize("seed", [50036, 111])  # seed-holdout: synthetic-fixture
def test_trace_generator_refuses_held_out_before_scenario_or_reset(
    monkeypatch, tmp_path, capsys, seed
):
    """CLI refusal happens before any scenario or episode code can reset an environment."""

    def forbidden(*args, **kwargs):
        pytest.fail("held-out seed reached scenario/episode code before refusal")

    monkeypatch.setattr(generator, "load_classic_matrix", forbidden)
    monkeypatch.setattr(generator, "_run_map_episode", forbidden)
    out = tmp_path / "forbidden.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["generate", "--algo", "goal", "--seeds", str(seed), "--out", str(out)]
    )
    assert generator.main() == 2
    assert "refusing" in capsys.readouterr().err
    assert not out.exists()


def test_trace_generator_accepts_dev_seed_at_episode_boundary(monkeypatch, tmp_path):
    """The guard permits dev1001 and forwards it unchanged, without a test environment step."""
    seen = []
    monkeypatch.setattr(generator, "load_classic_matrix", lambda _: [{"name": "dev-canary"}])

    def episode(scenario, seed, **kwargs):
        seen.append(seed)
        assert kwargs["record_simulation_step_trace"] is True
        return {"termination_reason": "max_steps", "steps": 1}

    monkeypatch.setattr(generator, "_run_map_episode", episode)
    out = tmp_path / "dev.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["generate", "--algo", "goal", "--seeds", "1001", "--out", str(out)]
    )
    assert generator.main() == 0
    assert seen == [1001]
    assert json.loads(out.read_text())["steps"] == 1
