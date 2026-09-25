"""Fail-closed identity tests for the separately staged 0.0.7 force observer."""

import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.ped_npc.ped_robot_force import PedRobotForce

OBSERVER = (
    Path(__file__).parents[2] / "scripts/validation/issue_9671_force_observer_sitecustomize.py"
)
SPEC = importlib.util.spec_from_file_location("issue_9671_observer_test", OBSERVER)
assert SPEC is not None and SPEC.loader is not None
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


def _row() -> dict:
    return {
        "episode_id": "doorway--113--test",
        "scenario_id": "doorway",
        "seed": 113,
        "algo": "ppo",
        "steps": 1,
        "algorithm_metadata": {
            "simulation_step_trace": {
                "reset": {
                    "pedestrians": [
                        {"actor_id": "simulator-slot-0", "position": [0.0, 0.0]},
                        {"actor_id": "simulator-slot-1", "position": [1.0, 0.0]},
                    ]
                },
                "steps": [
                    {
                        "step": 0,
                        "pedestrians": [
                            {"actor_id": "simulator-slot-0", "position": [0.1, 0.0]},
                            {"actor_id": "simulator-slot-1", "position": [1.1, 0.0]},
                        ],
                        "planner": {"ammv": {"pedestrian_force_vectors": [[1.0, 0.0], [2.0, 0.0]]}},
                    }
                ],
            },
        },
    }


def _capture() -> dict:
    return {
        "episode_id": "doorway--113--test",
        "scenario_id": "doorway",
        "seed": 113,
        "algo": "ppo",
        "steps": [
            {
                "step": 0,
                "step_entry_positions": [[0.0, 0.0], [1.0, 0.0]],
                "force_input_positions": [[0.0, 0.0], [1.0, 0.0]],
                "post_step_positions": [[0.1, 0.0], [1.1, 0.0]],
                "total_forces": [[1.0, 0.0], [2.0, 0.0]],
                "robot_forces": [[0.2, 0.0], [0.3, 0.0]],
                "component_ids": ["ped_robot:robot_0"],
            }
        ],
    }


def test_binds_each_force_slot_to_reset_actor_id() -> None:
    bound = observer.bind_episode(_capture(), _row())
    assert bound["actor_ids"] == ["simulator-slot-0", "simulator-slot-1"]
    assert bound["steps"][0]["actor_ids"] == bound["actor_ids"]


@pytest.mark.parametrize("field", ["step_entry_positions", "force_input_positions"])
def test_rejects_permuted_force_slots(field: str) -> None:
    capture = _capture()
    capture["steps"][0][field].reverse()
    with pytest.raises(observer.ObserverIdentityError, match="positions"):
        observer.bind_episode(capture, _row())


def test_rejects_episode_misbinding() -> None:
    capture = _capture()
    capture["episode_id"] = "doorway--114--other"
    with pytest.raises(observer.ObserverIdentityError, match="episode ID"):
        observer.bind_episode(capture, _row())


def test_rejects_actor_permutation_in_trace() -> None:
    row = _row()
    row["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["pedestrians"].reverse()
    with pytest.raises(observer.ObserverIdentityError, match="actor IDs"):
        observer.bind_episode(_capture(), row)


def test_rejects_coincident_positions_that_hide_permutation() -> None:
    row = _row()
    row["algorithm_metadata"]["simulation_step_trace"]["reset"]["pedestrians"][1]["position"] = [
        0.0,
        0.0,
    ]
    with pytest.raises(observer.ObserverIdentityError, match="coincident"):
        observer.bind_episode(_capture(), row)


def test_rejects_wrong_prior_trace_state_on_second_step() -> None:
    capture = _capture()
    second = dict(capture["steps"][0])
    second["step"] = 1
    capture["steps"].append(second)
    row = _row()
    second_trace = dict(row["algorithm_metadata"]["simulation_step_trace"]["steps"][0])
    second_trace["step"] = 1
    row["algorithm_metadata"]["simulation_step_trace"]["steps"].append(second_trace)
    row["steps"] = 2
    with pytest.raises(observer.ObserverIdentityError, match="pre-step positions"):
        observer.bind_episode(capture, row)


def test_rejects_changed_component_roster_on_second_step() -> None:
    capture = _capture()
    second = dict(capture["steps"][0])
    second["step"] = 1
    second["step_entry_positions"] = [[0.1, 0.0], [1.1, 0.0]]
    second["force_input_positions"] = [[0.1, 0.0], [1.1, 0.0]]
    second["component_ids"] = ["ped_robot:robot_1"]
    capture["steps"].append(second)
    row = _row()
    second_trace = dict(row["algorithm_metadata"]["simulation_step_trace"]["steps"][0])
    second_trace["step"] = 1
    row["algorithm_metadata"]["simulation_step_trace"]["steps"].append(second_trace)
    row["steps"] = 2
    with pytest.raises(observer.ObserverIdentityError, match="component roster"):
        observer.bind_episode(capture, row)


def test_live_dispatch_rejects_episode_on_second_thread(tmp_path: Path) -> None:
    source_root = tmp_path / "frozen"
    runner = source_root / observer.RUNNER_FILE
    runner.parent.mkdir(parents=True)
    code = "def run_map_episode(scenario, seed, algo):\n    return None\n"
    runner.write_text(code)
    namespace: dict = {
        "__name__": observer.SOURCE_MODULES[observer.RUNNER_FILE],
        "__file__": str(runner),
    }
    exec(  # noqa: S102 - compile a synthetic frozen filename to exercise live trace dispatch
        compile(code, str(runner), "exec"),
        namespace,
    )
    watched = observer.ForceObserver(tmp_path, {}, source_root)
    failures: list[Exception] = []

    def invoke() -> None:
        try:
            namespace["run_map_episode"]({"name": "doorway"}, 113, "ppo")
        except Exception as exc:
            failures.append(exc)

    old_sys_trace = sys.gettrace()
    old_thread_trace = threading.gettrace()
    try:
        sys.settrace(watched)
        threading.settrace(watched)
        thread = threading.Thread(target=invoke)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
    finally:
        sys.settrace(old_sys_trace)
        threading.settrace(old_thread_trace)
    assert len(failures) == 1
    assert isinstance(failures[0], observer.ObserverIdentityError)
    assert "process or thread" in str(failures[0])


def test_live_dispatch_rejects_shadowed_import_root(tmp_path: Path) -> None:
    verified_root = tmp_path / "frozen"
    verified = verified_root / observer.RUNNER_FILE
    shadowed = tmp_path / "shadow" / observer.RUNNER_FILE
    for path in (verified, shadowed):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def run_map_episode(scenario, seed, algo):\n    return None\n")
    namespace: dict = {
        "__name__": observer.SOURCE_MODULES[observer.RUNNER_FILE],
        "__file__": str(shadowed),
    }
    exec(  # noqa: S102 - live dispatch must see a real shadow-root filename
        compile(shadowed.read_text(), str(shadowed), "exec"), namespace
    )
    watched = observer.ForceObserver(tmp_path, {}, verified_root)
    previous = sys.gettrace()
    try:
        sys.settrace(watched)
        with pytest.raises(observer.ObserverIdentityError, match="outside pinned frozen source"):
            namespace["run_map_episode"]({"name": "doorway"}, 113, "ppo")
    finally:
        sys.settrace(previous)
    assert list(tmp_path.glob("*.robot-force.json")) == []


def test_copies_actual_last_forces_without_reinvoking_provider(tmp_path: Path) -> None:
    component = PedRobotForce.__new__(PedRobotForce)
    component.component_type = "pedestrian_robot"
    component.last_forces = np.asarray([[0.2, 0.0], [0.3, 0.0]])
    component.config = SimpleNamespace(force_multiplier=10.0)
    component.get_robot_pos = lambda: pytest.fail("observer re-invoked provider")
    watched = observer.ForceObserver(tmp_path, {}, tmp_path)
    watched.episode = {"steps": []}
    watched.step = {"step_entry_positions": [[0.0, 0.0], [1.0, 0.0]]}
    frame = SimpleNamespace(
        f_locals={
            "self": component,
            "ped_positions": np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            "robot_pos": np.asarray([2.0, 0.0]),
        }
    )
    watched._force_event(frame, component.last_forces)
    assert watched.force_returns[id(component)]["forces"] == [[0.2, 0.0], [0.3, 0.0]]
