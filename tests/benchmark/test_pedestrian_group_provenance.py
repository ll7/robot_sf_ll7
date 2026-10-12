"""Reset membership must survive episode execution and JSON serialization."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from robot_sf.benchmark import map_runner_episode
from robot_sf.benchmark.result_provenance import capture_pedestrian_group_provenance
from scripts.analysis import replay_pedestrian_groups as replay
from tests.benchmark.test_map_runner_episode_algo_meta_isolation import (
    _EpisodeEnv,
    _patch_episode_runtime,
)


def test_episode_preserves_reset_groups_after_runtime_membership_changes(monkeypatch) -> None:
    """The emitted row keeps original IDs/counts even when the runtime splits a group."""
    _patch_episode_runtime(monkeypatch)
    env = _EpisodeEnv()
    env.simulator.ped_pos = np.array([[2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
    env.simulator.groups = SimpleNamespace(
        groups={np.int64(7): {np.int64(0), np.int64(2)}, 9: {1}, 12: set()},
        group_by_ped_id={0: 7, 2: 7, 1: 9},
    )
    original_step = env.step

    def split_group(action):
        env.simulator.groups.groups[7].remove(2)
        env.simulator.groups.groups[13] = {2}
        env.simulator.groups.group_by_ped_id[2] = 13
        return original_step(action)

    env.step = split_group
    monkeypatch.setattr(map_runner_episode, "make_robot_env", lambda **_kwargs: env)
    row = map_runner_episode.run_map_episode(
        {"name": "reset-groups", "simulation_config": {"max_episode_steps": 1}},
        seed=1001,
        horizon=1,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        scenario_path=Path(__file__),
        policy_builder=lambda *_args, **_kwargs: (lambda _obs: (0.0, 0.0), {"algorithm": "goal"}),
    )
    snapshot = json.loads(json.dumps(row))["pedestrian_group_provenance"]
    assert snapshot == {
        "schema_version": "pedestrian_group_provenance.v1",
        "status": "recorded",
        "capture_phase": "reset",
        "pedestrian_id_scope": "reset_simulator_slot",
        "pedestrian_count": 4,
        "groups": [
            {"group_id": 7, "pedestrian_ids": [0, 2], "member_count": 2},
            {"group_id": 9, "pedestrian_ids": [1], "member_count": 1},
            {"group_id": 12, "pedestrian_ids": [], "member_count": 0},
        ],
        "unassigned_pedestrian_ids": [3],
        "multi_member_group_count": 1,
        "pedestrians_in_multi_member_groups": 2,
    }
    assert env.simulator.groups.groups[7] == {0}
    schema = json.loads(Path("robot_sf/benchmark/schemas/episode.schema.v1.json").read_text())
    Draft202012Validator(schema["properties"]["pedestrian_group_provenance"]).validate(snapshot)


def test_empty_crowd_is_recorded_but_missing_groups_are_unavailable() -> None:
    """Distinguish measured emptiness from absent runtime provenance."""
    empty = capture_pedestrian_group_provenance(
        SimpleNamespace(ped_pos=np.empty((0, 2)), groups=SimpleNamespace(groups={}))
    )
    assert empty["status"] == "recorded"
    assert empty["pedestrian_count"] == 0
    assert empty["groups"] == []
    assert empty["pedestrians_in_multi_member_groups"] == 0
    missing = capture_pedestrian_group_provenance(SimpleNamespace(ped_pos=np.empty((0, 2))))
    assert missing["status"] == "unavailable"
    assert "groups" not in missing


def test_authorized_replay_is_zero_step_and_preserves_inputs(tmp_path, monkeypatch) -> None:
    """An authorized export creates derived provenance without stepping or overwriting."""
    episodes = tmp_path / "episodes.jsonl"
    original = json.dumps(
        {"scenario_id": "one", "seed": 1001, "episode_id": "old", "git_hash": "old-source"}
    )
    episodes.write_text(original)
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("fixture")
    output = tmp_path / "derived.jsonl"
    env = _EpisodeEnv()
    env.simulator.groups = SimpleNamespace(groups={8: {0}})
    seeds = []

    def reset(*, seed):
        seeds.append(seed)

    monkeypatch.setattr(env, "reset", reset)
    monkeypatch.setattr(env, "step", lambda *_args: pytest.fail("replay must not step"))
    monkeypatch.setattr(replay, "load_scenarios", lambda _path: [{"name": "one"}])
    monkeypatch.setattr(
        replay,
        "build_env_config",
        lambda *_args, **_kwargs: SimpleNamespace(sim_config=SimpleNamespace()),
    )
    monkeypatch.setattr(replay, "make_robot_env", lambda **_kwargs: env)
    assert (
        replay.export_reset_groups(episodes, matrix, output, authorize_dev_reset_replay=True) == 1
    )
    assert seeds == [1001]
    assert episodes.read_text() == original
    assert matrix.read_text() == "fixture"
    derived = json.loads(output.read_text())
    assert derived["historical_equivalence"] == "not_established"
    assert derived["input_git_hash"] == "old-source"
    assert derived["pedestrian_group_provenance"]["groups"] == [
        {"group_id": 8, "pedestrian_ids": [0], "member_count": 1}
    ]
    with pytest.raises(ValueError, match="new file"):
        replay.export_reset_groups(episodes, matrix, output, authorize_dev_reset_replay=True)


def test_replay_rejects_missing_authorization_and_non_dev_batch_before_reset(
    tmp_path, monkeypatch
) -> None:
    """A mixed unsupported seed batch is rejected before any simulator is constructed."""
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text("\n".join(json.dumps({"seed": seed}) for seed in [1001, 1000]))
    matrix = tmp_path / "matrix.yaml"
    output = tmp_path / "derived.jsonl"
    monkeypatch.setattr(
        replay, "make_robot_env", lambda **_kwargs: pytest.fail("unauthorized reset")
    )
    with pytest.raises(ValueError, match="explicit"):
        replay.export_reset_groups(episodes, matrix, output)
    with pytest.raises(ValueError, match="development seeds"):
        replay.export_reset_groups(episodes, matrix, output, authorize_dev_reset_replay=True)
    assert not output.exists()
