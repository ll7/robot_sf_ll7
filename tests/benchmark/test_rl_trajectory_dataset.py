"""Tests for the RL trajectory dataset contract and recorder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from robot_sf.benchmark.rl_trajectory_dataset import (
    RLTrajectoryEpisode,
    compute_return_to_go,
    flatten_rl_trajectory_episodes,
    load_rl_trajectory_dataset,
    row_from_episode,
    write_rl_trajectory_dataset,
)
from robot_sf.benchmark.validation.trajectory_dataset import TrajectoryDatasetValidator

_RECORDER_PATH = (
    Path(__file__).resolve().parents[2] / "scripts/benchmark/record_rl_trajectory_dataset.py"
)
_RECORDER_SPEC = importlib.util.spec_from_file_location(
    "record_rl_trajectory_dataset",
    _RECORDER_PATH,
)
assert _RECORDER_SPEC is not None
assert _RECORDER_SPEC.loader is not None
record_rl_trajectory_dataset = importlib.util.module_from_spec(_RECORDER_SPEC)
_RECORDER_SPEC.loader.exec_module(record_rl_trajectory_dataset)


def _episode() -> RLTrajectoryEpisode:
    return RLTrajectoryEpisode(
        dataset_id="issue_4011_smoke",
        episode_id="classic_cross_trap_low:seed101:goal:000000",
        scenario_id="classic_cross_trap_low",
        seed=101,
        source_policy_id="goal",
        split="train",
        observations=({"robot": {"position": [0.0, 0.0]}}, {"robot": {"position": [1.0, 0.0]}}),
        actions=([0.5, 0.0], [0.4, 0.1]),
        rewards=(1.0, 2.0),
        return_to_go=(3.0, 2.0),
        terminated=(False, True),
        truncated=(False, False),
        pedestrians=([], [{"id": "0", "position": [2.0, 1.0]}]),
        robot_states=({"position": [0.0, 0.0]}, {"position": [1.0, 0.0]}),
        provenance={"source": "unit_test"},
    )


def test_compute_return_to_go_uses_undiscounted_future_returns() -> None:
    """Return-to-go uses the issue #4011 undiscounted convention."""
    assert compute_return_to_go([1.0, 2.0, 3.0]) == [6.0, 5.0, 3.0]


def test_loader_round_trips_rl_trajectory_episode(tmp_path) -> None:
    """Loader returns the same typed episode written to JSONL."""
    dataset_path = tmp_path / "issue_4011_smoke.jsonl"
    write_rl_trajectory_dataset([_episode()], dataset_path)

    loaded = load_rl_trajectory_dataset(dataset_path)
    batch = flatten_rl_trajectory_episodes(loaded)

    assert loaded == [_episode()]
    assert batch["rewards"].tolist() == [1.0, 2.0]
    assert batch["episode_id"] == [_episode().episode_id, _episode().episode_id]


def test_trajectory_dataset_validator_accepts_rl_jsonl_dataset(tmp_path) -> None:
    """TrajectoryDatasetValidator reports RL JSONL metadata and coverage."""
    dataset_path = tmp_path / "issue_4011_smoke.jsonl"
    write_rl_trajectory_dataset([_episode()], dataset_path)

    result = TrajectoryDatasetValidator(dataset_path).validate(minimum_episodes=1)

    assert result.episode_count == 1
    assert result.scenario_coverage == {"classic_cross_trap_low": 1}
    assert result.integrity_report["dataset_schema"] == "RLTrajectoryDataset.v1"
    assert result.integrity_report["step_count"] == 2
    assert result.quality_status.value == "validated"


def test_loader_rejects_mismatched_step_lengths(tmp_path) -> None:
    """Loader fails closed when trajectory field lengths diverge."""
    row = row_from_episode(_episode())
    row["trajectory"]["actions"] = [[0.5, 0.0]]
    dataset_path = tmp_path / "bad.jsonl"
    dataset_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="field lengths"):
        load_rl_trajectory_dataset(dataset_path)


def test_recorder_converts_synthetic_simulation_step_trace(tmp_path) -> None:
    """Recorder converts simulation-step traces into RL episode rows."""
    source_path = tmp_path / "episodes.jsonl"
    source_path.write_text(json.dumps(_source_record()) + "\n", encoding="utf-8")

    output_dir = tmp_path / "dataset"
    exit_code = record_rl_trajectory_dataset.main(
        [
            "--source-jsonl",
            str(source_path),
            "--output-dir",
            str(output_dir),
            "--dataset-id",
            "issue_4011_smoke",
        ]
    )

    assert exit_code == 0
    loaded = load_rl_trajectory_dataset(output_dir / "issue_4011_smoke.jsonl")
    assert len(loaded) == 1
    assert loaded[0].rewards == (1.0, 2.0)
    assert loaded[0].return_to_go == (3.0, 2.0)
    assert (output_dir / "issue_4011_smoke.manifest.json").exists()


def test_recorder_fails_closed_when_reward_fields_missing(tmp_path) -> None:
    """Recorder rejects source traces without per-step reward fields."""
    source = _source_record()
    del source["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["rl"]
    source_path = tmp_path / "episodes.jsonl"
    source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing rl.reward|rl must be an object"):
        record_rl_trajectory_dataset.convert_source_records(
            [source],
            dataset_id="issue_4011_smoke",
            source_jsonl=source_path,
        )


def _source_record() -> dict:
    return {
        "episode_id": "classic_cross_trap_low:seed101:goal:000000",
        "scenario_id": "classic_cross_trap_low",
        "seed": 101,
        "algo": "goal",
        "algorithm_metadata": {
            "simulation_step_trace": {
                "schema_version": "simulation-step-trace.v1",
                "steps": [
                    {
                        "step": 0,
                        "robot": {"position": [0.0, 0.0], "heading": 0.0},
                        "pedestrians": [],
                        "planner": {"selected_action": [0.5, 0.0]},
                        "rl": {"reward": 1.0, "terminated": False, "truncated": False},
                    },
                    {
                        "step": 1,
                        "robot": {"position": [1.0, 0.0], "heading": 0.0},
                        "pedestrians": [{"id": "0", "position": [2.0, 1.0]}],
                        "planner": {"selected_action": [0.4, 0.1]},
                        "rl": {"reward": 2.0, "terminated": True, "truncated": False},
                    },
                ],
            }
        },
    }
