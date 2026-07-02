"""Tests for the trajectory dataset validation CLI."""

from __future__ import annotations

import json

import numpy as np

from robot_sf.benchmark.rl_trajectory_dataset import (
    RLTrajectoryEpisode,
    write_rl_trajectory_dataset,
)
from scripts.validation import validate_trajectory_dataset


def _write_dataset(path, *, include_rewards: bool = True) -> None:
    """Write a tiny trajectory dataset fixture."""
    payload = {
        "positions": np.zeros((1, 2, 2), dtype=float),
        "actions": np.zeros((1, 2, 2), dtype=float),
        "observations": np.zeros((1, 2, 3), dtype=float),
        "episode_count": np.array(1),
        "metadata": {
            "scenario_coverage": {"demo": 1},
            "status_policy": {"handling": "exclude_or_explicitly_label_before_training"},
        },
    }
    if include_rewards:
        payload.update(
            {
                "rewards": np.array([[1.0, 2.0]], dtype=float),
                "terminated": np.array([[False, True]], dtype=bool),
                "truncated": np.array([[False, False]], dtype=bool),
                "return_to_go": np.array([[3.0, 2.0]], dtype=float),
            }
        )
    np.savez(path, **payload)


def test_validate_trajectory_dataset_cli_reports_validated(tmp_path, capsys) -> None:
    """CLI prints JSON validation output for a complete dataset."""
    dataset_path = tmp_path / "complete.npz"
    _write_dataset(dataset_path)

    exit_code = validate_trajectory_dataset.main(
        [
            "--path",
            str(dataset_path),
            "--min-episodes",
            "1",
            "--fail-on-quarantine",
            "--require-decision-transformer-fields",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["quality_status"] == "validated"
    assert payload["integrity_report"]["missing_arrays"] == []


def test_validate_trajectory_dataset_cli_fails_on_quarantine(tmp_path, capsys) -> None:
    """CLI can fail closed for datasets missing Decision Transformer labels."""
    dataset_path = tmp_path / "missing_rewards.npz"
    _write_dataset(dataset_path, include_rewards=False)

    exit_code = validate_trajectory_dataset.main(
        [
            "--path",
            str(dataset_path),
            "--min-episodes",
            "1",
            "--fail-on-quarantine",
            "--require-decision-transformer-fields",
        ]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["quality_status"] == "quarantined"
    assert "rewards" in payload["integrity_report"]["missing_arrays"]


def test_validate_trajectory_dataset_cli_accepts_rl_jsonl_dataset(tmp_path, capsys) -> None:
    """CLI recognizes first-class episode-major RL trajectory JSONL datasets."""
    dataset_path = tmp_path / "rl.jsonl"
    write_rl_trajectory_dataset(
        [
            RLTrajectoryEpisode(
                dataset_id="issue_4011_smoke",
                episode_id="demo:seed1:goal:000000",
                scenario_id="demo",
                seed=1,
                source_policy_id="goal",
                split="train",
                observations=({"robot": {}},),
                actions=([0.0, 0.0],),
                rewards=(1.0,),
                return_to_go=(1.0,),
                terminated=(True,),
                truncated=(False,),
                pedestrians=([],),
                robot_states=({"position": [0.0, 0.0]},),
                provenance={"source": "unit_test"},
            )
        ],
        dataset_path,
    )

    exit_code = validate_trajectory_dataset.main(
        [
            "--path",
            str(dataset_path),
            "--min-episodes",
            "1",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["quality_status"] == "validated"
    assert payload["integrity_report"]["dataset_schema"] == "RLTrajectoryDataset.v1"
    assert payload["integrity_report"]["missing_arrays"] == []
