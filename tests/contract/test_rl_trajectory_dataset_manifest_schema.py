"""Contract tests for RL trajectory dataset manifests."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.rl_trajectory_dataset import (
    RLTrajectoryEpisode,
    build_rl_trajectory_manifest,
    write_rl_trajectory_dataset,
)
from robot_sf.benchmark.schemas.rl_trajectory_dataset_schema import (
    validate_rl_trajectory_dataset_manifest,
)


def test_manifest_schema_accepts_complete_manifest(tmp_path) -> None:
    """Schema accepts a complete manifest including empty held-out splits."""
    episode = _episode("train")
    dataset_path = tmp_path / "dataset.jsonl"
    write_rl_trajectory_dataset([episode], dataset_path)
    manifest = build_rl_trajectory_manifest(
        dataset_id="issue_4011_smoke",
        dataset_path=dataset_path,
        episodes=[episode],
        created_at_utc="2026-07-02T00:00:00+00:00",
        provenance={"artifact_durability": "worktree_local_until_promoted"},
    )

    validate_rl_trajectory_dataset_manifest(manifest)
    assert manifest["splits"]["validation"]["episode_count"] == 0
    assert manifest["splits"]["test"]["episode_count"] == 0


def test_manifest_validation_rejects_scenario_leakage() -> None:
    """Manifest semantics reject scenario IDs reused across splits."""
    manifest = _manifest_with_split_ids(
        train_scenarios=["same"],
        validation_scenarios=["same"],
        train_seed_keys=["same:1"],
        validation_seed_keys=["same:2"],
    )

    with pytest.raises(ValueError, match="scenario_id"):
        validate_rl_trajectory_dataset_manifest(manifest)


def test_manifest_validation_rejects_scenario_seed_leakage() -> None:
    """Manifest semantics reject scenario-seed keys reused across splits."""
    manifest = _manifest_with_split_ids(
        train_scenarios=["same"],
        validation_scenarios=["other"],
        train_seed_keys=["same:1"],
        validation_seed_keys=["same:1"],
    )

    with pytest.raises(ValueError, match="scenario_seed_key"):
        validate_rl_trajectory_dataset_manifest(manifest)


def _episode(split: str) -> RLTrajectoryEpisode:
    return RLTrajectoryEpisode(
        dataset_id="issue_4011_smoke",
        episode_id=f"demo:{split}:0",
        scenario_id="demo",
        seed=1,
        source_policy_id="goal",
        split=split,
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


def _manifest_with_split_ids(
    *,
    train_scenarios: list[str],
    validation_scenarios: list[str],
    train_seed_keys: list[str],
    validation_seed_keys: list[str],
) -> dict:
    return {
        "schema_version": "rl_trajectory_dataset_manifest.v1",
        "dataset_schema_version": "RLTrajectoryDataset.v1",
        "dataset_id": "issue_4011_smoke",
        "created_at_utc": "2026-07-02T00:00:00+00:00",
        "dataset_path": "dataset.jsonl",
        "episode_count": 2,
        "step_count": 2,
        "format": "jsonl",
        "reward_convention": "environment_step_reward",
        "return_convention": "undiscounted_future_return_to_go",
        "split_policy": {
            "strategy": "deterministic_scenario_seed_hash",
            "leakage_prevention": ["scenario_ids", "scenario_seed_keys"],
            "split_names": ["train", "validation", "test"],
        },
        "splits": {
            "train": {
                "episode_count": 1,
                "step_count": 1,
                "scenario_ids": train_scenarios,
                "scenario_seed_keys": train_seed_keys,
                "episode_ids": ["train-0"],
            },
            "validation": {
                "episode_count": 1,
                "step_count": 1,
                "scenario_ids": validation_scenarios,
                "scenario_seed_keys": validation_seed_keys,
                "episode_ids": ["validation-0"],
            },
            "test": {
                "episode_count": 0,
                "step_count": 0,
                "scenario_ids": [],
                "scenario_seed_keys": [],
                "episode_ids": [],
            },
        },
        "provenance": {},
        "dataset_sha256": "0" * 64,
    }
