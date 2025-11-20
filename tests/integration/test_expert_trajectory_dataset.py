from __future__ import annotations

import numpy as np

from robot_sf import common
from robot_sf.benchmark.validation.trajectory_dataset import TrajectoryDatasetValidator


def _write_npz_dataset(path, episode_count: int) -> None:
    positions = np.zeros((episode_count, 4, 2), dtype=float)
    actions = np.zeros((episode_count, 4, 2), dtype=float)
    observations = np.zeros((episode_count, 4, 5), dtype=float)
    metadata = {"scenario_coverage": {"classic_interactions": episode_count}}
    np.savez(
        path,
        positions=positions,
        actions=actions,
        observations=observations,
        episode_count=np.array(episode_count),
        metadata=metadata,
    )


def test_validate_npz_dataset(tmp_path):
    dataset_path = tmp_path / "expert_dataset.npz"
    _write_npz_dataset(dataset_path, episode_count=8)

    validator = TrajectoryDatasetValidator(dataset_path)
    result = validator.validate(minimum_episodes=4)

    assert result.dataset_id == "expert_dataset"
    assert result.quality_status == common.TrajectoryQuality.VALIDATED
    assert result.integrity_report["missing_arrays"] == []
    assert result.integrity_report["episode_count"] == 8
    coverage = result.scenario_coverage
    assert coverage.get("classic_interactions") == 8


def test_incomplete_dataset_quarantines(tmp_path):
    dataset_path = tmp_path / "missing_actions.npz"
    positions = np.zeros((2, 3, 2), dtype=float)
    np.savez(dataset_path, positions=positions)

    validator = TrajectoryDatasetValidator(dataset_path)
    result = validator.validate(minimum_episodes=4)

    assert result.quality_status == common.TrajectoryQuality.QUARANTINED
    assert "actions" in result.integrity_report["missing_arrays"]


def test_jsonl_dataset_drafts_when_small(tmp_path):
    dataset_path = tmp_path / "todo.jsonl_frames"
    dataset_path.write_text("{}\n{}\n", encoding="utf-8")

    validator = TrajectoryDatasetValidator(dataset_path)
    result = validator.validate(minimum_episodes=3)

    assert result.episode_count == 2
    assert result.quality_status == common.TrajectoryQuality.DRAFT
