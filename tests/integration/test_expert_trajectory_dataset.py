"""TODO docstring. Document this module."""

from __future__ import annotations

import numpy as np

from robot_sf import common
from robot_sf.benchmark.validation.trajectory_dataset import TrajectoryDatasetValidator


def _write_npz_dataset(path, episode_count: int) -> None:
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.
        episode_count: TODO docstring.
    """
    positions = np.zeros((episode_count, 4, 2), dtype=float)
    actions = np.zeros((episode_count, 4, 2), dtype=float)
    observations = np.zeros((episode_count, 4, 5), dtype=float)
    rewards = np.ones((episode_count, 4), dtype=float)
    terminated = np.zeros((episode_count, 4), dtype=bool)
    terminated[:, -1] = True
    truncated = np.zeros((episode_count, 4), dtype=bool)
    return_to_go = np.tile(np.array([4.0, 3.0, 2.0, 1.0]), (episode_count, 1))
    metadata = {
        "scenario_coverage": {"classic_interactions": episode_count},
        "status_policy": {"handling": "exclude_or_explicitly_label_before_training"},
    }
    np.savez(
        path,
        positions=positions,
        actions=actions,
        observations=observations,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        return_to_go=return_to_go,
        episode_count=np.array(episode_count),
        metadata=metadata,
    )


def test_validate_npz_dataset(tmp_path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    dataset_path = tmp_path / "expert_dataset.npz"
    _write_npz_dataset(dataset_path, episode_count=8)

    validator = TrajectoryDatasetValidator(dataset_path)
    result = validator.validate(minimum_episodes=4)

    assert result.dataset_id == "expert_dataset"
    assert result.quality_status == common.TrajectoryQuality.VALIDATED
    assert result.integrity_report["missing_arrays"] == []
    assert result.integrity_report["alignment_issues"] == []
    assert result.integrity_report["episode_count"] == 8
    coverage = result.scenario_coverage
    assert coverage.get("classic_interactions") == 8


def test_incomplete_dataset_quarantines(tmp_path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    dataset_path = tmp_path / "missing_actions.npz"
    positions = np.zeros((2, 3, 2), dtype=float)
    np.savez(dataset_path, positions=positions)

    validator = TrajectoryDatasetValidator(dataset_path)
    result = validator.validate(minimum_episodes=4)

    assert result.quality_status == common.TrajectoryQuality.QUARANTINED
    assert "actions" in result.integrity_report["missing_arrays"]


def test_legacy_bc_dataset_without_reward_labels_remains_valid(tmp_path):
    """Base BC trajectory validation stays compatible unless DT fields are required."""
    dataset_path = tmp_path / "legacy_bc.npz"
    positions = np.zeros((2, 3, 2), dtype=float)
    actions = np.zeros((2, 3, 2), dtype=float)
    observations = np.zeros((2, 3, 5), dtype=float)
    np.savez(
        dataset_path,
        positions=positions,
        actions=actions,
        observations=observations,
        episode_count=np.array(2),
        metadata={"scenario_coverage": {"classic_interactions": 2}},
    )

    validator = TrajectoryDatasetValidator(dataset_path)
    legacy_result = validator.validate(minimum_episodes=2)
    dt_result = validator.validate(
        minimum_episodes=2,
        require_decision_transformer_fields=True,
    )

    assert legacy_result.quality_status == common.TrajectoryQuality.VALIDATED
    assert legacy_result.integrity_report["decision_transformer_preflight"] is False
    assert dt_result.quality_status == common.TrajectoryQuality.QUARANTINED
    assert "rewards" in dt_result.integrity_report["missing_arrays"]


def test_reward_label_misalignment_quarantines(tmp_path):
    """Decision Transformer preflight labels must align step-for-step."""
    dataset_path = tmp_path / "misaligned_rewards.npz"
    positions = np.zeros((1, 3, 2), dtype=float)
    actions = np.zeros((1, 3, 2), dtype=float)
    observations = np.zeros((1, 3, 5), dtype=float)
    rewards = np.zeros((1, 2), dtype=float)
    terminated = np.zeros((1, 3), dtype=bool)
    truncated = np.zeros((1, 3), dtype=bool)
    return_to_go = np.zeros((1, 3), dtype=float)
    np.savez(
        dataset_path,
        positions=positions,
        actions=actions,
        observations=observations,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        return_to_go=return_to_go,
        episode_count=np.array(1),
        metadata={"scenario_coverage": {"classic_interactions": 1}},
    )

    result = TrajectoryDatasetValidator(dataset_path).validate(minimum_episodes=1)

    assert result.quality_status == common.TrajectoryQuality.QUARANTINED
    assert result.integrity_report["alignment_issues"][0]["episode_index"] == 0
    assert result.integrity_report["alignment_issues"][0]["lengths"]["rewards"] == 2


def test_missing_reward_episode_quarantines(tmp_path):
    """Decision Transformer preflight labels must align episode-for-episode."""
    dataset_path = tmp_path / "missing_reward_episode.npz"
    positions = np.zeros((2, 3, 2), dtype=float)
    actions = np.zeros((2, 3, 2), dtype=float)
    observations = np.zeros((2, 3, 5), dtype=float)
    rewards = np.zeros((1, 3), dtype=float)
    terminated = np.zeros((2, 3), dtype=bool)
    truncated = np.zeros((2, 3), dtype=bool)
    return_to_go = np.zeros((1, 3), dtype=float)
    np.savez(
        dataset_path,
        positions=positions,
        actions=actions,
        observations=observations,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        return_to_go=return_to_go,
        episode_count=np.array(2),
        metadata={"scenario_coverage": {"classic_interactions": 2}},
    )

    result = TrajectoryDatasetValidator(dataset_path).validate(minimum_episodes=2)

    assert result.quality_status == common.TrajectoryQuality.QUARANTINED
    issue = result.integrity_report["alignment_issues"][0]
    assert issue["episode_index"] is None
    assert issue["expected_episodes"] == 2
    assert issue["array_episode_counts"]["rewards"] == 1


def test_unlabeled_fallback_rows_quarantine_dataset(tmp_path):
    """Fallback/degraded/not_available rows need an explicit exclusion policy."""
    dataset_path = tmp_path / "fallback_rows.npz"
    _write_npz_dataset(dataset_path, episode_count=1)
    with np.load(dataset_path, allow_pickle=True) as data:
        payload = {name: data[name] for name in data.files}
    payload["readiness_status"] = np.array(["fallback"], dtype=object)
    payload["metadata"] = {"scenario_coverage": {"classic_interactions": 1}}
    np.savez(dataset_path, **payload)

    result = TrajectoryDatasetValidator(dataset_path).validate(minimum_episodes=1)

    assert result.quality_status == common.TrajectoryQuality.QUARANTINED
    status_report = result.integrity_report["status_report"]
    assert status_report["excluded_rows"] == 1
    assert status_report["unlabeled_excluded_rows"] == 1


def test_jsonl_dataset_drafts_when_small(tmp_path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    dataset_path = tmp_path / "todo.jsonl_frames"
    dataset_path.write_text("{}\n{}\n", encoding="utf-8")

    validator = TrajectoryDatasetValidator(dataset_path)
    result = validator.validate(minimum_episodes=3)

    assert result.episode_count == 2
    assert result.quality_status == common.TrajectoryQuality.DRAFT
