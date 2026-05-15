"""Regression tests for PPO fine-tuning dataset metadata helpers.

These tests verify fail-closed behavior for dataset metadata resolution because the PPO warm-start
path reconstructs scenario and observation contracts from the stored dataset metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.training.train_ppo_with_pretrained_policy import _load_dataset_metadata

if TYPE_CHECKING:
    from pathlib import Path


def test_load_dataset_metadata_rejects_directory_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject directory dataset paths so invalid metadata inputs do not silently look empty."""
    dataset_dir = tmp_path / "dataset_dir"
    dataset_dir.mkdir()

    monkeypatch.setattr(
        "scripts.training.train_ppo_with_pretrained_policy.common.get_trajectory_dataset_path",
        lambda dataset_id: dataset_dir,
    )

    with pytest.raises(FileNotFoundError, match="not a file"):
        _load_dataset_metadata("demo-dataset")
