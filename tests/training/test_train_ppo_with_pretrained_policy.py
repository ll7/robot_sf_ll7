"""Regression tests for PPO fine-tuning dataset metadata helpers.

These tests verify fail-closed behavior for dataset metadata resolution because the PPO warm-start
path reconstructs scenario and observation contracts from the stored dataset metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf.training.imitation_config import PPOFineTuningConfig
from scripts.training.train_ppo_with_pretrained_policy import (
    _checkpoint_callback,
    _describe_num_envs_resolution,
    _load_dataset_metadata,
    _parse_num_envs,
    _resolve_num_envs,
)

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


def test_parse_num_envs_supports_auto_and_fixed_values() -> None:
    """Fine-tune YAML parser should accept the same common env-count forms."""
    assert _parse_num_envs(None) is None
    assert _parse_num_envs("auto") == "auto_throughput"
    assert _parse_num_envs("auto_throughput") == "auto_throughput"
    assert _parse_num_envs(22) == 22


def test_resolve_num_envs_defaults_to_legacy_single_env() -> None:
    """Existing fine-tune configs without num_envs keep their single-env behavior."""
    config = PPOFineTuningConfig.from_raw(
        run_id="demo",
        pretrained_policy_id="bc",
        total_timesteps=1000,
        random_seeds=(1,),
    )

    details = _describe_num_envs_resolution(config)

    assert _resolve_num_envs(config) == 1
    assert details["mode"] == "single_env_legacy"


def test_checkpoint_callback_divides_frequency_by_vec_env_count(tmp_path: Path) -> None:
    """Checkpoint cadence is configured in aggregate SB3 timesteps, not raw env calls."""
    config = PPOFineTuningConfig.from_raw(
        run_id="demo",
        pretrained_policy_id="bc",
        total_timesteps=1000,
        random_seeds=(1,),
        checkpoint_freq=500_000,
        checkpoint_dir=tmp_path,
    )

    callback = _checkpoint_callback(config, num_envs=22)

    assert callback is not None
    assert callback.save_freq == 22_727
    assert callback.save_path == str(tmp_path)
