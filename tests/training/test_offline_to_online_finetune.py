"""Tests for manifest-driven offline-to-online fine-tuning."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces as gym_spaces

from robot_sf.training.offline_pretraining_manifest import (
    build_offline_checkpoint_manifest,
    space_fingerprint,
    write_normalizer_state,
)
from scripts.training import offline_to_online_finetune


def test_offline_to_online_finetune_records_parent_manifest_chain(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Fine-tune CLI records parent manifest and output checkpoint hashes."""

    parent_path = _parent_manifest(tmp_path, action_space=_FakeVecEnv.action_space)
    config_path = tmp_path / "finetune.yaml"
    config_path.write_text("policy_id: fine\n", encoding="utf-8")
    monkeypatch.setattr(
        offline_to_online_finetune,
        "load_sac_training_config",
        lambda _path: _config(tmp_path),
    )
    monkeypatch.setattr(
        offline_to_online_finetune, "load_scenarios", lambda _path: [{"name": "unit"}]
    )
    monkeypatch.setattr(
        offline_to_online_finetune, "_build_env", lambda *_args, **_kwargs: _FakeVecEnv()
    )
    monkeypatch.setattr(offline_to_online_finetune, "SAC", _FakeSAC)

    manifest = offline_to_online_finetune.offline_to_online_finetune(
        config_path=config_path,
        pretrained_manifest=parent_path,
        output_dir=tmp_path / "out",
        manifest_out=tmp_path / "out" / "finetune_manifest.json",
    )

    assert manifest["schema_version"] == "offline-to-online-finetune-manifest.v1"
    assert manifest["parent_offline_checkpoint_manifest_path"] == str(parent_path.resolve())
    assert Path(str(manifest["checkpoint_path"])).is_file()
    assert manifest["eligible_for_claim"] is False
    assert _FakeSAC.latest is not None
    assert _FakeSAC.latest.learn_calls == [1]


def test_offline_to_online_finetune_rejects_environment_mismatch(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Action-space fingerprint mismatch fails closed."""

    parent_path = _parent_manifest(
        tmp_path,
        action_space=gym_spaces.Box(-0.5, 0.5, shape=(2,), dtype=np.float32),
    )
    config_path = tmp_path / "finetune.yaml"
    config_path.write_text("policy_id: fine\n", encoding="utf-8")
    monkeypatch.setattr(
        offline_to_online_finetune,
        "load_sac_training_config",
        lambda _path: _config(tmp_path),
    )
    monkeypatch.setattr(
        offline_to_online_finetune, "load_scenarios", lambda _path: [{"name": "unit"}]
    )
    monkeypatch.setattr(
        offline_to_online_finetune, "_build_env", lambda *_args, **_kwargs: _FakeVecEnv()
    )

    with pytest.raises(ValueError, match="action_space_fingerprint mismatch"):
        offline_to_online_finetune.offline_to_online_finetune(
            config_path=config_path,
            pretrained_manifest=parent_path,
            output_dir=tmp_path / "out",
            manifest_out=tmp_path / "out" / "finetune_manifest.json",
        )


def _parent_manifest(tmp_path: Path, *, action_space: gym_spaces.Space[Any]) -> Path:
    checkpoint = _file(tmp_path / "checkpoint.zip", "checkpoint\n")
    normalizer = _normalizer_state(tmp_path / "normalizer.json")
    config = _file(tmp_path / "pretrain.yaml", "policy_id: pretrain\n")
    dataset = _file(tmp_path / "dataset.jsonl", "{}\n")
    dataset_manifest = _file(
        tmp_path / "dataset.manifest.json", '{"dataset_id":"issue_4245_unit"}\n'
    )
    manifest = build_offline_checkpoint_manifest(
        checkpoint_path=checkpoint,
        normalizer_path=normalizer,
        training_config_path=config,
        dataset={
            "dataset_path": str(dataset),
            "dataset_manifest_path": str(dataset_manifest),
            "dataset_sha256": _sha(dataset),
            "dataset_manifest_sha256": _sha(dataset_manifest),
            "dataset_schema_version": "RLTrajectoryDataset.v1",
            "dataset_id": "issue_4245_unit",
            "split": "train",
            "episode_count": 1,
            "accepted_transitions": 2,
            "dropped_terminal_transitions": 0,
        },
        offline_training={
            "gradient_steps": 1,
            "batch_size": 2,
            "seed": 4245,
            "observation_contract": "dataset_observation",
            "action_contract": "box_direct",
        },
        environment_contract={
            "scenario_config": "unit.yaml",
            "observation_space_fingerprint": space_fingerprint(_FakeVecEnv.observation_space),
            "action_space_fingerprint": space_fingerprint(action_space),
        },
        policy_type="MlpPolicy",
        created_at_utc="2026-07-03T00:00:00Z",
    )
    manifest_path = tmp_path / "offline_manifest.json"
    import json

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        policy_id="issue_4245_fine",
        scenario_config=tmp_path / "scenarios.yaml",
        total_timesteps=1,
        seed=4245,
        device="cpu",
    )


def _file(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _normalizer_state(path: Path) -> Path:
    write_normalizer_state(path, present=False, reason="unit test explicit absence")
    return path


def _sha(path: Path) -> str:
    from robot_sf.benchmark.rl_trajectory_dataset import sha256_file

    return sha256_file(path)


class _FakeSAC:
    latest: _FakeSAC | None = None

    def __init__(self) -> None:
        self.learn_calls: list[int] = []
        _FakeSAC.latest = self

    @classmethod
    def load(cls, *_args: Any, **_kwargs: Any) -> _FakeSAC:
        return cls()

    def learn(self, *, total_timesteps: int, **_kwargs: Any) -> None:
        self.learn_calls.append(total_timesteps)

    def save(self, path: str) -> None:
        Path(path).write_text("finetuned\n", encoding="utf-8")


class _FakeVecEnv:
    observation_space = gym_spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32)
    action_space = gym_spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def close(self) -> None:
        pass
