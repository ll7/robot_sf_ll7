"""Tests for the standalone offline pretraining CLI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from gymnasium import spaces as gym_spaces

from robot_sf.benchmark.rl_trajectory_dataset import (
    RLTrajectoryEpisode,
    compute_return_to_go,
    write_rl_trajectory_dataset,
)
from scripts.training import pretrain_offline_policy


def test_pretrain_offline_policy_writes_checkpoint_normalizer_manifest(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Offline pretrain CLI emits the durable manifest handoff."""

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_manifest_path = tmp_path / "dataset.manifest.json"
    write_rl_trajectory_dataset([_episode()], dataset_path)
    dataset_manifest_path.write_text(
        '{"schema_version":"rl_trajectory_dataset_manifest.v1","dataset_id":"issue_4245_unit"}\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "pretrain.yaml"
    config_path.write_text("policy_id: issue_4245_unit\n", encoding="utf-8")
    config = _config(tmp_path, dataset_path, dataset_manifest_path)

    monkeypatch.setattr(pretrain_offline_policy, "load_sac_training_config", lambda _path: config)
    monkeypatch.setattr(pretrain_offline_policy, "load_scenarios", lambda _path: [{"name": "unit"}])
    monkeypatch.setattr(
        pretrain_offline_policy, "_build_env", lambda *_args, **_kwargs: _FakeVecEnv()
    )
    monkeypatch.setattr(pretrain_offline_policy, "SAC", _FakeSAC)
    monkeypatch.setattr(
        pretrain_offline_policy, "_seed_offline_online_replay_buffer", lambda *a, **k: None
    )

    manifest = pretrain_offline_policy.pretrain_offline_policy(
        config_path=config_path,
        output_dir=tmp_path / "out",
        manifest_out=tmp_path / "out" / "offline_manifest.json",
    )

    assert manifest["schema_version"] == "offline-policy-checkpoint-manifest.v1"
    assert Path(str(manifest["checkpoint_path"])).is_file()
    assert Path(str(manifest["normalizer_path"])).is_file()
    assert manifest["dataset"]["dataset_id"] == "issue_4245_unit"
    assert manifest["eligible_for_claim"] is False
    assert _FakeSAC.latest is not None
    assert _FakeSAC.latest.train_calls == [(1, 2)]


def _config(tmp_path: Path, dataset_path: Path, manifest_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        policy_id="issue_4245_unit",
        scenario_config=tmp_path / "scenarios.yaml",
        total_timesteps=1,
        sac_hyperparams={"batch_size": 2, "buffer_size": 8, "learning_starts": 0},
        output_dir=tmp_path / "checkpoints",
        seed=4245,
        device="cpu",
        obs_transform="none",
        offline_online=SimpleNamespace(
            enabled=True,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            dataset_split="train",
            min_transitions=2,
            offline_gradient_steps=1,
            action_contract="box_direct",
            observation_contract="dataset_observation",
        ),
    )


def _episode() -> RLTrajectoryEpisode:
    rewards = (1.0, 1.0)
    return RLTrajectoryEpisode(
        dataset_id="issue_4245_unit",
        episode_id="ep-train",
        scenario_id="unit",
        seed=4245,
        source_policy_id="unit",
        split="train",
        observations=([0.0, 0.0], [0.1, 0.1]),
        actions=([0.0, 0.0], [0.0, 0.0]),
        rewards=rewards,
        return_to_go=tuple(compute_return_to_go(rewards)),
        terminated=(False, True),
        truncated=(False, False),
        pedestrians=({}, {}),
        robot_states=({}, {}),
        provenance={"issue": 4245},
    )


class _FakeSAC:
    latest: _FakeSAC | None = None

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.train_calls: list[tuple[int, int]] = []
        _FakeSAC.latest = self

    def train(self, *, gradient_steps: int, batch_size: int) -> None:
        self.train_calls.append((gradient_steps, batch_size))

    def save(self, path: str) -> None:
        Path(path).write_text("checkpoint\n", encoding="utf-8")


class _FakeVecEnv:
    observation_space = gym_spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32)
    action_space = gym_spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def close(self) -> None:
        pass
