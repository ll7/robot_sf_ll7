"""SAC config and dry-run wiring tests for issue #4012."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces as gym_spaces

from robot_sf.benchmark.rl_trajectory_dataset import write_rl_trajectory_dataset
from scripts.training import train_sac_sb3
from scripts.training.train_sac_sb3 import load_sac_training_config, run_sac_training
from tests.training.test_offline_online_rl import _episode


def test_sac_config_loads_offline_online_block(tmp_path: Path) -> None:
    """SAC config parser accepts the opt-in offline-online block."""

    dataset_path = tmp_path / "dataset.jsonl"
    cfg = _config(tmp_path, dataset_path, enabled=True)

    config = load_sac_training_config(cfg)

    assert config.offline_online.enabled
    assert config.offline_online.dataset_path == dataset_path
    assert config.offline_online.min_transitions == 1


def test_sac_config_rejects_unknown_offline_online_key(tmp_path: Path) -> None:
    """Misspelled offline-online config fields fail closed."""

    dataset_path = tmp_path / "dataset.jsonl"
    cfg = _config(tmp_path, dataset_path, enabled=True, extra="  typo: true\n")

    with pytest.raises(ValueError, match="Unknown offline_online"):
        load_sac_training_config(cfg)


def test_sac_config_offline_online_disabled_preserves_defaults(tmp_path: Path) -> None:
    """Existing SAC configs keep offline-online disabled by default."""

    cfg = _config(tmp_path, tmp_path / "dataset.jsonl", enabled=False)

    config = load_sac_training_config(cfg)

    assert not config.offline_online.enabled
    assert config.offline_online.dataset_path is None


def test_sac_training_dry_run_validates_and_seeds_offline_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dry-run validates offline rows and seeds replay before learning."""

    dataset_path = tmp_path / "dataset.jsonl"
    write_rl_trajectory_dataset([_episode("train", "ep-train")], dataset_path)
    config = load_sac_training_config(_config(tmp_path, dataset_path, enabled=True))
    config.output_dir = tmp_path / "checkpoints"
    _FakeSAC.install(monkeypatch)
    monkeypatch.setattr(train_sac_sb3, "load_scenarios", lambda _path: [{"name": "unit"}])
    monkeypatch.setattr(train_sac_sb3, "_build_env", lambda *_args, **_kwargs: _FakeVecEnv())

    checkpoint = run_sac_training(config, dry_run=True)

    assert checkpoint == tmp_path / "checkpoints" / "issue_4012_unit.zip"
    assert _FakeSAC.latest is not None
    assert _FakeSAC.latest.replay_buffer.add_count == 1
    assert _FakeSAC.latest.learn_calls == [train_sac_sb3._DRY_RUN_TIMESTEPS]


def test_sac_training_fails_closed_on_offline_action_space_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incompatible offline actions stop before online SAC learning."""

    dataset_path = tmp_path / "dataset.jsonl"
    write_rl_trajectory_dataset([_episode("train", "ep-train")], dataset_path)
    config = load_sac_training_config(_config(tmp_path, dataset_path, enabled=True))
    _FakeSAC.install(monkeypatch)
    monkeypatch.setattr(train_sac_sb3, "load_scenarios", lambda _path: [{"name": "unit"}])
    monkeypatch.setattr(
        train_sac_sb3, "_build_env", lambda *_args, **_kwargs: _FakeVecEnv(action_high=0.05)
    )

    with pytest.raises(ValueError, match="action at transition 0"):
        run_sac_training(config, dry_run=True)

    assert _FakeSAC.latest is not None
    assert _FakeSAC.latest.learn_calls == []


def _config(
    tmp_path: Path,
    dataset_path: Path,
    *,
    enabled: bool,
    extra: str = "",
) -> Path:
    offline_block = (
        f"""offline_online:
  enabled: true
  dataset_path: {dataset_path}
  dataset_split: train
  min_transitions: 1
  offline_gradient_steps: 0
{extra}"""
        if enabled
        else ""
    )
    cfg = tmp_path / "sac.yaml"
    cfg.write_text(
        f"""policy_id: issue_4012_unit
scenario_config: {tmp_path / "scenarios.yaml"}
total_timesteps: 32
seed: 4012
sac_hyperparams:
  buffer_size: 64
  learning_starts: 0
  batch_size: 2
tracking:
  enabled: false
{offline_block}
""",
        encoding="utf-8",
    )
    return cfg


class _FakeReplayBuffer:
    def __init__(self) -> None:
        self.add_count = 0

    def add(self, *_args: Any, **_kwargs: Any) -> None:
        self.add_count += 1


class _FakeSAC:
    latest: _FakeSAC | None = None

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.replay_buffer = _FakeReplayBuffer()
        self.learn_calls: list[int] = []
        self.train_calls: list[tuple[int, int]] = []
        _FakeSAC.latest = self

    @classmethod
    def install(cls, monkeypatch: pytest.MonkeyPatch) -> None:
        cls.latest = None
        monkeypatch.setattr(train_sac_sb3, "SAC", cls)

    def learn(self, *, total_timesteps: int, **_kwargs: Any) -> None:
        self.learn_calls.append(total_timesteps)

    def train(self, *, gradient_steps: int, batch_size: int) -> None:
        self.train_calls.append((gradient_steps, batch_size))

    def save(self, path: str) -> None:
        Path(path).write_text("fake checkpoint\n", encoding="utf-8")


class _FakeVecEnv:
    def __init__(self, *, action_high: float = 1.0) -> None:
        self.observation_space = gym_spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32)
        self.action_space = gym_spaces.Box(-action_high, action_high, shape=(2,), dtype=np.float32)
        self.closed = False

    def close(self) -> None:
        self.closed = True
