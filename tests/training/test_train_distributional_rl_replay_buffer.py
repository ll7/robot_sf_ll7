"""Replay-buffer eviction tests for issue #6508.

These tests lock the bounded-deque replay contract introduced in
``scripts/training/train_distributional_rl.py``: the replay buffer caps at
``config.dqn.replay_size``, evicts the oldest appended transition first (FIFO),
and ``random.sample`` minibatch sampling still draws ``batch_size`` items from
the capped deque.
"""

from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path

import torch

from scripts.training.train_distributional_rl import (
    DQNConfig,
    _sample_replay,
    load_distributional_rl_training_config,
    run_distributional_rl_training,
)


def _write_config(
    tmp_path: Path,
    *,
    replay_size: int,
    total_timesteps: int,
    seed: int = 6508,
) -> Path:
    path = tmp_path / "qr_dqn.yaml"
    path.write_text(
        f"""
policy_id: qr_dqn_replay_test
algorithm: qr_dqn
scenario_config: configs/scenarios/sets/classic_cross_trap_subset.yaml
total_timesteps: {total_timesteps}
seed: {seed}
device: cpu
num_envs: 1
observation:
  synthetic_observation_dim: 4
action_lattice:
  linear_values: [0.0, 0.5]
  angular_values: [-0.5, 0.5]
  max_linear_speed: 0.5
  max_angular_speed: 0.5
critic:
  hidden_sizes: [8]
  num_quantiles: 4
  target_update_interval: 4
risk_selection:
  objective: cvar_lower
  alpha: 0.5
dqn:
  replay_size: {replay_size}
  batch_size: 2
  learning_starts: 2
  train_freq: 1
  gradient_steps: 1
output_dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )
    return path


def _transition(
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a deterministic transition 5-tuple tagged by ``seed``."""

    return (
        torch.tensor([float(seed)], dtype=torch.float32),
        torch.tensor(seed, dtype=torch.long),
        torch.tensor(float(seed), dtype=torch.float32),
        torch.tensor([float(seed)], dtype=torch.float32),
        torch.tensor(0.0, dtype=torch.float32),
    )


def test_replay_deque_caps_at_replay_size_and_evicts_fifo() -> None:
    """The trainer's replay construction caps at replay_size, evicting oldest-first (FIFO)."""

    config = DQNConfig(replay_size=4, batch_size=2)
    # Mirror the trainer's construction exactly: collections.deque(maxlen=replay_size).
    replay: deque[int] = deque(maxlen=config.replay_size)
    appended = list(range(config.replay_size + 5))
    for value in appended:
        replay.append(value)

    assert len(replay) == config.replay_size
    # The oldest appended items are evicted first; the newest replay_size items remain.
    assert list(replay) == appended[-config.replay_size :]


def test_random_sample_draws_batch_size_from_capped_deque() -> None:
    """``random.sample`` minibatch sampling remains valid on the capped deque."""

    config = DQNConfig(replay_size=3, batch_size=2)
    replay: deque[int] = deque(maxlen=config.replay_size)
    for value in range(config.replay_size * 3):
        replay.append(value)

    random.seed(6508)
    batch = random.sample(replay, k=config.batch_size)

    assert len(batch) == config.batch_size
    assert set(batch).issubset(set(replay))


def test_sample_replay_stacks_batch_from_bounded_deque() -> None:
    """The trainer's own ``_sample_replay`` path works on a bounded deque of transitions."""

    batch_size = 4
    replay: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = (
        deque(maxlen=8)
    )
    for seed in range(8):
        replay.append(_transition(seed))

    observations, actions, rewards, next_observations, dones = _sample_replay(replay, batch_size)

    assert observations.shape == (batch_size, 1)
    assert actions.shape == (batch_size,)
    assert rewards.shape == (batch_size,)
    assert next_observations.shape == (batch_size, 1)
    assert dones.shape == (batch_size,)
    assert set(actions.tolist()).issubset(set(range(8)))


def test_training_with_eviction_active_is_deterministic(tmp_path: Path) -> None:
    """With total_timesteps >> replay_size the bounded-deque loop stays stable and reproducible."""

    config = load_distributional_rl_training_config(
        _write_config(tmp_path, replay_size=4, total_timesteps=12)
    )
    # Eviction is genuinely exercised: more steps than the buffer can hold.
    assert config.total_timesteps > config.dqn.replay_size

    first = run_distributional_rl_training(config, dry_run=False)
    first_trace = Path(first["training_trace_path"]).read_text(encoding="utf-8").strip()

    second = run_distributional_rl_training(config, dry_run=False)
    second_trace = Path(second["training_trace_path"]).read_text(encoding="utf-8").strip()

    assert first["train_steps"] > 0
    assert first["train_steps"] == second["train_steps"]
    assert first_trace == second_trace
    rows = [json.loads(line) for line in first_trace.splitlines()]
    assert rows
    assert all("loss" in row and "train_step" in row for row in rows)
