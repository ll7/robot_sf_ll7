"""Tests for Decision Transformer trajectory preflight fields."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from gymnasium import spaces

from scripts.training import collect_expert_trajectories as collector


class _FakeEnv:
    """Small deterministic env for collector unit tests."""

    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def __init__(self) -> None:
        self.state = SimpleNamespace(max_sim_steps=3, nav=SimpleNamespace(pos=(0.0, 0.0)))
        self._step = 0

    def reset(self):
        self._step = 0
        return {"obs": np.array([0.0], dtype=np.float32)}, {}

    def step(self, action):
        self._step += 1
        self.state.nav.pos = (float(self._step), float(self._step + 1))
        reward = float(self._step)
        terminated = self._step == 3
        truncated = False
        return {"obs": np.array([self._step], dtype=np.float32)}, reward, terminated, truncated, {}


def test_record_episode_exports_reward_terminal_and_return_to_go_fields() -> None:
    """Recorded episodes include the labels needed by offline sequence models."""
    record = collector._record_episode(_FakeEnv(), policy=None, dry_run=True)

    assert record["rewards"] == [1.0, 2.0, 3.0]
    assert record["terminated"] == [False, False, True]
    assert record["truncated"] == [False, False, False]
    assert record["return_to_go"] == [6.0, 5.0, 3.0]
    assert len(record["actions"]) == len(record["observations"]) == 3


def test_write_dataset_persists_decision_transformer_arrays(tmp_path) -> None:
    """The NPZ payload keeps reward, terminal, and return labels alongside BC fields."""
    dataset_path = tmp_path / "dt_preflight.npz"
    arrays = {
        "positions": [np.array([(1.0, 2.0), (2.0, 3.0)], dtype=object)],
        "actions": [np.zeros((2, 2), dtype=float)],
        "observations": [np.array([{"obs": 1}, {"obs": 2}], dtype=object)],
        "rewards": [np.array([1.0, -0.5], dtype=float)],
        "terminated": [np.array([False, True], dtype=bool)],
        "truncated": [np.array([False, False], dtype=bool)],
        "return_to_go": [np.array([0.5, -0.5], dtype=float)],
        "episode_ids": [np.array(["demo:episode_000000"], dtype=object)],
        "scenario_ids": [np.array(["demo"], dtype=object)],
        "seeds": [np.array([11], dtype=int)],
    }
    metadata = {
        "dataset_schema": "trajectory_dataset.v2.decision_transformer_preflight",
        "reward_convention": collector.DECISION_TRANSFORMER_REWARD_CONVENTION,
        "return_convention": collector.DECISION_TRANSFORMER_RETURN_CONVENTION,
        "status_policy": collector.DECISION_TRANSFORMER_STATUS_POLICY,
    }

    collector._write_dataset(dataset_path, arrays, episode_count=1, metadata=metadata)

    with np.load(dataset_path, allow_pickle=True) as data:
        assert data["rewards"][0].tolist() == [1.0, -0.5]
        assert data["terminated"][0].tolist() == [False, True]
        assert data["truncated"][0].tolist() == [False, False]
        assert data["return_to_go"][0].tolist() == [0.5, -0.5]
        stored_metadata = data["metadata"].item()

    assert stored_metadata["reward_convention"] == "environment_step_reward"
    assert stored_metadata["status_policy"]["availability_status_excluded"] == [
        "not_available"
    ]
