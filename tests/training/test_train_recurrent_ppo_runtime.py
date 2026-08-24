"""Production runtime-contract tests for the #7847 RecurrentPPO lane."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robot_sf.training.recurrent_runtime import RecurrentStateError
from scripts.training import train_recurrent_ppo

CONFIG_PATH = "configs/training/ppo/issue_4014_recurrent_ppo_lstm_smoke_matched.yaml"


class _StubRecurrentModel:
    """Minimal RecurrentPPO stand-in exposing the predict/save contract."""

    def __init__(self, fail_non_finite: bool = False) -> None:
        self.fail_non_finite = fail_non_finite
        self.save_calls: list[Path] = []

    def predict(
        self,
        obs: Any,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = True,
    ) -> tuple[Any, Any]:
        del obs, episode_start, deterministic
        if self.fail_non_finite:
            action = np.array([[float("nan"), float("inf")]], dtype=np.float32)
            return action, state
        action = np.zeros((1, 2), dtype=np.float32)
        hidden_shape = (1, 1, 4)
        if state is None:
            state = (
                np.ones(hidden_shape, dtype=np.float32),
                np.zeros(hidden_shape, dtype=np.float32),
            )
        return action, state

    def save(self, path: str | Path) -> None:
        self.save_calls.append(Path(path))
        Path(path).write_bytes(b"stub-model")


class _StubVecEnv:
    """Single-index vectorized environment returning terminal episodes."""

    def __init__(self, episodes_to_terminate: int) -> None:
        self.episodes_to_terminate = episodes_to_terminate
        self.step_count = 0
        self.resets = 0

    def reset(self) -> Any:
        self.resets += 1
        self.step_count = 0
        return np.zeros((1, 4), dtype=np.float32)

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
        del action
        self.step_count += 1
        terminated = self.step_count >= 3
        obs = np.zeros((1, 4), dtype=np.float32)
        reward = np.array([0.5], dtype=np.float32)
        return obs, reward, terminated, False, {}

    def close(self) -> None:
        return None


def _load_config() -> train_recurrent_ppo.RecurrentPPOConfig:
    return train_recurrent_ppo.load_recurrent_ppo_config(CONFIG_PATH)


def test_plan_seed_runs_single_seed_uses_base_directory(tmp_path: Path) -> None:
    """A single-seed config keeps the plain run id and output directory."""
    plan = train_recurrent_ppo.plan_seed_runs(
        config=_load_config(),
        config_path=Path(CONFIG_PATH),
        run_id="smoke",
        output_dir=tmp_path,
    )
    assert len(plan) == 1
    assert plan[0].seed == 4014
    assert plan[0].run_id == "smoke"
    assert plan[0].output_dir == tmp_path


def test_evaluate_recurrently_propagates_state_and_counts_resets() -> None:
    """Stubbed evaluation records resets per boundary and returns a summary."""
    model = _StubRecurrentModel()
    env = _StubVecEnv(episodes_to_terminate=3)
    summary = train_recurrent_ppo._evaluate_recurrently(
        model=model,
        eval_env=env,
        episodes=2,
    )
    assert summary["episodes"] == 2
    assert summary["mean_episode_return"] == pytest.approx(3 * 0.5)
    assert summary["mean_episode_length"] == 3.0
    assert summary["reset_counts"]["env_reset"] == 2
    assert summary["reset_counts"]["terminated"] == 2
    assert summary["reset_counts"]["truncated"] == 0
    assert summary["non_finite_action_count"] == 0
    assert summary["state_norms"]["hidden_norm_max"] > 0.0


def test_evaluate_recurrently_fails_closed_on_non_finite_actions() -> None:
    """Non-finite evaluation actions stop the run instead of recording them."""
    model = _StubRecurrentModel(fail_non_finite=True)
    env = _StubVecEnv(episodes_to_terminate=3)
    with pytest.raises(RecurrentStateError, match="non-finite actions"):
        train_recurrent_ppo._evaluate_recurrently(model=model, eval_env=env, episodes=1)


def test_resume_identity_mismatch_rejected(tmp_path: Path) -> None:
    """Resume from an unrelated run fails closed on identity mismatch."""
    config = _load_config()
    identity = {
        **train_recurrent_ppo._checkpoint_identity_payload(config, "deadbeef"),
        "seed": 4014,
        "run_id": "prior-run",
        "completed_timesteps": 1024,
    }
    (tmp_path / "run_identity.json").write_text(
        json.dumps(identity),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        train_recurrent_ppo._validate_resume_identity(
            run_output_dir=tmp_path,
            config=config,
            source_sha="different-sha",
        )


def test_resume_step_regression_rejected() -> None:
    """A resumed step count below the prior count is rejected."""
    prior = {"completed_timesteps": 2048}
    with pytest.raises(RuntimeError, match="step regression"):
        train_recurrent_ppo._record_resume_boundary(prior=prior, resumed_steps=1024)


def test_missing_resume_identity_rejected(tmp_path: Path) -> None:
    """Only runs produced by this lane are resumable."""
    with pytest.raises(RuntimeError, match="missing run_identity.json"):
        train_recurrent_ppo._validate_resume_identity(
            run_output_dir=tmp_path,
            config=_load_config(),
            source_sha="deadbeef",
        )
