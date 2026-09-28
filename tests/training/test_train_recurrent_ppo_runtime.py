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


class _SegmentTrainingModel(_StubRecurrentModel):
    """Stub model for checkpoint selection over scheduled train segments."""

    def __init__(self) -> None:
        super().__init__()
        self.learned = 0

    def learn(self, total_timesteps: int, reset_num_timesteps: bool = False) -> None:
        del reset_num_timesteps
        self.learned += int(total_timesteps)


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


class _MetricVecEnv:
    """Single-index VecEnv stub exposing terminal metadata for metric checks."""

    def __init__(self, terminal_metas: list[dict[str, object]], *, legacy_api: bool) -> None:
        self.terminal_metas = terminal_metas
        self.legacy_api = legacy_api
        self.step_count = 0
        self.resets = 0

    def reset(self) -> Any:
        self.resets += 1
        self.step_count = 0
        return np.zeros((1, 4), dtype=np.float32)

    def step(self, action: Any) -> tuple[Any, ...]:
        del action
        self.step_count += 1
        done = self.step_count >= 2
        obs = np.zeros((1, 4), dtype=np.float32)
        reward = np.array([1.0], dtype=np.float32)
        info: dict[str, object] = {}
        if done:
            info = {"meta": self.terminal_metas[self.resets - 1]}
        if self.legacy_api:
            return obs, reward, np.asarray([done]), [info]
        return obs, reward, np.asarray([done]), np.asarray([False]), [info]

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


@pytest.mark.parametrize("legacy_api", [False, True])
def test_evaluate_recurrently_emits_native_success_and_collision_metrics(
    legacy_api: bool,
) -> None:
    """Both VecEnv step APIs preserve terminal metadata for native metrics."""
    model = _StubRecurrentModel()
    env = _MetricVecEnv(
        [
            {
                "is_route_complete": True,
                "is_pedestrian_collision": False,
                "is_robot_collision": False,
                "is_obstacle_collision": False,
                "max_sim_steps": 2,
            },
            {
                "is_route_complete": False,
                "is_pedestrian_collision": True,
                "is_robot_collision": False,
                "is_obstacle_collision": False,
                "max_sim_steps": 2,
            },
        ],
        legacy_api=legacy_api,
    )

    summary = train_recurrent_ppo._evaluate_recurrently(model=model, eval_env=env, episodes=2)

    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["collision_rate"] == pytest.approx(0.5)
    assert summary["eval_episode_return"] == pytest.approx(2.0)
    assert [row["success_rate"] for row in summary["episode_metrics"]] == [1.0, 0.0]
    assert [row["collision_rate"] for row in summary["episode_metrics"]] == [0.0, 1.0]


def test_checkpoint_selection_uses_configured_native_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpoint records use success_rate instead of a return proxy."""
    config = _load_config()
    model = _SegmentTrainingModel()
    summaries = iter(
        [
            {
                "episodes": 1,
                "mean_episode_return": 99.0,
                "mean_episode_length": 2.0,
                "reset_counts": {},
                "state_norms": None,
                "non_finite_action_count": 0,
                "episode_metrics": [],
                "success_rate": 0.25,
                "collision_rate": 0.0,
                "path_efficiency": 0.5,
                "comfort_exposure": 0.0,
                "snqi": 0.25,
                "eval_episode_return": 99.0,
                "eval_avg_step_reward": 1.0,
            },
            {
                "episodes": 1,
                "mean_episode_return": 1.0,
                "mean_episode_length": 2.0,
                "reset_counts": {},
                "state_norms": None,
                "non_finite_action_count": 0,
                "episode_metrics": [],
                "success_rate": 0.75,
                "collision_rate": 0.0,
                "path_efficiency": 0.5,
                "comfort_exposure": 0.0,
                "snqi": 0.75,
                "eval_episode_return": 1.0,
                "eval_avg_step_reward": 0.5,
            },
        ],
    )
    monkeypatch.setattr(
        train_recurrent_ppo,
        "_evaluate_recurrently",
        lambda **_: next(summaries),
    )
    output_dir = tmp_path / "segments"
    output_dir.mkdir()

    total_learned, best_score, best_path, index_entries = (
        train_recurrent_ppo._train_and_evaluate_segments(
            model=model,
            output_dir=output_dir,
            eval_vec_env=object(),
            config=config,
            hyperparams={},
            metric_name="success_rate",
            higher_is_better=True,
            source_sha="source-sha",
            seed=123,
            snqi_context=train_recurrent_ppo.train_ppo.resolve_training_snqi_context(),
        )
    )

    assert total_learned == config.base.total_timesteps == 2_048
    assert model.learned == 2_048
    assert best_score == pytest.approx(0.75)
    assert best_path == output_dir / "best.zip"
    best_entry = next(entry for entry in index_entries if entry["kind"] == "best")
    assert best_entry["metric"] == "success_rate"
    assert best_entry["score"] == pytest.approx(0.75)
    history = [
        json.loads(line)
        for line in (output_dir / "evaluation_history.jsonl").read_text().splitlines()
    ]
    assert [row["metric"] for row in history] == ["success_rate", "success_rate"]
    assert [row["score"] for row in history] == pytest.approx([0.25, 0.75])
    assert [row["mean_episode_return"] for row in history] == [99.0, 1.0]


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
