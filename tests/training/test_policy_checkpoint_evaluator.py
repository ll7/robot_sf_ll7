"""Tests for the shared policy checkpoint evaluator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from robot_sf.training.policy_checkpoint_evaluator import evaluate_policy_episodes


class _EpisodeEnv:
    """Tiny deterministic env stub that exposes one configured termination mode."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.state = SimpleNamespace(max_sim_steps=1)
        self._done = False

    def reset(self):
        self._done = False
        return 0.0, {}

    def step(self, _action):
        self._done = True
        meta = {
            "step_of_episode": 1,
            "max_sim_steps": 1,
            "is_route_complete": self.mode == "success",
            "is_pedestrian_collision": self.mode == "collision",
            "is_robot_collision": False,
            "is_obstacle_collision": False,
            "is_timesteps_exceeded": self.mode == "timeout",
        }
        terminated = self.mode in {"success", "collision"}
        return 0.0, 1.0, terminated, False, {"meta": meta}

    def close(self) -> None:
        return None


def test_evaluate_policy_episodes_reports_termination_reason_breakdown() -> None:
    """Standardized evaluation should aggregate success/collision/timeout outcomes."""
    modes = ["success", "collision", "timeout"]

    def _make_env(episode_idx: int, _seed: int | None):
        return _EpisodeEnv(modes[episode_idx]), f"scenario-{episode_idx}"

    result = evaluate_policy_episodes(
        episodes=3,
        make_env=_make_env,
        action_fn=lambda _obs: 0.0,
        eval_step=12,
    )

    assert result.summary["episodes"] == 3
    assert result.summary["termination_reason_counts"] == {
        "collision": 1,
        "max_steps": 1,
        "success": 1,
    }
    assert result.summary["metric_means"]["success_rate"] == pytest.approx(1.0 / 3.0)
    assert result.summary["metric_means"]["collision_rate"] == pytest.approx(1.0 / 3.0)
    assert result.episode_records[0]["termination_reason"] == "success"
    assert result.episode_records[1]["termination_reason"] == "collision"
    assert result.episode_records[2]["termination_reason"] == "max_steps"
