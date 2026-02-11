"""Tests for Optuna runner W&B defaults and trial grouping metadata."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from robot_sf.training.imitation_config import EvaluationSchedule
from scripts.training.optuna_expert_ppo import _build_trial_config


@dataclass
class _BaseConfigStub:
    policy_id: str = "ppo_expert_reference"
    best_checkpoint_metric: str = "eval_episode_return"
    total_timesteps: int = 100_000
    evaluation: EvaluationSchedule = field(
        default_factory=lambda: EvaluationSchedule(
            frequency_episodes=10,
            evaluation_episodes=5,
            hold_out_scenarios=(),
            step_schedule=(),
        )
    )
    randomize_seeds: bool = False
    seeds: tuple[int, ...] = (123,)
    tracking: dict[str, object] | None = None
    ppo_hyperparams: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.tracking is None:
            self.tracking = {}
        if self.ppo_hyperparams is None:
            self.ppo_hyperparams = {}


class _FakeTrial:
    def __init__(self, number: int = 3) -> None:
        self.number = number

    def suggest_categorical(self, _name: str, choices):
        return choices[0]

    def suggest_float(self, _name: str, low: float, _high: float, **_kwargs):
        return low

    def suggest_int(self, _name: str, low: int, _high: int, **_kwargs):
        return low


def _args(*, disable_wandb: bool) -> argparse.Namespace:
    return argparse.Namespace(
        trial_timesteps=1_000_000,
        eval_episodes=10,
        eval_every=250_000,
        deterministic=False,
        disable_wandb=disable_wandb,
    )


def test_build_trial_config_enables_and_groups_wandb_by_default():
    """Optuna trials should enable W&B and set study-centric grouping metadata."""
    config = _build_trial_config(
        _FakeTrial(number=7),
        base_config=_BaseConfigStub(),
        metric_name="eval_episode_return",
        objective_mode="last_n_mean",
        study_name="weekend_optuna_test",
        args=_args(disable_wandb=False),
        resolve_num_envs_fn=lambda _cfg: 2,
        evaluation_schedule_cls=EvaluationSchedule,
    )

    wandb_cfg = dict(config.tracking["wandb"])  # type: ignore[index]
    assert wandb_cfg["enabled"] is True
    assert wandb_cfg["group"] == "weekend_optuna_test"
    assert wandb_cfg["job_type"] == "optuna-trial-last_n_mean"
    assert "optuna" in wandb_cfg["tags"]
    assert "metric:eval_episode_return" in wandb_cfg["tags"]
    assert str(wandb_cfg["name"]).endswith("_optuna_007")


def test_build_trial_config_can_disable_wandb():
    """disable_wandb should force W&B off even if base config enabled it."""
    base = _BaseConfigStub(tracking={"wandb": {"enabled": True, "project": "robot_sf"}})
    config = _build_trial_config(
        _FakeTrial(number=1),
        base_config=base,
        metric_name="eval_episode_return",
        objective_mode="auc",
        study_name="weekend_optuna_test",
        args=_args(disable_wandb=True),
        resolve_num_envs_fn=lambda _cfg: 1,
        evaluation_schedule_cls=EvaluationSchedule,
    )

    wandb_cfg = dict(config.tracking["wandb"])  # type: ignore[index]
    assert wandb_cfg["enabled"] is False
