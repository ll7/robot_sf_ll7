"""Contract tests for expert PPO training runtime helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

from robot_sf.training.imitation_config import (
    ConvergenceCriteria,
    EvaluationSchedule,
    ExpertTrainingConfig,
)
from scripts.training import train_ppo

if TYPE_CHECKING:
    from pathlib import Path


def test_warn_frequency_episodes_deprecated_warns_once(monkeypatch) -> None:
    """frequency_episodes deprecation warning should be emitted once per process."""
    calls: list[str] = []

    def _fake_warning(msg: str, *_args) -> None:
        calls.append(msg)

    monkeypatch.setattr(train_ppo.logger, "warning", _fake_warning)
    monkeypatch.setattr(train_ppo, "_FREQUENCY_EPISODES_DEPRECATION_WARNED", False)

    train_ppo._warn_frequency_episodes_deprecated(10)
    train_ppo._warn_frequency_episodes_deprecated(20)

    assert len(calls) == 1
    assert "ignored" in calls[0]


def test_legacy_training_ppo_entrypoint_fails_with_migration_command() -> None:
    """Legacy PPO entrypoint should fail closed before users launch invalid runs."""
    result = subprocess.run(
        [sys.executable, "scripts/training_ppo.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "scripts/training/train_ppo.py" in result.stderr
    assert "--config" in result.stderr
    assert "docs/training/ppo_training_workflow.md" in result.stderr


def _capture_startup_summary(monkeypatch, tmp_path: Path) -> str:
    """Capture startup summary logs for focused contract assertions."""
    messages: list[str] = []

    def _fake_info(message: str, *args) -> None:
        try:
            messages.append(message.format(*args) if args else message)
        except (IndexError, KeyError, ValueError):
            messages.append(f"{message} | args={args}")

    monkeypatch.setattr(train_ppo.logger, "info", _fake_info)
    monkeypatch.setattr(train_ppo, "_host_memory_gib", lambda: 64.0)
    monkeypatch.setattr(train_ppo, "_slurm_allocated_cpus", lambda: None)
    monkeypatch.setattr(train_ppo.os, "cpu_count", lambda: 12)

    config = ExpertTrainingConfig.from_raw(
        scenario_config=tmp_path / "scenarios.yaml",
        seeds=(123,),
        total_timesteps=120_000,
        policy_id="ppo_startup_summary_test",
        convergence=ConvergenceCriteria(
            success_rate=0.9,
            collision_rate=0.05,
            plateau_window=1000,
        ),
        evaluation=EvaluationSchedule(
            frequency_episodes=0,
            evaluation_episodes=4,
            step_schedule=((None, 60_000),),
            randomize_seeds=False,
        ),
        env_factory_kwargs={"reward_name": "route_completion_v3"},
        scenario_sampling={"strategy": "random"},
        num_envs="auto_stable",
        worker_mode="subproc",
        resume_model_id="ppo_registry_source",
    )

    train_ppo._log_startup_summary(
        config=config,
        config_path=tmp_path / "train.yaml",
        num_envs=3,
        worker_mode="subproc",
    )

    return "\n".join(messages)


def test_log_startup_summary_reports_run_identity(monkeypatch, tmp_path: Path) -> None:
    """Startup summary should report policy, config, and training horizon identity."""
    summary = _capture_startup_summary(monkeypatch, tmp_path)

    assert "Training startup summary" in summary
    assert "policy_id=ppo_startup_summary_test" in summary
    assert "config_path=" in summary
    assert "total_timesteps=120000" in summary


def test_log_startup_summary_reports_training_settings(monkeypatch, tmp_path: Path) -> None:
    """Startup summary should report resolved reward, worker, and resume settings."""
    summary = _capture_startup_summary(monkeypatch, tmp_path)

    assert "reward_profile=route_completion_v3" in summary
    assert "requested_num_envs=auto_stable" in summary
    assert "num_envs=3" in summary
    assert "worker_mode=subproc" in summary
    assert "model_id:ppo_registry_source" in summary


def test_log_startup_summary_reports_num_envs_resolution(monkeypatch, tmp_path: Path) -> None:
    """Startup summary should include the auto num-envs resolution explanation."""
    summary = _capture_startup_summary(monkeypatch, tmp_path)

    assert "num_envs resolution" in summary
    assert "mode=auto_stable" in summary


def test_write_perf_summary_writes_expected_keys(tmp_path: Path, monkeypatch) -> None:
    """Perf summary writer should produce machine-readable aggregate keys."""
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))
    path = train_ppo._write_perf_summary(
        run_id="demo_run",
        startup_sec=1.2,
        per_checkpoint_perf=[
            {
                "eval_step": 100,
                "train_steps": 100,
                "num_envs": 2,
                "train_wall_sec": 2.0,
                "eval_wall_sec": 3.0,
                "train_env_steps_per_sec": 100.0,
            }
        ],
        total_wall_clock_sec=12.0,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "demo_run"
    assert payload["startup_sec"] == 1.2
    assert payload["total_wall_clock_sec"] == 12.0
    assert payload["train_env_steps_per_sec_mean"] == 100.0
    assert payload["eval_sec_per_checkpoint"] == 3.0
