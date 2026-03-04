"""Contract tests for expert PPO training runtime helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

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
