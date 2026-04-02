"""Contract tests for expert PPO training runtime helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace
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


def test_resolve_env_max_sim_steps_reads_wrapper_state() -> None:
    """Wrapped eval envs should still expose max_sim_steps through wrapper attr lookup."""
    env = SimpleNamespace(
        get_wrapper_attr=lambda name: SimpleNamespace(max_sim_steps=77)
        if name == "state"
        else None,
        unwrapped=SimpleNamespace(state=SimpleNamespace(max_sim_steps=12)),
    )

    assert train_ppo._resolve_env_max_sim_steps(env) == 77


def test_resolve_env_max_sim_steps_falls_back_to_unwrapped_state() -> None:
    """Fallback lookup should support wrappers that only expose the unwrapped env state."""

    def _missing_attr(_name: str) -> object:
        raise AttributeError("missing")

    env = SimpleNamespace(
        get_wrapper_attr=_missing_attr,
        unwrapped=SimpleNamespace(state=SimpleNamespace(max_sim_steps=33)),
    )

    assert train_ppo._resolve_env_max_sim_steps(env) == 33
