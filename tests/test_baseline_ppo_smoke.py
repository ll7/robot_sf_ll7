from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.benchmark.runner import run_episode


def _default_model_path() -> Path:
    # Must mirror PPOPlannerConfig default
    return Path("model/ppo_model_retrained_10m_2025-02-01.zip")


@pytest.mark.timeout(30)
def test_ppo_baseline_smoke_runs_or_skips():
    model_path = _default_model_path()
    algo_cfg_path = None

    # Skip if no model present in repo checkout
    if not model_path.exists():
        pytest.skip(f"Skipping PPO smoke: model not found at {model_path}")

    # Minimal scenario: single-lane sparse crowd
    scenario = {
        "id": "smoke",
        "width": 10.0,
        "height": 6.0,
        "n_peds": 5,
        "ped_speed": 1.0,
        "repeats": 1,
    }

    # Run for a short horizon
    rec = run_episode(
        scenario_params=scenario,
        seed=123,
        horizon=20,
        dt=0.1,
        record_forces=False,
        algo="ppo",
        algo_config_path=algo_cfg_path,  # Use PPO defaults (which should match model file)
    )

    assert "metrics" in rec and isinstance(rec["metrics"], dict)
    # Ensure we produced a valid basic record
    assert rec["scenario_id"] == "smoke"
