from __future__ import annotations

import numpy as np
import pytest

from robot_sf.training import ExtractorRunRecord, HardwareProfile
from scripts.multi_extractor_training import (
    RunContext,
    RunSettings,
    _enrich_records_with_analysis,
)


def _write_eval_history(path, timesteps, rewards):
    eval_dir = path / "eval_logs"
    eval_dir.mkdir(parents=True, exist_ok=True)
    np.savez(eval_dir / "evaluations.npz", timesteps=np.array(timesteps), results=np.array(rewards))


def test_enrich_records_computes_convergence_and_figures(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    base_dir = run_dir / "extractors" / "base"
    cand_dir = run_dir / "extractors" / "cand"
    base_dir.mkdir(parents=True)
    cand_dir.mkdir(parents=True)

    _write_eval_history(base_dir, [10, 20, 30], [[0.5, 0.6], [1.0, 1.1], [1.2, 1.15]])
    _write_eval_history(cand_dir, [10, 20, 30], [[0.2, 0.3], [0.4, 0.5], [1.0, 1.05]])

    hardware = HardwareProfile(
        platform="macOS",
        arch="arm64",
        python_version="3.11",
        workers=1,
        gpu_model=None,
        cuda_version=None,
    )

    base_record = ExtractorRunRecord(
        config_name="base",
        status="success",
        start_time="2025-10-02T12:00:00Z",
        end_time="2025-10-02T12:05:00Z",
        duration_seconds=300.0,
        hardware_profile=hardware,
        worker_mode="single-thread",
        training_steps=128,
        metrics={"best_mean_reward": 1.1},
        artifacts={"extractor_dir": str(base_dir.relative_to(run_dir))},
        reason=None,
    )
    cand_record = ExtractorRunRecord(
        config_name="cand",
        status="success",
        start_time="2025-10-02T12:00:00Z",
        end_time="2025-10-02T12:05:00Z",
        duration_seconds=320.0,
        hardware_profile=hardware,
        worker_mode="single-thread",
        training_steps=128,
        metrics={"best_mean_reward": 1.0},
        artifacts={"extractor_dir": str(cand_dir.relative_to(run_dir))},
        reason=None,
    )

    settings = RunSettings(total_timesteps=40, eval_freq=10, save_freq=20, n_eval_episodes=2)
    settings.baseline_extractor = "base"
    context = RunContext(
        run_id="demo",
        timestamp="ts",
        created_at="now",
        run_dir=run_dir,
        hardware_profile=hardware,
        settings=settings,
        test_mode=False,
    )

    baseline_target, baseline_conv = _enrich_records_with_analysis(
        [base_record, cand_record],
        context,
    )

    assert baseline_target == pytest.approx(1.1)
    assert baseline_conv == pytest.approx(20)

    assert base_record.metrics["convergence_timestep"] == pytest.approx(20)
    assert base_record.metrics["sample_efficiency_ratio"] == pytest.approx(20 / 30)

    assert cand_record.metrics["sample_efficiency_timestep"] == pytest.approx(40.0)
    assert cand_record.metrics["sample_efficiency_ratio"] == pytest.approx(0.5)

    assert "learning_curve" in cand_record.artifacts
    assert (run_dir / cand_record.artifacts["learning_curve"]).exists()
    assert "reward_distribution" in cand_record.artifacts
    assert (run_dir / cand_record.artifacts["reward_distribution"]).exists()
