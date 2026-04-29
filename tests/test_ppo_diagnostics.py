"""Focused tests for PPO diagnostics used in feature extractor triage."""

from __future__ import annotations

import json

import pytest

from robot_sf.feature_extractors.lightweight_cnn_extractor import LightweightCNNExtractor
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.training.ppo_diagnostics import DiagnosticPPO


def _require_sb3():
    """Import StableBaselines3 dependencies or skip tests cleanly."""
    pytest.importorskip("stable_baselines3", reason="StableBaselines3 not installed")
    from stable_baselines3.common.env_util import make_vec_env

    return make_vec_env


def _build_fast_config() -> EnvSettings:
    """Return a small environment config for PPO smoke tests."""
    config = EnvSettings()
    config.sim_config.time_per_step_in_secs = 0.02
    config.sim_config.sim_time_in_secs = 1
    return config


def test_diagnostic_ppo_writes_grad_and_feature_stats(tmp_path):
    """Verify diagnostics JSONL captures gradient and feature statistics."""
    make_vec_env = _require_sb3()
    config = _build_fast_config()

    def make_env():
        """Create the smoke-test environment."""
        return RobotEnv(config)

    env = make_vec_env(make_env, n_envs=1)
    diagnostics_path = tmp_path / "training_diagnostics.jsonl"
    model = DiagnosticPPO(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": LightweightCNNExtractor,
            "features_extractor_kwargs": {
                "num_filters": [16, 8],
                "record_feature_stats": True,
            },
        },
        diagnostics_path=diagnostics_path,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        gamma=0.95,
        learning_rate=3e-4,
        verbose=0,
    )

    try:
        model.learn(total_timesteps=8)
    finally:
        env.close()

    assert diagnostics_path.exists()
    lines = diagnostics_path.read_text(encoding="utf-8").splitlines()
    assert lines

    payload = json.loads(lines[-1])
    assert payload["num_timesteps"] >= 8
    assert payload["mini_batch_count"] >= 1
    assert payload["grad_norm_total_mean"] > 0.0
    assert payload["grad_norm_features_extractor_mean"] > 0.0
    assert payload["feature_combined_max_abs_mean"] > 0.0
