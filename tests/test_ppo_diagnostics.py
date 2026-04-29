"""Focused tests for PPO diagnostics used in feature extractor triage."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch as th
from gymnasium import spaces

from robot_sf.feature_extractors.lightweight_cnn_extractor import LightweightCNNExtractor
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS
from robot_sf.training.ppo_diagnostics import (
    DiagnosticPPO,
    _grad_l2_norm,
    _module_grad_norm,
    _summarize_samples,
)


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


def _observation_space() -> spaces.Dict:
    """Return the compact dict space used by diagnostics unit tests."""
    return spaces.Dict(
        {
            OBS_DRIVE_STATE: spaces.Box(low=-10, high=10, shape=(5, 5), dtype=float),
            OBS_RAYS: spaces.Box(low=0, high=10, shape=(5, 64), dtype=float),
        }
    )


def _sample_observation(batch_size: int = 2) -> dict[str, th.Tensor]:
    """Return a deterministic observation batch for feature-stat assertions."""
    return {
        OBS_DRIVE_STATE: th.ones(batch_size, 5, 5),
        OBS_RAYS: th.linspace(0.0, 1.0, steps=batch_size * 5 * 64).reshape(batch_size, 5, 64),
    }


def test_lightweight_cnn_records_feature_stats():
    """Feature-stat recording should be opt-in and expose a defensive copy."""
    extractor = LightweightCNNExtractor(
        _observation_space(),
        num_filters=[16, 8],
        kernel_sizes=[3, 3],
        dropout_rate=0.0,
        drive_hidden_dims=[8],
        record_feature_stats=True,
    )
    features = extractor(_sample_observation())

    assert features.shape == (2, extractor.features_dim)
    stats = extractor.latest_feature_stats()
    assert stats["combined_mean_abs"] > 0.0
    assert stats["combined_std"] >= 0.0
    assert stats["combined_max_abs"] >= stats["combined_mean_abs"]

    stats["combined_mean_abs"] = -1.0
    assert extractor.latest_feature_stats()["combined_mean_abs"] > 0.0


def test_diagnostic_helpers_cover_empty_and_populated_gradients(tmp_path):
    """Low-level helpers should handle empty gradients and aggregate populated values."""
    linear = th.nn.Linear(2, 1)
    assert _grad_l2_norm(linear.parameters()) == 0.0
    assert _module_grad_norm(None) == 0.0
    assert _summarize_samples([]) == {}

    output = linear(th.ones(1, 2)).sum()
    output.backward()
    assert _grad_l2_norm(linear.parameters()) > 0.0
    assert _module_grad_norm(linear) > 0.0

    summary = _summarize_samples(
        [{"grad_norm_total": 2.0}, {"grad_norm_total": 4.0, "feature_scale": 3.0}]
    )
    assert summary["mini_batch_count"] == 2.0
    assert summary["grad_norm_total_mean"] == 3.0
    assert summary["grad_norm_total_max"] == 4.0
    assert summary["feature_scale_mean"] == 3.0

    model = object.__new__(DiagnosticPPO)
    model._diagnostics_path = None
    model._append_training_diagnostics({"ignored": 1.0})

    model._diagnostics_path = tmp_path / "diagnostics.jsonl"
    model._append_training_diagnostics({"step": 1.0})
    assert json.loads(model._diagnostics_path.read_text(encoding="utf-8")) == {"step": 1.0}


def test_diagnostic_collect_batch_includes_feature_stats():
    """Batch diagnostics should include module gradient norms and extractor stats when available."""
    model = object.__new__(DiagnosticPPO)
    feature_extractor = LightweightCNNExtractor(
        _observation_space(),
        num_filters=[16, 8],
        kernel_sizes=[3, 3],
        dropout_rate=0.0,
        record_feature_stats=True,
    )
    _ = feature_extractor(_sample_observation())
    action_head = th.nn.Linear(feature_extractor.features_dim, 1)
    value_head = th.nn.Linear(feature_extractor.features_dim, 1)
    policy_net = th.nn.Linear(feature_extractor.features_dim, 2)
    value_net = th.nn.Linear(feature_extractor.features_dim, 2)
    model.policy = SimpleNamespace(
        features_extractor=feature_extractor,
        mlp_extractor=SimpleNamespace(policy_net=policy_net, value_net=value_net),
        action_net=action_head,
        value_net=value_head,
    )

    modules = [feature_extractor, action_head, value_head, policy_net, value_net]
    loss = sum(parameter.sum() for module in modules for parameter in module.parameters())
    loss.backward()

    diagnostics = model._collect_batch_diagnostics(
        parameter for module in modules for parameter in module.parameters()
    )
    assert diagnostics["grad_norm_total"] > 0.0
    assert diagnostics["grad_norm_features_extractor"] > 0.0
    assert diagnostics["grad_norm_policy_net"] > 0.0
    assert diagnostics["feature_combined_mean_abs"] > 0.0


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
