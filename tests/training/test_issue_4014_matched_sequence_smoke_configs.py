"""Tests for issue #4014 matched PPO sequence encoder smoke configs."""

from __future__ import annotations

from pathlib import Path

from scripts.training import train_ppo, train_recurrent_ppo

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "training" / "ppo"


def _ppo_config(name: str) -> train_ppo.ExpertTrainingConfig:
    return train_ppo.load_expert_training_config(CONFIG_DIR / name)


def test_issue_4014_matched_configs_share_smoke_budget() -> None:
    """All three rows share scenario, seed, reward, timesteps, and PPO budget."""
    ppo = _ppo_config("issue_4014_ppo_smoke_matched.yaml")
    lstm = train_recurrent_ppo.load_recurrent_ppo_config(
        CONFIG_DIR / "issue_4014_recurrent_ppo_lstm_smoke_matched.yaml"
    )
    mamba = _ppo_config("issue_4014_ppo_mamba_smoke_matched.yaml")

    bases = [ppo, lstm.base, mamba]
    assert {config.scenario_config for config in bases} == {ppo.scenario_config}
    assert {config.total_timesteps for config in bases} == {2048}
    assert {config.seeds for config in bases} == {(4014,)}
    assert {config.randomize_seeds for config in bases} == {False}
    assert {config.env_factory_kwargs["reward_name"] for config in bases} == {"route_completion_v2"}
    assert {config.num_envs for config in bases} == {1}
    assert {config.worker_mode for config in bases} == {"dummy"}

    assert ppo.feature_extractor == "default"
    assert lstm.recurrent_policy == "MultiInputLstmPolicy"
    assert mamba.feature_extractor == "mamba"
    assert mamba.feature_extractor_kwargs["backend"] == "torch_ssm_lite"


def test_issue_4014_mamba_matched_config_keeps_claim_boundary_visible() -> None:
    """Mamba row remains explicit about the CPU-safe fallback backend."""
    config = _ppo_config("issue_4014_ppo_mamba_smoke_matched.yaml")

    assert config.env_overrides["observation_mode"] == "default_gym"
    assert config.feature_extractor_kwargs["sequence_source"] == "rays"
