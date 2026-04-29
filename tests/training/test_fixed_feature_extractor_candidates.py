"""Tests for the fixed feature-extractor candidate runner helpers."""

from __future__ import annotations

from types import SimpleNamespace

import scripts.training.fixed_feature_extractor_candidates as fixed_runner

_matrix_metadata = fixed_runner.__dict__["_matrix_metadata"]
_configure_candidate = fixed_runner.__dict__["_configure_candidate"]


def _base_config() -> SimpleNamespace:
    """Build a minimal config-like object for runner helper tests.

    This verifies matrix metadata wiring without depending on training runtime.
    """

    return SimpleNamespace(
        policy_id="feature_extractor_base",
        total_timesteps=32_000,
        seeds=(123,),
        randomize_seeds=False,
        feature_extractor="mlp",
        feature_extractor_kwargs={},
        policy_net_arch=(64, 64),
        evaluation=SimpleNamespace(
            frequency_episodes=10,
            evaluation_episodes=5,
            hold_out_scenarios=[],
            scenario_config=None,
        ),
        env_factory_kwargs={"reward_name": "route_completion_v2"},
        tracking={},
    )


def test_matrix_metadata_defaults_when_optional_fields_missing() -> None:
    """Fixed matrices should have stable default W&B metadata when unspecified."""
    job_type, tags = _matrix_metadata({})

    assert job_type == "feat-extractor-fixed"
    assert tags == []


def test_configure_candidate_uses_matrix_specific_wandb_metadata() -> None:
    """Issue follow-up matrices need truthful W&B tags instead of hardcoded issue-193 tags."""
    config = _configure_candidate(
        _base_config(),
        candidate=fixed_runner.FixedCandidate(
            candidate_id="dyn_large_med_s231_rcv3",
            extractor_type="dynamics",
            arch_size="large",
            policy_arch_size="medium",
            policy_arch=[128, 128],
            dropout_rate=0.0,
            seed=231,
            extractor_kwargs={"num_filters": [128, 32, 32, 16]},
        ),
        matrix={
            "total_timesteps": 12_000_000,
            "eval_every": 48_000,
            "eval_episodes": 5,
            "wandb_job_type": "feat-extractor-fixed-issue850",
            "wandb_tags": ["issue-850", "reward-mitigation"],
            "env_factory_kwargs": {
                "reward_name": "route_completion_v3",
                "reward_kwargs": {"weights": {"collision": -15.0}},
            },
        },
        study_name="issue850_reward_v3",
        disable_wandb=False,
    )

    wandb_cfg = config.tracking["wandb"]
    assert wandb_cfg["job_type"] == "feat-extractor-fixed-issue850"
    assert wandb_cfg["group"] == "issue850_reward_v3"
    assert "issue-850" in wandb_cfg["tags"]
    assert "reward-mitigation" in wandb_cfg["tags"]
    assert "issue-193" not in wandb_cfg["tags"]
    assert "extractor:dynamics" in wandb_cfg["tags"]
    assert "seed:231" in wandb_cfg["tags"]
    assert config.env_factory_kwargs["reward_name"] == "route_completion_v3"
    assert config.env_factory_kwargs["reward_kwargs"] == {"weights": {"collision": -15.0}}
