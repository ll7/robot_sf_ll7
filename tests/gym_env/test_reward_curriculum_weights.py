"""Curriculum stages must explicitly inherit or replace top-level reward weights."""

import pytest

from robot_sf.gym_env.reward import build_reward_curriculum_function


def _build(mode="merge"):
    return build_reward_curriculum_function(
        {
            "stage_weight_merge": mode,
            "stages": [{"until_episodes": 2, "reward_kwargs": {"weights": {"progress": 1.1}}}, {}],
        },
        default_reward_name="route_completion_v3",
        default_reward_kwargs={"weights": {"living": -0.04, "near_miss": -2.0, "progress": 2.0}},
    )


def test_stage_weights_inherit_top_level_without_silent_profile_fallback():
    curriculum = _build()
    meta = {"prev_distance_to_goal": 2.0, "distance_to_goal": 1.0, "near_misses": 1.0}
    curriculum(meta)
    assert meta["reward_terms"]["living"] == -0.04
    assert meta["reward_terms"]["near_miss"] == -2.0
    assert meta["reward_terms"]["progress"] == 1.1


def test_explicit_replace_mode_preserves_historical_stage_defaults():
    curriculum = _build("replace")
    meta = {"near_misses": 1.0}
    curriculum(meta)
    assert meta["reward_terms"]["living"] == -0.01
    assert meta["reward_terms"]["near_miss"] == -1.0


def test_unknown_stage_weight_merge_mode_fails_closed():
    with pytest.raises(ValueError, match="stage_weight_merge"):
        _build("merg")


def test_leader_config_stages_resolve_the_declared_top_level_weight_table():
    """The leader's sparse first stage inherits all ten declared reward weights."""
    from pathlib import Path

    import yaml

    source = (
        Path(__file__).resolve().parents[2]
        / "configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_base.yaml"
    )
    kwargs = yaml.safe_load(source.read_text())["env_factory_kwargs"]
    curriculum = build_reward_curriculum_function(
        kwargs["reward_curriculum"],
        default_reward_name=kwargs["reward_name"],
        default_reward_kwargs=kwargs["reward_kwargs"],
    )
    expected = {
        "progress": 1.1,
        "living": -0.015,
        "collision": -15.0,
        "near_miss": -1.5,
        "ttc_risk": -1.2,
        "comfort": -0.5,
        "smoothness": -0.18,
        "timeout": -6.5,
        "stagnation": -0.8,
        "terminal_bonus": 20.0,
    }
    for stage in curriculum.stage_callables:
        assert stage.keywords["weights"] == expected
