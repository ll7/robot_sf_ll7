"""Adversarial caller mutation and falsy-weight validation for staged rewards."""

import pytest

from robot_sf.gym_env.reward import build_reward_curriculum_function


def _build(stages, *, default_weights, mode="merge"):
    return build_reward_curriculum_function(
        {"stage_weight_merge": mode, "stages": stages},
        default_reward_name="route_completion_v3",
        default_reward_kwargs={"weights": default_weights},
    )


def _living(curriculum):
    meta = {"step_of_episode": 1}
    curriculum(meta)
    return meta["reward_terms"]["living"]


@pytest.mark.parametrize("mode", ["merge", "replace"])
@pytest.mark.parametrize("stage", [{}, {"reward_kwargs": {}}])
def test_mutating_default_weights_cannot_change_built_stage(mode, stage):
    defaults = {"living": -0.04}
    curriculum = _build([stage], default_weights=defaults, mode=mode)
    defaults["living"] = -9.0
    assert _living(curriculum) == -0.04


def test_mutating_replace_stage_weights_cannot_change_built_stage():
    stages = [{"reward_kwargs": {"weights": {"living": -0.04}}}]
    curriculum = _build(stages, default_weights={}, mode="replace")
    stages[0]["reward_kwargs"]["weights"]["living"] = -9.0
    assert _living(curriculum) == -0.04


def test_built_stages_do_not_share_nested_weight_maps():
    curriculum = _build([{"until_episodes": 2}, {}], default_weights={"living": -0.04})
    curriculum.stage_callables[0].keywords["weights"]["living"] = -9.0
    meta = {"step_of_episode": 1}
    curriculum.stage_callables[1](meta)
    assert meta["reward_terms"]["living"] == -0.04


@pytest.mark.parametrize("mode", ["merge", "replace"])
@pytest.mark.parametrize("invalid", [None, False, 0, "", []])
def test_falsy_default_weights_are_rejected_even_without_stage_overrides(mode, invalid):
    with pytest.raises(ValueError, match="weights.*mapping"):
        _build([{}], default_weights=invalid, mode=mode)


@pytest.mark.parametrize("mode", ["merge", "replace"])
@pytest.mark.parametrize("invalid", [None, False, 0, "", []])
def test_falsy_stage_weights_are_rejected_in_both_merge_modes(mode, invalid):
    with pytest.raises(ValueError, match="weights.*mapping"):
        _build([{"reward_kwargs": {"weights": invalid}}], default_weights={}, mode=mode)


@pytest.mark.parametrize(("mode", "expected"), [("merge", -0.04), ("replace", -0.01)])
def test_empty_weight_mapping_is_valid_and_retains_mode_semantics(mode, expected):
    curriculum = _build(
        [{"reward_kwargs": {"weights": {}}}], default_weights={"living": -0.04}, mode=mode
    )
    assert _living(curriculum) == expected


@pytest.mark.parametrize("mode", ["merge", "replace"])
@pytest.mark.parametrize("invalid", [None, False, 0, "", []])
def test_valid_stage_mapping_cannot_mask_falsy_non_mapping_defaults(mode, invalid):
    with pytest.raises(ValueError, match="weights.*mapping"):
        _build([{"reward_kwargs": {"weights": {}}}], default_weights=invalid, mode=mode)
