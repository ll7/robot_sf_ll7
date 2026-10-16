"""Independent odds oracle and default-off contract for family oversampling."""

import math

import pytest

from robot_sf.training.scenario_family_weights import compile_family_weights
from robot_sf.training.scenario_sampling import ScenarioSampler


def test_family_oversampling_retains_other_scenarios_and_changes_odds():
    """Doorway gets three times the probability of an otherwise uniform open cell."""
    ids = ("classic_doorway_low", "francis2023_narrow_doorway", "open", "notdoorway")
    weights = compile_family_weights(ids, {"doorway": 3})
    assert weights == {ids[0]: 3.0, ids[1]: 3.0, "open": 1.0, "notdoorway": 1.0}
    sampler = ScenarioSampler([{"name": sid} for sid in ids], weights=weights, seed=100001)
    assert sampler._weights_array.tolist() == [3 / 8, 3 / 8, 1 / 8, 1 / 8]
    assert compile_family_weights(ids) is None
    assert compile_family_weights(ids, {}, {"open": 2}) == {"open": 2}


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf])
def test_invalid_multiplier_fails_closed(value):
    """Refuse disabling, nonfinite, or downweighting values in an oversampling option."""
    with pytest.raises(ValueError, match="finite and >= 1"):
        compile_family_weights(["classic_doorway_low"], {"doorway": value})


def test_unknown_unmatched_and_zero_mass_families_rejected():
    """A misspelled or filtered family must not silently produce a uniform run."""
    with pytest.raises(ValueError, match="Unknown scenario families"):
        compile_family_weights(["open"], {"doors": 2})
    with pytest.raises(ValueError, match="no sampleable"):
        compile_family_weights(["open"], {"doorway": 2})
    with pytest.raises(ValueError, match="no sampleable"):
        compile_family_weights(["classic_doorway_low"], {"doorway": 2}, {"classic_doorway_low": 0})


def test_real_config_compilation_keeps_evaluation_unweighted():
    """The entry-point transform reaches canonical training sampling only."""
    from dataclasses import replace

    from robot_sf.training.scenario_family_weights import with_family_weights
    from scripts.training.train_ppo import load_expert_training_config

    original = load_expert_training_config("configs/training/ppo/dev_train010_fixed_5m.yaml")
    source = replace(
        original, scenario_sampling={"strategy": "random", "family_weights": {"doorway": 2}}
    )
    resolved = with_family_weights(source)
    assert resolved.scenario_sampling["weights"]["classic_doorway_low"] == 2
    assert resolved.scenario_sampling["weights"]["francis2023_frontal_approach"] == 1
    assert resolved.evaluation is original.evaluation
    assert "weights" not in source.scenario_sampling
    excluded = replace(
        source,
        scenario_sampling={
            "strategy": "random",
            "family_weights": {"doorway": 2},
            "include_scenarios": ["francis2023_frontal_approach"],
        },
    )
    with pytest.raises(ValueError, match="no sampleable"):
        with_family_weights(excluded)


def test_empty_base_weights_keep_uniform_mass_through_config_and_sampler():
    """An explicit uniform mapping retains open-family mass after oversampling."""
    from dataclasses import replace

    from robot_sf.training.scenario_family_weights import with_family_weights
    from scripts.training.train_ppo import load_expert_training_config

    ids = ("classic_doorway_low", "francis2023_frontal_approach")
    original = load_expert_training_config("configs/training/ppo/dev_train010_fixed_5m.yaml")
    configured = replace(
        original,
        scenario_sampling={
            "strategy": "random",
            "include_scenarios": list(ids),
            "weights": {},
            "family_weights": {"doorway": 3},
        },
    )
    resolved = with_family_weights(configured)
    assert compile_family_weights(ids, {"doorway": 3}, {}) == {ids[0]: 3.0, ids[1]: 1.0}
    sampler = ScenarioSampler(
        [{"name": sid} for sid in ids], weights=resolved.scenario_sampling["weights"], seed=100001
    )
    assert sampler._weights_array.tolist() == [0.75, 0.25]
    assert resolved.evaluation is original.evaluation
    assert configured.scenario_sampling["weights"] == {}
