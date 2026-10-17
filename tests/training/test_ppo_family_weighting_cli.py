"""Compare configured family odds through the public pre/post-feature CLI paths."""

import importlib
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import yaml

from robot_sf.training.scenario_loader import load_scenarios
from robot_sf.training.scenario_sampling import ScenarioSampler
from scripts.training import train_ppo


@pytest.mark.parametrize(
    ("families", "include_ids", "error", "expected"),
    [
        (None, None, None, (0.5, 0.5)),
        ({}, None, None, (0.5, 0.5)),
        ({"doorway": 3}, None, None, (0.75, 0.25)),
        ({"doorway": 0}, None, "finite and >= 1", None),
        ({"doorway": -1}, None, "finite and >= 1", None),
        ({"doorway": math.nan}, None, "finite and >= 1", None),
        ({"doorway": math.inf}, None, "finite and >= 1", None),
        ({"doors": 2}, None, "Unknown scenario families", None),
        (
            {"doorway": 2},
            ["francis2023_frontal_approach"],
            "no sampleable",
            None,
        ),
        ({"doorway": True}, None, "values numeric", None),
        ("doorway", None, "must be a mapping", None),
    ],
)
def test_requested_family_odds_reach_canonical_sampler_before_training(
    monkeypatch, tmp_path, families, include_ids, error, expected
):
    """The public CLI applies requested odds or refuses invalid options before dispatch."""
    ids = ("classic_doorway_low", "francis2023_frontal_approach")
    sampling = {
        "strategy": "random",
        "include_scenarios": include_ids or list(ids),
        "episode_seed_pool": [100001],
        "weights": {},
    }
    if families is not None:
        sampling["family_weights"] = families
    recipe = Path(__file__).resolve().parents[2] / "configs/training/ppo/dev_train010_fixed_5m.yaml"
    data = train_ppo._load_expert_training_config_mapping(recipe)
    data.pop("base_config", None)
    data["scenario_config"] = str((recipe.parent / data["scenario_config"]).resolve())
    evaluation = data["evaluation"]
    if evaluation.get("scenario_config"):
        evaluation["scenario_config"] = str(
            (recipe.parent / evaluation["scenario_config"]).resolve()
        )
    evaluation["evaluation_seed_manifest"] = None
    evaluation["evaluation_seeds"] = [1031, 1032]
    evaluation["randomize_seeds"] = False
    data.update(seeds=[100001], randomize_seeds=False, scenario_sampling=sampling)
    source = tmp_path / "training.yaml"
    source.write_text(yaml.safe_dump(data))
    original = train_ppo.load_expert_training_config(source)
    dispatched = []

    def record_dispatch(config, **kwargs):
        assert kwargs["dry_run"] is True
        dispatched.append(config)

    monkeypatch.setattr(train_ppo, "run_expert_training", record_dispatch)
    monkeypatch.setattr(sys, "argv", ["training", "--config", str(source), "--dry-run"])
    family_cli = "scripts.training.train_ppo_with_family_weights"
    # Main has only the original public CLI. Compare that actual pre-feature
    # path rather than failing to import an API that did not exist there.
    entrypoint = (
        importlib.import_module(family_cli)
        if importlib.util.find_spec(family_cli) is not None
        else train_ppo
    )
    if error is not None:
        with pytest.raises(ValueError, match=error):
            entrypoint.main()
        assert dispatched == []
        return
    entrypoint.main()
    assert len(dispatched) == 1
    configured = dispatched[0]
    sampler = ScenarioSampler(
        load_scenarios(configured.scenario_config),
        include_scenarios=ids,
        weights=configured.scenario_sampling.get("weights"),
        seed=100001,
    )
    probabilities = sampler._weights_array
    actual = (
        dict.fromkeys(sampler.scenario_ids, 1 / len(sampler.scenario_ids))
        if probabilities is None
        else dict(zip(sampler.scenario_ids, probabilities.tolist(), strict=True))
    )
    assert actual == pytest.approx(dict(zip(ids, expected, strict=True)))
    assert configured.evaluation == original.evaluation
    assert "family_weights" not in configured.scenario_sampling
