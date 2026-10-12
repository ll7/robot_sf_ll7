"""Exercise shipped predictive seed recipes through the production collection boundary."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.training import collect_predictive_hardcase_data as collector
from scripts.training import run_predictive_training_pipeline as pipeline

CONFIGS = [
    Path("configs/training/predictive") / name
    for name in (
        "predictive_br07_all_maps_randomized.yaml",
        "predictive_br07_all_maps_randomized_full.yaml",
        "predictive_br07_all_maps_randomized_xl.yaml",
        "predictive_br07_all_maps_randomized_xl_ego.yaml",
        "predictive_crossing_conflict_weighted_issue_3254.yaml",
        "predictive_obstacle_features_v1_issue_1167.yaml",
    )
]


class CollectionStarted(Exception):
    """Stop at environment construction; do not collect or train in this integration test."""


def _start_collection(monkeypatch):
    original = collector.make_robot_env

    def make_env(*, config, seed, **kwargs):
        assert 100000 <= seed <= 199999
        env = original(config=config, seed=seed, **kwargs)
        try:
            assert env.simulator is not None
        finally:
            env.close()
        raise CollectionStarted

    monkeypatch.setattr(collector, "make_robot_env", make_env)


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.stem)
def test_shipped_randomized_pipeline_reaches_training_collection(
    monkeypatch, tmp_path, config_path
):
    """The real config, generator, command, parser and collector must agree on seed roles."""
    original_paths = pipeline._paths_from_config

    def paths(cfg, **kwargs):
        kwargs["output_base_dir"] = tmp_path
        return original_paths(cfg, **kwargs)

    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        assert "collect_predictive_hardcase_data.py" in cmd[1]
        monkeypatch.setattr(sys, "argv", cmd[1:])
        collector.main()
        pytest.fail("The collection startup sentinel must stop the pipeline before training")

    _start_collection(monkeypatch)
    monkeypatch.setattr(pipeline, "_paths_from_config", paths)
    monkeypatch.setattr(pipeline, "_run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_predictive_training_pipeline.py",
            "--config",
            str(config_path),
            "--run-id",
            "seed_role_startup",
        ],
    )
    with pytest.raises(CollectionStarted):
        pipeline.main()
    assert len(commands) == 1
    assert "--training-seeds-only" in commands[0]
    manifest_path = Path(commands[0][commands[0].index("--seed-manifest") + 1])
    manifest = yaml.safe_load(manifest_path.read_text())
    seeds = [seed for values in manifest.values() for seed in values]
    assert seeds and all(100000 <= seed <= 199999 for seed in seeds)
    assert len(seeds) == len(set(seeds))


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.stem)
def test_shipped_hardcase_collection_uses_separate_training_manifest(monkeypatch, config_path):
    """A diagnostic manifest must never double as hardcase training supervision."""
    cfg = pipeline._read_yaml(config_path)
    train_path = pipeline._resolve(
        cfg["hardcase_collection"]["seed_manifest"], base=config_path.parent
    )
    eval_path = pipeline._resolve(cfg["scenarios"]["hard_seed_manifest"], base=config_path.parent)
    assert train_path != eval_path
    train = yaml.safe_load(train_path.read_text())
    dev = yaml.safe_load(eval_path.read_text())
    seeds = [seed for values in train.values() for seed in values]
    dev_seeds = [seed for values in dev.values() for seed in values]
    assert seeds and all(100000 <= seed <= 199999 for seed in seeds)
    assert dev_seeds and all(1001 <= seed <= 1200 for seed in dev_seeds)
    assert set(seeds).isdisjoint(dev_seeds)
    expected = (
        np.random.default_rng(cfg["hardcase_collection"]["master_seed"]).choice(
            100000, size=len(seeds), replace=False
        )
        + 100000
    )
    assert seeds == expected.tolist()
    scenario_matrix = pipeline._resolve(
        cfg["scenarios"]["scenario_matrix"], base=config_path.parent
    )
    _start_collection(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_predictive_hardcase_data.py",
            "--scenario-matrix",
            str(scenario_matrix),
            "--seed-manifest",
            str(train_path),
            "--training-seeds-only",
        ],
    )
    with pytest.raises(CollectionStarted):
        collector.main()


def test_training_collection_refuses_development_manifest_before_environment(monkeypatch, tmp_path):
    """Training mode must reject a diagnostic manifest before starting simulation."""
    manifest = tmp_path / "dev.yaml"
    manifest.write_text("classic_cross_trap_low: [1001]\n")
    monkeypatch.setattr(
        collector,
        "make_robot_env",
        lambda **kwargs: pytest.fail("Development seeds reached training simulation"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_predictive_hardcase_data.py",
            "--scenario-matrix",
            "configs/scenarios/classic_interactions.yaml",
            "--seed-manifest",
            str(manifest),
            "--training-seeds-only",
        ],
    )
    with pytest.raises(ValueError, match="training-role"):
        collector.main()


def test_generator_refuses_exhausted_pool_without_creating_manifest(monkeypatch, tmp_path):
    """Never wrap or escape the training seed pool when a recipe cannot fit."""
    monkeypatch.setattr(pipeline, "load_scenarios", lambda path: [{"name": "one"}, {"name": "two"}])
    output = tmp_path / "overflow.yaml"
    with pytest.raises(ValueError, match="exceeds"):
        pipeline._build_random_seed_manifest(
            scenario_matrix=tmp_path / "scenarios.yaml",
            seeds_per_scenario=50001,
            master_seed=100001,
            output_path=output,
        )
    assert not output.exists()
