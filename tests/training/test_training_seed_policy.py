"""Resolution regressions for disjoint training and held-out episode identities."""

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark import seed_bands
from robot_sf.training.seed_policy import TrainingSeedPolicy, load_collection_seed_policy
from scripts.training import train_ppo


def _config(tmp_path: Path, pool: list[int]) -> Path:
    """Write an otherwise valid legacy config with a bounded reset pool."""
    manifest = tmp_path / "selection.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "robot-sf-evaluation-seed-manifest.v1",
                "evaluation_seeds": [1031, 1032],
            }
        )
    )
    path = tmp_path / "training.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "scenario_config": "scenarios.yaml",
                "seeds": [1001],
                "randomize_seeds": True,
                "total_timesteps": 100,
                "policy_id": "seed-policy-regression",
                "convergence": {"success_rate": 1.1, "collision_rate": 0.0, "plateau_window": 10},
                "scenario_sampling": {"episode_seed_pool": pool},
                "evaluation": {
                    "evaluation_seed_manifest": str(manifest),
                    "evaluation_episodes": 2,
                    "randomize_seeds": False,
                    "step_schedule": [{"every_steps": 100}],
                },
            }
        )
    )
    return path


def test_resolution_refuses_reset_pool_selection_overlap(tmp_path: Path) -> None:
    """An explicit reset pool must never include checkpoint-selection identities."""
    path = _config(tmp_path, [1001, 1031])
    with pytest.raises(ValueError, match="training.*overlap.*selection"):
        train_ppo.load_expert_training_config(path)


@pytest.mark.parametrize("band", ["current", "retired"])
def test_resolution_refuses_sealed_interior_in_reset_pool(
    tmp_path: Path, band: str, monkeypatch
) -> None:
    """Reject both protected bands, including unlisted points in the current interval."""
    # Synthetic policy double: no protected identity is evaluated or emitted.
    # seed-holdout: synthetic-fixture begin
    monkeypatch.setattr(seed_bands, "EVAL_SEEDS_0_0_8", (100050, 100060))
    monkeypatch.setattr(seed_bands, "RETIRED_EVAL_SEEDS_0_0_7", frozenset({100070, 100071}))
    interior = 100051 if band == "current" else 100071
    # seed-holdout: synthetic-fixture end
    path = _config(tmp_path, [interior])
    with pytest.raises(ValueError, match="training.*sealed"):
        train_ppo.load_expert_training_config(path)


def test_resolution_refuses_runtime_seed_fields_in_user_sampling_config(tmp_path: Path) -> None:
    """Open sampling metadata must not provide an unvalidated alternate range path."""
    path = _config(tmp_path, [])
    raw = yaml.safe_load(path.read_text())
    raw["scenario_sampling"].update(
        training_seed_range=[1031, 1032],
        training_master_seed=20261012,
        seed_worker_index=0,
        seed_worker_count=1,
    )
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="runtime-owned"):
        train_ppo.load_expert_training_config(path)


def _range_config(tmp_path: Path, **changes) -> Path:
    """Bind a valid training-only policy with independent fixed held-out splits."""
    path = _config(tmp_path, [])
    raw = yaml.safe_load(path.read_text())
    raw.update(
        seeds=[],
        training_seed_range=[100000, 199999],
        training_master_seed=20261012,
        development_evaluation_seeds=[1101, 1130],
    )
    raw.update(changes)
    path.write_text(yaml.safe_dump(raw))
    return path


def test_range_resolution_refuses_inherited_random_evaluation(tmp_path: Path) -> None:
    """The inherited randomized flag must not override declared fixed selection identities."""
    path = _range_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["evaluation"].pop("randomize_seeds")
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="fixed checkpoint-selection seeds"):
        train_ppo.load_expert_training_config(path)


@pytest.mark.parametrize("split", ["selection", "evaluation"])
def test_range_resolution_refuses_actual_heldout_overlap(tmp_path: Path, split: str) -> None:
    """Range validation covers checkpoint selection and later development evaluation."""
    path = _range_config(tmp_path, training_seed_range=[1000, 2000])
    with pytest.raises(ValueError, match=f"training seed range overlaps {split}"):
        if split == "evaluation":
            (tmp_path / "selection.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "robot-sf-evaluation-seed-manifest.v1",
                        "evaluation_seeds": [2001],
                    }
                )
            )
        train_ppo.load_expert_training_config(path)


def test_range_resolution_refuses_sealed_interior(tmp_path: Path, monkeypatch) -> None:
    """A one-value range inside the sealed interval fails before simulation."""
    # seed-holdout: synthetic-fixture begin
    monkeypatch.setattr(seed_bands, "EVAL_SEEDS_0_0_8", (100050, 100060))
    interior = 100051
    # seed-holdout: synthetic-fixture end
    path = _range_config(tmp_path, training_seed_range=[interior, interior])
    with pytest.raises(ValueError, match="training seed range overlaps sealed"):
        train_ppo.load_expert_training_config(path)


@pytest.mark.parametrize(
    "changes",
    [
        {"training_master_seed": None},
        {"training_master_seed": True},
        {"training_seed_range": [100000.5, 199999]},
        {"training_seed_range": [199999, 100000]},
        {"randomize_seeds": False},
        {"seeds": [1001]},
        {"development_evaluation_seeds": []},
        {"resume_from": "checkpoint.zip"},
    ],
)
def test_range_resolution_refuses_ambiguous_policy(tmp_path: Path, changes: dict) -> None:
    """Malformed or conflicting seed modes cannot silently fall back to broad randomness."""
    with pytest.raises(ValueError):
        train_ppo.load_expert_training_config(_range_config(tmp_path, **changes))


def test_master_permutation_is_reproducible_disjoint_and_exhausts() -> None:
    """Workers see fresh identities regardless of their relative reset scheduling."""
    policy = TrainingSeedPolicy.resolve([100000, 100009], 20261012)
    streams = [policy.stream(index, 2) for index in range(2)]
    draws = [[stream.next_seed() for _ in range(5)] for stream in streams]
    assert sorted(draws[0] + draws[1]) == list(range(100000, 100010))
    replay_zero = policy.stream(0, 2)
    assert draws[0] == [replay_zero.next_seed() for _ in range(5)]
    replay = policy.stream(1, 2)
    assert draws[1] == [replay.next_seed() for _ in range(5)]
    with pytest.raises(ValueError, match="exhausted"):
        streams[0].next_seed()


def test_training_factory_uses_range_at_construction_and_every_reset(
    monkeypatch, tmp_path: Path
) -> None:
    """Observe actual constructor and reset identities, including SB3 master-seed reset."""
    import gymnasium as gym
    import numpy as np

    constructions, resets = [], []

    class Env(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            resets.append(seed)
            return np.zeros(1, dtype=np.float32), {}

    def factory(**kwargs):
        constructions.append(kwargs["seed"])
        return Env()

    monkeypatch.setattr(train_ppo, "make_robot_env", factory)
    monkeypatch.setattr("robot_sf.training.scenario_sampling.make_robot_env", factory)
    # The switching wrapper's default factory is bound at import time; bind the
    # real factory explicitly via its canonical owner, avoiding simulator work.
    original = train_ppo.ScenarioSwitchingEnv
    monkeypatch.setattr(
        train_ppo, "ScenarioSwitchingEnv", lambda **kw: original(env_factory=factory, **kw)
    )
    monkeypatch.setattr(train_ppo, "build_robot_config_from_scenario", lambda *a, **kw: object())
    monkeypatch.setattr(train_ppo, "_apply_env_overrides", lambda *a: None)
    env = train_ppo._make_training_env(
        20261012,
        scenario={"name": "dev"},
        scenario_definitions=None,
        scenario_path=tmp_path / "scenarios.yaml",
        exclude_scenarios=(),
        suite_name="test",
        algorithm_name="test",
        env_overrides={},
        env_factory_kwargs={},
        scenario_sampling={
            "training_seed_range": (100000, 100009),
            "training_master_seed": 20261012,
            "seed_worker_index": 0,
            "seed_worker_count": 1,
        },
    )()
    try:
        env.reset(seed=20261012)
        for _ in range(4):
            env.reset()
        assert constructions == resets
        assert len(set(resets)) == 5
        assert all(100000 <= seed <= 100009 for seed in resets)
    finally:
        env.close()


@pytest.mark.parametrize("arm", ["fixed", "safety"])
def test_seedpol_ppo_preserves_train010_scientific_settings(arm: str) -> None:
    """Resolved reward, plant, optimization, horizon and budget remain train010's values."""
    root = Path(__file__).resolve().parents[2] / "configs/training/ppo"
    old = asdict(train_ppo.load_expert_training_config(root / f"dev_train010_{arm}_5m.yaml"))
    config = train_ppo.load_expert_training_config(root / f"dev_seedpol_{arm}_5m.yaml")
    new = asdict(config)
    assert config.training_seed_range == (100000, 199999)
    assert config.training_master_seed == 20261012
    assert config.evaluation.evaluation_seeds == tuple(range(1031, 1061))
    assert config.development_evaluation_seeds == tuple(range(1101, 1131))
    for key in (
        "policy_id",
        "seeds",
        "randomize_seeds",
        "training_seed_range",
        "training_master_seed",
        "development_evaluation_seeds",
        "tracking",
    ):
        old.pop(key)
        new.pop(key)
    old["scenario_sampling"].pop("episode_seed_pool")
    new["scenario_sampling"].pop("episode_seed_pool")
    assert new["evaluation"]["evaluation_episodes"] == 48 * 30
    for key in ("evaluation_seeds", "evaluation_seed_manifest", "evaluation_episodes"):
        old["evaluation"].pop(key)
        new["evaluation"].pop(key)
    assert old == new


@pytest.mark.parametrize("consumer", ["forecast", "prediction"])
def test_predictive_collection_resolution_preserves_science_and_guards_splits(
    tmp_path: Path, consumer: str
) -> None:
    """Both predictive consumers retain train010 learning and collection physics."""
    root = Path(__file__).resolve().parents[2] / "configs/training/predictive"
    old = yaml.safe_load((root / "dev_train010_identity.yaml").read_text())
    new, policy = load_collection_seed_policy(root / f"dev_seedpol_{consumer}_collection.yaml")
    assert policy.seed_range == (100000, 199999)
    for key in (
        "training",
        "base_max_steps",
        "hardcase_max_steps",
        "forecast_steps",
        "max_agents",
        "max_speed",
        "hardcase_repeat",
        "supervision_schema",
        "allow_legacy_supervision",
    ):
        assert old[key] == new[key]
    new["selection_seeds"] = [100001]
    path = tmp_path / "collection.yaml"
    path.write_text(yaml.safe_dump(new))
    with pytest.raises(ValueError, match="development band|overlaps selection"):
        load_collection_seed_policy(path)


@pytest.mark.parametrize("split", ["training", "selection", "evaluation"])
def test_collector_materializes_separate_split_manifests(tmp_path: Path, split: str) -> None:
    """The supported collector binds actual episode identities and archives their source."""
    from argparse import Namespace

    from scripts.training.collect_predictive_hardcase_data import _resolve_collection_seed_manifest

    root = Path(__file__).resolve().parents[2]
    args = Namespace(
        training_config=root / "configs/training/predictive/dev_seedpol_forecast_collection.yaml",
        seed_split=split,
        seed_worker_index=0,
        seed_worker_count=2,
        output=tmp_path / "data.npz",
    )
    manifest = _resolve_collection_seed_manifest(args, [{"name": "A"}, {"name": "B"}])
    if split == "training":
        flattened = manifest["A"] + manifest["B"]
        assert len(set(flattened)) == 30
        assert all(100000 <= seed <= 199999 for seed in flattened)
        args.seed_worker_index = 1
        second = _resolve_collection_seed_manifest(args, [{"name": "A"}, {"name": "B"}])
        assert not set(flattened).intersection(second["A"] + second["B"])
    else:
        expected = list(range(1031, 1061)) if split == "selection" else list(range(1101, 1131))
        assert manifest == {"A": expected, "B": expected}
    saved = yaml.safe_load(args.output.with_suffix(".seed-manifest.yaml").read_text())
    assert saved["training_master_seed"] == 20261012
    assert saved["split"] == split


def test_collection_seed_policy_retains_train010_hardcase_filter(tmp_path: Path) -> None:
    """A new seed policy must not expand the scientific hardcase portfolio."""
    from argparse import Namespace

    from scripts.training.collect_predictive_hardcase_data import _resolve_collection_seed_manifest

    root = Path(__file__).resolve().parents[2]
    args = Namespace(
        training_config=root / "configs/training/predictive/dev_seedpol_forecast_collection.yaml",
        scenario_matrix=root
        / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml",
        collection_subset=None,
        seed_split="training",
        seed_worker_index=1,
        seed_worker_count=2,
        output=tmp_path / "data.npz",
    )
    manifest = _resolve_collection_seed_manifest(
        args,
        [
            {"name": "classic_bottleneck_medium"},
            {"name": "classic_overtaking_low"},
        ],
    )
    assert set(manifest) == {"classic_bottleneck_medium"}
    assert len(manifest["classic_bottleneck_medium"]) == 15
    sidecar = yaml.safe_load(args.output.with_suffix(".seed-manifest.yaml").read_text())
    assert sidecar["collection_subset"] == "hardcase"
