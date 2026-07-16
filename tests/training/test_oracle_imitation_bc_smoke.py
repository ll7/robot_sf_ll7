"""Tests for the issue #1496 oracle-imitation BC loader/overfit smoke.

These tests exercise the bounded smoke delivered for issue #1496: the ``expert_traj_v1`` NPZ
loader, the split/leakage contract enforcement, and the tiny BC overfit probe. They build a
schema-faithful synthetic NPZ fixture (same arrays and observation contract as the job 13520
artifact) so the smoke is deterministic on CPU without requiring the private 363 MB dataset.

Claim boundary reminder: a passing overfit here is *smoke* evidence that the loader and a BC
training step execute end to end and memorize the tiny training split. It is NOT evidence that
the real warm-start policy is good, that BC beats RL, or anything benchmark-facing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robot_sf.training.oracle_imitation_bc_smoke import (
    CLAIM_BOUNDARY,
    SCHEMA_VERSION,
    BCSmokeConfig,
    OracleImitationBcSmokeError,
    flatten_observation_action_pairs,
    load_expert_trajectory_dataset,
    run_bc_overfit_smoke,
    validate_split_leakage_contract,
)

_PED_SLOTS = 64


def _observation(seed: int, step: int) -> dict[str, Any]:
    """Build one structured observation dict with the expert_traj_v1 contract keys."""
    rng = np.random.default_rng(seed * 1000 + step)
    return {
        "robot_position": rng.standard_normal(2).astype(np.float32),
        "robot_heading": np.array([rng.standard_normal()], dtype=np.float32),
        "robot_speed": rng.standard_normal(2).astype(np.float32),
        "robot_velocity_xy": rng.standard_normal(2).astype(np.float32),
        "robot_angular_velocity": np.array([rng.standard_normal()], dtype=np.float32),
        "robot_radius": np.array([0.3], dtype=np.float32),
        "goal_current": rng.standard_normal(2).astype(np.float32),
        "goal_next": rng.standard_normal(2).astype(np.float32),
        "pedestrians_positions": rng.standard_normal((_PED_SLOTS, 2)).astype(np.float32),
        "pedestrians_velocities": rng.standard_normal((_PED_SLOTS, 2)).astype(np.float32),
        "pedestrians_radius": np.array([0.4], dtype=np.float32),
        "pedestrians_count": np.array([float(_PED_SLOTS)], dtype=np.float32),
        "map_size": np.array([20.0, 10.0], dtype=np.float32),
        "sim_timestep": np.array([0.1], dtype=np.float32),
    }


def _action_for(observation: dict[str, Any], scenario_bias: int) -> np.ndarray:
    """Deterministic, learnable action mapping so the overfit probe can succeed.

    The action is a small linear function of compact observation fields plus a per-scenario
    bias. This gives the tiny MLP a memorizable mapping without being a constant target.
    """
    features = np.concatenate(
        [
            np.asarray(observation["robot_position"], dtype=np.float32),
            np.asarray(observation["robot_heading"], dtype=np.float32),
            np.asarray(observation["goal_current"], dtype=np.float32),
        ]
    ).astype(np.float32)
    weight = np.array([0.5, -0.3, 0.8, 0.2, -0.4, 0.9], dtype=np.float32)[: features.shape[0]]
    speed = float(features @ weight) + 0.3 * scenario_bias
    steering = float(features.sum() * 0.1) - 0.1 * scenario_bias
    return np.array([speed, steering], dtype=np.float32)


def _build_episode(
    *,
    episode_id: str,
    scenario_id: str,
    seed: int,
    split: str,
    steps: int,
    scenario_bias: int,
) -> dict[str, Any]:
    """Build one episode record matching the expert_traj_v1 NPZ layout."""
    observations = np.empty(steps, dtype=object)
    actions = np.empty((steps, 2), dtype=np.float32)
    for step in range(steps):
        observation = _observation(seed, step)
        observations[step] = observation
        actions[step] = _action_for(observation, scenario_bias)
    return {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "split": split,
        "observations": observations,
        "actions": actions,
    }


def _write_npz(
    path: Path,
    *,
    episodes: list[dict[str, Any]],
    splits_mapping: dict[str, list[str]],
) -> None:
    """Serialize episodes into the expert_traj_v1 NPZ schema (object arrays + metadata)."""
    episode_count = len(episodes)
    max_steps = max(int(episode["actions"].shape[0]) for episode in episodes)

    observations = np.empty((episode_count, max_steps), dtype=object)
    actions = np.empty((episode_count, max_steps, 2), dtype=object)
    positions = np.empty((episode_count, max_steps, 2), dtype=object)
    rewards = np.empty((episode_count, max_steps), dtype=object)
    return_to_go = np.empty((episode_count, max_steps), dtype=object)
    terminated = np.empty((episode_count, max_steps), dtype=object)
    truncated = np.empty((episode_count, max_steps), dtype=object)
    episode_ids = np.empty((episode_count, 1), dtype=object)
    scenario_ids = np.empty((episode_count, 1), dtype=object)
    seeds = np.empty((episode_count, 1), dtype=object)
    split_tags = np.empty((episode_count, 1), dtype=object)

    for index, episode in enumerate(episodes):
        steps = int(episode["actions"].shape[0])
        for step in range(steps):
            observations[index, step] = episode["observations"][step]
            actions[index, step] = episode["actions"][step]
            positions[index, step] = np.zeros(2, dtype=np.float32)
            rewards[index, step] = 0.05
            return_to_go[index, step] = 0.05 * (steps - step)
            terminated[index, step] = step == steps - 1
            truncated[index, step] = False
        for step in range(steps, max_steps):
            observations[index, step] = None
            actions[index, step] = np.zeros(2, dtype=np.float32)
            positions[index, step] = np.zeros(2, dtype=np.float32)
            rewards[index, step] = 0.0
            return_to_go[index, step] = 0.0
            terminated[index, step] = False
            truncated[index, step] = False
        episode_ids[index, 0] = episode["episode_id"]
        scenario_ids[index, 0] = episode["scenario_id"]
        seeds[index, 0] = episode["seed"]
        split_tags[index, 0] = episode["split"]

    metadata = {
        "dataset_id": "expert_traj_v1",
        "source_policy_id": "hybrid_rule_v3_static_margin0_waypoint2",
        "dataset_schema": "trajectory_dataset.v2.decision_transformer_preflight",
        "splits": {split: {"episode_ids": ids} for split, ids in splits_mapping.items()},
        "observation_contract": {"keys": sorted(_observation(0, 0).keys())},
        "data_collection_only": True,
        "training_performed": False,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        observations=observations,
        actions=actions,
        positions=positions,
        rewards=rewards,
        return_to_go=return_to_go,
        terminated=terminated,
        truncated=truncated,
        episode_ids=episode_ids,
        scenario_ids=scenario_ids,
        seeds=seeds,
        splits=split_tags,
        episode_count=np.array(episode_count),
        metadata=np.array(metadata, dtype=object),
    )


def _build_fixture(
    tmp_path: Path,
    *,
    episodes: list[dict[str, Any]] | None = None,
    splits_mapping: dict[str, list[str]] | None = None,
) -> Path:
    """Build a schema-faithful expert_traj_v1 NPZ fixture and return its path."""
    if episodes is None:
        episodes = [
            _build_episode(
                episode_id="train__planner_sanity_simple__seed201",
                scenario_id="planner_sanity_simple",
                seed=201,
                split="train",
                steps=12,
                scenario_bias=1,
            ),
            _build_episode(
                episode_id="train__classic_head_on__seed202",
                scenario_id="classic_head_on",
                seed=202,
                split="train",
                steps=10,
                scenario_bias=2,
            ),
            _build_episode(
                episode_id="validation__classic_crossing__seed101",
                scenario_id="classic_crossing",
                seed=101,
                split="validation",
                steps=8,
                scenario_bias=3,
            ),
            _build_episode(
                episode_id="evaluation__classic_doorway__seed111",
                scenario_id="classic_doorway",
                seed=111,
                split="evaluation",
                steps=8,
                scenario_bias=4,
            ),
        ]
    if splits_mapping is None:
        loaded = load_expert_trajectory_dataset  # noqa: F841 (readability anchor)
        splits_mapping = {"train": [], "validation": [], "evaluation": []}
        for episode in episodes:
            splits_mapping.setdefault(episode["split"], []).append(episode["episode_id"])
    dataset_path = tmp_path / "expert_traj_v1.npz"
    _write_npz(dataset_path, episodes=episodes, splits_mapping=splits_mapping)
    return dataset_path


def test_loader_validates_schema_and_partitions_by_split(tmp_path: Path) -> None:
    """The loader accepts the expert_traj_v1 schema and reports per-split episode mapping."""
    dataset_path = _build_fixture(tmp_path)
    dataset = load_expert_trajectory_dataset(dataset_path)

    assert dataset["episode_count"] == 4
    assert set(dataset["splits"]) == {"train", "validation", "evaluation"}
    assert dataset["splits"]["train"] == [
        "train__planner_sanity_simple__seed201",
        "train__classic_head_on__seed202",
    ]
    assert dataset["splits"]["evaluation"] == ["evaluation__classic_doorway__seed111"]


def test_loader_fails_closed_when_dataset_missing(tmp_path: Path) -> None:
    """A missing artifact path must fail closed rather than fabricate data."""
    with pytest.raises(OracleImitationBcSmokeError, match="not found"):
        load_expert_trajectory_dataset(tmp_path / "does_not_exist.npz")


def test_loader_fails_closed_on_missing_required_array(tmp_path: Path) -> None:
    """Dropping a required array must surface a fail-closed schema error."""
    dataset_path = _build_fixture(tmp_path)
    with np.load(dataset_path, allow_pickle=True) as npz:
        keep = {key: npz[key] for key in npz.files if key != "rewards"}
    broken = dataset_path.with_name("broken.npz")
    np.savez(broken, **keep)
    with pytest.raises(OracleImitationBcSmokeError, match="missing required array 'rewards'"):
        load_expert_trajectory_dataset(broken)


def test_split_leakage_contract_passes_on_disjoint_splits(tmp_path: Path) -> None:
    """Disjoint episode ids and seeds satisfy the issue #1496 split/leakage gate."""
    dataset_path = _build_fixture(tmp_path)
    dataset = load_expert_trajectory_dataset(dataset_path)
    report = validate_split_leakage_contract(dataset)
    assert report["episode_counts"] == {"train": 2, "validation": 1, "evaluation": 1}
    assert report["train_seeds_disjoint_from_holdout"] is True
    assert report["evaluation_held_out_from_training"] is True


def test_split_leakage_contract_fails_closed_on_cross_split_episode(tmp_path: Path) -> None:
    """An episode id reused across splits must fail closed before any training."""
    base = [
        _build_episode(
            episode_id="dup__ep__seed1",
            scenario_id="a",
            seed=1,
            split="train",
            steps=6,
            scenario_bias=1,
        ),
        _build_episode(
            episode_id="dup__ep__seed1",
            scenario_id="b",
            seed=2,
            split="validation",
            steps=5,
            scenario_bias=2,
        ),
        _build_episode(
            episode_id="evaluation__ep__seed3",
            scenario_id="c",
            seed=3,
            split="evaluation",
            steps=5,
            scenario_bias=3,
        ),
    ]
    splits_mapping = {
        "train": ["dup__ep__seed1"],
        "validation": ["dup__ep__seed1"],
        "evaluation": ["evaluation__ep__seed3"],
    }
    dataset_path = _build_fixture(tmp_path, episodes=base, splits_mapping=splits_mapping)
    dataset = load_expert_trajectory_dataset(dataset_path)
    with pytest.raises(OracleImitationBcSmokeError, match="appears in both"):
        validate_split_leakage_contract(dataset)


def test_split_leakage_contract_fails_closed_on_shared_seed(tmp_path: Path) -> None:
    """A train seed reused in a holdout split must fail closed (leakage)."""
    base = [
        _build_episode(
            episode_id="train__ep__seed7",
            scenario_id="a",
            seed=7,
            split="train",
            steps=6,
            scenario_bias=1,
        ),
        _build_episode(
            episode_id="validation__ep__seed7",
            scenario_id="b",
            seed=7,
            split="validation",
            steps=5,
            scenario_bias=2,
        ),
        _build_episode(
            episode_id="evaluation__ep__seed9",
            scenario_id="c",
            seed=9,
            split="evaluation",
            steps=5,
            scenario_bias=3,
        ),
    ]
    splits_mapping = {
        "train": ["train__ep__seed7"],
        "validation": ["validation__ep__seed7"],
        "evaluation": ["evaluation__ep__seed9"],
    }
    dataset_path = _build_fixture(tmp_path, episodes=base, splits_mapping=splits_mapping)
    dataset = load_expert_trajectory_dataset(dataset_path)
    with pytest.raises(OracleImitationBcSmokeError, match="train seeds reused"):
        validate_split_leakage_contract(dataset)


def test_flatten_pairs_only_includes_the_training_split(tmp_path: Path) -> None:
    """The flatten step must keep training steps only and exclude holdout steps."""
    dataset_path = _build_fixture(tmp_path)
    dataset = load_expert_trajectory_dataset(dataset_path)
    features, targets, per_split_steps = flatten_observation_action_pairs(dataset)
    # 12 + 10 training steps; validation/evaluation held out.
    assert features.shape[0] == 22
    assert targets.shape == (22, 2)
    assert features.shape[1] > 0
    assert per_split_steps == {"train": 22}


def test_bc_overfit_smoke_reduces_loss_and_writes_artifacts(tmp_path: Path) -> None:
    """The full smoke trains, reduces loss, and writes checkpoint + manifest + metrics."""
    dataset_path = _build_fixture(tmp_path)
    output_dir = tmp_path / "smoke_out"
    result = run_bc_overfit_smoke(
        BCSmokeConfig(
            dataset_path=str(dataset_path),
            output_dir=str(output_dir),
            epochs=300,
            learning_rate=5e-3,
            hidden_dim=64,
            seed=1496,
        )
    )

    assert result.loss_reduction > 0.0
    assert result.final_loss < result.initial_loss
    assert result.num_train_steps == 22
    assert result.episode_counts == {"train": 2, "validation": 1, "evaluation": 1}
    assert result.evidence_tier == "real-trajectory-smoke"

    checkpoint = Path(result.checkpoint_path)
    manifest_path = Path(result.manifest_path)
    metrics_path = Path(result.metrics_path)
    assert checkpoint.is_file()
    assert manifest_path.is_file()
    assert metrics_path.is_file()

    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["smoke_not_quality"] is True
    assert manifest["training_performed"] is True
    assert manifest["not_full_warm_start_comparison"] is True
    assert manifest["evidence_tier"] == "real-trajectory-smoke"
    assert "not benchmark" in CLAIM_BOUNDARY.lower()
    assert manifest["metrics"]["loss_reduction"] == pytest.approx(result.loss_reduction)
    # The split/leakage contract is recorded inside the smoke manifest.
    assert manifest["split_contract"]["train_seeds_disjoint_from_holdout"] is True


def test_bc_overfit_smoke_fails_closed_when_loss_does_not_decrease(tmp_path: Path) -> None:
    """A zero-epoch run that cannot reduce loss must fail closed instead of claiming success."""
    dataset_path = _build_fixture(tmp_path)
    output_dir = tmp_path / "smoke_out_zero"
    # With epochs=0 the smoke must detect no loss reduction and fail closed.
    with pytest.raises(OracleImitationBcSmokeError, match="did not decrease"):
        run_bc_overfit_smoke(
            BCSmokeConfig(
                dataset_path=str(dataset_path),
                output_dir=str(output_dir),
                epochs=0,
            )
        )
