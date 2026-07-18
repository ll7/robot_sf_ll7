"""Real-schema conformance for the issue #1496 BC loader/overfit smoke.

PR #5917 shipped the BC loader/overfit smoke with tests built on a *rectangular* synthetic NPZ
fixture (every episode padded to a common step width, observations stored as a 2-D
``(episodes, steps)`` object array). The actual ``expert_traj_v1.npz`` produced by the job 13520
collection spec is **ragged**: episodes have unequal lengths (job 13520 ran 1..241 steps per
episode), so the collector's ``np.asarray(per_episode_arrays, dtype=object)`` writes
observations/actions as a **1-D** ``(episodes,)`` object array of per-episode arrays -- not a
rectangular 2-D array.

These tests build a fixture with the *exact* writer layout the issue #1496 collection spec uses
(``np.savez(... observations=np.asarray(list_of_per_episode_arrays, dtype=object), ...)``) and
assert the loader, the split/leakage gate, the flatten step, and the full overfit smoke run
against it. They are the regression guard that the loader cannot silently drift away from the
real collector schema again.

Claim boundary reminder: a passing overfit here is *smoke* evidence that the loader and a BC
training step execute end to end on the real collector schema and memorize the tiny training
split. It is NOT evidence that the real warm-start policy is good, that BC beats RL, or anything
benchmark-facing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robot_sf.training.oracle_imitation_bc_smoke import (
    BCSmokeConfig,
    OracleImitationBcSmokeError,
    flatten_observation_action_pairs,
    load_expert_trajectory_dataset,
    run_bc_overfit_smoke,
    validate_split_leakage_contract,
)

_PED_SLOTS = 64
_FEATURE_OBS_KEYS = (
    "robot_position",
    "robot_heading",
    "robot_speed",
    "robot_velocity_xy",
    "robot_angular_velocity",
    "robot_radius",
    "goal_current",
    "goal_next",
    "pedestrians_positions",
    "pedestrians_velocities",
    "pedestrians_radius",
    "pedestrians_count",
    "map_size",
    "sim_timestep",
)


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


def _action_for(observation: dict[str, Any], bias: int) -> np.ndarray:
    """Deterministic, learnable action mapping so the overfit probe can succeed."""
    features = np.concatenate(
        [
            np.asarray(observation["robot_position"], dtype=np.float32),
            np.asarray(observation["robot_heading"], dtype=np.float32),
            np.asarray(observation["goal_current"], dtype=np.float32),
        ]
    ).astype(np.float32)
    weight = np.array([0.5, -0.3, 0.8, 0.2, -0.4, 0.9], dtype=np.float32)[: features.shape[0]]
    speed = float(features @ weight) + 0.3 * bias
    steering = float(features.sum() * 0.1) - 0.1 * bias
    return np.array([speed, steering], dtype=np.float32)


def _write_ragged_npz(
    path: Path,
    *,
    episodes: list[dict[str, Any]],
    splits_mapping: dict[str, list[str]],
) -> None:
    """Serialize ragged episodes with the EXACT writer layout the job 13520 spec uses.

    Mirrors the collection spec: each per-episode array is built independently, then the whole
    field is written via ``np.asarray(list_of_per_episode_arrays, dtype=object)``. Because
    episodes have unequal step counts, numpy stores each field as a 1-D ``(episodes,)`` object
    array of per-episode arrays -- the layout the real 363 MB artifact uses.
    """
    episode_count = len(episodes)

    per_episode: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "observations",
            "actions",
            "positions",
            "rewards",
            "return_to_go",
            "terminated",
            "truncated",
        )
    }
    episode_ids: list[np.ndarray] = []
    scenario_ids: list[np.ndarray] = []
    seeds: list[np.ndarray] = []
    split_tags: list[np.ndarray] = []

    for episode in episodes:
        steps = int(episode["steps"])
        obs_arrays = np.asarray(episode["observations"], dtype=object)
        act_arrays = np.asarray(episode["actions"], dtype=np.float32)
        assert obs_arrays.shape == (steps,)
        assert act_arrays.shape == (steps, 2)
        per_episode["observations"].append(obs_arrays)
        per_episode["actions"].append(act_arrays)
        per_episode["positions"].append(np.zeros((steps, 2), dtype=np.float32))
        per_episode["rewards"].append(np.full(steps, 0.05, dtype=np.float32))
        per_episode["return_to_go"].append(
            np.asarray([0.05 * (steps - i) for i in range(steps)], dtype=np.float32)
        )
        per_episode["terminated"].append(
            np.asarray([i == steps - 1 for i in range(steps)], dtype=bool)
        )
        per_episode["truncated"].append(np.zeros(steps, dtype=bool))
        episode_ids.append(np.asarray([episode["episode_id"]], dtype=object))
        scenario_ids.append(np.asarray([episode["scenario_id"]], dtype=object))
        seeds.append(np.asarray([episode["seed"]], dtype=np.int64))
        split_tags.append(np.asarray([episode["split"]], dtype=object))

    metadata = {
        "dataset_id": "expert_traj_v1",
        "source_policy_id": "hybrid_rule_v3_static_margin0_waypoint2",
        "dataset_schema": "trajectory_dataset.v2.decision_transformer_preflight",
        "splits": {split: {"episode_ids": ids} for split, ids in splits_mapping.items()},
        "observation_contract": {"keys": sorted(_FEATURE_OBS_KEYS)},
        "data_collection_only": True,
        "training_performed": False,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        observations=np.asarray(per_episode["observations"], dtype=object),
        actions=np.asarray(per_episode["actions"], dtype=object),
        positions=np.asarray(per_episode["positions"], dtype=object),
        rewards=np.asarray(per_episode["rewards"], dtype=object),
        return_to_go=np.asarray(per_episode["return_to_go"], dtype=object),
        terminated=np.asarray(per_episode["terminated"], dtype=object),
        truncated=np.asarray(per_episode["truncated"], dtype=object),
        episode_ids=np.asarray(episode_ids, dtype=object),
        scenario_ids=np.asarray(scenario_ids, dtype=object),
        seeds=np.asarray(seeds, dtype=object),
        splits=np.asarray(split_tags, dtype=object),
        episode_count=np.array(episode_count),
        metadata=np.array(metadata, dtype=object),
    )


def _build_ragged_fixture(tmp_path: Path) -> Path:
    """Build a ragged NPZ matching the real job 13520 writer (unequal episode lengths)."""
    episodes: list[dict[str, Any]] = []
    # Deliberately unequal step counts, including a 1-step episode exactly like job 13520's
    # minimum (the registration records minimum_episode_steps=1, maximum=241).
    spec = [
        ("train__planner_sanity_simple__seed201", "planner_sanity_simple", 201, "train", 12, 1),
        ("train__classic_head_on__seed202", "classic_head_on", 202, "train", 1, 2),
        ("validation__classic_crossing__seed101", "classic_crossing", 101, "validation", 8, 3),
        ("evaluation__classic_doorway__seed111", "classic_doorway", 111, "evaluation", 5, 4),
    ]
    for episode_id, scenario_id, seed, split, steps, bias in spec:
        observations = [_observation(seed, step) for step in range(steps)]
        actions = np.stack([_action_for(observations[step], bias) for step in range(steps)])
        episodes.append(
            {
                "episode_id": episode_id,
                "scenario_id": scenario_id,
                "seed": seed,
                "split": split,
                "steps": steps,
                "observations": observations,
                "actions": actions,
            }
        )
    splits_mapping = {split: [] for split in ("train", "validation", "evaluation")}
    for episode in episodes:
        splits_mapping[episode["split"]].append(episode["episode_id"])
    dataset_path = tmp_path / "expert_traj_v1.npz"
    _write_ragged_npz(dataset_path, episodes=episodes, splits_mapping=splits_mapping)
    return dataset_path


def test_ragged_observations_field_is_one_dimensional(tmp_path: Path) -> None:
    """The real writer stores observations as a 1-D ragged object array, not 2-D rectangular.

    This pins the schema assumption these conformance tests guard: if numpy ever stores equal
    lengths as 2-D, the fixture would no longer reproduce the real ragged layout.
    """
    dataset_path = _build_ragged_fixture(tmp_path)
    with np.load(dataset_path, allow_pickle=True) as npz:
        observations = npz["observations"]
        actions = npz["actions"]
    assert observations.ndim == 1, observations.shape
    assert observations.shape == (4,)
    # actions are also ragged 1-D (each element a (steps, 2) array).
    assert actions.ndim == 1, actions.shape
    # Per-episode step counts differ -> this is the ragged layout, not rectangular padding.
    per_episode_steps = [len(observations[i]) for i in range(observations.shape[0])]
    assert per_episode_steps == [12, 1, 8, 5]


def test_loader_accepts_ragged_real_schema_and_partitions_by_split(tmp_path: Path) -> None:
    """The loader must load the real ragged collector schema, not only rectangular fixtures."""
    dataset_path = _build_ragged_fixture(tmp_path)
    dataset = load_expert_trajectory_dataset(dataset_path)

    assert dataset["episode_count"] == 4
    assert dataset["splits"]["train"] == [
        "train__planner_sanity_simple__seed201",
        "train__classic_head_on__seed202",
    ]
    assert dataset["splits"]["evaluation"] == ["evaluation__classic_doorway__seed111"]


def test_split_leakage_contract_passes_on_ragged_dataset(tmp_path: Path) -> None:
    """The split/leakage gate must hold on the real ragged schema before training starts."""
    dataset_path = _build_ragged_fixture(tmp_path)
    dataset = load_expert_trajectory_dataset(dataset_path)
    report = validate_split_leakage_contract(dataset)
    assert report["episode_counts"] == {"train": 2, "validation": 1, "evaluation": 1}
    assert report["train_seeds_disjoint_from_holdout"] is True
    assert report["evaluation_held_out_from_training"] is True


def test_flatten_keeps_only_training_steps_on_ragged_dataset(tmp_path: Path) -> None:
    """The flatten step must walk per-episode arrays on the ragged schema (train only)."""
    dataset_path = _build_ragged_fixture(tmp_path)
    dataset = load_expert_trajectory_dataset(dataset_path)
    features, targets, per_split_steps = flatten_observation_action_pairs(dataset)
    # 12 + 1 training steps; validation/evaluation held out.
    assert features.shape[0] == 13
    assert targets.shape == (13, 2)
    assert per_split_steps == {"train": 13}


def test_overfit_smoke_runs_end_to_end_on_ragged_real_schema(tmp_path: Path) -> None:
    """The full overfit smoke must run against the real ragged collector schema end to end."""
    dataset_path = _build_ragged_fixture(tmp_path)
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
    assert result.num_train_steps == 13
    assert result.episode_counts == {"train": 2, "validation": 1, "evaluation": 1}
    assert Path(result.checkpoint_path).is_file()
    assert Path(result.manifest_path).is_file()

    import json

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["smoke_not_quality"] is True
    assert manifest["not_full_warm_start_comparison"] is True
    assert manifest["per_split_steps"] == {"train": 13}


def test_loader_rejects_per_episode_observation_action_length_mismatch(tmp_path: Path) -> None:
    """A per-episode observation/action step mismatch must fail closed on the ragged schema."""
    dataset_path = _build_ragged_fixture(tmp_path)
    with np.load(dataset_path, allow_pickle=True) as npz:
        keep = {key: npz[key] for key in npz.files}
    # Corrupt one episode's actions to a different step count than its observations.
    observations = keep["observations"]
    actions = keep["actions"]
    episode_index = 0
    obs_steps = len(observations[episode_index])
    actions[episode_index] = np.zeros((obs_steps + 5, 2), dtype=np.float32)
    broken = dataset_path.with_name("broken.npz")
    np.savez(broken, **keep)
    with pytest.raises(OracleImitationBcSmokeError, match="disagree on step count"):
        load_expert_trajectory_dataset(broken)
