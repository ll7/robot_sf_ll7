"""Unit tests for balanced oracle imitation dataset collection and accounting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robot_sf.training.action_bin_accounting import compute_action_bin_accounting
from robot_sf.training.balanced_oracle_dataset_collector import (
    BalancedDatasetCollectionError,
    BalancedOracleCollector,
    validate_split_and_episode_invariants,
)
from robot_sf.training.oracle_imitation_bc_smoke import (
    BCSmokeConfig,
    run_bc_overfit_smoke,
)
from scripts.training.collect_balanced_oracle_imitation_dataset import main as cli_main

TEST_PACKET_PATH = Path(
    "configs/training/ppo_imitation/oracle_dataset_issue_6127_balanced_launch_packet.yaml"
)


def _make_obs(step: int) -> dict[str, Any]:
    """Build structured observation dict matching expert_traj_v1 schema."""
    return {
        "robot_position": np.array([1.0 + step * 0.1, 2.0], dtype=np.float32),
        "robot_heading": np.array([0.5], dtype=np.float32),
        "robot_speed": np.array([1.0, 0.0], dtype=np.float32),
        "robot_velocity_xy": np.array([1.0, 0.0], dtype=np.float32),
        "robot_angular_velocity": np.array([0.1], dtype=np.float32),
        "robot_radius": np.array([0.3], dtype=np.float32),
        "goal_current": np.array([10.0, 10.0], dtype=np.float32),
        "goal_next": np.array([20.0, 20.0], dtype=np.float32),
        "pedestrians_positions": np.zeros((64, 2), dtype=np.float32),
        "pedestrians_velocities": np.zeros((64, 2), dtype=np.float32),
        "pedestrians_radius": np.array([0.4], dtype=np.float32),
        "pedestrians_count": np.array([64.0], dtype=np.float32),
        "map_size": np.array([20.0, 10.0], dtype=np.float32),
        "sim_timestep": np.array([0.1], dtype=np.float32),
    }


def _make_episode(
    episode_id: str,
    scenario_id: str,
    seed: int,
    split: str,
    steps: int = 5,
    *,
    failed: bool = False,
    fallback: bool = False,
    leakage_invalid: bool = False,
) -> dict[str, Any]:
    obs_list = [_make_obs(s) for s in range(steps)]
    act_list = [np.array([0.5, 0.1], dtype=np.float32) for _ in range(steps)]
    pos_list = [np.array([float(s), 0.0], dtype=np.float32) for s in range(steps)]
    rew_list = [0.05] * steps
    return {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "split": split,
        "steps": steps,
        "actions": act_list,
        "observations": obs_list,
        "positions": pos_list,
        "rewards": rew_list,
        "failed": failed,
        "fallback": fallback,
        "leakage_invalid": leakage_invalid,
    }


def test_preflight_only(tmp_path: Path) -> None:
    """The --preflight flag performs no simulation and writes a deterministic launch plan JSON."""
    output_root = tmp_path / "preflight_out"
    exit_code = cli_main(
        [
            "--config",
            str(TEST_PACKET_PATH),
            "--output-root",
            str(output_root),
            "--preflight",
            "--json",
        ]
    )
    assert exit_code == 0
    plan_path = output_root / "balanced_oracle_collection_plan.json"
    assert plan_path.is_file()
    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan_data["schema_version"] == "balanced-oracle-collection-plan.v1"
    assert plan_data["dataset_id"] == "issue_6127_balanced_oracle_v1"
    assert "planned_strata" in plan_data


def test_split_overlap_detection() -> None:
    """Validate that seed overlap between splits raises BalancedDatasetCollectionError."""
    bad_packet = {
        "seeds_by_split": {
            "train": [101, 102],
            "validation": [102, 103],
            "evaluation": [104],
        },
        "episode_ids_by_split": {
            "train": ["train__sc__seed101", "train__sc__seed102"],
            "validation": ["validation__sc__seed102_dup", "validation__sc__seed103"],
            "evaluation": ["evaluation__sc__seed104"],
        },
    }
    with pytest.raises(BalancedDatasetCollectionError, match="Seed overlap detected"):
        validate_split_and_episode_invariants(bad_packet)


def test_duplicate_episode_id_detection() -> None:
    """Validate that duplicate episode IDs across splits raise BalancedDatasetCollectionError."""
    dup_packet = {
        "seeds_by_split": {
            "train": [101],
            "validation": [102],
            "evaluation": [103],
        },
        "episode_ids_by_split": {
            "train": ["same_id"],
            "validation": ["same_id"],
            "evaluation": ["other_id"],
        },
    }
    with pytest.raises(BalancedDatasetCollectionError, match="Duplicate episode ID"):
        validate_split_and_episode_invariants(dup_packet)


def test_one_step_and_fallback_exclusion(tmp_path: Path) -> None:
    """One-step, failed, and fallback episodes are retained in exclusions and excluded from usable counts."""
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=tmp_path)
    scenarios = collector.scenario_ids

    episodes: list[dict[str, Any]] = []
    for sc in scenarios:
        for seed in range(10):
            episodes.append(
                _make_episode(f"train__{sc}__seed200{seed}", sc, 200 + seed, "train", steps=5)
            )

    episodes.append(
        _make_episode(
            "train__planner_sanity_simple__seed999", "planner_sanity_simple", 999, "train", steps=1
        )
    )
    episodes.append(
        _make_episode(
            "train__planner_sanity_simple__seed998",
            "planner_sanity_simple",
            998,
            "train",
            steps=5,
            fallback=True,
        )
    )
    episodes.append(
        _make_episode(
            "train__planner_sanity_simple__seed997",
            "planner_sanity_simple",
            997,
            "train",
            steps=5,
            failed=True,
        )
    )

    manifest = collector.collect_dataset(
        episodes_override=episodes,
        allow_insufficient_yield=True,
    )

    exclusions = manifest["exclusions"]
    reasons = [e["reason"] for e in exclusions]
    assert "one-step" in reasons
    assert "fallback" in reasons
    assert "failed" in reasons

    balance_summary = manifest["balance_summary"]
    assert balance_summary["total_excluded_episodes"] >= 3


def test_action_bin_accounting_determinism() -> None:
    """Action-bin accounting produces deterministic weights and summary for identical inputs."""
    actions1 = np.array([[0.5, -0.2], [1.2, 0.1], [-0.5, 0.8], [0.5, -0.2]], dtype=np.float32)
    actions2 = np.array([[0.5, -0.2], [1.2, 0.1], [-0.5, 0.8], [0.5, -0.2]], dtype=np.float32)

    weights1, summary1 = compute_action_bin_accounting(actions1)
    weights2, summary2 = compute_action_bin_accounting(actions2)

    np.testing.assert_array_almost_equal(weights1, weights2)
    assert summary1 == summary2
    assert summary1["total_transitions"] == 4
    assert summary1["active_bins"] == 3


def test_insufficient_yield_termination(tmp_path: Path) -> None:
    """Collector fails closed when usable transitions < min_usable_transitions or stratum < min_episodes_per_stratum."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1000,
        min_episodes_per_stratum=5,
    )
    scenarios = collector.scenario_ids

    episodes: list[dict[str, Any]] = []
    for sc in scenarios[:1]:
        for seed in range(2):
            episodes.append(
                _make_episode(f"train__{sc}__seed200{seed}", sc, 200 + seed, "train", steps=10)
            )

    with pytest.raises(BalancedDatasetCollectionError, match="Insufficient yield"):
        collector.collect_dataset(episodes_override=episodes, allow_insufficient_yield=False)


def test_bc_loader_smoke_fixture_pass(tmp_path: Path) -> None:
    """Produce a schema-faithful fixture and verify it passes existing BC loader smoke without fallback."""
    output_root = tmp_path / "bc_smoke_fixture"
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=output_root)
    scenarios = collector.scenario_ids

    episodes: list[dict[str, Any]] = []
    for sc in scenarios:
        for seed in range(2):
            ep_id = f"train__{sc}__seed20{seed}"
            episodes.append(_make_episode(ep_id, sc, 200 + seed, "train", steps=10))

    for ep_id in collector.packet["episode_ids_by_split"]["validation"]:
        _, sc, seed = ep_id.split("__")
        s_num = int(seed.replace("seed", ""))
        episodes.append(_make_episode(ep_id, sc, s_num, "validation", steps=10))

    for ep_id in collector.packet["episode_ids_by_split"]["evaluation"]:
        _, sc, seed = ep_id.split("__")
        s_num = int(seed.replace("seed", ""))
        episodes.append(_make_episode(ep_id, sc, s_num, "evaluation", steps=10))

    manifest = collector.collect_dataset(
        episodes_override=episodes,
        allow_insufficient_yield=True,
    )
    npz_path = Path(manifest["npz_path"])
    assert npz_path.is_file()

    smoke_out = tmp_path / "smoke_out"
    smoke_config = BCSmokeConfig(
        dataset_path=str(npz_path),
        output_dir=str(smoke_out),
        epochs=10,
        seed=42,
    )
    result = run_bc_overfit_smoke(smoke_config)

    assert result.num_train_steps > 0
    assert result.loss_reduction > 0.0
    assert result.episode_counts["train"] > 0
    assert result.episode_counts["validation"] > 0
    assert result.episode_counts["evaluation"] > 0
