"""Unit tests for balanced oracle imitation dataset collection and accounting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robot_sf.training import balanced_oracle_dataset_collector as collector_module
from robot_sf.training.action_bin_accounting import compute_action_bin_accounting
from robot_sf.training.balanced_oracle_dataset_collector import (
    BalancedDatasetCollectionError,
    BalancedOracleCollector,
    check_yield_status,
    compute_packet_fingerprint,
    validate_split_and_episode_invariants,
)
from robot_sf.training.oracle_imitation_bc_smoke import (
    BCSmokeConfig,
    run_bc_overfit_smoke,
)
from scripts.training.collect_balanced_oracle_imitation_dataset import main as cli_main
from scripts.validation.validate_balanced_oracle_manifest import validate_balanced_manifest

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
        "terminated": [False] * (steps - 1) + [True],
        "truncated": [False] * steps,
        "failed": failed,
        "fallback": fallback,
        "leakage_invalid": leakage_invalid,
        "provenance": {"fixture": True},
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
    assert set(plan_data["planned_train_episodes_per_stratum"].values()) == {50}
    assert plan_data["git_commit"] == plan_data["git_commit"].lower()

    second_root = tmp_path / "preflight_out_again"
    assert (
        cli_main(
            [
                "--config",
                str(TEST_PACKET_PATH),
                "--output-root",
                str(second_root),
                "--preflight",
                "--json",
            ]
        )
        == 0
    )
    assert (
        plan_path.read_bytes()
        == (second_root / "balanced_oracle_collection_plan.json").read_bytes()
    )


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
            "train": ["train__sc__seed101"],
            "validation": ["train__sc__seed101"],
            "evaluation": ["evaluation__sc__seed103"],
        },
    }
    with pytest.raises(BalancedDatasetCollectionError, match="Duplicate episode ID"):
        validate_split_and_episode_invariants(dup_packet)


def test_one_step_and_fallback_exclusion(tmp_path: Path) -> None:
    """One-step, failed, and fallback episodes are retained in exclusions and excluded from usable counts."""
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=tmp_path)
    scenarios = collector.scenario_ids
    episodes: list[dict[str, Any]] = []
    ids_by_scenario: dict[str, list[str]] = {scenario: [] for scenario in scenarios}
    for episode_id in collector.episodes_by_split["train"]:
        _, scenario, _ = episode_id.split("__")
        ids_by_scenario[scenario].append(episode_id)
    for scenario in scenarios:
        for episode_id in ids_by_scenario[scenario][:10]:
            _, parsed_scenario, seed_token = episode_id.split("__")
            episodes.append(
                _make_episode(
                    episode_id,
                    parsed_scenario,
                    int(seed_token.removeprefix("seed")),
                    "train",
                    steps=5,
                )
            )

    flagged = ids_by_scenario[scenarios[0]][10:13]
    for episode_id, kwargs in zip(
        flagged,
        ({"steps": 1}, {"steps": 5, "fallback": True}, {"steps": 5, "failed": True}),
        strict=True,
    ):
        _, scenario, seed_token = episode_id.split("__")
        episodes.append(
            _make_episode(
                episode_id,
                scenario,
                int(seed_token.removeprefix("seed")),
                "train",
                **kwargs,
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


def test_invalid_and_failed_rows_remain_distinct(tmp_path: Path) -> None:
    """Malformed and failed zero-step rows stay visible under separate reasons."""
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=tmp_path)
    episode_id = collector.episodes_by_split["train"][0]
    _, scenario, seed_token = episode_id.split("__")
    invalid = _make_episode(
        episode_id,
        scenario,
        int(seed_token.removeprefix("seed")),
        "train",
        steps=2,
    )
    invalid["actions"] = None
    failed = _make_episode(
        collector.episodes_by_split["train"][1],
        scenario,
        int(seed_token.removeprefix("seed")) + 1,
        "train",
        steps=0,
        failed=True,
    )
    failed["provenance"] = {"collection_error": "simulator failed"}

    _usable, exclusions = collector._filter_episodes([invalid, failed])

    assert [item["reason"] for item in exclusions] == ["invalid", "failed"]
    assert exclusions[0]["steps"] == 0
    assert exclusions[1]["steps"] == 0


def test_misaligned_trajectory_is_invalid_not_usable(tmp_path: Path) -> None:
    """A row with misaligned trajectory fields is preserved as an invalid exclusion."""
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=tmp_path)
    episode_id = collector.episodes_by_split["train"][0]
    _, scenario, seed_token = episode_id.split("__")
    malformed = _make_episode(
        episode_id,
        scenario,
        int(seed_token.removeprefix("seed")),
        "train",
        steps=2,
    )
    malformed["observations"] = malformed["observations"][:1]

    usable, exclusions = collector._filter_episodes([malformed])

    assert usable == []
    assert exclusions[0]["reason"] == "invalid"
    assert exclusions[0]["steps"] == 0


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
    episodes: list[dict[str, Any]] = []
    for episode_id in collector.episodes_by_split["train"][:2]:
        _, scenario, seed_token = episode_id.split("__")
        episodes.append(
            _make_episode(
                episode_id,
                scenario,
                int(seed_token.removeprefix("seed")),
                "train",
                steps=10,
            )
        )

    with pytest.raises(BalancedDatasetCollectionError, match="Insufficient yield"):
        collector.collect_dataset(episodes_override=episodes, allow_insufficient_yield=False)


def test_bc_loader_smoke_fixture_pass(tmp_path: Path) -> None:
    """Produce a schema-faithful fixture and verify it passes existing BC loader smoke without fallback."""
    output_root = tmp_path / "bc_smoke_fixture"
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=output_root)
    scenarios = collector.scenario_ids

    episodes: list[dict[str, Any]] = []
    for scenario in scenarios:
        matching_ids = [
            episode_id
            for episode_id in collector.episodes_by_split["train"]
            if episode_id.startswith(f"train__{scenario}__")
        ]
        for episode_id in matching_ids[:2]:
            _, parsed_scenario, seed_token = episode_id.split("__")
            episodes.append(
                _make_episode(
                    episode_id,
                    parsed_scenario,
                    int(seed_token.removeprefix("seed")),
                    "train",
                    steps=10,
                )
            )

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


def _all_packet_episodes(
    collector: BalancedOracleCollector, *, steps: int = 2
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for split, episode_ids in collector.episodes_by_split.items():
        for episode_id in episode_ids:
            _, scenario, seed_token = episode_id.split("__")
            episodes.append(
                _make_episode(
                    episode_id,
                    scenario,
                    int(seed_token.removeprefix("seed")),
                    split,
                    steps=steps,
                )
            )
    return episodes


def test_production_cli_collects_without_test_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The non-preflight CLI must call the real source-collection seam."""
    calls: list[tuple[int, float]] = []

    def fake_collect(
        collector: BalancedOracleCollector, *, horizon: int, dt: float
    ) -> list[dict[str, Any]]:
        calls.append((horizon, dt))
        return _all_packet_episodes(collector)

    monkeypatch.setattr(BalancedOracleCollector, "collect_source_episodes", fake_collect)
    output_root = tmp_path / "real_cli"
    assert (
        cli_main(
            [
                "--config",
                str(TEST_PACKET_PATH),
                "--output-root",
                str(output_root),
                "--min-usable-transitions",
                "1",
                "--min-episodes-per-stratum",
                "1",
                "--horizon",
                "7",
                "--dt",
                "0.2",
                "--json",
            ]
        )
        == 0
    )
    assert calls == [(7, 0.2)]
    report = validate_balanced_manifest(output_root / "balanced_oracle_dataset_manifest.json")
    manifest = json.loads(
        (output_root / "balanced_oracle_dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "valid"
    assert manifest["eligibility_status"] == "training_ready"
    assert len(manifest["exact_public_sha"]) == 40
    assert manifest["exact_public_sha"] != manifest["dataset_sha256"]
    assert manifest["generating_commit"] == manifest["exact_public_sha"]
    assert manifest["source_candidate_config"].endswith(
        "hybrid_rule_v3_static_margin0_waypoint2.yaml"
    )
    assert (
        manifest["scenario_ids"]
        == collector_module.BalancedOracleCollector(
            TEST_PACKET_PATH, output_root=tmp_path / "packet_contract"
        ).scenario_ids
    )
    assert manifest["seeds_by_split"]["train"] == list(range(1001, 1301))
    assert len(manifest["episode_ids_by_split"]["train"]) == 300
    assert manifest["hard_slice_assignment"]
    assert manifest["relabeling_policy"] is None
    assert manifest["checksums"] == manifest["sha256_inventory"]
    assert set(manifest["artifact_paths"].values()) == set(manifest["checksums"])

    with np.load(Path(manifest["npz_path"]), allow_pickle=True) as dataset:
        assert dataset["observations"].ndim == 1
        assert dataset["actions"].ndim == 1
        assert dataset["action_balance_weights"].ndim == 1
        assert int(dataset["episode_count"].item()) == 306
        metadata = dataset["metadata"].item()
        assert isinstance(metadata, dict)
        assert (
            metadata["splits"]["train"]["episode_ids"]
            == manifest["private_artifact_registry_candidate"]["splits"]["train"]["episode_ids"]
        )


def test_capture_episode_records_policy_io_and_fallback_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generalized job-13520 CaptureEnv seam records aligned policy I/O."""

    class FakeEnv:
        def reset(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
            return _make_obs(0), {}

        def step(
            self, action: Any
        ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
            return _make_obs(1), 0.1, False, False, {}

    monkeypatch.setattr(collector_module.map_runner, "make_robot_env", FakeEnv)
    episode_factory_before = collector_module.map_runner._map_runner_episode_module.make_robot_env
    execution_mode = "adapter"

    def fake_run(scenario: dict[str, Any], seed: int, **_kwargs: Any) -> dict[str, Any]:
        env = collector_module.map_runner.make_robot_env()
        env.reset()
        env.step(np.array([0.2, -0.1], dtype=np.float32))
        env.step(np.array([0.3, 0.1], dtype=np.float32))
        return {
            "status": "success",
            "scenario_id": "planner_sanity_simple",
            "seed": seed,
            "algorithm_metadata": {
                "simulation_step_trace": {
                    "steps": [
                        {"robot": {"position": [0.0, 0.0]}},
                        {"robot": {"position": [0.1, 0.0]}},
                    ]
                },
                "planner_kinematics": {"execution_mode": execution_mode},
                "planner_runtime": {"fallback_count": 0},
            },
            "pedestrian_model": {"fallback_degraded_status": "native"},
        }

    monkeypatch.setattr(collector_module, "_run_map_episode", fake_run)
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=tmp_path)
    episode = collector._capture_episode(
        {"name": "planner_sanity_simple"},
        seed=1001,
        split="train",
        episode_id="train__planner_sanity_simple__seed1001",
        algo="hybrid_rule_local_planner",
        algo_config={},
        scenario_path=Path("configs/policy_search/nominal_sanity_matrix.yaml"),
        horizon=2,
        dt=0.1,
    )
    assert len(episode["actions"]) == len(episode["observations"]) == 2
    np.testing.assert_array_equal(
        episode["observations"][0]["robot_position"], _make_obs(0)["robot_position"]
    )
    np.testing.assert_array_equal(
        episode["observations"][1]["robot_position"], _make_obs(1)["robot_position"]
    )
    assert episode["fallback"] is False
    assert episode["degraded"] is False
    assert episode["leakage_invalid"] is False
    assert (
        collector_module.map_runner._map_runner_episode_module.make_robot_env
        is episode_factory_before
    )

    execution_mode = "mixed"
    mixed = collector._capture_episode(
        {"name": "planner_sanity_simple"},
        seed=1001,
        split="train",
        episode_id="train__planner_sanity_simple__seed1001",
        algo="hybrid_rule_local_planner",
        algo_config={},
        scenario_path=Path("configs/policy_search/nominal_sanity_matrix.yaml"),
        horizon=2,
        dt=0.1,
    )
    assert mixed["fallback"] is False

    def fake_degraded_run(scenario: dict[str, Any], seed: int, **kwargs: Any) -> dict[str, Any]:
        record = fake_run(scenario, seed, **kwargs)
        record["algorithm_metadata"]["planner_runtime"]["nested"] = {"fallback_or_degraded": True}
        return record

    monkeypatch.setattr(collector_module, "_run_map_episode", fake_degraded_run)
    degraded = collector._capture_episode(
        {"name": "planner_sanity_simple"},
        seed=1001,
        split="train",
        episode_id="train__planner_sanity_simple__seed1001",
        algo="hybrid_rule_local_planner",
        algo_config={},
        scenario_path=Path("configs/policy_search/nominal_sanity_matrix.yaml"),
        horizon=2,
        dt=0.1,
    )
    assert degraded["fallback"] is True

    def fake_missing_identity_run(
        scenario: dict[str, Any], seed: int, **kwargs: Any
    ) -> dict[str, Any]:
        record = fake_run(scenario, seed, **kwargs)
        record.pop("scenario_id")
        record.pop("seed")
        return record

    monkeypatch.setattr(collector_module, "_run_map_episode", fake_missing_identity_run)
    missing_identity = collector._capture_episode(
        {"name": "planner_sanity_simple"},
        seed=1001,
        split="train",
        episode_id="train__planner_sanity_simple__seed1001",
        algo="hybrid_rule_local_planner",
        algo_config={},
        scenario_path=Path("configs/policy_search/nominal_sanity_matrix.yaml"),
        horizon=2,
        dt=0.1,
    )
    assert missing_identity["leakage_invalid"] is True


@pytest.mark.parametrize("raw_seed", [1001.0, "1001", True], ids=["float", "string", "bool"])
def test_capture_episode_rejects_coerced_runtime_seed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_seed: object
) -> None:
    """Runtime seeds must be strict integers before identity validation can pass."""
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=tmp_path)

    def fake_run(_scenario: dict[str, Any], _seed: int, **_kwargs: Any) -> dict[str, Any]:
        return {
            "status": "success",
            "scenario_id": "planner_sanity_simple",
            "seed": raw_seed,
            "algorithm_metadata": {
                "simulation_step_trace": {"steps": []},
                "planner_kinematics": {"execution_mode": "native"},
                "planner_runtime": {"fallback_count": 0},
            },
            "pedestrian_model": {"fallback_degraded_status": "native"},
        }

    monkeypatch.setattr(collector_module, "_run_map_episode", fake_run)
    episode = collector._capture_episode(
        {"name": "planner_sanity_simple"},
        seed=1001,
        split="train",
        episode_id="train__planner_sanity_simple__seed1001",
        algo="hybrid_rule_local_planner",
        algo_config={},
        scenario_path=Path("configs/policy_search/nominal_sanity_matrix.yaml"),
        horizon=2,
        dt=0.1,
    )

    assert episode["leakage_invalid"] is True
    assert episode["provenance"]["identity_missing"] is True


def test_diagnostic_insufficient_manifest_is_rejected(
    tmp_path: Path,
) -> None:
    """A bypass artifact remains diagnostic and cannot pass manifest validation."""
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=tmp_path)
    episode_id = collector.episodes_by_split["train"][0]
    _, scenario, seed_token = episode_id.split("__")
    manifest = collector.collect_dataset(
        episodes_override=[
            _make_episode(
                episode_id,
                scenario,
                int(seed_token.removeprefix("seed")),
                "train",
                steps=2,
            )
        ],
        allow_insufficient_yield=True,
    )
    assert manifest["eligibility_status"] == "diagnostic_insufficient_yield"
    assert manifest["private_artifact_registry_candidate"] is None
    with pytest.raises(ValueError, match="eligibility_status"):
        validate_balanced_manifest(Path(manifest["manifest_path"]))


def test_identity_mismatch_is_excluded_as_leakage_invalid(tmp_path: Path) -> None:
    """Runtime identity drift is retained in provenance but excluded from usable evidence."""
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=tmp_path)
    episode_id = collector.episodes_by_split["train"][0]
    _, scenario, seed_token = episode_id.split("__")
    episode = _make_episode(
        episode_id,
        scenario,
        int(seed_token.removeprefix("seed")) + 1,
        "train",
        steps=2,
    )
    manifest = collector.collect_dataset(episodes_override=[episode], allow_insufficient_yield=True)
    assert manifest["exclusions"][0]["reason"] == "leakage_invalid"
    provenance_lines = Path(manifest["raw_provenance_path"]).read_text(encoding="utf-8")
    assert "identity_mismatches" in provenance_lines


def test_one_step_identity_mismatch_cannot_become_training_ready(tmp_path: Path) -> None:
    """Leakage is terminal even when a stronger exclusion would otherwise hide its reason."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    episodes = _all_packet_episodes(collector)
    mismatched = next(ep for ep in episodes if ep["split"] == "validation")
    mismatched["seed"] += 1
    for field in ("actions", "observations", "positions", "rewards", "terminated", "truncated"):
        mismatched[field] = mismatched[field][:1]

    manifest = collector.collect_dataset(
        episodes_override=episodes,
        allow_insufficient_yield=True,
    )

    assert manifest["eligibility_status"] == "diagnostic_insufficient_yield"
    assert manifest["private_artifact_registry_candidate"] is None
    assert any(row["reason"] == "leakage_invalid" for row in manifest["exclusions"])
    with pytest.raises(ValueError, match="eligibility_status|leakage"):
        validate_balanced_manifest(Path(manifest["manifest_path"]))

    strict_collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path / "strict",
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    with pytest.raises(BalancedDatasetCollectionError, match="Leakage-invalid"):
        strict_collector.collect_dataset(episodes_override=episodes)


def test_manifest_validator_requires_split_contract_fields(tmp_path: Path) -> None:
    """A training-ready manifest cannot omit the inherited #1397 split contract."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    manifest = collector.collect_dataset(episodes_override=_all_packet_episodes(collector))
    manifest_path = Path(manifest["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("source_candidate_config", None)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source_candidate_config"):
        validate_balanced_manifest(manifest_path)


def test_yield_ledger_present_in_manifest(tmp_path: Path) -> None:
    """Manifest includes yield_ledger with per-stratum declared/attempted/completed/usable counts."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    manifest = collector.collect_dataset(episodes_override=_all_packet_episodes(collector))

    ledger = manifest.get("yield_ledger")
    assert isinstance(ledger, dict)
    assert ledger["schema_version"] == "yield-ledger.v1"
    assert "strata" in ledger
    assert "totals" in ledger
    assert "lineage" in ledger
    assert "differential_remedy" in ledger

    for split in ("train", "validation", "evaluation"):
        assert split in ledger["strata"]
        for sc_id in collector.scenario_ids:
            stats = ledger["strata"][split][sc_id]
            for field in (
                "declared",
                "attempted",
                "completed",
                "usable",
                "nondegenerate",
                "excluded",
                "failed",
                "missing",
                "usable_transitions",
                "reason_counts",
                "target_minimum",
                "shortfall",
            ):
                assert field in stats, f"Missing {field} in {split}/{sc_id}"


def test_yield_ledger_totals_consistent(tmp_path: Path) -> None:
    """Yield ledger totals are consistent with per-stratum counts."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    manifest = collector.collect_dataset(episodes_override=_all_packet_episodes(collector))
    ledger = manifest["yield_ledger"]

    totals = ledger["totals"]
    assert totals["declared"] == 306
    assert totals["attempted"] == 306
    assert totals["usable"] == 306
    assert totals["excluded"] == 0
    assert totals["missing"] == 0


def test_yield_ledger_observed_shortfall(tmp_path: Path) -> None:
    """Yield ledger reports shortfall when usable < minimum per stratum."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=10,
    )
    episodes = _all_packet_episodes(collector)
    doorway_eps = [ep for ep in episodes if "classic_doorway_low" in ep["episode_id"]]
    non_doorway_eps = [ep for ep in episodes if "classic_doorway_low" not in ep["episode_id"]]
    only_one_doorway = doorway_eps[:1]
    manifest = collector.collect_dataset(
        episodes_override=non_doorway_eps + only_one_doorway,
        allow_insufficient_yield=True,
    )
    ledger = manifest["yield_ledger"]
    doorway_stats = ledger["strata"]["train"]["classic_doorway_low"]
    assert doorway_stats["usable"] == 1
    assert doorway_stats["nondegenerate"] == 1
    assert doorway_stats["target_minimum"] == 10
    assert doorway_stats["shortfall"] == 9


def test_decision_packet_emitted(tmp_path: Path) -> None:
    """A public-safe decision packet JSON is written alongside the manifest."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    collector.collect_dataset(episodes_override=_all_packet_episodes(collector))

    dp_path = tmp_path / "yield_decision_packet.json"
    assert dp_path.is_file()
    dp = json.loads(dp_path.read_text(encoding="utf-8"))
    assert dp["schema_version"] == "yield-decision-packet.v1"
    assert dp["check_status"] in {
        "eligible_complete",
        "blocked_scientific_yield",
        "blocked_integrity_or_lineage",
        "inconclusive_missing_input",
    }
    assert dp["dataset_id"] == "issue_6127_balanced_oracle_v1"
    assert isinstance(dp["packet_fingerprint"], str)
    assert len(dp["packet_fingerprint"]) == 64


def test_decision_packet_shows_shortfall(tmp_path: Path) -> None:
    """Decision packet reports observed shortfall for strata below minimum."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=10,
    )
    episodes = _all_packet_episodes(collector)
    doorway_eps = [ep for ep in episodes if "classic_doorway_low" in ep["episode_id"]]
    non_doorway_eps = [ep for ep in episodes if "classic_doorway_low" not in ep["episode_id"]]
    only_one_doorway = doorway_eps[:1]
    collector.collect_dataset(
        episodes_override=non_doorway_eps + only_one_doorway,
        allow_insufficient_yield=True,
    )
    dp = json.loads((tmp_path / "yield_decision_packet.json").read_text(encoding="utf-8"))
    assert "train/classic_doorway_low" in dp["observed"]
    observed = dp["observed"]["train/classic_doorway_low"]
    assert observed["usable"] == 1
    assert observed["minimum"] == 10
    assert observed["shortfall"] == 9


def test_packet_fingerprint_in_manifest(tmp_path: Path) -> None:
    """Manifest includes a canonical scientific/execution packet fingerprint."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    manifest = collector.collect_dataset(episodes_override=_all_packet_episodes(collector))
    assert isinstance(manifest["packet_fingerprint"], str)
    assert len(manifest["packet_fingerprint"]) == 64
    expected = compute_packet_fingerprint(collector.packet)
    assert manifest["packet_fingerprint"] == expected


def test_yield_check_eligible_complete(tmp_path: Path) -> None:
    """check_yield_status returns eligible_complete when all gates pass."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    collector.collect_dataset(episodes_override=_all_packet_episodes(collector))
    result = check_yield_status(tmp_path)
    assert result["check_status"] == "eligible_complete"


def test_yield_check_missing_manifest(tmp_path: Path) -> None:
    """check_yield_status returns inconclusive_missing_input when no manifest exists."""
    result = check_yield_status(tmp_path / "nonexistent")
    assert result["check_status"] == "inconclusive_missing_input"


def test_yield_check_blocked_yield(tmp_path: Path) -> None:
    """check_yield_status returns blocked_scientific_yield when yield gates fail."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=10000,
        min_episodes_per_stratum=10,
    )
    episodes = _all_packet_episodes(collector, steps=2)
    collector.collect_dataset(
        episodes_override=episodes,
        allow_insufficient_yield=True,
    )
    result = check_yield_status(tmp_path)
    assert result["check_status"] == "blocked_scientific_yield"


def test_yield_check_blocked_integrity_leakage(tmp_path: Path) -> None:
    """check_yield_status returns blocked_integrity_or_lineage for leakage."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    episodes = _all_packet_episodes(collector)
    mismatched = episodes[0]
    mismatched["seed"] += 1
    for field in ("actions", "observations", "positions", "rewards", "terminated", "truncated"):
        mismatched[field] = mismatched[field][:1]
    collector.collect_dataset(
        episodes_override=episodes,
        allow_insufficient_yield=True,
    )
    result = check_yield_status(tmp_path)
    assert result["check_status"] == "blocked_integrity_or_lineage"


def test_yield_check_fingerprint_mismatch(tmp_path: Path) -> None:
    """check_yield_status returns blocked_integrity_or_lineage on fingerprint mismatch."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    collector.collect_dataset(episodes_override=_all_packet_episodes(collector))
    fake_config = tmp_path / "different_packet.yaml"
    fake_config.write_text("different: content\n", encoding="utf-8")
    result = check_yield_status(tmp_path, config_path=fake_config)
    assert result["check_status"] == "blocked_integrity_or_lineage"
    assert "fingerprint" in result["reason"].lower()


def test_yield_check_missing_config_is_inconclusive(tmp_path: Path) -> None:
    """A missing configured packet cannot be silently treated as an unverified check."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    collector.collect_dataset(episodes_override=_all_packet_episodes(collector))

    missing_config = tmp_path / "missing_packet.yaml"
    result = check_yield_status(tmp_path, config_path=missing_config)
    assert result["check_status"] == "inconclusive_missing_input"
    assert "unavailable" in result["reason"].lower()

    exit_code = cli_main(
        [
            "--config",
            str(missing_config),
            "--output-root",
            str(tmp_path),
            "--yield-check",
            "--json",
        ]
    )
    assert exit_code == 0


def test_yield_check_rejects_malformed_manifest_shape(tmp_path: Path) -> None:
    """A syntactically valid non-object manifest is missing required check input."""
    manifest_path = tmp_path / "balanced_oracle_dataset_manifest.json"
    manifest_path.write_text("[]\n", encoding="utf-8")

    result = check_yield_status(tmp_path)
    assert result["check_status"] == "inconclusive_missing_input"
    assert "json object" in result["reason"].lower()


def test_yield_check_deterministic(tmp_path: Path) -> None:
    """check_yield_status is deterministic for identical inputs."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    collector.collect_dataset(episodes_override=_all_packet_episodes(collector))
    r1 = check_yield_status(tmp_path)
    r2 = check_yield_status(tmp_path)
    assert r1 == r2


def test_yield_check_rejects_malformed_packet_fingerprint(tmp_path: Path) -> None:
    """A yield check must not trust a non-hash packet fingerprint without config input."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    manifest = collector.collect_dataset(episodes_override=_all_packet_episodes(collector))
    manifest["packet_fingerprint"] = "not-a-sha256"
    Path(manifest["manifest_path"]).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result = check_yield_status(tmp_path)

    assert result["check_status"] == "blocked_integrity_or_lineage"
    assert "packet fingerprint" in result["reason"].lower()


def test_yield_check_rejects_tampered_packet_fingerprint_payload(tmp_path: Path) -> None:
    """A packet fingerprint must cover the durable payload it claims to identify."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    manifest = collector.collect_dataset(episodes_override=_all_packet_episodes(collector))
    manifest["packet_fingerprint_payload"]["dataset_id"] = "tampered"
    Path(manifest["manifest_path"]).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result = check_yield_status(tmp_path)

    assert result["check_status"] == "blocked_integrity_or_lineage"
    assert "does not match" in result["reason"].lower()


def test_yield_check_rejects_ledger_identity_defects(tmp_path: Path) -> None:
    """The durable ledger identity-defect field is part of the fail-closed check."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    manifest = collector.collect_dataset(episodes_override=_all_packet_episodes(collector))
    manifest["identity_defects"] = []
    manifest["yield_ledger"]["identity_defects"] = [{"kind": "tampered_identity"}]
    Path(manifest["manifest_path"]).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result = check_yield_status(tmp_path)

    assert result["check_status"] == "blocked_integrity_or_lineage"
    assert "identity defect" in result["reason"].lower()


def test_yield_check_rejects_inconsistent_pass_gate(tmp_path: Path) -> None:
    """A forged pass flag cannot override inconsistent per-stratum counts."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    collector.collect_dataset(episodes_override=_all_packet_episodes(collector))
    manifest_path = tmp_path / "balanced_oracle_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = manifest["yield_ledger"]["strata"]["train"][collector.scenario_ids[0]]
    stats["usable"] = 0
    stats["nondegenerate"] = 0
    stats["shortfall"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = check_yield_status(tmp_path)

    assert result["check_status"] == "blocked_integrity_or_lineage"
    assert "Yield ledger" in result["reason"]


def test_yield_check_rejects_unaccounted_attempted_rows(tmp_path: Path) -> None:
    """A forged attempted count cannot exceed the declared packet roster."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    collector.collect_dataset(episodes_override=_all_packet_episodes(collector))
    manifest_path = tmp_path / "balanced_oracle_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = manifest["yield_ledger"]["strata"]["train"][collector.scenario_ids[0]]
    stats["attempted"] += 1
    manifest["yield_ledger"]["totals"]["attempted"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = check_yield_status(tmp_path)

    assert result["check_status"] == "blocked_integrity_or_lineage"
    assert "attempted" in result["reason"]


def test_yield_check_rejects_selected_differential_remedy(tmp_path: Path) -> None:
    """A status check cannot treat a selected remedy as an implementation result."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    collector.collect_dataset(episodes_override=_all_packet_episodes(collector))
    manifest_path = tmp_path / "balanced_oracle_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["yield_ledger"]["differential_remedy"]["selected"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = check_yield_status(tmp_path)

    assert result["check_status"] == "blocked_integrity_or_lineage"
    assert "unselected" in result["reason"]


def test_yield_check_rejects_unexpected_ledger_keys(tmp_path: Path) -> None:
    """Unexpected ledger splits or strata cannot crash or bypass exact-key validation."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    collector.collect_dataset(episodes_override=_all_packet_episodes(collector))
    manifest_path = tmp_path / "balanced_oracle_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["yield_ledger"]["strata"]["unexpected"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = check_yield_status(tmp_path)

    assert result["check_status"] == "blocked_integrity_or_lineage"
    assert "strata keys" in result["reason"]

    del manifest["yield_ledger"]["strata"]["unexpected"]
    manifest["yield_ledger"]["strata"]["train"]["unexpected"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = check_yield_status(tmp_path)

    assert result["check_status"] == "blocked_integrity_or_lineage"
    assert "unexpected scenarios" in result["reason"]


def test_yield_check_side_effect_free(tmp_path: Path) -> None:
    """check_yield_status does not modify files in the output root."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    collector.collect_dataset(episodes_override=_all_packet_episodes(collector))
    manifest_mtime = (tmp_path / "balanced_oracle_dataset_manifest.json").stat().st_mtime_ns
    check_yield_status(tmp_path)
    assert (tmp_path / "balanced_oracle_dataset_manifest.json").stat().st_mtime_ns == manifest_mtime


def test_differential_remedy_in_ledger(tmp_path: Path) -> None:
    """yield_ledger includes exactly one differential_remedy with category and rationale."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    manifest = collector.collect_dataset(episodes_override=_all_packet_episodes(collector))
    remedy = manifest["yield_ledger"]["differential_remedy"]
    assert "category" in remedy
    assert "rationale" in remedy
    assert isinstance(remedy["category"], str)
    assert isinstance(remedy["rationale"], str)


def test_differential_remedy_shortfall_category(tmp_path: Path) -> None:
    """differential_remedy records an unselected allowed category for a shortfall."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=10,
    )
    episodes = _all_packet_episodes(collector)
    doorway_eps = [ep for ep in episodes if "classic_doorway_low" in ep["episode_id"]]
    non_doorway_eps = [ep for ep in episodes if "classic_doorway_low" not in ep["episode_id"]]
    only_one_doorway = doorway_eps[:1]
    manifest = collector.collect_dataset(
        episodes_override=non_doorway_eps + only_one_doorway,
        allow_insufficient_yield=True,
    )
    remedy = manifest["yield_ledger"]["differential_remedy"]
    assert remedy["category"] == "budget_or_sampling_change"
    assert remedy["selected"] is False
    assert remedy["selection_status"] == "pending_maintainer_ruling"
    assert "classic_doorway_low" in remedy["rationale"]


def test_existing_valid_behavior_unchanged(tmp_path: Path) -> None:
    """Existing preflight behavior is unchanged by yield diagnostics additions."""
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


def test_excluded_failed_rows_preserved(tmp_path: Path) -> None:
    """Excluded and failed rows are preserved in manifest exclusions list."""
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=tmp_path)
    scenarios = collector.scenario_ids
    episodes: list[dict[str, Any]] = []
    ids_by_scenario: dict[str, list[str]] = {scenario: [] for scenario in scenarios}
    for episode_id in collector.episodes_by_split["train"]:
        _, scenario, _ = episode_id.split("__")
        ids_by_scenario[scenario].append(episode_id)
    for scenario in scenarios:
        for episode_id in ids_by_scenario[scenario][:10]:
            _, parsed_scenario, seed_token = episode_id.split("__")
            episodes.append(
                _make_episode(
                    episode_id,
                    parsed_scenario,
                    int(seed_token.removeprefix("seed")),
                    "train",
                    steps=5,
                )
            )

    flagged = ids_by_scenario[scenarios[0]][10:13]
    for episode_id, kwargs in zip(
        flagged,
        ({"steps": 1}, {"steps": 5, "fallback": True}, {"steps": 5, "failed": True}),
        strict=True,
    ):
        _, scenario, seed_token = episode_id.split("__")
        episodes.append(
            _make_episode(
                episode_id,
                scenario,
                int(seed_token.removeprefix("seed")),
                "train",
                **kwargs,
            )
        )

    manifest = collector.collect_dataset(
        episodes_override=episodes,
        allow_insufficient_yield=True,
    )

    exclusions = manifest["exclusions"]
    assert len(exclusions) >= 3
    reasons = [e["reason"] for e in exclusions]
    assert "one-step" in reasons
    assert "fallback" in reasons
    assert "failed" in reasons


def test_yield_cli_check(tmp_path: Path) -> None:
    """The --yield-check CLI flag performs a side-effect-free check."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    collector.collect_dataset(episodes_override=_all_packet_episodes(collector))
    exit_code = cli_main(
        [
            "--config",
            str(TEST_PACKET_PATH),
            "--output-root",
            str(tmp_path),
            "--yield-check",
            "--json",
        ]
    )
    assert exit_code == 0


@pytest.mark.parametrize(
    "contents",
    [None, "{not-json", "[1]", '{"attempts": [1]}'],
    ids=["missing", "malformed-json", "non-mapping-items", "invalid-attempts"],
)
def test_cli_rejects_invalid_exhausted_attempts_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], contents: str | None
) -> None:
    """Malformed exhausted-attempt input returns a structured CLI error without a traceback."""
    attempts_path = tmp_path / "exhausted_attempts.json"
    if contents is not None:
        attempts_path.write_text(contents, encoding="utf-8")

    exit_code = cli_main(
        [
            "--config",
            str(TEST_PACKET_PATH),
            "--output-root",
            str(tmp_path / "out"),
            "--preflight",
            "--exhausted-attempts-file",
            str(attempts_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "out" / "balanced_oracle_collection_plan.json").exists()


def test_packet_difference_rejects_unchanged_exhausted_attempt(tmp_path: Path) -> None:
    """An unchanged exhausted packet is rejected before a preflight plan is written."""
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=tmp_path)
    attempt = {
        "packet_fingerprint": collector._compute_packet_fingerprint(),
        "packet_fingerprint_payload": collector.packet_fingerprint_payload(),
    }

    with pytest.raises(BalancedDatasetCollectionError, match="Unchanged exhausted packet"):
        collector.build_preflight_plan(exhausted_attempts=[attempt])
    assert not (tmp_path / "balanced_oracle_collection_plan.json").exists()


def test_packet_difference_reports_exact_changed_fields(tmp_path: Path) -> None:
    """A changed packet reports the changed scientific/execution fingerprint fields."""
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=tmp_path)
    changed_packet = dict(collector.packet)
    changed_packet["scenario_ids"] = list(collector.scenario_ids) + ["diagnostic_only"]
    attempt = {
        "packet_fingerprint": compute_packet_fingerprint(collector.packet),
        "packet_fingerprint_payload": collector.packet_fingerprint_payload(),
    }

    report = collector.validate_packet_difference([])
    assert report["status"] == "not_checked"
    changed_report = collector_module.validate_packet_difference(changed_packet, [attempt])
    assert changed_report["status"] == "changed"
    assert changed_report["comparisons"][0]["changed_fields"] == ["scenario_ids"]


def test_packet_difference_rejects_fingerprint_payload_mismatch(tmp_path: Path) -> None:
    """A stale exhausted-attempt digest cannot authorize a packet comparison."""
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=tmp_path)
    with pytest.raises(BalancedDatasetCollectionError, match="does not match its payload"):
        collector_module.validate_packet_difference(
            collector.packet,
            [
                {
                    "packet_fingerprint": "0" * 64,
                    "packet_fingerprint_payload": collector.packet_fingerprint_payload(),
                }
            ],
        )


def test_packet_difference_requires_complete_reference_payload(tmp_path: Path) -> None:
    """Hash-only exhausted attempts cannot claim exact changed packet fields."""
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=tmp_path)
    with pytest.raises(BalancedDatasetCollectionError, match="complete packet fingerprint payload"):
        collector_module.validate_packet_difference(
            collector.packet,
            [{"packet_fingerprint": collector._compute_packet_fingerprint()}],
        )


def test_yield_check_missing_lineage_is_integrity_blocked(tmp_path: Path) -> None:
    """A summary missing required lineage cannot be treated as complete evidence."""
    collector = BalancedOracleCollector(
        TEST_PACKET_PATH,
        output_root=tmp_path,
        min_usable_transitions=1,
        min_episodes_per_stratum=1,
    )
    manifest = collector.collect_dataset(episodes_override=_all_packet_episodes(collector))
    del manifest["yield_ledger"]["lineage"]["commit"]
    Path(manifest["manifest_path"]).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result = check_yield_status(tmp_path)
    assert result["check_status"] == "blocked_integrity_or_lineage"
    assert "commit" in result["reason"]


def test_duplicate_and_provenance_incomplete_rows_are_not_usable(tmp_path: Path) -> None:
    """Duplicate and provenance-incomplete rows remain explicit diagnostic exclusions."""
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=tmp_path)
    episodes = _all_packet_episodes(collector, steps=2)
    duplicate = dict(episodes[0])
    incomplete = dict(episodes[1])
    incomplete["provenance_incomplete"] = True
    manifest = collector.collect_dataset(
        episodes_override=[episodes[0], *episodes[2:], duplicate, incomplete],
        allow_insufficient_yield=True,
    )

    assert any(item["kind"] == "duplicate_episode_id" for item in manifest["identity_defects"])
    reasons = {item["reason"] for item in manifest["exclusions"]}
    assert "leakage_invalid" in reasons
    assert "provenance_incomplete" in reasons
    assert manifest["yield_ledger"]["totals"]["usable"] < len(episodes) + 2


def test_missing_or_non_mapping_provenance_is_not_usable(tmp_path: Path) -> None:
    """Absent and malformed provenance must remain explicit diagnostic exclusions."""
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=tmp_path)
    episode_id = collector.episodes_by_split["train"][:2]
    complete = [
        _make_episode(
            ep_id,
            "_".join(ep_id.split("__")[1:-1]),
            int(ep_id.rsplit("seed", 1)[1]),
            "train",
            steps=2,
        )
        for ep_id in episode_id
    ]
    missing = dict(complete[0])
    missing.pop("provenance")
    non_mapping = dict(complete[1])
    non_mapping["provenance"] = ["not", "a", "mapping"]

    usable, exclusions = collector._filter_episodes([missing, non_mapping])

    assert usable == []
    assert [item["reason"] for item in exclusions] == [
        "provenance_incomplete",
        "provenance_incomplete",
    ]


def test_malformed_identity_rows_are_preserved_as_defects(tmp_path: Path) -> None:
    """Malformed identity fields fail closed without aborting diagnostic accounting."""
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=tmp_path)
    expected_id = collector.episodes_by_split["train"][0]
    malformed = _make_episode(
        expected_id,
        "classic_doorway_low",
        1001,
        "train",
        steps=2,
    )
    malformed["seed"] = "not-an-integer"
    malformed["provenance"] = ["not", "a", "mapping"]
    unexpected = dict(malformed)
    unexpected["episode_id"] = "train__unexpected__seed9999"

    manifest = collector.collect_dataset(
        episodes_override=[malformed, unexpected],
        allow_insufficient_yield=True,
    )

    defect_kinds = {item["kind"] for item in manifest["identity_defects"]}
    assert {"identity_mismatch", "unexpected_episode_id"} <= defect_kinds
    assert all(item["reason"] == "leakage_invalid" for item in manifest["exclusions"])
    assert manifest["yield_ledger"]["totals"]["usable"] == 0


@pytest.mark.parametrize("seed", [1001.0, "1001"])
def test_non_integer_identity_seed_is_preserved_as_a_defect(tmp_path: Path, seed: object) -> None:
    """Numeric coercion cannot turn a non-integer runtime seed into usable evidence."""
    collector = BalancedOracleCollector(TEST_PACKET_PATH, output_root=tmp_path)
    expected_id = collector.episodes_by_split["train"][0]
    _, scenario, seed_token = expected_id.split("__")
    episode = _make_episode(
        expected_id,
        scenario,
        int(seed_token.removeprefix("seed")),
        "train",
        steps=2,
    )
    episode["seed"] = seed

    manifest = collector.collect_dataset(
        episodes_override=[episode],
        allow_insufficient_yield=True,
    )

    assert manifest["identity_defects"][0]["fields"] == ["seed"]
    assert manifest["exclusions"][0]["reason"] == "leakage_invalid"
    assert manifest["yield_ledger"]["totals"]["usable"] == 0
