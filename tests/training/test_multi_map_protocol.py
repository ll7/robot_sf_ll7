"""Contract tests for Issue #4982 multi-map PPO training support."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.training.multi_map_protocol import (
    DomainRandomization,
    MultiMapTrainTestProtocol,
    apply_domain_randomization,
)
from scripts.training import train_ppo


def test_multi_map_protocol_rejects_overlapping_splits() -> None:
    """A scenario cannot belong to both optimization and zero-shot surfaces."""
    with pytest.raises(ValueError, match="overlap"):
        MultiMapTrainTestProtocol.from_raw(
            {
                "schema_version": "multi-map-train-test.v1",
                "train_scenarios": ["train-map"],
                "held_out_scenarios": ["TRAIN-map"],
                "zero_shot_decay_metric": "success_rate",
            }
        )


def test_domain_randomization_changes_only_declared_simulation_knobs() -> None:
    """Randomization should be reproducible and preserve unrelated scenario data."""
    profile = DomainRandomization.from_raw(
        {
            "schema_version": "training-domain-randomization.v1",
            "peds_speed_mult": [1.1, 1.1],
            "ped_density_multiplier": [0.5, 0.5],
            "route_spawn_jitter_frac": [0.2, 0.2],
        }
    )
    assert profile is not None
    scenario, sampled = apply_domain_randomization(
        {"name": "map-a", "simulation_config": {"ped_density": 0.4}, "metadata": {"keep": 1}},
        profile,
        rng=np.random.default_rng(123),
    )

    assert scenario["simulation_config"] == {
        "ped_density": 0.2,
        "peds_speed_mult": 1.1,
        "route_spawn_jitter_frac": 0.2,
    }
    assert scenario["metadata"] == {"keep": 1}
    assert sampled == {
        "peds_speed_mult": 1.1,
        "ped_density_multiplier": 0.5,
        "route_spawn_jitter_frac": 0.2,
    }
    assert profile.as_dict()["schema_version"] == "training-domain-randomization.v1"


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("not-a-mapping", "must be a mapping"),
        ({"schema_version": "wrong"}, "schema_version"),
        (
            {
                "schema_version": "training-domain-randomization.v1",
                "peds_speed_mult": [1.0],
                "ped_density_multiplier": [1.0, 1.0],
                "route_spawn_jitter_frac": [0.0, 0.0],
            },
            "two-item",
        ),
    ],
)
def test_domain_randomization_rejects_malformed_profiles(raw: object, match: str) -> None:
    """Malformed range profiles should fail before training starts."""
    with pytest.raises(ValueError, match=match):
        DomainRandomization.from_raw(raw)


def test_domain_randomization_preserves_legacy_scenarios_and_rejects_bad_sim_config() -> None:
    """Optional randomization must leave legacy inputs untouched and fail closed when enabled."""
    scenario, sampled = apply_domain_randomization(
        {"name": "legacy", "metadata": {"keep": True}}, None, rng=np.random.default_rng(1)
    )
    assert scenario == {"name": "legacy", "metadata": {"keep": True}}
    assert sampled == {}

    profile = DomainRandomization.from_raw(
        {
            "schema_version": "training-domain-randomization.v1",
            "peds_speed_mult": [1.0, 1.0],
            "ped_density_multiplier": [1.0, 1.0],
            "route_spawn_jitter_frac": [0.0, 0.0],
        }
    )
    assert profile is not None
    with pytest.raises(ValueError, match="simulation_config"):
        apply_domain_randomization(
            {"name": "bad", "simulation_config": []}, profile, rng=np.random.default_rng(1)
        )


def test_issue_4982_config_declares_disjoint_train_and_held_out_maps() -> None:
    """The checked-in launch config should load through the canonical PPO parser."""
    config = train_ppo.load_expert_training_config(
        "configs/training/ppo/issue_4982_multi_map_train_test_protocol.yaml"
    )

    assert config.multi_map_protocol is not None
    assert config.evaluation.hold_out_scenarios == config.multi_map_protocol.held_out_scenarios
    assert config.scenario_sampling["include_scenarios"] == list(
        config.multi_map_protocol.train_scenarios
    )
    assert config.domain_randomization is not None


def test_zero_shot_decay_uses_final_checkpoint_and_positive_drop() -> None:
    """Positive decay means held-out performance fell below the train split."""
    protocol = MultiMapTrainTestProtocol.from_raw(
        {
            "schema_version": "multi-map-train-test.v1",
            "train_scenarios": ["train-map"],
            "held_out_scenarios": ["held-out-map"],
            "zero_shot_decay_metric": "success_rate",
        }
    )
    assert protocol is not None
    records = [
        {"eval_step": 10, "split": "train", "metrics": {"success_rate": 0.1}},
        {"eval_step": 10, "split": "held_out", "metrics": {"success_rate": 0.0}},
        {"eval_step": 20, "split": "train", "metrics": {"success_rate": 0.9}},
        {"eval_step": 20, "split": "held_out", "metrics": {"success_rate": 0.6}},
    ]

    assert train_ppo._zero_shot_decay_metric(records, protocol) == (
        "zero_shot_success_rate_decay",
        pytest.approx(0.3),
    )


def test_zero_shot_decay_keeps_step_zero_and_excludes_nonfinite_values() -> None:
    """Initial evaluation is valid, while non-finite metric values cannot enter the aggregate."""
    protocol = MultiMapTrainTestProtocol.from_raw(
        {
            "schema_version": "multi-map-train-test.v1",
            "train_scenarios": ["train-map"],
            "held_out_scenarios": ["held-out-map"],
            "zero_shot_decay_metric": "success_rate",
        }
    )
    assert protocol is not None
    records = [
        {"eval_step": 0, "split": "train", "metrics": {"success_rate": 1.0}},
        {"eval_step": 0, "split": "train", "metrics": {"success_rate": float("nan")}},
        {"eval_step": 0, "split": "held_out", "metrics": {"success_rate": 0.5}},
        {"eval_step": 0, "split": "held_out", "metrics": {"success_rate": float("inf")}},
    ]

    assert train_ppo._zero_shot_decay_metric(records, protocol) == (
        "zero_shot_success_rate_decay",
        pytest.approx(0.5),
    )


def test_protocol_validates_known_scenarios_and_serializes() -> None:
    """Protocol IDs must resolve in the scenario manifest before PPO starts."""
    protocol = MultiMapTrainTestProtocol.from_raw(
        {
            "schema_version": "multi-map-train-test.v1",
            "train_scenarios": ["train-map"],
            "held_out_scenarios": ["held-out-map"],
            "zero_shot_decay_metric": "snqi",
        }
    )
    assert protocol is not None
    protocol.validate_scenarios([{"name": "train-map"}, {"scenario_id": "held-out-map"}])
    assert protocol.as_dict()["zero_shot_decay_metric"] == "snqi"
    with pytest.raises(ValueError, match="unknown"):
        protocol.validate_scenarios([{"name": "train-map"}])


def test_per_scenario_report_preserves_split_identity() -> None:
    """The report must not merge train and held-out rows for the same scenario."""
    rows = train_ppo._per_scenario_eval_rows_from_episode_records(
        episode_records=[
            {
                "eval_step": 10,
                "split": "train",
                "scenario_id": "map-a",
                "metrics": {"success_rate": 1.0},
            },
            {
                "eval_step": 10,
                "split": "held_out",
                "scenario_id": "map-a",
                "metrics": {"success_rate": 0.0},
            },
        ]
    )

    assert [(row["split"], row["success_rate"]) for row in rows] == [
        ("held_out", 0.0),
        ("train", 1.0),
    ]
