"""Validation for the guarded-PPO tuning-compare workers=1 benchmark config."""

from __future__ import annotations

from pathlib import Path

import yaml

from robot_sf.benchmark.algorithm_readiness import get_algorithm_readiness
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config

ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = (
    ROOT / "configs/benchmarks/paper_experiment_matrix_v1_guarded_ppo_tuning_compare.yaml"
)
TARGET_CONFIG = (
    ROOT / "configs/benchmarks/paper_experiment_matrix_v1_guarded_ppo_tuning_compare_workers1.yaml"
)
H600_PROBE_CONFIG = (
    ROOT / "configs/benchmarks/paper_experiment_matrix_v1_issue_791_horizon600_probe.yaml"
)
WORKER_NEUTRAL_FIELDS = {"name", "workers", "planners"}
PLANNER_WORKER_NEUTRAL_FIELDS = {"workers"}


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_workers1_config_preserves_tuning_compare_matrix_contract() -> None:
    """Workers=1 config keeps the tuning-compare scientific surface unchanged."""
    source_payload = _load_yaml(SOURCE_CONFIG)
    target_payload = _load_yaml(TARGET_CONFIG)

    assert (
        target_payload["name"] == "paper_experiment_matrix_v1_guarded_ppo_tuning_compare_workers1"
    )
    assert target_payload["workers"] == 1

    source_scientific_fields = {
        key: value for key, value in source_payload.items() if key not in WORKER_NEUTRAL_FIELDS
    }
    target_scientific_fields = {
        key: value for key, value in target_payload.items() if key not in WORKER_NEUTRAL_FIELDS
    }
    assert target_scientific_fields == source_scientific_fields

    source_planners = source_payload["planners"]
    target_planners = target_payload["planners"]
    assert [planner["key"] for planner in target_planners] == [
        "ppo",
        "guarded_ppo",
        "guarded_ppo_relaxed_v1",
        "guarded_ppo_relaxed_v2",
    ]
    assert [
        (planner["key"], planner["algo"], planner["algo_config"]) for planner in target_planners
    ] == [(planner["key"], planner["algo"], planner["algo_config"]) for planner in source_planners]

    for source_planner, target_planner in zip(source_planners, target_planners, strict=True):
        assert {
            key: value
            for key, value in target_planner.items()
            if key not in PLANNER_WORKER_NEUTRAL_FIELDS
        } == source_planner
        assert target_planner["workers"] == 1


def test_workers1_config_matches_h600_single_worker_override_convention() -> None:
    """Every learned PPO-family row carries the h600-style planner-local workers override."""
    h600_payload = _load_yaml(H600_PROBE_CONFIG)
    target_payload = _load_yaml(TARGET_CONFIG)

    h600_ppo = next(planner for planner in h600_payload["planners"] if planner["key"] == "ppo")
    assert h600_ppo["workers"] == 1

    for planner in target_payload["planners"]:
        assert planner["workers"] == h600_ppo["workers"] == 1


def test_workers1_config_loads_and_resolves_all_planners() -> None:
    """Loader accepts the config and keeps all planner rows explicitly single-worker."""
    cfg = load_campaign_config(TARGET_CONFIG)

    assert cfg.name == "paper_experiment_matrix_v1_guarded_ppo_tuning_compare_workers1"
    assert cfg.workers == 1
    assert [planner.key for planner in cfg.planners] == [
        "ppo",
        "guarded_ppo",
        "guarded_ppo_relaxed_v1",
        "guarded_ppo_relaxed_v2",
    ]

    for planner in cfg.planners:
        readiness = get_algorithm_readiness(planner.algo)
        assert readiness is not None
        assert planner.algo_config_path is not None
        assert planner.algo_config_path.exists()
        assert planner.workers_override == 1
