"""Preflight tests for the issue #6481 social-compliance smoke config.

Validates that the frozen smoke config expands to exactly nine
planner-scenario-seed identities and that the social-compliance
diagnostic block survives the aggregation path without zero imputation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robot_sf.benchmark.aggregate import compute_aggregates, flatten_metrics
from robot_sf.benchmark.camera_ready._config import load_campaign_config
from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics
from robot_sf.benchmark.social_compliance import SOCIAL_COMPLIANCE_SCHEMA_VERSION

REPO_ROOT = Path(__file__).parents[2]
CONFIG_PATH = REPO_ROOT / "configs/benchmarks/issue_6481_social_compliance_preflight_smoke.yaml"
SCENARIO_MATRIX_PATH = REPO_ROOT / "configs/scenarios/issue_6481_social_compliance_preflight.yaml"

EXPECTED_PLANNERS = ("goal", "social_force", "orca")
EXPECTED_SEEDS = (111, 112, 113)
EXPECTED_ROW_COUNT = 9


def _load_config():
    """Load the frozen smoke campaign config."""
    return load_campaign_config(CONFIG_PATH)


def _load_scenarios(cfg) -> list[dict[str, Any]]:
    """Load scenarios from the campaign config's scenario matrix."""
    from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios

    return _load_campaign_scenarios(cfg)


def test_config_loads_and_is_not_paper_facing() -> None:
    """The preflight smoke config must be non-paper-facing and CPU-bounded."""
    cfg = _load_config()

    assert cfg.paper_facing is False
    assert cfg.export_publication_bundle is False
    assert cfg.workers == 1
    assert cfg.horizon == 250
    assert cfg.record_forces is False


def test_config_expands_to_exactly_nine_rows() -> None:
    """Three planners times one scenario times three seeds equals nine rows."""
    cfg = _load_config()

    planner_keys = [p.key for p in cfg.planners]
    assert planner_keys == list(EXPECTED_PLANNERS)
    assert list(cfg.seed_policy.seeds) == list(EXPECTED_SEEDS)

    scenarios = _load_scenarios(cfg)
    assert len(scenarios) == 1
    assert scenarios[0]["name"] == "single_ped_crossing_orthogonal"

    total_rows = len(planner_keys) * len(scenarios) * len(cfg.seed_policy.seeds)
    assert total_rows == EXPECTED_ROW_COUNT


def test_all_planners_are_baseline_safe_core() -> None:
    """Every planner must be core group with baseline-safe profile."""
    cfg = _load_config()

    for planner in cfg.planners:
        assert planner.planner_group == "core"
        assert planner.benchmark_profile == "baseline-safe"


def test_social_compliance_block_survives_flatten_and_aggregate() -> None:
    """The diagnostic block must pass through flatten and aggregate without zero imputation."""
    episode = EpisodeData(
        robot_pos=np.zeros((3, 2), dtype=float),
        robot_vel=np.zeros((3, 2), dtype=float),
        robot_acc=np.zeros((3, 2), dtype=float),
        peds_pos=np.asarray([[[0.5, 0.0]], [[2.0, 0.0]], [[0.5, 0.0]]], dtype=float),
        ped_forces=np.ones((3, 1, 2), dtype=float),
        goal=np.asarray([1.0, 0.0]),
        dt=0.5,
        reached_goal_step=2,
        robot_radius=0.1,
        ped_radius=0.1,
    )
    metrics = compute_all_metrics(episode, horizon=3)
    record = {
        "episode_id": "preflight-1",
        "scenario_id": "single_ped_crossing_orthogonal",
        "seed": 111,
        "scenario_params": {"algo": "goal"},
        "metrics": metrics,
    }

    flat = flatten_metrics(record)
    assert flat["social_compliance.comfort_exposure_person_s.status"] == "available"
    assert flat["social_compliance.comfort_exposure_person_s.support_count"] == 3
    assert flat["social_compliance.pedestrian_deviation_mean_m.status"] == "unavailable"

    aggregate = compute_aggregates([record])
    social = aggregate["goal"]["social_compliance"]
    assert social["schema_version"] == SOCIAL_COMPLIANCE_SCHEMA_VERSION

    comfort = social["metrics"]["comfort_exposure_person_s"]
    assert comfort["status_counts"] == {"available": 1}
    assert comfort["support_count"] == 3
    assert comfort["mean"] == pytest.approx(1.0)

    deviation = social["metrics"]["pedestrian_deviation_mean_m"]
    assert deviation["status_counts"] == {"unavailable": 1}
    assert deviation["support_count"] == 0
    assert "mean" not in deviation


def test_unavailable_families_are_not_zero_imputed() -> None:
    """Unavailable metric families must not produce numeric aggregate values."""
    episode = EpisodeData(
        robot_pos=np.zeros((3, 2), dtype=float),
        robot_vel=np.zeros((3, 2), dtype=float),
        robot_acc=np.zeros((3, 2), dtype=float),
        peds_pos=np.asarray([[[0.5, 0.0]], [[2.0, 0.0]], [[0.5, 0.0]]], dtype=float),
        ped_forces=np.ones((3, 1, 2), dtype=float),
        goal=np.asarray([1.0, 0.0]),
        dt=0.5,
        reached_goal_step=2,
        robot_radius=0.1,
        ped_radius=0.1,
    )
    metrics = compute_all_metrics(episode, horizon=3)
    record = {
        "episode_id": "preflight-2",
        "scenario_id": "single_ped_crossing_orthogonal",
        "seed": 112,
        "scenario_params": {"algo": "orca"},
        "metrics": metrics,
    }

    aggregate = compute_aggregates([record])
    social = aggregate["orca"]["social_compliance"]

    for metric_id in (
        "pedestrian_deviation_mean_m",
        "flow_disruption_delay_s",
        "legibility_progress_deficit_m",
        "distributional_inconvenience_p90_p50_gap",
    ):
        metric = social["metrics"][metric_id]
        assert metric["status_counts"] == {"unavailable": 1}
        assert metric["support_count"] == 0
        assert "mean" not in metric
        assert "median" not in metric
        assert "p95" not in metric


def test_scenario_matrix_selects_one_pedestrian_scenario() -> None:
    """The preflight scenario matrix must select exactly one scenario with pedestrians."""
    import yaml

    payload = yaml.safe_load(SCENARIO_MATRIX_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "robot_sf.scenario_matrix.v1"
    assert payload["select_scenarios"] == ["single_ped_crossing_orthogonal"]
