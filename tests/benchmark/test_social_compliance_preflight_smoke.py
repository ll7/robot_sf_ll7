"""Preflight tests for the issue #6481 social-compliance smoke config.

Validates that the frozen smoke config expands to exactly nine
planner-scenario-seed identities and that the social-compliance
diagnostic block survives the aggregation path without zero imputation.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robot_sf.benchmark.aggregate import compute_aggregates, flatten_metrics
from robot_sf.benchmark.camera_ready._config import load_campaign_config
from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics
from robot_sf.benchmark.social_compliance import SOCIAL_COMPLIANCE_SCHEMA_VERSION
from scripts.validation.preflight_social_compliance_smoke_issue_6481 import (
    _aggregate_contract_is_ok,
    _aggregate_metric_contract_is_ok,
    _classify_row,
    build_receipt,
)

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
    assert flat["social_compliance.comfort_exposure_person_s.denominator"] == "pedestrian_steps"
    assert flat["social_compliance.pedestrian_deviation_mean_m.status"] == "unavailable"

    aggregate = compute_aggregates([record])
    social = aggregate["goal"]["social_compliance"]
    assert social["schema_version"] == SOCIAL_COMPLIANCE_SCHEMA_VERSION

    comfort = social["metrics"]["comfort_exposure_person_s"]
    assert comfort["status_counts"] == {"available": 1}
    assert comfort["support_count"] == 3
    assert comfort["denominators"] == {"pedestrian_steps": 1}
    assert comfort["mean"] == pytest.approx(1.0)

    deviation = social["metrics"]["pedestrian_deviation_mean_m"]
    assert deviation["status_counts"] == {"unavailable": 1}
    assert deviation["support_count"] == 0
    assert deviation["denominators"] == {"tracked_pedestrian_steps_with_baseline": 1}
    assert deviation["unavailable_reasons"] == {
        "matched pedestrian reference trajectory is unavailable": 1
    }
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


def test_receipt_preserves_contract_metadata_and_classifies_execution_modes() -> None:
    """Receipt keeps metadata; native/adapter are benchmark-capable, fallback/degraded excluded."""
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
    record = {
        "episode_id": "preflight-contract",
        "scenario_id": "single_ped_crossing_orthogonal",
        "seed": 111,
        "execution_mode": "native",
        "scenario_params": {"algo": "goal"},
        "metrics": compute_all_metrics(episode, horizon=3),
    }

    classified = _classify_row(record)
    assert classified["schema_valid"] is True
    assert classified["execution_mode"] == "native"
    assert classified["denominators"]["comfort_exposure_person_s"] == "pedestrian_steps"
    assert (
        classified["unavailable_reasons"]["pedestrian_deviation_mean_m"]
        == "matched pedestrian reference trajectory is unavailable"
    )

    invalid_contract_record = {
        **record,
        "metrics": {
            **record["metrics"],
            "social_compliance": {
                **record["metrics"]["social_compliance"],
                "metrics": {
                    **record["metrics"]["social_compliance"]["metrics"],
                    "comfort_exposure_person_s": {
                        **record["metrics"]["social_compliance"]["metrics"][
                            "comfort_exposure_person_s"
                        ],
                        "units": "seconds",
                    },
                },
            },
        },
    }
    assert _classify_row(invalid_contract_record)["schema_valid"] is False

    aggregate = compute_aggregates([record])["goal"]
    campaign_with_aggregate = {
        "runs": [
            {
                "planner": {"key": "goal", "algo": "goal"},
                "aggregates": {"goal": aggregate},
            }
        ]
    }
    assert _aggregate_contract_is_ok(campaign_with_aggregate, [classified]) is True

    incorrect_aggregate = deepcopy(aggregate)
    incorrect_aggregate["social_compliance"]["metrics"]["comfort_exposure_person_s"]["mean"] = 99.0
    assert (
        _aggregate_contract_is_ok(
            {"runs": [{"planner": {"key": "goal"}, "aggregates": {"goal": incorrect_aggregate}}]},
            [classified],
        )
        is False
    )

    missing_aggregate_metadata = {
        **aggregate,
        "social_compliance": {
            **aggregate["social_compliance"],
            "metrics": {
                **aggregate["social_compliance"]["metrics"],
                "comfort_exposure_person_s": {
                    **aggregate["social_compliance"]["metrics"]["comfort_exposure_person_s"],
                    "denominators": {},
                },
            },
        },
    }
    assert (
        _aggregate_contract_is_ok(
            {
                "runs": [
                    {"planner": {"key": "goal"}, "aggregates": {"goal": missing_aggregate_metadata}}
                ]
            },
            [classified],
        )
        is False
    )

    campaign_result = {
        "_runner_returncode": 0,
        "campaign_root": "output/unused",
        "campaign_execution_status": "completed",
        "exit_code": 0,
    }

    # Native rows satisfy both the literal-native and benchmark-capable contracts.
    native_receipt = build_receipt(campaign_result, [record], Path("output/unused"))
    assert native_receipt["all_native"] is True
    assert native_receipt["all_benchmark_capable_execution"] is True
    assert native_receipt["campaign_ok"] is True

    # Declared adapter rows (e.g. social_force/orca, which are inherently adapter planners per
    # ``_KINEMATICS_PROFILE_BY_CANONICAL`` with ``supports_native_commands: False``) are
    # benchmark-capable execution under the issue #691 fallback policy and
    # ``NATIVE_EXECUTION_MODES = {native, adapter}``; they are not fallback/degraded and so
    # remain eligible for the preflight pass even though they are not literally "native".
    adapter_receipt = build_receipt(
        campaign_result, [{**record, "execution_mode": "adapter"}], Path("output/unused")
    )
    assert adapter_receipt["all_native"] is False
    assert adapter_receipt["all_benchmark_capable_execution"] is True

    # Fallback, degraded, and unavailable rows are excluded from successful evidence and
    # must fail the benchmark-capable execution contract.
    for bad_mode in ("fallback", "degraded", "unavailable"):
        bad_receipt = build_receipt(
            campaign_result, [{**record, "execution_mode": bad_mode}], Path("output/unused")
        )
        assert bad_receipt["all_native"] is False
        assert bad_receipt["all_benchmark_capable_execution"] is False
        assert bad_receipt["passed"] is False

    # ``algorithm_metadata.planner_kinematics.execution_mode`` is canonical provenance and
    # takes precedence over the legacy top-level ``execution_mode`` label.
    metadata_overrides_top_level = {
        **record,
        "execution_mode": "native",
        "algorithm_metadata": {"planner_kinematics": {"execution_mode": "adapter"}},
    }
    assert _classify_row(metadata_overrides_top_level)["execution_mode"] == "adapter"


def test_receipt_requires_completed_zero_exit_campaign() -> None:
    """A successful-looking row set cannot hide a failed campaign process."""
    receipt = build_receipt(
        {
            "_runner_returncode": 1,
            "campaign_execution_status": "failed",
            "exit_code": 2,
        },
        [],
        Path("output/unused"),
    )
    assert receipt["campaign_ok"] is False
    assert receipt["passed"] is False
    assert receipt["all_native"] is False


@pytest.mark.parametrize("invalid_support_count", [True, -1, float("nan"), float("inf")])
def test_aggregate_contract_ignores_invalid_support_counts(
    invalid_support_count: object,
) -> None:
    """Receipt validation mirrors the aggregator's fail-closed support-count filter."""
    metric_id = "comfort_exposure_person_s"
    rows = [
        {
            "statuses": {metric_id: "available"},
            "support_counts": {metric_id: invalid_support_count},
            "denominators": {metric_id: "pedestrian_steps"},
            "unavailable_reasons": {},
            "values": {metric_id: 1.0},
        }
    ]
    aggregate_metric = {
        "status_counts": {"available": 1},
        "support_count": 0,
        "denominators": {"pedestrian_steps": 1},
        "unavailable_reasons": {},
        "mean": 1.0,
        "median": 1.0,
        "p95": 1.0,
    }

    assert _aggregate_metric_contract_is_ok(metric_id, aggregate_metric, rows) is True


def test_scenario_matrix_selects_one_pedestrian_scenario() -> None:
    """The preflight scenario matrix must select exactly one scenario with pedestrians."""
    import yaml

    payload = yaml.safe_load(SCENARIO_MATRIX_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "robot_sf.scenario_matrix.v1"
    assert payload["select_scenarios"] == ["single_ped_crossing_orthogonal"]
