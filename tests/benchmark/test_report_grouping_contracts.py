"""Regression tests for shared benchmark report grouping contracts."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.ablation import compute_snqi_ablation
from robot_sf.benchmark.aggregate import compute_aggregates
from robot_sf.benchmark.distributions import collect_grouped_values
from robot_sf.benchmark.errors import AggregationMetadataError
from robot_sf.benchmark.grouping import resolve_report_group_key
from robot_sf.benchmark.plots import compute_pareto_points
from robot_sf.benchmark.ranking import compute_ranking
from robot_sf.benchmark.report_table import compute_table
from robot_sf.benchmark.seed_variance import compute_seed_variance


def _metadata_algorithm_records() -> list[dict]:
    """Build legacy-compatible records that only identify the planner in algorithm metadata."""
    return [
        {
            "episode_id": "legacy-meta-1",
            "scenario_id": "scenario-fallback",
            "seed": 1,
            "algorithm_metadata": {"algorithm": "metadata-planner"},
            "metrics": {
                "collisions": 1.0,
                "comfort_exposure": 0.4,
                "success": 1.0,
                "time_to_goal_norm": 0.3,
            },
        },
        {
            "episode_id": "legacy-meta-2",
            "scenario_id": "scenario-fallback",
            "seed": 2,
            "algorithm_metadata": {"algorithm": "metadata-planner"},
            "metrics": {
                "collisions": 3.0,
                "comfort_exposure": 0.6,
                "success": 0.8,
                "time_to_goal_norm": 0.5,
            },
        },
    ]


def test_report_group_key_uses_shared_legacy_algorithm_fallbacks() -> None:
    """Top-level and algorithm metadata planner ids should win before scenario fallback."""
    assert (
        resolve_report_group_key(
            {"algo": "top-level", "scenario_id": "scenario"},
            group_by="scenario_params.algo",
            fallback_group_by="scenario_id",
        )
        == "top-level"
    )
    assert (
        resolve_report_group_key(
            {
                "algorithm_metadata": {"algorithm": "metadata-planner"},
                "scenario_id": "scenario",
            },
            group_by="scenario_params.algo",
            fallback_group_by="scenario_id",
        )
        == "metadata-planner"
    )
    assert (
        resolve_report_group_key(
            {"scenario_id": "scenario"},
            group_by="scenario_params.algo",
            fallback_group_by="scenario_id",
        )
        == "scenario"
    )


def test_report_group_key_missing_policy_is_explicit() -> None:
    """Report surfaces should opt into skip, unknown, or error behavior deliberately."""
    record = {"episode_id": "missing-group", "metrics": {"success": 1.0}}
    assert (
        resolve_report_group_key(
            record,
            group_by="scenario_params.algo",
            fallback_group_by="scenario_id",
            missing="skip",
        )
        is None
    )
    assert (
        resolve_report_group_key(
            record,
            group_by="scenario_params.algo",
            fallback_group_by="scenario_id",
            missing="unknown",
        )
        == "unknown"
    )
    with pytest.raises(AggregationMetadataError):
        resolve_report_group_key(
            record,
            group_by="scenario_params.algo",
            fallback_group_by="scenario_id",
            missing="error",
        )


def test_report_surfaces_share_algorithm_metadata_grouping_contract() -> None:
    """Legacy rows with algorithm metadata should group consistently across report surfaces."""
    records = _metadata_algorithm_records()
    weights = {"w_success": 1.0, "w_time": 1.0, "w_collisions": 1.0}
    baseline = {"collisions": {"med": 0.0, "p95": 4.0}}

    aggregate = compute_aggregates(records, group_by="scenario_params.algo")
    assert aggregate["metadata-planner"]["collisions"]["mean"] == 2.0

    table = compute_table(records, metrics=["collisions"])
    assert [(row.group, row.values["collisions"]) for row in table] == [("metadata-planner", 2.0)]

    ranking = compute_ranking(records, metric="collisions")
    assert [(row.group, row.mean, row.count) for row in ranking] == [("metadata-planner", 2.0, 2)]

    distributions = collect_grouped_values(records, metrics=["collisions"])
    assert distributions["metadata-planner"]["collisions"] == [1.0, 3.0]

    _, pareto_labels = compute_pareto_points(records, "collisions", "comfort_exposure")
    assert pareto_labels == ["metadata-planner"]

    ablation = compute_snqi_ablation(
        records,
        weights=weights,
        baseline=baseline,
        group_by="scenario_params.algo",
    )
    assert [row.group for row in ablation] == ["metadata-planner"]

    seed_variance = compute_seed_variance(records, group_by="scenario_params.algo")
    assert seed_variance["metadata-planner"]["collisions"]["mean"] == 2.0


def test_unknown_bucket_surfaces_keep_fully_missing_legacy_rows() -> None:
    """SNQI ablation and seed variance keep an explicit bucket when all group metadata is absent."""
    records = [
        {
            "episode_id": "unknown-1",
            "seed": 1,
            "metrics": {"success": 1.0, "time_to_goal_norm": 0.2, "collisions": 0.0},
        },
        {
            "episode_id": "unknown-2",
            "seed": 2,
            "metrics": {"success": 0.5, "time_to_goal_norm": 0.4, "collisions": 2.0},
        },
    ]
    weights = {"w_success": 1.0, "w_time": 1.0, "w_collisions": 1.0}
    baseline = {"collisions": {"med": 0.0, "p95": 2.0}}

    ablation = compute_snqi_ablation(
        records,
        weights=weights,
        baseline=baseline,
        group_by="scenario_params.algo",
    )
    assert [row.group for row in ablation] == ["unknown"]

    seed_variance = compute_seed_variance(records, group_by="scenario_params.algo")
    assert seed_variance["unknown"]["collisions"]["count"] == 2.0
