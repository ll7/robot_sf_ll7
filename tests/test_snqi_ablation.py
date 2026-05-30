"""TODO docstring. Document this module."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.ablation import (
    compute_ablation_summary,
    compute_snqi_ablation,
    format_csv,
    format_markdown,
    to_json,
)
from robot_sf.benchmark.errors import AggregationMetadataError


def _episodes_two_groups():
    # Two groups (A, B) with two episodes each; metrics chosen so that
    # base ranking favors B, but removing collisions weight favors A.
    """TODO docstring. Document this function."""
    base = {
        "scenario_id": "sc-1",
        "seed": 0,
    }
    return [
        {
            **base,
            "episode_id": "A-1",
            "scenario_params": {"algo": "A"},
            "algo": "A",
            "metrics": {"success": 1.0, "time_to_goal_norm": 0.5, "collisions": 2.0},
        },
        {
            **base,
            "episode_id": "A-2",
            "scenario_params": {"algo": "A"},
            "algo": "A",
            "metrics": {"success": 1.0, "time_to_goal_norm": 0.5, "collisions": 2.0},
        },
        {
            **base,
            "episode_id": "B-1",
            "scenario_params": {"algo": "B"},
            "algo": "B",
            "metrics": {"success": 0.5, "time_to_goal_norm": 0.2, "collisions": 0.0},
        },
        {
            **base,
            "episode_id": "B-2",
            "scenario_params": {"algo": "B"},
            "algo": "B",
            "metrics": {"success": 0.5, "time_to_goal_norm": 0.2, "collisions": 0.0},
        },
    ]


def test_compute_snqi_ablation_rank_shifts():
    """TODO docstring. Document this function."""
    records = _episodes_two_groups()
    weights = {"w_success": 2.0, "w_time": 1.0, "w_collisions": 2.0}
    baseline = {"collisions": {"med": 0.0, "p95": 2.0}}
    rows = compute_snqi_ablation(
        records,
        weights=weights,
        baseline=baseline,
        group_by="scenario_params.algo",
    )
    # Expect two groups
    assert len(rows) == 2
    # Base SNQI: B > A (see analysis), hence base ranks: B=1, A=2
    ranks = {r.group: r.base_rank for r in rows}
    assert ranks == {"B": 1, "A": 2}
    # Ablating collisions should flip: A improves by -1, B degrades by +1
    deltas = {r.group: r.deltas for r in rows}
    assert deltas["A"].get("w_collisions") == -1.0
    assert deltas["B"].get("w_collisions") == +1.0


def test_compute_snqi_ablation_requires_explicit_cross_track_mode() -> None:
    """SNQI ablation should not silently rank pooled observation tracks."""
    records = [
        {
            "episode_id": "grid-a",
            "scenario_id": "sc-1",
            "seed": 0,
            "benchmark_track": "grid_socnav_v1",
            "scenario_params": {"algo": "A", "benchmark_track": "grid_socnav_v1"},
            "metrics": {"success": 1.0, "time_to_goal_norm": 0.5, "collisions": 1.0},
        },
        {
            "episode_id": "lidar-a",
            "scenario_id": "sc-1",
            "seed": 0,
            "benchmark_track": "lidar_2d_v1",
            "scenario_params": {"algo": "A", "benchmark_track": "lidar_2d_v1"},
            "metrics": {"success": 0.8, "time_to_goal_norm": 0.4, "collisions": 0.0},
        },
    ]
    weights = {"w_success": 2.0, "w_time": 1.0, "w_collisions": 2.0}
    baseline = {"collisions": {"med": 0.0, "p95": 2.0}}

    with pytest.raises(AggregationMetadataError):
        compute_snqi_ablation(
            records,
            weights=weights,
            baseline=baseline,
            group_by="scenario_params.algo",
        )

    rows = compute_snqi_ablation(
        records,
        weights=weights,
        baseline=baseline,
        group_by="scenario_params.algo",
        observation_track_mode="diagnostic-cross-track",
    )
    assert {row.group for row in rows} == {
        "grid_socnav_v1 :: A",
        "lidar_2d_v1 :: A",
    }


def test_ablation_formatters_and_summary_cover_json_contract() -> None:
    """Ablation helper outputs should keep stable Markdown, CSV, JSON, and summaries."""
    weights = {"w_success": 2.0, "w_time": 1.0, "w_collisions": 2.0}
    baseline = {"collisions": {"med": 0.0, "p95": 2.0}}
    rows = compute_snqi_ablation(
        _episodes_two_groups(),
        weights=weights,
        baseline=baseline,
        group_by="scenario_params.algo",
        top=1,
    )

    assert len(rows) == 1
    assert format_markdown(rows).startswith("| Rank | Group | base_mean")
    assert format_csv(rows).splitlines()[0].startswith("rank,group,base_mean")
    payload = to_json(rows)
    assert payload[0]["group"] == "B"
    summary = compute_ablation_summary(rows)
    assert "w_collisions" in summary
    assert set(summary["w_collisions"]) == {
        "changed",
        "mean_abs",
        "max_abs",
        "pos",
        "neg",
        "mean",
    }
