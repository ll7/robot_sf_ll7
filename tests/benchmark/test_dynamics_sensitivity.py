"""Tests for deterministic dynamics-sensitivity ranking summaries."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.dynamics_sensitivity import (
    DYNAMICS_SENSITIVITY_SCHEMA,
    analyze_dynamics_sensitivity,
)


def test_dynamics_sensitivity_reports_stable_ranking_and_metric_table() -> None:
    """Stable planner order returns no flipping dynamics and preserves metrics."""

    report = analyze_dynamics_sensitivity(
        {
            "holonomic_disc": {
                "orca": {"collision_rate": 0.0, "near_miss_rate": 0.1},
                "rvo": {"collision_rate": 0.2, "near_miss_rate": 0.4},
            },
            "unicycle": {
                "orca": {"collision_rate": 0.1, "near_miss_rate": 0.2},
                "rvo": {"collision_rate": 0.3, "near_miss_rate": 0.4},
            },
        },
        reference_dynamics="holonomic_disc",
    )

    payload = report.to_dict()

    assert payload["schema_version"] == DYNAMICS_SENSITIVITY_SCHEMA
    assert payload["reference_ranking"] == ["orca", "rvo"]
    assert payload["rank_stable"] is True
    assert payload["flipping_dynamics"] == []
    assert payload["metric_table"]["unicycle"]["orca"]["near_miss_rate"] == pytest.approx(0.2)


def test_dynamics_sensitivity_flags_rank_flip() -> None:
    """A changed planner order is reported as a dynamics-sensitive rank flip."""

    report = analyze_dynamics_sensitivity(
        {
            "holonomic_disc": {
                "planner_a": {"collision_rate": 0.0, "tracking_error": 1.0},
                "planner_b": {"collision_rate": 0.2, "tracking_error": 0.5},
            },
            "kinematic_bicycle": {
                "planner_a": {"collision_rate": 0.3, "tracking_error": 0.6},
                "planner_b": {"collision_rate": 0.1, "tracking_error": 0.4},
            },
        },
        reference_dynamics="holonomic_disc",
    )

    payload = report.to_dict()

    assert payload["rank_stable"] is False
    assert payload["flipping_dynamics"] == ["kinematic_bicycle"]
    bicycle = next(row for row in payload["dynamics"] if row["dynamics"] == "kinematic_bicycle")
    assert bicycle["ranking"] == ["planner_b", "planner_a"]
    assert bicycle["rank_flips"] == 1
    assert bicycle["top1_changed"] is True
    assert bicycle["kendall_tau"] == pytest.approx(-1.0)


def test_dynamics_sensitivity_requires_reference_with_two_planners() -> None:
    """The reference table must identify a ranking over at least two planners."""

    with pytest.raises(ValueError, match="at least two finite planner"):
        analyze_dynamics_sensitivity(
            {"holonomic_disc": {"orca": {"collision_rate": 0.0}}},
            reference_dynamics="holonomic_disc",
        )
