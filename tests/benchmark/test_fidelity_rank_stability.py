"""Tests for fidelity rank-stability analysis (issue #3237, child of #3207)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.fidelity_rank_stability import (
    FIDELITY_RANK_STABILITY_SCHEMA,
    analyze_fidelity_sensitivity,
    count_rank_flips,
    kendall_tau,
    metric_drift,
    rank_planners,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _REPO_ROOT / "configs" / "research" / "fidelity_sensitivity_v1.yaml"


# --- primitives -----------------------------------------------------------


def test_rank_planners_orders_best_first() -> None:
    """Higher primary metric ranks first; ties broken by name."""
    table = {
        "p_a": {"success_rate": 0.9},
        "p_b": {"success_rate": 0.7},
        "p_c": {"success_rate": 0.9},
    }
    assert rank_planners(table, "success_rate", higher_is_better=True) == ["p_a", "p_c", "p_b"]


def test_rank_planners_lower_is_better() -> None:
    """Lower-is-better metrics rank ascending."""
    table = {"p_a": {"collision_rate": 0.1}, "p_b": {"collision_rate": 0.3}}
    assert rank_planners(table, "collision_rate", higher_is_better=False) == ["p_a", "p_b"]


def test_kendall_tau_identical_and_reversed() -> None:
    """Identical order gives 1.0, reversed gives -1.0, singletons give 1.0."""
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    assert kendall_tau(["a"], ["a"]) == 1.0


def test_count_rank_flips() -> None:
    """Discordant pairs are counted; identical order has zero flips."""
    assert count_rank_flips(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert count_rank_flips(["a", "b", "c"], ["b", "a", "c"]) == 1
    assert count_rank_flips(["a", "b", "c"], ["c", "b", "a"]) == 3


def test_metric_drift_relative_change() -> None:
    """Drift is the mean absolute relative change per metric."""
    nominal = {"p": {"m": 1.0}}
    axis = {"p": {"m": 1.5}}
    assert metric_drift(nominal, axis, ["m"]) == {"m": 0.5}


# --- end-to-end analysis --------------------------------------------------

_NOMINAL = {
    "planner_x": {"success_rate": 0.90, "collision_rate": 0.05},
    "planner_y": {"success_rate": 0.80, "collision_rate": 0.10},
    "planner_z": {"success_rate": 0.70, "collision_rate": 0.15},
}


def test_analyze_reports_rank_stable_when_order_preserved() -> None:
    """Axes that preserve the nominal order yield a rank-stable verdict."""
    axis_tables = {
        "timestep": {
            "planner_x": {"success_rate": 0.88, "collision_rate": 0.06},
            "planner_y": {"success_rate": 0.79, "collision_rate": 0.11},
            "planner_z": {"success_rate": 0.69, "collision_rate": 0.16},
        },
    }
    report = analyze_fidelity_sensitivity(
        _NOMINAL, axis_tables, primary_metric="success_rate", drift_metrics=["success_rate"]
    )
    assert report.nominal_ranking == ["planner_x", "planner_y", "planner_z"]
    assert report.rank_stable is True
    assert report.flipping_axes == []
    axis = report.axes[0]
    assert axis.kendall_tau == 1.0
    assert axis.rank_flips == 0
    assert axis.top1_changed is False
    assert axis.metric_drift["success_rate"] > 0


def test_analyze_flags_ranking_flipping_axis() -> None:
    """An axis that reorders the top planners is flagged ranking-sensitive."""
    axis_tables = {
        "observation_noise": {  # noise flips x and y at the top
            "planner_x": {"success_rate": 0.60, "collision_rate": 0.20},
            "planner_y": {"success_rate": 0.85, "collision_rate": 0.08},
            "planner_z": {"success_rate": 0.55, "collision_rate": 0.22},
        },
    }
    report = analyze_fidelity_sensitivity(
        _NOMINAL, axis_tables, primary_metric="success_rate", drift_metrics=["success_rate"]
    )
    assert report.rank_stable is False
    assert "observation_noise" in report.flipping_axes
    axis = report.axes[0]
    assert axis.rank_flips > 0
    assert axis.top1_changed is True
    assert axis.kendall_tau < 1.0


def test_analyze_rejects_planner_set_mismatch() -> None:
    """An axis missing a nominal planner is rejected."""
    axis_tables = {"bad": {"planner_x": {"success_rate": 0.9}}}
    with pytest.raises(ValueError, match="planner set does not match"):
        analyze_fidelity_sensitivity(_NOMINAL, axis_tables, primary_metric="success_rate")


def test_report_to_dict_uses_schema() -> None:
    """The report payload carries the fidelity_rank_stability.v1 schema."""
    report = analyze_fidelity_sensitivity(
        _NOMINAL, {}, primary_metric="success_rate", drift_metrics=["success_rate"]
    )
    payload = report.to_dict()
    assert payload["schema_version"] == FIDELITY_RANK_STABILITY_SCHEMA
    assert payload["nominal_ranking"] == ["planner_x", "planner_y", "planner_z"]
    assert payload["rank_stable"] is True


# --- shipped config -------------------------------------------------------


def test_config_declares_three_axes_and_primary_metric() -> None:
    """The shipped contract declares >=3 fidelity axes and a primary metric."""
    data = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    assert data["primary_metric"]
    assert len(data["fidelity_axes"]) >= 3
    assert {"timestep", "sfm_params", "observation_noise"} <= set(data["fidelity_axes"])
