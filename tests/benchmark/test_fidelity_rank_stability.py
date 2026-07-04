"""Tests for fidelity rank-stability analysis (issue #3237, child of #3207)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.fidelity_rank_stability import (
    AXIS_PRIMARY_METRIC_NON_IDENTIFIABLE_REASON,
    FIDELITY_RANK_STABILITY_SCHEMA,
    PRIMARY_METRIC_ZERO_VARIANCE_REASON,
    RANK_STABILITY_REPORT_SCHEMA,
    PostRunContractResult,
    analyze_fidelity_sensitivity,
    check_rank_identifiability_contract,
    count_rank_flips,
    kendall_tau,
    metric_drift,
    rank_planners,
    write_rank_identifiability_report,
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


def test_rank_planners_sorts_missing_and_invalid_values_last() -> None:
    """Missing, non-numeric, and non-finite metric values sort last deterministically."""
    table = {
        "valid_b": {"success_rate": 0.6},
        "missing": {},
        "invalid_text": {"success_rate": "not-a-number"},
        "nan": {"success_rate": float("nan")},
        "valid_a": {"success_rate": 0.9},
        "inf": {"success_rate": float("inf")},
    }

    assert rank_planners(table, "success_rate", higher_is_better=True) == [
        "valid_a",
        "valid_b",
        "inf",
        "invalid_text",
        "missing",
        "nan",
    ]


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


def test_metric_drift_uses_documented_unit_floor_denominator() -> None:
    """Sub-unit nominal values use max(abs(nominal), 1.0) as the denominator."""
    nominal = {"p": {"success_rate": 0.02}}
    axis = {"p": {"success_rate": 0.04}}
    assert metric_drift(nominal, axis, ["success_rate"]) == {"success_rate": 0.02}


def test_metric_drift_skips_missing_invalid_and_non_finite_values() -> None:
    """Drift ignores rows that cannot produce a finite numeric comparison."""
    nominal = {
        "valid": {"m": 1.0},
        "missing_axis_metric": {"m": 1.0},
        "nan": {"m": float("nan")},
        "text": {"m": "bad"},
        "inf_axis": {"m": 1.0},
    }
    axis = {
        "valid": {"m": 1.5},
        "missing_axis_metric": {},
        "nan": {"m": 1.0},
        "text": {"m": 1.0},
        "inf_axis": {"m": float("inf")},
    }
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


def test_analyze_marks_all_tie_nominal_rank_non_identifiable() -> None:
    """All-tied nominal primary metrics preserve order only for serialization."""
    nominal = {
        "planner_b": {"success_rate": 0.0, "collision_rate": 1.0},
        "planner_a": {"success_rate": 0.0, "collision_rate": 1.0},
    }
    axis_tables = {
        "timestep": {
            "planner_b": {"success_rate": 0.0, "collision_rate": 1.0},
            "planner_a": {"success_rate": 0.0, "collision_rate": 1.0},
        }
    }

    report = analyze_fidelity_sensitivity(
        nominal, axis_tables, primary_metric="success_rate", drift_metrics=["success_rate"]
    )

    assert report.nominal_ranking == ["planner_a", "planner_b"]
    assert report.rank_identifiable is False
    assert report.rank_identifiability_reason == PRIMARY_METRIC_ZERO_VARIANCE_REASON
    assert report.rank_stable is None
    assert report.flipping_axes == []
    assert report.non_identifiable_axes == ["timestep"]
    axis = report.axes[0]
    assert axis.ranking == ["planner_a", "planner_b"]
    assert axis.rank_identifiable is False
    assert axis.rank_identifiability_reason == PRIMARY_METRIC_ZERO_VARIANCE_REASON
    assert axis.kendall_tau is None
    assert axis.rank_flips is None
    assert axis.top1_changed is None


def test_analyze_marks_axis_all_tie_rank_non_identifiable() -> None:
    """A tied axis cannot support tau/flip evidence even when nominal ranks exist."""
    axis_tables = {
        "observation_noise": {
            "planner_x": {"success_rate": 0.0, "collision_rate": 1.0},
            "planner_y": {"success_rate": 0.0, "collision_rate": 1.0},
            "planner_z": {"success_rate": 0.0, "collision_rate": 1.0},
        },
    }

    report = analyze_fidelity_sensitivity(
        _NOMINAL, axis_tables, primary_metric="success_rate", drift_metrics=["success_rate"]
    )

    assert report.rank_identifiable is False
    assert report.rank_identifiability_reason == AXIS_PRIMARY_METRIC_NON_IDENTIFIABLE_REASON
    assert report.rank_stable is None
    assert report.flipping_axes == []
    assert report.non_identifiable_axes == ["observation_noise"]
    axis = report.axes[0]
    assert axis.rank_identifiable is False
    assert axis.rank_identifiability_reason == PRIMARY_METRIC_ZERO_VARIANCE_REASON
    assert axis.kendall_tau is None
    assert axis.rank_flips is None
    assert axis.top1_changed is None


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
    assert payload["rank_identifiable"] is True
    assert payload["rank_identifiability_reason"] is None
    assert payload["rank_stable"] is True
    assert payload["non_identifiable_axes"] == []


# --- shipped config -------------------------------------------------------


def test_config_declares_three_axes_and_primary_metric() -> None:
    """The shipped contract declares >=3 fidelity axes and a primary metric."""
    data = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    assert data["primary_metric"]
    assert len(data["fidelity_axes"]) >= 3
    assert {"timestep", "sfm_params", "observation_noise"} <= set(data["fidelity_axes"])


# --- post-run contract: standalone report write -----------------------------


def test_write_rank_identifiability_report_roundtrips(tmp_path: Path) -> None:
    """The standalone report file carries the wrapper schema and report content."""
    report = analyze_fidelity_sensitivity(
        _NOMINAL, {}, primary_metric="success_rate", drift_metrics=["success_rate"]
    ).to_dict()
    path = write_rank_identifiability_report(report, tmp_path)
    assert path.name == "fidelity_rank_stability_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == RANK_STABILITY_REPORT_SCHEMA
    assert payload["rank_identifiable"] is True
    assert payload["primary_metric"] == "success_rate"


# --- post-run contract: check_rank_identifiability_contract -----------------


_CONTRACT_SPEC = {
    "id": "runtime_rank_identifiability_recheck",
    "report": "fidelity_rank_stability_report.json",
    "builder": "robot_sf/benchmark/fidelity_rank_stability.py",
    "metric": "success_rate",
    "threshold": "non_zero_variance_and_rank_identifiable",
    "output_path": "output/fidelity_sensitivity/<campaign>/rank_identifiability.json",
    "blocks_claims_when_failed": True,
}


def test_check_contract_passes_when_identifiable() -> None:
    """An identifiable report satisfies the contract."""
    report = analyze_fidelity_sensitivity(
        _NOMINAL, {}, primary_metric="success_rate", drift_metrics=["success_rate"]
    ).to_dict()
    result = check_rank_identifiability_contract(report, _CONTRACT_SPEC)
    assert result.satisfied is True
    assert result.reason is None
    assert result.contract_id == "runtime_rank_identifiability_recheck"


def test_check_contract_fails_when_not_identifiable() -> None:
    """All-tied metrics fail the identifiability contract."""
    tied = {
        "p_a": {"success_rate": 0.0},
        "p_b": {"success_rate": 0.0},
    }
    report = analyze_fidelity_sensitivity(
        tied, {}, primary_metric="success_rate", drift_metrics=["success_rate"]
    ).to_dict()
    result = check_rank_identifiability_contract(report, _CONTRACT_SPEC)
    assert result.satisfied is False
    assert result.reason is not None
    assert "rank not identifiable" in result.reason
    assert "blocks_claims_when_failed=True" in result.reason


def test_check_contract_rejects_unsupported_threshold() -> None:
    """An unknown threshold raises ValueError."""
    bad_spec = {**_CONTRACT_SPEC, "threshold": "bogus"}
    report = analyze_fidelity_sensitivity(
        _NOMINAL, {}, primary_metric="success_rate", drift_metrics=["success_rate"]
    ).to_dict()
    with pytest.raises(ValueError, match="unsupported post-run contract threshold"):
        check_rank_identifiability_contract(report, bad_spec)


def test_check_contract_result_to_dict_roundtrips() -> None:
    """PostRunContractResult.to_dict is JSON-safe."""
    result = PostRunContractResult(contract_id="test", satisfied=True, reason=None)
    payload = result.to_dict()
    assert json.dumps(payload)
    assert payload == {"contract_id": "test", "satisfied": True, "reason": None}
