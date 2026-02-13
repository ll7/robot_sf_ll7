"""Tests for Optuna safety-constraint helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import optuna
import pytest

from robot_sf.training.optuna_objective import (
    episodic_metric_from_records,
    eval_metric_series,
    objective_from_series,
)
from scripts.training.optuna_expert_ppo import (
    _apply_constraint_handling,
    _build_safety_constraints,
    _evaluate_safety_constraints,
    _resolve_trial_metric,
)


@dataclass
class _MetricAggregateStub:
    mean: float


@dataclass
class _BestCheckpointStub:
    metrics: dict[str, float]


@dataclass
class _ResultStub:
    metrics: dict[str, _MetricAggregateStub]
    best_checkpoint: _BestCheckpointStub | None = None


def _constraint_args(
    *,
    collision: float | None = None,
    comfort: float | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        constraint_collision_rate_max=collision,
        constraint_comfort_exposure_max=comfort,
    )


def test_build_safety_constraints_returns_empty_mapping_when_unset() -> None:
    """No CLI thresholds should produce no active constraints."""
    assert _build_safety_constraints(_constraint_args()) == {}


@pytest.mark.parametrize("collision", [-0.1, float("nan"), float("inf")])
def test_build_safety_constraints_rejects_invalid_collision_threshold(collision: float) -> None:
    """Constraint thresholds must be finite and non-negative."""
    with pytest.raises(ValueError, match="collision_rate"):
        _build_safety_constraints(_constraint_args(collision=collision))


def test_evaluate_safety_constraints_marks_feasible_trial() -> None:
    """Constraint evaluation should pass when all thresholds are satisfied."""
    result = _ResultStub(
        metrics={
            "collision_rate": _MetricAggregateStub(mean=0.1),
            "comfort_exposure": _MetricAggregateStub(mean=0.08),
        }
    )
    records = [
        {"eval_step": 100, "metrics": {"collision_rate": 0.2, "comfort_exposure": 0.1}},
        {"eval_step": 200, "metrics": {"collision_rate": 0.05, "comfort_exposure": 0.08}},
    ]
    feasible, values, violations, missing = _evaluate_safety_constraints(
        result=result,
        records=records,
        constraints={"collision_rate": 0.2, "comfort_exposure": 0.12},
        objective_mode="last_n_mean",
        objective_window=2,
        eval_metric_series_fn=eval_metric_series,
        objective_from_series_fn=objective_from_series,
    )

    assert feasible is True
    assert values["collision_rate"] == pytest.approx(0.125)
    assert values["comfort_exposure"] == pytest.approx(0.09)
    assert violations == {}
    assert missing == []


def test_evaluate_safety_constraints_tracks_violations_and_missing_metrics() -> None:
    """Missing/violating metrics should mark trial infeasible with details."""
    result = _ResultStub(metrics={"collision_rate": _MetricAggregateStub(mean=0.5)})
    records = [{"eval_step": 100, "metrics": {"collision_rate": 0.5}}]
    feasible, values, violations, missing = _evaluate_safety_constraints(
        result=result,
        records=records,
        constraints={"collision_rate": 0.2, "comfort_exposure": 0.1},
        objective_mode="last_n_mean",
        objective_window=1,
        eval_metric_series_fn=eval_metric_series,
        objective_from_series_fn=objective_from_series,
    )

    assert feasible is False
    assert values["collision_rate"] == pytest.approx(0.5)
    assert violations["collision_rate"] == pytest.approx(0.3)
    assert missing == ["comfort_exposure"]


def test_apply_constraint_handling_penalizes_objective_in_correct_direction() -> None:
    """Penalty mode should push infeasible objectives away from feasible candidates."""
    penalized_max = _apply_constraint_handling(
        objective_value=12.5,
        direction="maximize",
        handling="penalize",
        violations={"collision_rate": 0.3},
        missing_metrics=[],
    )
    penalized_min = _apply_constraint_handling(
        objective_value=0.4,
        direction="minimize",
        handling="penalize",
        violations={"collision_rate": 0.3},
        missing_metrics=["comfort_exposure"],
    )

    assert penalized_max < -100_000.0
    assert penalized_min > 100_000.0


def test_apply_constraint_handling_prune_raises_trial_pruned() -> None:
    """Prune mode should mark infeasible trials as pruned."""
    with pytest.raises(optuna.TrialPruned):
        _apply_constraint_handling(
            objective_value=1.0,
            direction="maximize",
            handling="prune",
            violations={"collision_rate": 0.1},
            missing_metrics=[],
        )


def test_resolve_trial_metric_episodic_mode_falls_back_when_logs_missing() -> None:
    """Episodic mode should fall back to aggregate metrics when episode logs are absent."""
    result = _ResultStub(metrics={"snqi": _MetricAggregateStub(mean=0.42)})
    metric_value, series = _resolve_trial_metric(
        result=result,
        records=[],
        metric_name="snqi",
        objective_mode="episodic_snqi",
        objective_window=3,
        eval_metric_series_fn=eval_metric_series,
        objective_from_series_fn=objective_from_series,
        episodic_metric_from_records_fn=episodic_metric_from_records,
    )
    assert metric_value == pytest.approx(0.42)
    assert series == []


def test_resolve_trial_metric_episodic_mode_uses_episode_values() -> None:
    """Episodic mode should use full episode records when available."""
    result = _ResultStub(metrics={"snqi": _MetricAggregateStub(mean=0.0)})
    records = [
        {"eval_step": 100, "metrics": {"snqi": 0.0}},
        {"eval_step": 100, "metrics": {"snqi": 1.0}},
        {"eval_step": 200, "metrics": {"snqi": 1.0}},
    ]
    metric_value, _series = _resolve_trial_metric(
        result=result,
        records=records,
        metric_name="snqi",
        objective_mode="episodic_snqi",
        objective_window=3,
        eval_metric_series_fn=eval_metric_series,
        objective_from_series_fn=objective_from_series,
        episodic_metric_from_records_fn=episodic_metric_from_records,
    )
    assert metric_value == pytest.approx(2.0 / 3.0)
