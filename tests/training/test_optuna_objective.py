"""Tests for Optuna objective reduction helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.training.optuna_objective import (
    eval_metric_series,
    load_episode_records,
    objective_from_series,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_eval_metric_series_groups_episode_metrics_by_eval_step():
    """Series extraction should aggregate per-eval-step means."""
    records = [
        {"eval_step": 100, "metrics": {"snqi": 0.1}},
        {"eval_step": 100, "metrics": {"snqi": 0.3}},
        {"eval_step": 200, "metrics": {"snqi": 0.7}},
    ]
    series = eval_metric_series(records, metric_name="snqi")
    assert series == [(100, pytest.approx(0.2)), (200, pytest.approx(0.7))]


def test_objective_from_series_final_lastn_and_auc_modes():
    """Objective reducers should provide stable scalar summaries."""
    series = [(100, 1.0), (200, 3.0), (300, 5.0)]
    assert objective_from_series(series, mode="final_eval", window=3) == 5.0
    assert objective_from_series(series, mode="last_n_mean", window=2) == pytest.approx(4.0)
    assert objective_from_series(series, mode="auc", window=3) == pytest.approx(3.0)


def test_objective_from_series_best_checkpoint_mode_returns_none():
    """Best-checkpoint mode delegates scoring to checkpoint selection logic."""
    series = [(100, 1.0)]
    assert objective_from_series(series, mode="best_checkpoint", window=3) is None


def test_load_episode_records_reads_jsonl(tmp_path: Path):
    """Episode JSONL loader should return one dict per JSON object line."""
    path = tmp_path / "episodes.jsonl"
    payloads = [
        {"eval_step": 100, "metrics": {"snqi": 0.2}},
        {"eval_step": 200, "metrics": {"snqi": 0.4}},
    ]
    path.write_text("\n".join(json.dumps(item) for item in payloads), encoding="utf-8")

    loaded = load_episode_records(path)
    assert loaded == payloads
