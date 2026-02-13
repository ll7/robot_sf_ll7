"""Tests for Optuna objective reduction helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.training.optuna_objective import (
    episodic_metric_from_records,
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
    assert objective_from_series(series, mode="episodic_snqi", window=3) is None


def test_episodic_metric_from_records_uses_full_episode_values() -> None:
    """Episodic reducer should average episode-level values, not checkpoint means."""
    records = [
        {"eval_step": 100, "metrics": {"snqi": 0.0}},
        {"eval_step": 100, "metrics": {"snqi": 1.0}},
        {"eval_step": 200, "metrics": {"snqi": 1.0}},
    ]
    # Mean over all 3 episode values.
    assert episodic_metric_from_records(records, metric_name="snqi", window=5) == pytest.approx(
        2.0 / 3.0
    )
    # Last-window=1 keeps only eval_step=200 episodes.
    assert episodic_metric_from_records(records, metric_name="snqi", window=1) == pytest.approx(1.0)


def test_episodic_metric_from_records_returns_none_for_missing_metric() -> None:
    """Episodic reducer should return None when no valid values exist."""
    records = [{"eval_step": 100, "metrics": {"success_rate": 1.0}}]
    assert episodic_metric_from_records(records, metric_name="snqi", window=3) is None


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


def test_load_episode_records_skips_malformed_and_non_object_lines(tmp_path: Path):
    """Malformed JSONL rows should be ignored instead of crashing loading."""
    path = tmp_path / "episodes.jsonl"
    payloads = [
        {"eval_step": 100, "metrics": {"snqi": 0.2}},
        {"eval_step": 200, "metrics": {"snqi": 0.4}},
    ]
    path.write_text(
        "\n".join(
            [
                json.dumps(payloads[0]),
                "{bad json",
                json.dumps(["not", "a", "record"]),
                json.dumps(payloads[1]),
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_episode_records(path)
    assert loaded == payloads
