"""Tests for training SNQI utility helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.snqi import compute_snqi
from robot_sf.training import snqi_utils
from robot_sf.training.snqi_utils import (
    compute_training_snqi,
    default_training_snqi_context,
    resolve_training_snqi_context,
)

if TYPE_CHECKING:
    from pathlib import Path


class _WarningRecorder:
    """Capture loguru-style warning calls without depending on global logger sinks."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, message: str, *args: object) -> None:
        self.messages.append(message.format(*args))


def test_default_training_snqi_context_has_collision_penalty_baseline():
    """Default context should define normalization stats for key safety metrics."""
    context = default_training_snqi_context()
    assert context.weights_source == "default"
    assert context.baseline_source == "default"
    assert context.baseline_stats["collisions"] == {"med": 0.0, "p95": 1.0}
    assert context.weights["w_collisions"] == 2.0


def test_resolve_training_snqi_context_supports_nested_weight_payload(tmp_path: Path):
    """Weight payloads with a nested 'weights' mapping should be accepted."""
    weights_file = tmp_path / "weights.json"
    weights_file.write_text(
        json.dumps(
            {
                "weights": {
                    "w_success": 1.2,
                    "w_time": 0.7,
                    "w_collisions": 2.5,
                    "w_near": 0.8,
                    "w_comfort": 0.4,
                    "w_force_exceed": 1.1,
                    "w_jerk": 0.2,
                },
            },
        ),
        encoding="utf-8",
    )
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(
        json.dumps({"collisions": {"med": 0.0, "p95": 2.0}}),
        encoding="utf-8",
    )

    context = resolve_training_snqi_context(weights_path=weights_file, baseline_path=baseline_file)
    assert context.weights_source == str(weights_file)
    assert context.baseline_source == str(baseline_file)
    assert context.weights["w_success"] == 1.2
    assert "near_misses" in context.baseline_stats
    assert "near_misses" in context.baseline_fallback_keys


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        ("{not-json", "Failed to load SNQI weights"),
        (json.dumps({"weights": {"w_success": 1.0}}), "Missing weight keys"),
    ],
)
def test_resolve_training_snqi_context_invalid_weights_falls_back_to_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
    expected_message: str,
):
    """Malformed or structurally invalid weights should not crash config resolution."""
    recorder = _WarningRecorder()
    monkeypatch.setattr(snqi_utils, "logger", recorder)
    weights_file = tmp_path / "weights.json"
    weights_file.write_text(payload, encoding="utf-8")

    context = resolve_training_snqi_context(weights_path=weights_file)
    default_context = default_training_snqi_context()

    assert context.weights == default_context.weights
    assert context.weights_source == "default"
    assert context.baseline_source == "default"
    assert any(expected_message in message for message in recorder.messages)


def test_resolve_training_snqi_context_missing_baseline_falls_back_to_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Unreadable baseline paths should not crash config resolution."""
    recorder = _WarningRecorder()
    monkeypatch.setattr(snqi_utils, "logger", recorder)

    context = resolve_training_snqi_context(baseline_path=tmp_path / "missing-baseline.json")
    default_context = default_training_snqi_context()

    assert context.weights == default_context.weights
    assert context.baseline_stats == default_context.baseline_stats
    assert context.baseline_source == "default"
    assert context.baseline_fallback_keys == ()
    assert any("Failed to load SNQI baseline stats" in message for message in recorder.messages)


def test_compute_training_snqi_matches_canonical_benchmark_formula():
    """The training wrapper should delegate to canonical benchmark SNQI compute."""
    context = default_training_snqi_context()
    metric_values = {
        "success": 1.0,
        "time_to_goal_norm": 0.4,
        "collisions": 1.0,
        "near_misses": 0.0,
        "comfort_exposure": 0.1,
        "force_exceed_events": 0.0,
        "jerk_mean": 0.0,
    }
    expected = compute_snqi(metric_values, context.weights, context.baseline_stats)
    assert compute_training_snqi(metric_values, context=context) == expected
