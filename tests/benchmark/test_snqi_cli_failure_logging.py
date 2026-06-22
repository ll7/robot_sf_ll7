"""SNQI CLI failure logging regressions for benchmark readiness."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

from loguru import logger

from robot_sf.benchmark.snqi import cli as snqi_cli
from robot_sf.benchmark.snqi.compute import WEIGHT_NAMES
from robot_sf.benchmark.snqi.types import SNQIWeights

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _baseline_stats() -> dict[str, dict[str, float]]:
    """Return compact baseline stats for SNQI CLI tests."""
    return {
        "collisions": {"med": 0.0, "p95": 2.0},
        "near_misses": {"med": 1.0, "p95": 3.0},
        "force_exceed_events": {"med": 0.0, "p95": 1.0},
        "jerk_mean": {"med": 0.1, "p95": 0.4},
    }


def _capture_errors() -> tuple[list, int]:
    """Attach a loguru error sink for structured CLI assertions."""
    captured: list = []
    handle = logger.add(captured.append, level="ERROR")
    return captured, handle


def _assert_logged_stage(captured: list, stage: str) -> None:
    """Assert that the SNQI CLI emitted the expected structured failure stage."""
    assert any(
        msg.record["extra"].get("event") == "snqi_cli_failed"
        and msg.record["extra"].get("stage") == stage
        for msg in captured
    )


def test_cmd_recompute_weights_logs_expected_failure_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recompute should log missing-input, bad-method, and compute-failure stages."""
    baseline_path = tmp_path / "baseline.json"
    output_path = tmp_path / "weights.json"
    baseline_path.write_text(json.dumps(_baseline_stats()), encoding="utf-8")

    def _fail_recompute(**kwargs: object) -> SNQIWeights:
        raise ValueError("cannot recompute")

    monkeypatch.setattr(snqi_cli, "recompute_snqi_weights", _fail_recompute)
    captured, handle = _capture_errors()
    try:
        missing = snqi_cli.cmd_recompute_weights(
            argparse.Namespace(
                baseline_stats=str(tmp_path / "missing.json"),
                out=str(output_path),
                method="balanced",
                seed=7,
            ),
        )
        unknown = snqi_cli.cmd_recompute_weights(
            argparse.Namespace(
                baseline_stats=str(baseline_path),
                out=str(output_path),
                method="old_alias",
                seed=7,
            ),
        )
        failed = snqi_cli.cmd_recompute_weights(
            argparse.Namespace(
                baseline_stats=str(baseline_path),
                out=str(output_path),
                method="balanced",
                seed=7,
            ),
        )
    finally:
        logger.remove(handle)

    assert (missing, unknown, failed) == (1, 2, 1)
    _assert_logged_stage(captured, "load_baseline_stats")
    _assert_logged_stage(captured, "validate_method")
    _assert_logged_stage(captured, "recompute_weights")


def test_cmd_recompute_weights_logs_unexpected_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-level recompute command should log unexpected exceptions before returning."""
    baseline_path = tmp_path / "baseline.json"
    output_path = tmp_path / "weights.json"
    baseline_path.write_text(json.dumps(_baseline_stats()), encoding="utf-8")

    def _fail_recompute(**kwargs: object) -> SNQIWeights:
        raise AttributeError("unexpected recompute state")

    monkeypatch.setattr(snqi_cli, "recompute_snqi_weights", _fail_recompute)
    captured, handle = _capture_errors()
    try:
        exit_code = snqi_cli.cmd_recompute_weights(
            argparse.Namespace(
                baseline_stats=str(baseline_path),
                out=str(output_path),
                method="balanced",
                seed=7,
            ),
        )
    finally:
        logger.remove(handle)

    assert exit_code == 1
    _assert_logged_stage(captured, "recompute_weights")


def test_cmd_ablation_analysis_logs_expected_failure_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ablation should log missing episode, missing weight, and compute-failure stages."""
    empty_episodes = tmp_path / "empty.jsonl"
    empty_episodes.write_text("", encoding="utf-8")
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(
        json.dumps({"metrics": {"collisions": 0, "near_misses": 1, "jerk_mean": 0.1}}) + "\n",
        encoding="utf-8",
    )
    weights_path = tmp_path / "weights.json"
    SNQIWeights(
        weights_version="test",
        created_at="2026-01-01T00:00:00",
        git_sha="test",
        baseline_stats_path="baseline.json",
        baseline_stats_hash="hash",
        normalization_strategy="median_p95_clamp",
        components=list(WEIGHT_NAMES),
        weights=dict.fromkeys(WEIGHT_NAMES, 1.0),
    ).save(weights_path)

    def _fail_ablation(**kwargs: object) -> dict[str, float]:
        raise ValueError("cannot ablate")

    monkeypatch.setattr(snqi_cli, "compute_snqi_ablation", _fail_ablation)
    captured, handle = _capture_errors()
    try:
        missing_episodes = snqi_cli.cmd_ablation_analysis(
            argparse.Namespace(
                episodes=str(tmp_path / "missing.jsonl"),
                summary_out=str(tmp_path / "missing-out.json"),
                weights=None,
                components=None,
                seed=11,
            )
        )
        empty = snqi_cli.cmd_ablation_analysis(
            argparse.Namespace(
                episodes=str(empty_episodes),
                summary_out=str(tmp_path / "empty-out.json"),
                weights=None,
                components=None,
                seed=11,
            )
        )
        missing_weights = snqi_cli.cmd_ablation_analysis(
            argparse.Namespace(
                episodes=str(episodes_path),
                summary_out=str(tmp_path / "missing-weights-out.json"),
                weights=str(tmp_path / "missing-weights.json"),
                components=None,
                seed=11,
            )
        )
        failed = snqi_cli.cmd_ablation_analysis(
            argparse.Namespace(
                episodes=str(episodes_path),
                summary_out=str(tmp_path / "failed-out.json"),
                weights=str(weights_path),
                components=None,
                seed=11,
            )
        )
    finally:
        logger.remove(handle)

    assert (missing_episodes, empty, missing_weights, failed) == (1, 1, 1, 1)
    load_episode_logs = [
        msg
        for msg in captured
        if msg.record["extra"].get("event") == "snqi_cli_failed"
        and msg.record["extra"].get("stage") == "load_episodes"
    ]
    assert len(load_episode_logs) >= 2
    _assert_logged_stage(captured, "load_weights")
    _assert_logged_stage(captured, "ablation_analysis")


def test_cmd_ablation_analysis_logs_unexpected_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-level ablation command should log unexpected exceptions before returning."""
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(
        json.dumps({"metrics": {"collisions": 0, "near_misses": 1, "jerk_mean": 0.1}}) + "\n",
        encoding="utf-8",
    )

    def _fail_ablation(**kwargs: object) -> dict[str, float]:
        raise AttributeError("unexpected ablation state")

    monkeypatch.setattr(snqi_cli, "compute_snqi_ablation", _fail_ablation)
    captured, handle = _capture_errors()
    try:
        exit_code = snqi_cli.cmd_ablation_analysis(
            argparse.Namespace(
                episodes=str(episodes_path),
                summary_out=str(tmp_path / "failed-out.json"),
                weights=None,
                components=None,
                seed=11,
            )
        )
    finally:
        logger.remove(handle)

    assert exit_code == 1
    _assert_logged_stage(captured, "ablation_analysis")
