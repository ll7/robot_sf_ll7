"""Regression tests for aggregate SNQI recomputation controls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from loguru import logger

from robot_sf.benchmark import aggregate
from robot_sf.benchmark.aggregate import compute_aggregates

if TYPE_CHECKING:
    from pathlib import Path


def test_compute_aggregates_recomputes_stored_snqi_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SNQI aggregation should preserve stored scores unless recompute is explicit."""

    def _fake_snqi(
        metrics: dict[str, float],
        weights: dict[str, float],
        *,
        baseline_stats: dict[str, dict[str, float]] | None = None,
    ) -> float:
        return float(metrics["score"] * weights["score"])

    monkeypatch.setattr(aggregate, "snqi_fn", _fake_snqi)
    records = [
        {
            "episode_id": "ep-1",
            "scenario_id": "sc-1",
            "seed": 1,
            "algo": "planner-a",
            "metrics": {"score": 2.0, "snqi": 0.25},
        }
    ]

    summary = compute_aggregates(
        records,
        group_by="algo",
        snqi_weights={"score": 3.0},
    )
    assert summary["planner-a"]["snqi"]["mean"] == 0.25

    recomputed = compute_aggregates(
        records,
        group_by="algo",
        snqi_weights={"score": 3.0},
        recompute_snqi=True,
    )
    assert recomputed["planner-a"]["snqi"]["mean"] == 6.0


def test_compute_aggregates_logs_record_id_and_reraises_snqi_failure_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict aggregation should fail closed with record context when SNQI recompute fails."""

    def _failing_snqi(
        metrics: dict[str, float],
        weights: dict[str, float],
        *,
        baseline_stats: dict[str, dict[str, float]] | None = None,
    ) -> float:
        raise ValueError("bad snqi input")

    monkeypatch.setattr(aggregate, "snqi_fn", _failing_snqi)
    records = [
        {
            "episode_id": "ep-snqi-fail",
            "scenario_id": "sc-1",
            "seed": 1,
            "algo": "planner-a",
            "metrics": {"score": 2.0},
        }
    ]
    captured: list = []
    handle = logger.add(captured.append, level="ERROR")
    try:
        with pytest.raises(ValueError, match="bad snqi input"):
            compute_aggregates(records, group_by="algo", snqi_weights={"score": 3.0})
    finally:
        logger.remove(handle)

    assert any(
        msg.record["extra"].get("event") == "aggregation_snqi_compute_failed"
        and msg.record["extra"].get("episode_id") == "ep-snqi-fail"
        for msg in captured
    )


def test_compute_aggregates_logs_key_error_snqi_failure_in_diagnostic_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diagnostic aggregation should log missing SNQI inputs without serializing SNQI."""

    def _missing_metric_snqi(
        metrics: dict[str, float],
        weights: dict[str, float],
        *,
        baseline_stats: dict[str, dict[str, float]] | None = None,
    ) -> float:
        raise KeyError("renamed_metric")

    monkeypatch.setattr(aggregate, "snqi_fn", _missing_metric_snqi)
    records = [
        {
            "episode_id": "ep-missing-key",
            "scenario_id": "sc-1",
            "seed": 1,
            "algo": "planner-a",
            "observation_track": "features",
            "metrics": {"score": 2.0},
        }
    ]
    captured: list = []
    handle = logger.add(captured.append, level="ERROR")
    try:
        result = compute_aggregates(
            records,
            group_by="algo",
            snqi_weights={"score": 3.0},
            observation_track_mode="diagnostic-cross-track",
        )
    finally:
        logger.remove(handle)

    assert "snqi" not in records[0]["metrics"]
    score_summary = next(group["score"] for group in result.values() if "score" in group)
    assert score_summary["mean"] == 2.0
    assert any(
        msg.record["extra"].get("event") == "aggregation_snqi_compute_failed"
        and msg.record["extra"].get("episode_id") == "ep-missing-key"
        for msg in captured
    )


def test_aggregate_cli_passes_recompute_snqi_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The aggregate CLI should wire --recompute-snqi into aggregation."""
    pytest.importorskip("torch")

    from robot_sf.benchmark import cli

    src = tmp_path / "episodes.jsonl"
    src.write_text('{"episode_id":"ep-1","metrics":{}}\n', encoding="utf-8")
    weights_path = tmp_path / "weights.json"
    weights_path.write_text('{"score": 1.0}\n', encoding="utf-8")
    captured: dict[str, Any] = {}

    def _fake_compute(records: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"_meta": {"records": len(records)}}

    monkeypatch.setattr(cli, "_agg_compute", _fake_compute)

    exit_code = cli.cli_main(
        [
            "aggregate",
            "--in",
            str(src),
            "--out",
            str(tmp_path / "summary.json"),
            "--snqi-weights",
            str(weights_path),
            "--recompute-snqi",
        ]
    )

    assert exit_code == 0
    assert captured["snqi_weights"] == {"score": 1.0}
    assert captured["recompute_snqi"] is True
