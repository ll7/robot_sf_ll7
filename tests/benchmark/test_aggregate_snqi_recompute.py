"""Regression tests for aggregate SNQI recomputation controls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

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
