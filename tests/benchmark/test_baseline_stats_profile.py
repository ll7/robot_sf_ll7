"""Tests for baseline benchmark profile propagation."""

from __future__ import annotations

from typing import Any

from robot_sf.benchmark import baseline_stats


def test_run_and_compute_baseline_forwards_benchmark_profile(monkeypatch, tmp_path) -> None:
    """Experimental baselines must be able to reach map-runner readiness gates."""
    captured: dict[str, Any] = {}

    def fake_run_batch(*args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(baseline_stats, "run_batch", fake_run_batch)
    monkeypatch.setattr(baseline_stats, "read_jsonl", lambda _path: [{"metrics": {}}])
    monkeypatch.setattr(
        baseline_stats,
        "compute_baseline_stats_from_records",
        lambda _records, metrics=None: {"success": {"med": 1.0, "p95": 1.0}},
    )

    baseline_stats.run_and_compute_baseline(
        [{"id": "scenario"}],
        out_json=tmp_path / "stats.json",
        out_jsonl=tmp_path / "episodes.jsonl",
        schema_path="robot_sf/benchmark/schemas/episode.schema.v1.json",
        algo="distributional_rl",
        benchmark_profile="experimental",
    )

    assert captured["algo"] == "distributional_rl"
    assert captured["benchmark_profile"] == "experimental"
