from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.aggregate import (
    compute_aggregates,
    flatten_metrics,
    read_jsonl,
    write_episode_csv,
)
from robot_sf.benchmark.runner import run_batch

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def _make_sample_jsonl(tmp_path: Path) -> Path:
    # Use run_batch to generate 3 episodes across 2 algos (via scenario_params.algo)
    scenarios = [
        {
            "id": "agg-uni-low-open-A",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 2,
            "algo": "A",
        },
        {
            "id": "agg-uni-low-open-B",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 1,
            "algo": "B",
        },
    ]
    out_file = tmp_path / "episodes.jsonl"
    # run without forces for speed
    run_batch(
        scenarios,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=10,
        horizon=8,
        dt=0.1,
        record_forces=False,
        append=False,
    )
    return out_file


def test_read_and_flatten_and_write_csv(tmp_path: Path):
    jsonl_path = _make_sample_jsonl(tmp_path)
    recs = read_jsonl(jsonl_path)
    assert len(recs) == 3
    rows = [flatten_metrics(r) for r in recs]
    assert all("episode_id" in r for r in rows)
    # Write CSV
    csv_path = tmp_path / "episodes.csv"
    out = write_episode_csv(recs, csv_path)
    assert Path(out).exists()
    text = Path(out).read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("episode_id,scenario_id,seed")


def test_compute_aggregates_group_by_algo(tmp_path: Path):
    jsonl_path = _make_sample_jsonl(tmp_path)
    recs = read_jsonl(jsonl_path)
    # We stored algo at the top-level of scenario params; group path is scenario_params.algo
    summary = compute_aggregates(recs, group_by="scenario_params.algo")
    # Should contain two groups A and B
    assert set(summary.keys()) == {"A", "B"}
    # Each group should have numeric aggregates present for some core metric
    for metrics in summary.values():
        assert "time_to_goal_norm" in metrics
        assert set(metrics["time_to_goal_norm"].keys()) == {"mean", "median", "p95"}
