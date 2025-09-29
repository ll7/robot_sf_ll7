from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.baseline_stats import (
    compute_baseline_stats_from_records,
    run_and_compute_baseline,
)

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def test_compute_baseline_stats_from_records(tmp_path: Path):
    # Create a tiny JSONL with two records and known metrics
    p = tmp_path / "episodes.jsonl"
    recs = [
        {
            "episode_id": "e1",
            "scenario_id": "s1",
            "seed": 1,
            "metrics": {"time_to_goal_norm": 0.5, "collisions": 0, "energy": 1.0},
        },
        {
            "episode_id": "e2",
            "scenario_id": "s1",
            "seed": 2,
            "metrics": {"time_to_goal_norm": 0.7, "collisions": 1, "energy": 3.0},
        },
    ]
    with p.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    loaded = read_jsonl(p)
    stats = compute_baseline_stats_from_records(
        loaded,
        metrics=("time_to_goal_norm", "collisions", "energy"),
    )
    assert set(stats.keys()) == {"time_to_goal_norm", "collisions", "energy"}
    assert stats["time_to_goal_norm"]["med"] == 0.6
    assert stats["collisions"]["p95"] >= 0.95  # 95th percentile between 0 and 1
    assert stats["energy"]["med"] == 2.0


ess_min_matrix = [
    {
        "id": "bl-uni-low-open",
        "density": "low",
        "flow": "uni",
        "obstacle": "open",
        "groups": 0.0,
        "speed_var": "low",
        "goal_topology": "point",
        "robot_context": "embedded",
        "repeats": 2,
    },
]


def test_run_and_compute_baseline(tmp_path: Path):
    out_json = tmp_path / "baseline_stats.json"
    out_jsonl = tmp_path / "episodes.jsonl"
    stats = run_and_compute_baseline(
        ess_min_matrix,
        out_json=out_json,
        out_jsonl=out_jsonl,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=8,
        dt=0.1,
        record_forces=False,
    )
    assert out_json.exists()
    # sanity: some expected keys exist
    for k in ("time_to_goal_norm", "collisions", "energy"):
        assert k in stats
    # JSON round-trip
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved.keys() == stats.keys()
