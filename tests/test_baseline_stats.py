"""Tests for computing baseline benchmark summary statistics from records and runs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.baseline_stats import (
    compute_baseline_stats_from_records,
    run_and_compute_baseline,
)
from robot_sf.benchmark.metric_definitions import metric_schema_version
from robot_sf.benchmark.metrics import snqi

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def test_compute_baseline_stats_from_records(tmp_path: Path):
    # Create a tiny JSONL with two records and known metrics
    """Verify baseline statistic calculations from pre-generated episode records."""
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
    assert set(stats.keys()) == {"time_to_goal_norm", "collisions", "energy", "_metadata"}
    assert stats["_metadata"]["metric_schema_version"] == "robot-sf-metrics.v1"
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
    """Verify executing a minimal scenario matrix and computing baseline summary statistics."""
    out_json = tmp_path / "baseline_stats.json"
    out_jsonl = tmp_path / "episodes.jsonl"
    stats = run_and_compute_baseline(
        ess_min_matrix,
        out_json=out_json,
        out_jsonl=out_jsonl,
        schema_path=SCHEMA_PATH,
        base_seed=1001,
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


@pytest.mark.parametrize("version", ["robot-sf-metrics.v1", "robot-sf-metrics.v2"])
def test_derived_baseline_preserves_version_and_is_accepted(tmp_path: Path, version: str) -> None:
    """Persisted matching-version anchors must normalize records from either definition."""
    records = [
        {
            "metrics": {
                "metric_schema_version": version,
                "success": 1.0,
                "time_to_goal_norm": 0.25,
                "collisions": c,
            }
        }
        for c in (0.0, 2.0)
    ]
    stats = compute_baseline_stats_from_records(records, metrics=("collisions",))
    path = tmp_path / "anchors.json"
    path.write_text(json.dumps(stats), encoding="utf-8")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["_metadata"]["metric_schema_version"] == version
    assert metric_schema_version(saved) == version
    assert saved["collisions"] == {"med": 1.0, "p95": 1.9}
    assert (
        snqi(records[0]["metrics"], weights={"w_success": 1.0, "w_time": 0.0}, baseline_stats=saved)
        == 1.0
    )


def test_derived_baseline_refuses_mixed_versions() -> None:
    """Changed metric definitions may never be pooled into one baseline."""
    records = [
        {"metrics": {"metric_schema_version": version, "collisions": 0}}
        for version in ("robot-sf-metrics.v1", "robot-sf-metrics.v2")
    ]
    with pytest.raises(ValueError, match="incompatible metric definitions"):
        compute_baseline_stats_from_records(records)


def test_unmarked_baseline_is_refused_for_v2() -> None:
    """Historical anchors keep their v1 meaning; relabeling them is not the repair."""
    with pytest.raises(ValueError, match="incompatible metric definitions"):
        snqi(
            {"metric_schema_version": "robot-sf-metrics.v2", "success": 1.0},
            weights={"w_success": 1.0, "w_time": 0.0},
            baseline_stats={"collisions": {"med": 0, "p95": 1}},
        )
