"""Headless smoke for issue #3977 public-requirement scenario diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.map_runner import run_map_batch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_SET = REPO_ROOT / "configs/scenarios/sets/issue_3977_public_requirements.yaml"
EPISODE_SCHEMA = REPO_ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"


def test_issue_3977_public_requirement_headless_smoke(tmp_path: Path) -> None:
    """Four public-requirement scenarios emit diagnostic public_requirement rows."""
    out_path = tmp_path / "episodes.jsonl"

    summary = run_map_batch(
        SCENARIO_SET,
        out_path,
        EPISODE_SCHEMA,
        algo="goal",
        benchmark_profile="baseline-safe",
        horizon=90,
        dt=0.1,
        record_forces=False,
        workers=1,
        resume=False,
    )

    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    by_category = {
        record["public_requirement"]["category"]: record["public_requirement"] for record in records
    }

    assert summary["written"] == 4
    assert set(by_category) == {
        "safe_braking",
        "visibility_and_intent",
        "emergency_reaction",
        "speed_limit",
    }
    assert by_category["safe_braking"]["triggered"] is True
    assert by_category["emergency_reaction"]["triggered"] is True
    assert by_category["speed_limit"]["speed_limit_m_s"] == 0.8
    assert by_category["speed_limit"]["max_speed_m_s"] is not None
    assert by_category["speed_limit"]["speed_limit_violation_count"] > 0
    assert all(record["metrics"]["collisions"] >= 0.0 for record in records)
    assert all(
        record["algorithm_metadata"]["public_requirement"] == record["public_requirement"]
        for record in records
    )
