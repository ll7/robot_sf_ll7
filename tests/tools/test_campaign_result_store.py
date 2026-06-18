"""Tests for the canonical campaign result-store helper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb

from scripts.tools.campaign_result_store import validate_result_store, write_result_store

if TYPE_CHECKING:
    from pathlib import Path


def test_result_store_round_trips_episode_rows_through_parquet_and_duckdb(tmp_path: Path) -> None:
    """Episode rows should be queryable from the canonical Parquet store."""
    output_dir = tmp_path / "result-store"
    summary = write_result_store(
        output_dir,
        [
            {
                "run_id": "run-a",
                "episode_id": "run-a-001",
                "planner": "orca",
                "scenario_id": "crossing",
                "scenario_family": "crossing",
                "seed": 5,
                "row_status": "native",
                "success": True,
                "collision": False,
                "snqi": 0.5,
            },
            {
                "run_id": "run-a",
                "episode_id": "run-a-002",
                "planner": "social_force",
                "scenario_id": "crossing",
                "scenario_family": "crossing",
                "seed": 5,
                "row_status": "native",
                "success": False,
                "collision": True,
                "snqi": -0.2,
            },
        ],
        study_id="rsf-2026-06-ranking-transfer",
        command="uv run robot_sf_bench ...",
    )

    assert summary["episode_count"] == 2
    assert validate_result_store(output_dir).ok
    with duckdb.connect(database=":memory:") as connection:
        rows = connection.execute(
            "select planner, count(*) as n from read_parquet(?) group by planner order by planner",
            [str(output_dir / "episodes.parquet")],
        ).fetchall()
    assert rows == [("orca", 1), ("social_force", 1)]


def test_result_store_reports_missing_required_files(tmp_path: Path) -> None:
    """Validation should fail closed when required report surfaces are absent."""
    output_dir = tmp_path / "incomplete"
    output_dir.mkdir()

    result = validate_result_store(output_dir)

    assert not result.ok
    assert "missing required result-store file: episodes.parquet" in result.errors
