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
                "artifact_uri": "wandb://robot-sf/run-a/episodes/run-a-001.jsonl",
                "artifact_sha256": "a" * 64,
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
                "artifact_uri": "wandb://robot-sf/run-a/episodes/run-a-002.jsonl",
                "artifact_sha256": "b" * 64,
                "success": False,
                "collision": True,
                "snqi": -0.2,
            },
        ],
        study_id="rsf-2026-06-ranking-transfer",
        command="uv run robot_sf_bench ...",
        source_commit="abc123",
    )

    assert summary["episode_count"] == 2
    assert summary["run_count"] == 1
    assert summary["run_ids"] == ["run-a"]
    assert summary["source_commit"] == "abc123"
    assert validate_result_store(output_dir).ok
    assert (output_dir / "tables" / "manifest.json").is_file()
    assert (output_dir / "figures" / "manifest.json").is_file()
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


def test_result_store_reports_unreadable_artifacts(tmp_path: Path) -> None:
    """Validation should report corrupt artifacts instead of raising."""
    output_dir = tmp_path / "result-store"
    write_result_store(
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
                "artifact_uri": "wandb://robot-sf/run-a/episodes/run-a-001.jsonl",
                "artifact_sha256": "a" * 64,
            }
        ],
        study_id="rsf-2026-06-ranking-transfer",
        command="uv run robot_sf_bench ...",
    )
    (output_dir / "episodes.parquet").write_text("not parquet", encoding="utf-8")
    (output_dir / "summary.json").write_text("{not json", encoding="utf-8")

    result = validate_result_store(output_dir)

    assert not result.ok
    assert any(error.startswith("episodes.parquet could not be read:") for error in result.errors)
    assert any(error.startswith("summary.json could not be read:") for error in result.errors)


def test_result_store_rejects_missing_artifact_provenance(tmp_path: Path) -> None:
    """Artifact provenance must be present before rows can feed reports."""
    output_dir = tmp_path / "result-store"

    try:
        write_result_store(
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
                    "artifact_uri": "",
                    "artifact_sha256": "",
                }
            ],
            study_id="rsf-2026-06-ranking-transfer",
            command="uv run robot_sf_bench ...",
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive assertion for clearer failure output.
        raise AssertionError("write_result_store accepted missing artifact provenance")

    assert "missing artifact provenance field: artifact_uri" in message
    assert "missing artifact provenance field: artifact_sha256" in message


def test_result_store_rejects_invalid_row_status(tmp_path: Path) -> None:
    """Invalid row-status values should fail before report generation."""
    output_dir = tmp_path / "result-store"

    try:
        write_result_store(
            output_dir,
            [
                {
                    "run_id": "run-a",
                    "episode_id": "run-a-001",
                    "planner": "orca",
                    "scenario_id": "crossing",
                    "scenario_family": "crossing",
                    "seed": 5,
                    "row_status": "success",
                    "artifact_uri": "wandb://robot-sf/run-a/episodes/run-a-001.jsonl",
                    "artifact_sha256": "a" * 64,
                }
            ],
            study_id="rsf-2026-06-ranking-transfer",
            command="uv run robot_sf_bench ...",
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive assertion for clearer failure output.
        raise AssertionError("write_result_store accepted invalid row_status")

    assert "invalid row_status: 'success'" in message
