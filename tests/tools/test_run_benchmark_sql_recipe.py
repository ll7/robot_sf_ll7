"""Tests for DuckDB benchmark SQL recipe runner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robot_sf.benchmark.parquet_export import export_episodes_jsonl_to_parquet
from scripts.tools.run_benchmark_sql_recipe import (
    RecipeValidationError,
    load_recipe_manifest,
    main,
    run_recipe,
)

if TYPE_CHECKING:
    from pathlib import Path


def _episode(
    *,
    episode_id: str,
    algo: str,
    scenario_family: str,
    seed: int,
    metrics: dict[str, float | bool],
    execution_mode: str = "native",
) -> dict[str, Any]:
    """Build a tiny benchmark episode fixture."""

    return {
        "version": "v1",
        "episode_id": episode_id,
        "scenario_id": f"{scenario_family}_001",
        "seed": seed,
        "algo": algo,
        "scenario_params": {"scenario_family": scenario_family},
        "algorithm_metadata": {
            "algorithm": algo,
            "planner_kinematics": {"execution_mode": execution_mode},
        },
        "metrics": {
            "success": metrics["success"],
            "collisions": metrics["collisions"],
            "min_ttc": metrics["min_ttc"],
            "clearance": metrics["clearance"],
        },
        "termination_reason": "goal_reached" if metrics["success"] else "collision",
        "outcome": {"collision_event": bool(metrics["collisions"])},
        "integrity": {"seed": seed},
        "timestamps": {
            "start": "2026-05-16T08:00:00+00:00",
            "end": "2026-05-16T08:00:01+00:00",
        },
        "wall_time_sec": 1.0,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records to a JSONL fixture."""

    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _export_fixture(tmp_path: Path) -> Path:
    """Export a tiny benchmark fixture to Parquet and return the export dir."""

    episodes_path = tmp_path / "episodes.jsonl"
    _write_jsonl(
        episodes_path,
        [
            _episode(
                episode_id="ep-a",
                algo="planner_a",
                scenario_family="crossing",
                seed=11,
                metrics={"success": True, "collisions": 0.0, "min_ttc": 1.5, "clearance": 0.4},
            ),
            _episode(
                episode_id="ep-b",
                algo="planner_a",
                scenario_family="crossing",
                seed=12,
                metrics={
                    "success": False,
                    "collisions": 1.0,
                    "min_ttc": 0.1,
                    "clearance": 0.05,
                },
            ),
            _episode(
                episode_id="ep-c",
                algo="planner_b",
                scenario_family="overtake",
                seed=11,
                metrics={"success": True, "collisions": 0.0, "min_ttc": 2.0, "clearance": 0.6},
                execution_mode="adapter",
            ),
        ],
    )
    return export_episodes_jsonl_to_parquet(episodes_path, tmp_path / "parquet").output_dir


def test_recipe_manifest_exposes_stable_recipe_metadata() -> None:
    """Recipe manifest should describe stable IDs, inputs, outputs, and caveats."""

    manifest = load_recipe_manifest()

    recipe = manifest.recipe("planner_outcome_summary")
    assert recipe.recipe_id == "planner_outcome_summary"
    assert recipe.sql_file.name == "planner_outcome_summary.sql"
    assert recipe.required_tables == ("episodes", "metrics")
    assert "success_rate" in recipe.output_columns
    assert recipe.caveats


def test_run_recipe_writes_csv_and_markdown_outputs(tmp_path: Path) -> None:
    """Runner should execute a recipe against fixture Parquet and export stable outputs."""

    export_dir = _export_fixture(tmp_path)
    csv_path = tmp_path / "planner_outcome_summary.csv"
    md_path = tmp_path / "planner_outcome_summary.md"

    result = run_recipe(
        "planner_outcome_summary",
        export_dir=export_dir,
        output_csv=csv_path,
        output_markdown=md_path,
    )

    assert result.recipe_id == "planner_outcome_summary"
    assert result.row_count == 2
    assert csv_path.is_file()
    assert md_path.is_file()
    assert "planner_a" in csv_path.read_text(encoding="utf-8")
    assert "success_rate" in md_path.read_text(encoding="utf-8")


def test_cli_runs_recipe_against_fixture_export(tmp_path: Path) -> None:
    """CLI should expose the same recipe runner used by automation."""

    export_dir = _export_fixture(tmp_path)
    csv_path = tmp_path / "failure_rows.csv"

    exit_code = main(
        [
            "--recipe",
            "failure_near_miss_mining",
            "--export-dir",
            str(export_dir),
            "--output-csv",
            str(csv_path),
        ]
    )

    assert exit_code == 0
    text = csv_path.read_text(encoding="utf-8")
    assert "ep-b" in text
    assert "collision" in text


def test_missing_required_column_reports_actionable_error(tmp_path: Path) -> None:
    """Missing required columns should fail closed instead of silently filling zeros."""

    export_dir = _export_fixture(tmp_path)
    broken_episodes = export_dir / "episodes.parquet"
    table = pq.read_table(broken_episodes).drop(["scenario_family"])
    pq.write_table(pa.Table.from_batches(table.to_batches()), broken_episodes)

    with pytest.raises(RecipeValidationError, match="episodes\\.scenario_family"):
        run_recipe("planner_outcome_summary", export_dir=export_dir)
