"""Tests for DuckDB benchmark SQL recipe runner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robot_sf.benchmark.parquet_export import export_episodes_jsonl_to_parquet
from scripts.tools.run_benchmark_sql_recipe import (
    RECIPE_INVENTORY_SCHEMA,
    RecipeValidationError,
    inventory_recipes,
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


def _write_recipe_root(
    root: Path,
    *,
    manifest_recipe: str,
    sql: str = "SELECT COUNT(*) AS episode_count FROM {episodes}",
) -> Path:
    """Write a tiny custom recipe root for hardening tests."""

    (root / "sql").mkdir(parents=True)
    (root / "sql" / "recipe.sql").write_text(sql, encoding="utf-8")
    (root / "manifest.yaml").write_text(
        "\n".join(
            [
                "schema_version: benchmark_sql_recipes.v1",
                "recipes:",
                manifest_recipe,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return root


def _single_table_manifest_recipe() -> str:
    """Return one manifest recipe that only depends on the episodes table."""

    return "\n".join(
        [
            "  - recipe_id: custom",
            "    title: Custom recipe",
            "    sql_file: sql/recipe.sql",
            "    required_tables:",
            "      - episodes",
            "    required_columns:",
            "      episodes:",
            "        - episode_id",
            "    output_columns:",
            "      - episode_count",
            "    caveats:",
            "      - fixture only",
        ]
    )


def test_cli_reports_missing_manifest_without_traceback(tmp_path: Path, capsys) -> None:
    """Missing manifests should fail closed through the CLI."""

    export_dir = _export_fixture(tmp_path)

    exit_code = main(
        [
            "--recipe",
            "custom",
            "--export-dir",
            str(export_dir),
            "--recipe-root",
            str(tmp_path / "missing-recipes"),
        ]
    )

    assert exit_code == 2
    assert "Recipe manifest file not found" in capsys.readouterr().err


def test_malformed_manifest_reports_recipe_validation_error(tmp_path: Path) -> None:
    """Malformed YAML should be reported as recipe validation, not a parser traceback."""

    recipe_root = tmp_path / "recipes"
    recipe_root.mkdir()
    (recipe_root / "manifest.yaml").write_text("recipes: [", encoding="utf-8")

    with pytest.raises(RecipeValidationError, match="Failed to parse manifest"):
        load_recipe_manifest(recipe_root)


def test_manifest_missing_required_field_reports_actionable_error(tmp_path: Path) -> None:
    """Missing manifest fields should name the missing field."""

    recipe_root = tmp_path / "recipes"
    _write_recipe_root(
        recipe_root,
        manifest_recipe="\n".join(
            [
                "  - recipe_id: custom",
                "    title: Custom recipe",
            ]
        ),
    )

    with pytest.raises(RecipeValidationError, match="missing required field: sql_file"):
        load_recipe_manifest(recipe_root)


def test_manifest_invalid_recipe_structure_reports_actionable_error(tmp_path: Path) -> None:
    """Malformed recipe field shapes should not leak TypeError or ValueError."""

    recipe_root = tmp_path / "recipes"
    _write_recipe_root(
        recipe_root,
        manifest_recipe="\n".join(
            [
                "  - recipe_id: custom",
                "    title: Custom recipe",
                "    sql_file: sql/recipe.sql",
                "    required_tables: 42",
                "    required_columns: not-a-mapping",
                "    output_columns:",
                "      - episode_count",
            ]
        ),
    )

    with pytest.raises(RecipeValidationError, match="invalid structure"):
        load_recipe_manifest(recipe_root)


def test_sql_literal_braces_do_not_break_placeholder_substitution(tmp_path: Path) -> None:
    """SQL files may contain literal braces unrelated to table placeholders."""

    export_dir = _export_fixture(tmp_path)
    recipe_root = tmp_path / "recipes"
    _write_recipe_root(
        recipe_root,
        manifest_recipe=_single_table_manifest_recipe(),
        sql="SELECT '{literal brace}' AS note, COUNT(*) AS episode_count FROM {episodes}",
    )

    result = run_recipe("custom", export_dir=export_dir, recipe_root=recipe_root)

    assert result.rows == (("{literal brace}", 3),)


def test_missing_sql_file_reports_recipe_validation_error(tmp_path: Path) -> None:
    """Missing SQL files should fail before read_text raises FileNotFoundError."""

    export_dir = _export_fixture(tmp_path)
    recipe_root = tmp_path / "recipes"
    _write_recipe_root(recipe_root, manifest_recipe=_single_table_manifest_recipe())
    (recipe_root / "sql" / "recipe.sql").unlink()

    with pytest.raises(RecipeValidationError, match="SQL recipe file not found"):
        run_recipe("custom", export_dir=export_dir, recipe_root=recipe_root)


def test_sql_execution_errors_are_recipe_validation_errors(tmp_path: Path) -> None:
    """DuckDB execution failures should be converted to clean validation errors."""

    export_dir = _export_fixture(tmp_path)
    recipe_root = tmp_path / "recipes"
    _write_recipe_root(
        recipe_root,
        manifest_recipe=_single_table_manifest_recipe(),
        sql="SELECT missing_column AS episode_count FROM {episodes}",
    )

    with pytest.raises(RecipeValidationError, match="SQL execution failed"):
        run_recipe("custom", export_dir=export_dir, recipe_root=recipe_root)


def test_unreadable_parquet_schema_reports_recipe_validation_error(tmp_path: Path) -> None:
    """Unreadable Parquet files should fail closed during schema validation."""

    export_dir = _export_fixture(tmp_path)
    (export_dir / "episodes.parquet").write_bytes(b"not parquet")
    recipe_root = tmp_path / "recipes"
    _write_recipe_root(recipe_root, manifest_recipe=_single_table_manifest_recipe())

    with pytest.raises(RecipeValidationError, match="Failed to read schema"):
        run_recipe("custom", export_dir=export_dir, recipe_root=recipe_root)


def test_markdown_float_cells_use_compact_fixed_precision(tmp_path: Path) -> None:
    """Markdown output should avoid noisy binary float representations."""

    export_dir = _export_fixture(tmp_path)
    md_path = tmp_path / "summary.md"

    run_recipe(
        "planner_outcome_summary",
        export_dir=export_dir,
        output_markdown=md_path,
    )

    assert "0.5000" in md_path.read_text(encoding="utf-8")


def test_inventory_recipes_exposes_full_schema_contract() -> None:
    """The inventory must expose every recipe's declared schema contract (issue #3467)."""
    manifest = load_recipe_manifest()
    inventory = inventory_recipes()

    assert inventory["schema"] == RECIPE_INVENTORY_SCHEMA
    assert inventory["recipe_schema_version"] == manifest.schema_version
    assert inventory["recipe_count"] == len(manifest.recipes)

    inventoried = {entry["recipe_id"]: entry for entry in inventory["recipes"]}
    assert inventoried.keys() == {recipe.recipe_id for recipe in manifest.recipes}
    for recipe in manifest.recipes:
        entry = inventoried[recipe.recipe_id]
        assert entry["title"] == recipe.title
        assert entry["required_tables"] == list(recipe.required_tables)
        assert entry["output_columns"] == list(recipe.output_columns)
        assert entry["required_columns"] == {
            table: list(columns) for table, columns in recipe.required_columns.items()
        }


def test_cli_list_emits_recipe_inventory_json(capsys) -> None:
    """`--list` must emit a parseable inventory and skip execution (no --export-dir needed)."""
    exit_code = main(["--list"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == RECIPE_INVENTORY_SCHEMA
    assert payload["recipe_count"] == len(payload["recipes"])
    assert any(entry["recipe_id"] == "planner_outcome_summary" for entry in payload["recipes"])


def test_cli_requires_recipe_or_list(capsys) -> None:
    """Omitting both --recipe and --list must fail closed with an actionable message."""
    exit_code = main([])

    assert exit_code == 2
    assert "--recipe and --export-dir are required" in capsys.readouterr().err
