#!/usr/bin/env python3
"""Run curated DuckDB SQL recipes against benchmark Parquet exports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import yaml

RECIPE_SCHEMA_VERSION = "benchmark_sql_recipes.v1"
RECIPE_INVENTORY_SCHEMA = "benchmark_sql_recipe_inventory.v1"
DEFAULT_RECIPE_ROOT = Path(__file__).with_name("benchmark_sql_recipes")
TABLE_FILENAMES = {
    "episodes": "episodes.parquet",
    "metrics": "metrics.parquet",
    "scenario_params": "scenario_params.parquet",
    "algorithm_metadata": "algorithm_metadata.parquet",
}


class RecipeValidationError(ValueError):
    """Raised when recipe inputs do not satisfy the declared contract."""


@dataclass(frozen=True, slots=True)
class SqlRecipe:
    """One SQL recipe declared in the recipe manifest."""

    recipe_id: str
    title: str
    sql_file: Path
    required_tables: tuple[str, ...]
    required_columns: dict[str, tuple[str, ...]]
    output_columns: tuple[str, ...]
    caveats: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RecipeManifest:
    """Loaded SQL recipe manifest."""

    schema_version: str
    recipes: tuple[SqlRecipe, ...]

    def recipe(self, recipe_id: str) -> SqlRecipe:
        """Return a recipe by stable ID."""

        for recipe in self.recipes:
            if recipe.recipe_id == recipe_id:
                return recipe
        available = ", ".join(recipe.recipe_id for recipe in self.recipes)
        raise RecipeValidationError(f"unknown recipe_id '{recipe_id}'; available: {available}")


@dataclass(frozen=True, slots=True)
class RecipeRunResult:
    """Result metadata for a recipe execution."""

    recipe_id: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    row_count: int
    output_csv: Path | None = None
    output_markdown: Path | None = None


def load_recipe_manifest(recipe_root: Path = DEFAULT_RECIPE_ROOT) -> RecipeManifest:
    """Load the curated SQL recipe manifest."""

    manifest_path = recipe_root / "manifest.yaml"
    if not manifest_path.is_file():
        raise RecipeValidationError(f"Recipe manifest file not found: {manifest_path}")
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RecipeValidationError(f"Failed to parse manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RecipeValidationError(f"{manifest_path}: expected mapping payload")
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != RECIPE_SCHEMA_VERSION:
        raise RecipeValidationError(
            f"{manifest_path}: expected schema_version {RECIPE_SCHEMA_VERSION}, "
            f"got {schema_version!r}"
        )
    recipe_payloads = payload.get("recipes")
    if not isinstance(recipe_payloads, list):
        raise RecipeValidationError(f"{manifest_path}: recipes must be a list")
    recipes = tuple(_recipe_from_payload(item, recipe_root=recipe_root) for item in recipe_payloads)
    return RecipeManifest(schema_version=schema_version, recipes=recipes)


def inventory_recipes(recipe_root: Path = DEFAULT_RECIPE_ROOT) -> dict[str, Any]:
    """Return a machine-readable inventory of recipe entry points and their schema contract.

    This makes the versioned recipe/schema contract discoverable and auditable without
    executing any SQL, so schema drift can be reviewed against the declared inputs and
    outputs of every recipe.

    Returns:
        dict[str, Any]: Inventory payload tagged with ``RECIPE_INVENTORY_SCHEMA``.
    """

    manifest = load_recipe_manifest(recipe_root)
    return {
        "schema": RECIPE_INVENTORY_SCHEMA,
        "recipe_schema_version": manifest.schema_version,
        "recipe_root": recipe_root.as_posix(),
        "recipe_count": len(manifest.recipes),
        "recipes": [
            {
                "recipe_id": recipe.recipe_id,
                "title": recipe.title,
                "sql_file": recipe.sql_file.as_posix(),
                "required_tables": list(recipe.required_tables),
                "required_columns": {
                    table: list(columns) for table, columns in recipe.required_columns.items()
                },
                "output_columns": list(recipe.output_columns),
                "caveats": list(recipe.caveats),
            }
            for recipe in manifest.recipes
        ],
    }


def run_recipe(
    recipe_id: str,
    *,
    export_dir: Path,
    output_csv: Path | None = None,
    output_markdown: Path | None = None,
    recipe_root: Path = DEFAULT_RECIPE_ROOT,
) -> RecipeRunResult:
    """Validate inputs, execute a recipe, and optionally write tabular outputs."""

    manifest = load_recipe_manifest(recipe_root)
    recipe = manifest.recipe(recipe_id)
    table_paths = _validate_recipe_inputs(recipe, export_dir=export_dir)
    if not recipe.sql_file.is_file():
        raise RecipeValidationError(f"SQL recipe file not found: {recipe.sql_file}")
    try:
        sql = recipe.sql_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecipeValidationError(f"Failed to read SQL recipe {recipe.sql_file}: {exc}") from exc
    for table, path in table_paths.items():
        sql = sql.replace(f"{{{table}}}", _read_parquet_sql(path))
    try:
        with duckdb.connect(database=":memory:") as connection:
            cursor = connection.execute(sql)
            rows = tuple(tuple(row) for row in cursor.fetchall())
            columns = tuple(str(column[0]) for column in cursor.description or ())
    except Exception as exc:
        raise RecipeValidationError(
            f"SQL execution failed for recipe '{recipe_id}': {exc}"
        ) from exc
    _validate_output_columns(recipe, columns)
    if output_csv is not None:
        _write_csv(output_csv, columns, rows)
    if output_markdown is not None:
        _write_markdown(output_markdown, columns, rows, recipe=recipe)
    return RecipeRunResult(
        recipe_id=recipe.recipe_id,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        output_csv=output_csv,
        output_markdown=output_markdown,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the recipe runner argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        default=None,
        help="Stable recipe ID to run. Required unless --list is given.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_recipes",
        help="Inventory available recipes and their schema contract as JSON, then exit.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Directory containing benchmark Parquet export tables. Required unless --list.",
    )
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional Markdown table output path.",
    )
    parser.add_argument(
        "--recipe-root",
        type=Path,
        default=DEFAULT_RECIPE_ROOT,
        help="Recipe manifest directory.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON run summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one SQL recipe and return a shell-friendly exit code."""

    args = build_arg_parser().parse_args(argv)
    if args.list_recipes:
        try:
            inventory = inventory_recipes(args.recipe_root)
        except RecipeValidationError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
        sys.stdout.write(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
        return 0
    if args.recipe is None or args.export_dir is None:
        sys.stderr.write("--recipe and --export-dir are required unless --list is given\n")
        return 2
    try:
        result = run_recipe(
            args.recipe,
            export_dir=args.export_dir,
            output_csv=args.output_csv,
            output_markdown=args.output_markdown,
            recipe_root=args.recipe_root,
        )
    except RecipeValidationError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    if args.json:
        sys.stdout.write(
            json.dumps(
                {
                    "schema": "benchmark_sql_recipe_run.v1",
                    "recipe_id": result.recipe_id,
                    "row_count": result.row_count,
                    "columns": list(result.columns),
                    "output_csv": str(result.output_csv) if result.output_csv else None,
                    "output_markdown": (
                        str(result.output_markdown) if result.output_markdown else None
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    else:
        sys.stdout.write(f"recipe {result.recipe_id} returned {result.row_count} rows\n")
    return 0


def _recipe_from_payload(payload: dict[str, Any], *, recipe_root: Path) -> SqlRecipe:
    """Build one recipe from manifest data."""

    if not isinstance(payload, dict):
        raise RecipeValidationError("Manifest recipe entries must be mappings")
    try:
        recipe_id = str(payload["recipe_id"])
        required_columns = {
            str(table): tuple(str(column) for column in columns)
            for table, columns in dict(payload.get("required_columns") or {}).items()
        }
        return SqlRecipe(
            recipe_id=recipe_id,
            title=str(payload["title"]),
            sql_file=recipe_root / str(payload["sql_file"]),
            required_tables=tuple(str(table) for table in payload["required_tables"]),
            required_columns=required_columns,
            output_columns=tuple(str(column) for column in payload["output_columns"]),
            caveats=tuple(str(caveat) for caveat in payload.get("caveats") or ()),
        )
    except KeyError as exc:
        field = str(exc.args[0])
        raise RecipeValidationError(f"Manifest recipe is missing required field: {field}") from exc
    except (TypeError, ValueError) as exc:
        raise RecipeValidationError(f"Manifest recipe has invalid structure: {exc}") from exc


def _validate_recipe_inputs(recipe: SqlRecipe, *, export_dir: Path) -> dict[str, Path]:
    """Validate required Parquet files and declared columns for one recipe."""

    table_paths: dict[str, Path] = {}
    for table_name in recipe.required_tables:
        filename = TABLE_FILENAMES.get(table_name)
        if filename is None:
            raise RecipeValidationError(
                f"{recipe.recipe_id}: unknown required table '{table_name}'"
            )
        table_path = export_dir / filename
        if not table_path.is_file():
            raise RecipeValidationError(
                f"{recipe.recipe_id}: required table file is missing: {table_path}"
            )
        table_paths[table_name] = table_path
        _validate_columns(recipe, table_name=table_name, table_path=table_path)
    return table_paths


def _validate_columns(recipe: SqlRecipe, *, table_name: str, table_path: Path) -> None:
    """Validate declared table columns before running SQL."""

    required_columns = set(recipe.required_columns.get(table_name, ()))
    if not required_columns:
        return
    try:
        with duckdb.connect(database=":memory:") as connection:
            cursor = connection.execute(f"DESCRIBE SELECT * FROM {_read_parquet_sql(table_path)}")
            columns = {str(row[0]) for row in cursor.fetchall()}
    except Exception as exc:
        raise RecipeValidationError(f"Failed to read schema from {table_path}: {exc}") from exc
    missing = sorted(required_columns - columns)
    if missing:
        missing_text = ", ".join(f"{table_name}.{column}" for column in missing)
        raise RecipeValidationError(f"{recipe.recipe_id}: missing required columns: {missing_text}")


def _validate_output_columns(recipe: SqlRecipe, columns: tuple[str, ...]) -> None:
    """Validate the SQL result exposes the documented output columns."""

    missing = sorted(set(recipe.output_columns) - set(columns))
    if missing:
        raise RecipeValidationError(
            f"{recipe.recipe_id}: SQL output is missing columns: {', '.join(missing)}"
        )


def _read_parquet_sql(path: Path) -> str:
    """Return a DuckDB read_parquet expression for a path."""

    escaped = path.as_posix().replace("'", "''")
    return f"read_parquet('{escaped}')"


def _write_csv(path: Path, columns: tuple[str, ...], rows: tuple[tuple[Any, ...], ...]) -> None:
    """Write recipe rows as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    columns: tuple[str, ...],
    rows: tuple[tuple[Any, ...], ...],
    *,
    recipe: SqlRecipe,
) -> None:
    """Write recipe rows as a compact Markdown table with caveats."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {recipe.title}", ""]
    if rows:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in rows:
            lines.append("| " + " | ".join(_markdown_cell(value) for value in row) + " |")
    else:
        lines.append("_No rows returned._")
    if recipe.caveats:
        lines.extend(["", "## Caveats", ""])
        lines.extend(f"- {caveat}" for caveat in recipe.caveats)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_cell(value: Any) -> str:
    """Render one Markdown table cell."""

    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
