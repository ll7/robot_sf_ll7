"""Parquet analytics export for benchmark episode JSONL records."""

# ruff: noqa: DOC201

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.benchmark.errors import EpisodeRecordInputError

try:  # Optional analytics dependency; validated when export is invoked.
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    pa = None
    pq = None

EXPORT_SCHEMA_VERSION = "benchmark_parquet_export.v1"

_TABLE_FILENAMES = {
    "episodes": "episodes.parquet",
    "metrics": "metrics.parquet",
    "scenario_params": "scenario_params.parquet",
    "algorithm_metadata": "algorithm_metadata.parquet",
}


@dataclass(frozen=True)
class ParquetExportResult:
    """Summary of files written by a benchmark Parquet export."""

    output_dir: Path
    record_count: int
    table_paths: dict[str, Path]
    metadata_path: Path
    duckdb_examples_path: Path


def export_episodes_jsonl_to_parquet(
    input_paths: Sequence[str | Path] | str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> ParquetExportResult:
    """Convert benchmark episode JSONL records into Parquet analytics tables.

    JSONL remains the canonical source artifact. The exported tables are derived
    views intended for SQL analytics, campaign comparison, and failure mining.

    Args:
        input_paths: One or more benchmark episode JSONL files.
        output_dir: Directory that receives the Parquet tables and metadata.
        overwrite: Replace existing export files when True.

    Returns:
        Summary of the generated files and row counts.

    Raises:
        FileExistsError: If export files already exist and overwrite is False.
        RuntimeError: If the optional PyArrow dependency is unavailable.
    """
    pa, pq = _load_pyarrow()
    paths = _normalize_paths(input_paths)
    out_dir = Path(output_dir)
    table_paths = {name: out_dir / filename for name, filename in _TABLE_FILENAMES.items()}
    metadata_path = out_dir / "metadata.json"
    duckdb_examples_path = out_dir / "duckdb_examples.sql"
    _ensure_can_write([*table_paths.values(), metadata_path, duckdb_examples_path], overwrite)

    records = _read_jsonl_files(paths)
    rows = _build_rows(records)

    out_dir.mkdir(parents=True, exist_ok=True)
    schemas = _schemas(pa)
    for table_name, table_rows in rows.items():
        table = pa.Table.from_pylist(table_rows, schema=schemas[table_name])
        pq.write_table(table, table_paths[table_name])

    metadata = _build_metadata(
        paths=paths,
        record_count=len(records),
        rows=rows,
        table_paths=table_paths,
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    duckdb_examples_path.write_text(_duckdb_examples_sql(), encoding="utf-8")

    return ParquetExportResult(
        output_dir=out_dir,
        record_count=len(records),
        table_paths=table_paths,
        metadata_path=metadata_path,
        duckdb_examples_path=duckdb_examples_path,
    )


def _load_pyarrow() -> tuple[Any, Any]:
    """Load PyArrow modules only when the export path is used."""
    if pa is None or pq is None:  # pragma: no cover - environment dependent
        msg = (
            "Parquet export requires optional analytics dependencies. "
            "Install them with `uv sync --extra analytics` or `uv sync --all-extras`."
        )
        raise RuntimeError(msg)
    return pa, pq


def _normalize_paths(input_paths: Sequence[str | Path] | str | Path) -> list[Path]:
    """Normalize one or more input paths."""
    if isinstance(input_paths, str | Path):
        return [Path(input_paths)]
    return [Path(path) for path in input_paths]


def _ensure_can_write(paths: Sequence[Path], overwrite: bool) -> None:
    """Fail before partial writes when export files already exist."""
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Parquet export output already exists: {names}")


def _schemas(pa: Any) -> dict[str, Any]:
    """Return fixed PyArrow schemas for all exported tables."""
    typed_value_fields = [
        ("value_number", pa.float64()),
        ("value_bool", pa.bool_()),
        ("value_text", pa.string()),
        ("value_json", pa.string()),
    ]
    return {
        "episodes": pa.schema(
            [
                ("episode_id", pa.string()),
                ("scenario_id", pa.string()),
                ("seed", pa.int64()),
                ("started_at_utc", pa.string()),
                ("finished_at_utc", pa.string()),
                ("total_runtime_sec", pa.float64()),
                ("algo", pa.string()),
                ("scenario_family", pa.string()),
                ("termination_reason", pa.string()),
                ("version", pa.string()),
                ("outcome_json", pa.string()),
                ("integrity_json", pa.string()),
                ("record_json_sha256", pa.string()),
            ]
        ),
        "metrics": pa.schema(
            [
                ("episode_id", pa.string()),
                ("metric_path", pa.string()),
                *typed_value_fields,
            ]
        ),
        "scenario_params": pa.schema(
            [
                ("episode_id", pa.string()),
                ("param_path", pa.string()),
                *typed_value_fields,
            ]
        ),
        "algorithm_metadata": pa.schema(
            [
                ("episode_id", pa.string()),
                ("metadata_path", pa.string()),
                *typed_value_fields,
            ]
        ),
    }


def _build_rows(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build normalized table rows from benchmark episode records."""
    episode_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    scenario_param_rows: list[dict[str, Any]] = []
    algorithm_metadata_rows: list[dict[str, Any]] = []

    for record in records:
        episode_id = str(record.get("episode_id", ""))
        episode_rows.append(_episode_row(record, episode_id))
        metric_rows.extend(
            _key_value_rows(
                episode_id=episode_id,
                source=record.get("metrics"),
                path_column="metric_path",
            )
        )
        scenario_param_rows.extend(
            _key_value_rows(
                episode_id=episode_id,
                source=record.get("scenario_params"),
                path_column="param_path",
            )
        )
        algorithm_metadata_rows.extend(
            _key_value_rows(
                episode_id=episode_id,
                source=record.get("algorithm_metadata"),
                path_column="metadata_path",
            )
        )

    return {
        "episodes": episode_rows,
        "metrics": metric_rows,
        "scenario_params": scenario_param_rows,
        "algorithm_metadata": algorithm_metadata_rows,
    }


def _episode_row(record: Mapping[str, Any], episode_id: str) -> dict[str, Any]:
    """Build the fixed top-level episode row."""
    return {
        "episode_id": episode_id,
        "scenario_id": _string_or_none(record.get("scenario_id")),
        "seed": _int_or_none(record.get("seed")),
        "started_at_utc": _resolve_started_at_utc(record),
        "finished_at_utc": _resolve_finished_at_utc(record),
        "total_runtime_sec": _resolve_total_runtime_sec(record),
        "algo": _resolve_algo(record),
        "scenario_family": _resolve_scenario_family(record),
        "termination_reason": _string_or_none(record.get("termination_reason")),
        "version": _string_or_none(record.get("version")),
        "outcome_json": _json_or_none(record.get("outcome")),
        "integrity_json": _json_or_none(record.get("integrity")),
        "record_json_sha256": hashlib.sha256(_json_dumps(record).encode("utf-8")).hexdigest(),
    }


def _key_value_rows(
    *,
    episode_id: str,
    source: Any,
    path_column: str,
) -> list[dict[str, Any]]:
    """Convert a nested mapping into long-form typed key/value rows."""
    if not isinstance(source, Mapping):
        return []

    rows: list[dict[str, Any]] = []
    for path, value in _iter_leaf_values(source):
        value_columns = _typed_value_columns(value)
        rows.append({"episode_id": episode_id, path_column: path, **value_columns})
    return rows


def _iter_leaf_values(source: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested mappings into dotted leaf paths."""
    rows: list[tuple[str, Any]] = []
    for key in sorted(source):
        value = source[key]
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and value:
            rows.extend(_iter_leaf_values(value, path))
        else:
            rows.append((path, value))
    return rows


def _typed_value_columns(value: Any) -> dict[str, Any]:
    """Represent a Python value across stable typed Parquet columns."""
    value_number = None
    value_bool = None
    value_text = None
    value_json = None
    if isinstance(value, bool):
        value_bool = value
    elif isinstance(value, int | float):
        value_number = float(value)
    elif isinstance(value, str):
        value_text = value
    elif value is not None:
        value_json = _json_dumps(value)
    return {
        "value_number": value_number,
        "value_bool": value_bool,
        "value_text": value_text,
        "value_json": value_json,
    }


def _resolve_algo(record: Mapping[str, Any]) -> str | None:
    """Resolve the planner/algorithm identifier with benchmark metadata fallbacks."""
    for value in (
        _nested_value(record, "scenario_params.algo"),
        record.get("algo"),
        _nested_value(record, "algorithm_metadata.algorithm"),
        _nested_value(record, "algorithm_metadata.canonical_algorithm"),
    ):
        text = _string_or_none(value)
        if text:
            return text
    return None


def _resolve_scenario_family(record: Mapping[str, Any]) -> str | None:
    """Resolve a scenario-family key suitable for grouped analytics."""
    for value in (
        _nested_value(record, "scenario_params.scenario_family"),
        _nested_value(record, "scenario_params.family"),
        record.get("scenario_family"),
    ):
        text = _string_or_none(value)
        if text:
            return text
    scenario_id = _string_or_none(record.get("scenario_id"))
    if scenario_id and "_" in scenario_id:
        return scenario_id.split("_", maxsplit=1)[0]
    return scenario_id


def _nested_value(source: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted path from a nested mapping."""
    current: Any = source
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _int_or_none(value: Any) -> int | None:
    """Coerce an integer-like value when possible."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _float_or_none(value: Any) -> float | None:
    """Coerce a numeric value when possible."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _string_or_none(value: Any) -> str | None:
    """Coerce non-empty string-like values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _json_or_none(value: Any) -> str | None:
    """Serialize structured values for fixed top-level episode columns."""
    if value is None:
        return None
    return _json_dumps(value)


def _json_dumps(value: Any) -> str:
    """Serialize JSON deterministically for hashes and stored JSON columns."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _resolve_started_at_utc(record: Mapping[str, Any]) -> str | None:
    """Resolve episode start time from canonical or legacy provenance fields."""
    return _string_or_none(record.get("started_at_utc")) or _string_or_none(
        _nested_value(record, "timestamps.start")
    )


def _resolve_finished_at_utc(record: Mapping[str, Any]) -> str | None:
    """Resolve episode finish time from canonical or legacy provenance fields."""
    return _string_or_none(record.get("finished_at_utc")) or _string_or_none(
        _nested_value(record, "timestamps.end")
    )


def _resolve_total_runtime_sec(record: Mapping[str, Any]) -> float | None:
    """Resolve episode runtime from known benchmark provenance fields."""
    for value in (
        record.get("total_runtime_sec"),
        record.get("runtime_sec"),
        record.get("wall_time_sec"),
        record.get("total_runtime"),
    ):
        number = _float_or_none(value)
        if number is not None:
            return number
    return None


def _read_jsonl_files(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Read benchmark episode JSONL files, failing closed on malformed source data."""
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark episode JSONL input is not a file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    # Surface the canonical typed input error (a ValueError subclass, so
                    # backward-compatible) so the export-parquet CLI boundary reports it as a
                    # documented non-zero exit instead of a raw traceback. See issue #4988.
                    raise EpisodeRecordInputError(
                        f"{path}:{line_number} is not valid JSON: {exc.msg}"
                    ) from exc
                records.append(record)
    return records


def _build_metadata(
    *,
    paths: Sequence[Path],
    record_count: int,
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    table_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Build export metadata and provenance."""
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "jsonl_is_source_of_truth": True,
        "record_count": record_count,
        "source_files": [
            {
                "path": str(path),
                "sha256": _path_sha256(path) if path.is_file() else None,
            }
            for path in paths
        ],
        "tables": {
            table_name: {
                "file": table_paths[table_name].name,
                "rows": len(table_rows),
            }
            for table_name, table_rows in rows.items()
        },
    }


def _path_sha256(path: Path) -> str:
    """Return a SHA-256 digest for an input file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duckdb_examples_sql() -> str:
    """Return example DuckDB queries for the exported table layout."""
    return """-- Robot SF benchmark Parquet analytics examples.
-- Run from the export directory, or replace the file paths with absolute paths.

-- Grouped safety metrics by planner and scenario family.
WITH metric_values AS (
    SELECT
        e.algo,
        e.scenario_family,
        m.metric_path,
        m.value_number
    FROM read_parquet('episodes.parquet') AS e
    JOIN read_parquet('metrics.parquet') AS m USING (episode_id)
)
SELECT
    algo,
    scenario_family,
    AVG(CASE WHEN metric_path = 'min_ttc' THEN value_number END) AS avg_min_ttc,
    AVG(CASE WHEN metric_path = 'clearance' THEN value_number END) AS avg_clearance,
    SUM(CASE WHEN metric_path = 'collisions' THEN value_number ELSE 0 END) AS collisions
FROM metric_values
GROUP BY algo, scenario_family
ORDER BY algo, scenario_family;

-- Failure and near-miss mining.
SELECT
    e.episode_id,
    e.algo,
    e.scenario_id,
    e.scenario_family,
    e.termination_reason,
    m.value_number AS min_ttc
FROM read_parquet('episodes.parquet') AS e
LEFT JOIN read_parquet('metrics.parquet') AS m
    ON e.episode_id = m.episode_id AND m.metric_path = 'min_ttc'
WHERE e.termination_reason IN ('collision', 'deadlock', 'timeout')
   OR m.value_number < 0.5
ORDER BY e.algo, min_ttc NULLS LAST;
"""
