"""Canonical benchmark table exporters with provenance sidecars."""
# ruff: noqa: DOC201

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "benchmark_canonical_table_export.v1"


@dataclass(frozen=True)
class CanonicalTableSpec:
    """Column contract for one canonical benchmark table."""

    table_id: str
    columns: tuple[str, ...]
    sort_by: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class CanonicalTableExportResult:
    """Paths and metadata produced by one table export."""

    table_id: str
    output_paths: dict[str, Path]
    metadata_path: Path
    row_count: int


TABLE_SPECS: dict[str, CanonicalTableSpec] = {
    "planner_outcome_summary": CanonicalTableSpec(
        table_id="planner_outcome_summary",
        columns=(
            "planner_key",
            "execution_mode",
            "success_mean",
            "collisions_mean",
            "near_misses_mean",
            "snqi_mean",
            "runtime_mean_sec",
            "row_status",
        ),
        sort_by=("planner_key", "execution_mode", "row_status"),
        description="Planner-level outcome summary with row status preserved.",
    ),
    "scenario_family_summary": CanonicalTableSpec(
        table_id="scenario_family_summary",
        columns=(
            "scenario_family",
            "planner_key",
            "success_mean",
            "collisions_mean",
            "near_misses_mean",
            "timeout_or_low_progress_mean",
            "seed_count",
        ),
        sort_by=("scenario_family", "planner_key"),
        description="Scenario-family outcome summary by planner.",
    ),
    "seed_variability_summary": CanonicalTableSpec(
        table_id="seed_variability_summary",
        columns=(
            "planner_key",
            "metric",
            "mean",
            "interval_low",
            "interval_high",
            "rank_stability",
            "warning",
        ),
        sort_by=("planner_key", "metric"),
        description="Seed variability summary with interval and stability fields.",
    ),
    "execution_mode": CanonicalTableSpec(
        table_id="execution_mode",
        columns=(
            "planner_key",
            "availability_status",
            "execution_mode",
            "readiness_status",
            "exclusion_reason",
        ),
        sort_by=("planner_key", "execution_mode", "readiness_status"),
        description="Planner availability and execution-mode table.",
    ),
    "artifact_source": CanonicalTableSpec(
        table_id="artifact_source",
        columns=("artifact_id", "source_files", "checksums", "command", "generated_outputs"),
        sort_by=("artifact_id",),
        description="Artifact source/provenance table.",
    ),
}


def load_rows_json(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON list of row mappings."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("canonical table row input must be a JSON list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise TypeError(f"canonical table row {index} must be a mapping")
        rows.append(dict(row))
    return rows


def export_canonical_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    table_id: str,
    output_dir: str | Path,
    formats: Sequence[str] = ("csv", "md", "tex"),
    precision: int = 4,
    source_paths: Sequence[str | Path] = (),
    command: str | None = None,
) -> CanonicalTableExportResult:
    """Write a canonical benchmark table in requested formats plus metadata."""
    if table_id not in TABLE_SPECS:
        known = ", ".join(sorted(TABLE_SPECS))
        raise ValueError(f"unknown canonical table id {table_id!r}; expected one of: {known}")
    normalized_formats = _normalize_formats(formats)
    spec = TABLE_SPECS[table_id]
    normalized_rows = _normalize_rows(rows, spec=spec, precision=precision)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}
    for fmt in normalized_formats:
        output_path = out_dir / f"{table_id}.{fmt}"
        output_path.write_text(_format_rows(normalized_rows, spec.columns, fmt), encoding="utf-8")
        output_paths[fmt] = output_path

    metadata_path = out_dir / f"{table_id}.metadata.json"
    metadata = _build_metadata(
        spec=spec,
        rows=normalized_rows,
        output_paths=output_paths,
        precision=precision,
        source_paths=source_paths,
        command=command,
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return CanonicalTableExportResult(
        table_id=table_id,
        output_paths=output_paths,
        metadata_path=metadata_path,
        row_count=len(normalized_rows),
    )


def _normalize_formats(formats: Sequence[str]) -> tuple[str, ...]:
    """Validate and normalize requested output formats."""
    valid = {"csv", "md", "tex"}
    normalized = tuple(dict.fromkeys(fmt.strip().lower() for fmt in formats if fmt.strip()))
    if not normalized:
        raise ValueError("at least one canonical table format is required")
    unknown = sorted(set(normalized) - valid)
    if unknown:
        raise ValueError(f"unknown canonical table formats: {', '.join(unknown)}")
    return normalized


def _normalize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: CanonicalTableSpec,
    precision: int,
) -> list[dict[str, str]]:
    """Project rows onto spec columns, format values, and sort deterministically."""
    normalized: list[dict[str, str]] = []
    for row in rows:
        normalized.append(
            {column: _format_cell(row.get(column), precision=precision) for column in spec.columns}
        )
    normalized.sort(key=lambda row: tuple(row[column] for column in spec.sort_by))
    return normalized


def _format_cell(value: Any, *, precision: int) -> str:
    """Format one table cell without dropping status strings such as degraded/fallback."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    if isinstance(value, list | tuple):
        return ";".join(_format_cell(item, precision=precision) for item in value)
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _format_rows(rows: Sequence[Mapping[str, str]], columns: Sequence[str], fmt: str) -> str:
    """Format normalized rows as CSV, Markdown, or LaTeX."""
    if fmt == "csv":
        return _format_csv(rows, columns)
    if fmt == "md":
        return _format_markdown(rows, columns)
    if fmt == "tex":
        return _format_tex(rows, columns)
    raise ValueError(f"unsupported canonical table format: {fmt}")


def _format_csv(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> str:
    """Format rows as CSV."""
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _format_markdown(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> str:
    """Format rows as a GitHub-flavored Markdown table."""
    lines = [
        "| " + " | ".join(_markdown_escape(column) for column in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_escape(row[column]) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _format_tex(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> str:
    """Format rows as a LaTeX booktabs fragment."""
    lines = [
        "% Auto-generated by robot_sf.benchmark.canonical_table_export",
        "\\begin{tabular}{" + ("l" * len(columns)) + "}",
        "\\toprule",
        " & ".join(_latex_escape(column) for column in columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(row[column]) for column in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _markdown_escape(text: str) -> str:
    """Escape Markdown table separators and normalize whitespace."""
    return text.replace("\\", "\\\\").replace("|", r"\|").replace("\n", "<br>")


def _latex_escape(text: str) -> str:
    """Escape LaTeX-sensitive characters in text cells."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _build_metadata(
    *,
    spec: CanonicalTableSpec,
    rows: Sequence[Mapping[str, str]],
    output_paths: Mapping[str, Path],
    precision: int,
    source_paths: Sequence[str | Path],
    command: str | None,
) -> dict[str, Any]:
    """Build provenance metadata for a canonical table export."""
    return {
        "schema_version": SCHEMA_VERSION,
        "table_id": spec.table_id,
        "description": spec.description,
        "columns": list(spec.columns),
        "sort_by": list(spec.sort_by),
        "row_count": len(rows),
        "formats": sorted(output_paths),
        "outputs": {fmt: path.as_posix() for fmt, path in sorted(output_paths.items())},
        "precision": precision,
        "source_files": [_source_file_metadata(path) for path in source_paths],
        "command": command,
        "git_commit": _git_commit(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }


def _source_file_metadata(path: str | Path) -> dict[str, Any]:
    """Return checksum metadata for one source file path."""
    source_path = Path(path)
    payload: dict[str, Any] = {"path": source_path.as_posix(), "exists": source_path.is_file()}
    if source_path.is_file():
        payload["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return payload


def _git_commit() -> str | None:
    """Return current Git commit when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - environment defensive
        return None
