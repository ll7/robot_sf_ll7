"""Artifact writer helpers for camera-ready benchmark campaigns."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.camera_ready._util import _sanitize_csv_cell

if TYPE_CHECKING:
    from pathlib import Path


def _escape_markdown_cell(value: Any) -> str:
    """Escape markdown table cell content to prevent row/column injection.

    Returns:
        Escaped single-line markdown-safe cell value.
    """
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("|", "\\|")
    text = text.replace("\n", " ").replace("\r", " ")
    return text


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting and trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_markdown_table(path: Path, rows: list[dict[str, Any]], headers: tuple[str, ...]) -> None:
    """Write a table in Markdown format using explicit header order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        values = [_escape_markdown_cell(row.get(col, "")) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write campaign summary table in CSV format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _sanitize_csv_cell(value) for key, value in row.items()})


def _write_table_artifacts(
    reports_dir: Path,
    base_name: str,
    rows: list[dict[str, Any]],
    *,
    headers: tuple[str, ...],
) -> tuple[Path, Path]:
    """Write CSV and Markdown table artifacts for one table dataset.

    Returns:
        Tuple of generated ``(csv_path, markdown_path)``.
    """
    csv_path = reports_dir / f"{base_name}.csv"
    md_path = reports_dir / f"{base_name}.md"
    csv_rows = [{key: row.get(key, "") for key in headers} for row in rows]
    _write_csv(csv_path, csv_rows)
    _write_markdown_table(md_path, rows, headers=headers)
    return csv_path, md_path
