#!/usr/bin/env python3
"""Validate the active research-lane scientific-state table."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_PATH = Path("docs/context/research_lane_states.md")
REQUIRED_HEADERS = (
    "lane",
    "issue",
    "current_state",
    "last_evidence_source",
    "next_discriminating_experiment",
    "stop_condition",
)
ALLOWED_STATES = {
    "candidate",
    "diagnostic_signal",
    "mechanism_inactive",
    "active_but_irrelevant",
    "slice_local",
    "primary_route_overselected",
    "feasibility_only",
    "blocked_missing_trace",
    "revise",
    "stop",
}
PLACEHOLDER_VALUES = {"", "na", "n/a", "none", "tbd", "todo", "-"}
TABLE_HEADING = "## Active Lane Table"


def _split_row(line: str) -> list[str]:
    """Split a Markdown table row into stripped cells."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table_lines(text: str) -> list[str]:
    """Return the active-lane table lines following the canonical heading."""
    lines = text.splitlines()
    try:
        start = lines.index(TABLE_HEADING) + 1
    except ValueError as exc:
        raise ValueError(f"missing heading: {TABLE_HEADING}") from exc

    table: list[str] = []
    for line in lines[start:]:
        if line.startswith("|"):
            table.append(line)
            continue
        if table and line.strip():
            break
    if len(table) < 3:
        raise ValueError("active lane table must include a header, separator, and at least one row")
    return table


def _strip_code(value: str) -> str:
    """Remove Markdown code ticks around a cell value."""
    return value.strip().strip("`").strip()


def _local_markdown_links(cell: str) -> list[str]:
    """Return local Markdown link targets from a cell."""
    links = re.findall(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]+)?\)", cell)
    return [link for link in links if not re.match(r"[a-z]+://", link)]


def _validate_row(
    *,
    row_line: str,
    line_number: int,
    headers: list[str],
    note_path: Path,
) -> list[str]:
    """Return validation errors for one active-lane table row."""
    errors: list[str] = []
    cells = _split_row(row_line)
    if len(cells) != len(headers):
        return [f"row {line_number}: expected {len(headers)} cells, found {len(cells)}"]

    row = dict(zip(headers, cells, strict=True))
    lane = row.get("lane", f"row {line_number}")
    for header in REQUIRED_HEADERS:
        value = _strip_code(row.get(header, ""))
        if value.lower() in PLACEHOLDER_VALUES:
            errors.append(f"{lane}: {header} is missing or placeholder text")

    state = _strip_code(row.get("current_state", ""))
    if state and state not in ALLOWED_STATES:
        errors.append(f"{lane}: unknown current_state {state!r}")

    for target in _local_markdown_links(row.get("last_evidence_source", "")):
        target_path = (note_path.parent / target).resolve()
        if not target_path.is_file():
            errors.append(f"{lane}: missing linked evidence source {target}")

    return errors


def validate_research_lane_states(path: Path = DEFAULT_PATH) -> list[str]:
    """Return validation errors for the research-lane state table."""
    if not path.is_file():
        return [f"missing research-lane state file: {path}"]

    errors: list[str] = []
    table = _table_lines(path.read_text(encoding="utf-8"))
    headers = _split_row(table[0])
    missing_headers = [header for header in REQUIRED_HEADERS if header not in headers]
    if missing_headers:
        errors.append(f"missing required column(s): {', '.join(missing_headers)}")
        return errors

    rows = table[2:]
    if not rows:
        errors.append("active lane table must include at least one lane row")
        return errors

    for line_number, row_line in enumerate(rows, start=1):
        errors.extend(
            _validate_row(
                row_line=row_line,
                line_number=line_number,
                headers=headers,
                note_path=path,
            )
        )

    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help=f"Research-lane state Markdown file. Default: {DEFAULT_PATH}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate the configured research-lane state file."""
    args = _parser().parse_args(argv)
    try:
        errors = validate_research_lane_states(args.path)
    except ValueError as exc:
        errors = [str(exc)]
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"research lane states ok: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
