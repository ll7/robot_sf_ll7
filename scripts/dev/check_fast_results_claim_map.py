#!/usr/bin/env python3
"""Validate the fast-results claim map as an executable issue queue."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLAIM_MAP = REPO_ROOT / "docs/context/issue_2943_fast_results_claim_map_v0.md"
ISSUE_RE = re.compile(r"(?:#|/issues/)(\d+)\b")
PLACEHOLDERS = {"", "-", "none", "n/a", "na", "tbd", "todo", "missing"}
ALLOWED_STATUSES = {
    "ready",
    "blocked",
    "running",
    "completed",
    "diagnostic-only",
    "do-not-claim",
}
SECTION_HEADINGS = ("p0_now", "p1_after_gate", "parked_blocked")
REQUIRED_COLUMNS = {
    "item",
    "owner issue",
    "status",
    "next command or artifact",
    "evidence gate",
    "durable evidence",
}


@dataclass(frozen=True)
class ClaimMapError:
    """A single claim-map validation error."""

    section: str
    row: int
    field: str
    message: str


@dataclass(frozen=True)
class MarkdownTable:
    """A parsed Markdown table, including empty but well-formed tables."""

    found: bool
    header: tuple[str, ...] = ()
    rows: tuple[dict[str, str], ...] = ()


def _normalize_header(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _clean_cell(value: str) -> str:
    return value.strip().replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")


def _is_placeholder(value: str) -> bool:
    cleaned = _clean_cell(value).strip("`").strip().lower()
    return cleaned in PLACEHOLDERS


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|") or len(stripped) < 2:
        return []
    return [_clean_cell(cell) for cell in stripped[1:-1].split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-+:?", cell.strip()) for cell in cells)


def _extract_table(markdown: str, heading: str) -> MarkdownTable:
    lines = markdown.splitlines()
    heading_re = re.compile(rf"^###\s+{re.escape(heading)}\b")
    start = next((index for index, line in enumerate(lines) if heading_re.match(line)), None)
    if start is None:
        return MarkdownTable(found=False)

    table_lines: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("### "):
            break
        if line.strip().startswith("|"):
            table_lines.append(line)
        elif table_lines and line.strip():
            break

    if len(table_lines) < 2:
        return MarkdownTable(found=False)
    header = [_normalize_header(cell) for cell in _split_markdown_row(table_lines[0])]
    separator = _split_markdown_row(table_lines[1])
    if not _is_separator_row(separator):
        return MarkdownTable(found=False)

    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:
        cells = _split_markdown_row(line)
        if len(cells) != len(header):
            rows.append({"__parse_error__": line})
            continue
        rows.append(dict(zip(header, cells, strict=True)))
    return MarkdownTable(found=True, header=tuple(header), rows=tuple(rows))


def _issue_refs(value: str) -> tuple[str, ...]:
    return tuple(f"#{match}" for match in sorted(set(ISSUE_RE.findall(value)), key=int))


def _next_step_kinds(value: str) -> tuple[str, ...]:
    lowered = value.lower()
    kinds: list[str] = []
    if "command:" in lowered:
        kinds.append("command")
    if "artifact:" in lowered:
        kinds.append("artifact")
    return tuple(kinds)


def _is_local_output_only(value: str) -> bool:
    lowered = value.lower()
    return "output/" in lowered and not any(
        marker in lowered for marker in ("docs/context/evidence", "tracked", "external", "wandb")
    )


def _validate_status(section: str, row_index: int, row: dict[str, str]) -> list[ClaimMapError]:
    status = row["status"].strip("`").strip().lower()
    if status in ALLOWED_STATUSES:
        return []
    return [
        ClaimMapError(
            section=section,
            row=row_index,
            field="status",
            message=f"status must be one of {sorted(ALLOWED_STATUSES)}",
        )
    ]


def _validate_owner_issue(section: str, row_index: int, row: dict[str, str]) -> list[ClaimMapError]:
    if section != "p0_now" or len(_issue_refs(row["owner issue"])) == 1:
        return []
    return [
        ClaimMapError(
            section=section,
            row=row_index,
            field="owner issue",
            message="p0 rows must link exactly one owner issue",
        )
    ]


def _validate_next_step(section: str, row_index: int, row: dict[str, str]) -> list[ClaimMapError]:
    next_step = row["next command or artifact"]
    errors: list[ClaimMapError] = []
    if _is_placeholder(next_step):
        errors.append(
            ClaimMapError(
                section=section,
                row=row_index,
                field="next command or artifact",
                message="next step must not be empty or placeholder",
            )
        )
    if section == "p0_now" and len(_next_step_kinds(next_step)) != 1:
        errors.append(
            ClaimMapError(
                section=section,
                row=row_index,
                field="next command or artifact",
                message="p0 rows must declare exactly one of Command: or Artifact:",
            )
        )
    return errors


def _validate_evidence_gate(
    section: str, row_index: int, row: dict[str, str]
) -> list[ClaimMapError]:
    if not _is_placeholder(row["evidence gate"]):
        return []
    return [
        ClaimMapError(
            section=section,
            row=row_index,
            field="evidence gate",
            message="evidence gate must not be empty or placeholder",
        )
    ]


def _validate_durable_evidence(
    section: str, row_index: int, row: dict[str, str]
) -> list[ClaimMapError]:
    status = row["status"].strip("`").strip().lower()
    durable_evidence = row["durable evidence"]
    if status != "completed" or (
        not _is_placeholder(durable_evidence) and not _is_local_output_only(durable_evidence)
    ):
        return []
    return [
        ClaimMapError(
            section=section,
            row=row_index,
            field="durable evidence",
            message="completed rows must link durable evidence, not local output only",
        )
    ]


def _validate_row(section: str, row_index: int, row: dict[str, str]) -> list[ClaimMapError]:
    errors: list[ClaimMapError] = []
    if "__parse_error__" in row:
        return [
            ClaimMapError(
                section=section,
                row=row_index,
                field="table",
                message="row has a different number of cells than the header",
            )
        ]
    errors.extend(_validate_status(section, row_index, row))
    errors.extend(_validate_owner_issue(section, row_index, row))
    errors.extend(_validate_next_step(section, row_index, row))
    errors.extend(_validate_evidence_gate(section, row_index, row))
    errors.extend(_validate_durable_evidence(section, row_index, row))
    return errors


def validate_claim_map(path: Path = DEFAULT_CLAIM_MAP) -> dict[str, object]:
    """Validate *path* and return a compact machine-readable report."""
    markdown = path.read_text(encoding="utf-8")
    errors: list[ClaimMapError] = []
    sections: dict[str, MarkdownTable] = {}

    for heading in SECTION_HEADINGS:
        table = _extract_table(markdown, heading)
        sections[heading] = table
        if not table.found:
            errors.append(
                ClaimMapError(
                    section=heading,
                    row=0,
                    field="table",
                    message="missing executable priority table",
                )
            )
            continue

        observed_columns = set(table.header)
        missing_columns = REQUIRED_COLUMNS - observed_columns
        for column in sorted(missing_columns):
            errors.append(
                ClaimMapError(
                    section=heading,
                    row=0,
                    field=column,
                    message="required column missing",
                )
            )
        if missing_columns:
            continue

        for row_index, row in enumerate(table.rows, start=1):
            errors.extend(_validate_row(heading, row_index, row))

    return {
        "schema": "fast_results_claim_map_check.v1",
        "path": str(path),
        "ok": not errors,
        "sections": {
            heading: {
                "row_count": len(table.rows),
                "statuses": sorted(
                    {row.get("status", "").strip("`").strip() for row in table.rows}
                ),
            }
            for heading, table in sections.items()
        },
        "errors": [asdict(error) for error in errors],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the claim-map validator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-map", type=Path, default=DEFAULT_CLAIM_MAP)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text summary.")
    args = parser.parse_args(argv)

    report = validate_claim_map(args.claim_map)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["ok"]:
        print(f"OK: {args.claim_map} is an executable claim queue")
    else:
        print(f"FAIL: {args.claim_map} has {len(report['errors'])} claim-map issue(s)")
        for error in report["errors"]:
            print(f"- {error['section']} row {error['row']} {error['field']}: {error['message']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
