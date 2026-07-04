#!/usr/bin/env python3
"""Verify manuscript-asserted implementation numbers against declared source records."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DECLARATIONS = Path("configs/validation/issue_4366_manuscript_asserted_numbers.yaml")
DEFAULT_REPORT = Path("docs/context/evidence/issue_4366_manuscript_asserted_numbers_report.md")
REPORT_SCHEMA_VERSION = "manuscript-asserted-number-verification-report.v1"
DECLARATION_SCHEMA_VERSION = "manuscript-asserted-number-declarations.v1"
MATCH = "match"
MISMATCH = "mismatch"
NOT_VERIFIABLE = "not_verifiable"
BLOCKED = "blocked"
_SELECTOR_RE = re.compile(r"^(?P<key>[^\[]+)\[(?P<field>[^=\]]+)=(?P<value>[^\]]+)\]$")


class VerificationError(ValueError):
    """Raised when declarations are malformed or cannot be checked fail-closed."""


@dataclass(frozen=True)
class VerificationResult:
    """One declaration verification outcome."""

    id: str
    status: str
    manuscript_locator: str
    expected: Any
    actual: Any | None = None
    source_path: str | None = None
    pointer: str | None = None
    reason: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_structured_file(path: Path) -> Any:
    if not path.is_file():
        raise VerificationError(f"source file does not exist: {path}")
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise VerificationError(f"unsupported source file extension for {path}")


def _coerce_selector_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _resolve_pointer(payload: Any, pointer: str) -> Any:
    current = payload
    for segment in pointer.split("."):
        match = _SELECTOR_RE.match(segment)
        if match:
            key = match.group("key")
            field = match.group("field")
            value = _coerce_selector_value(match.group("value"))
            if not isinstance(current, dict) or key not in current:
                raise VerificationError(f"pointer segment missing mapping key: {segment}")
            candidates = current[key]
            if not isinstance(candidates, list):
                raise VerificationError(f"pointer selector target is not a list: {segment}")
            selected = [
                item for item in candidates if isinstance(item, dict) and item.get(field) == value
            ]
            if len(selected) != 1:
                raise VerificationError(
                    f"pointer selector {segment!r} matched {len(selected)} entries"
                )
            current = selected[0]
            continue
        if isinstance(current, dict):
            if segment not in current:
                raise VerificationError(f"pointer segment missing mapping key: {segment}")
            current = current[segment]
            continue
        if isinstance(current, list):
            try:
                current = current[int(segment)]
            except (ValueError, IndexError) as exc:
                raise VerificationError(f"invalid list pointer segment: {segment}") from exc
            continue
        raise VerificationError(f"cannot traverse pointer segment {segment!r}")
    return current


def _values_equal(expected: Any, actual: Any, *, tolerance: float) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return math.isclose(
                float(expected), float(actual), rel_tol=tolerance, abs_tol=tolerance
            )
        except (TypeError, ValueError):
            return False
    return expected == actual


def _verify_entry(
    entry: dict[str, Any], *, repo_root: Path, tolerance: float
) -> VerificationResult:
    entry_id = str(entry.get("id", ""))
    manuscript_locator = str(entry.get("manuscript_locator", ""))
    if not entry_id or not manuscript_locator:
        raise VerificationError("each entry requires id and manuscript_locator")

    expected_status = entry.get("expected_status", MATCH)
    if expected_status == NOT_VERIFIABLE:
        reason = str(entry.get("not_verifiable_reason", "")).strip()
        if not reason:
            raise VerificationError(f"{entry_id}: not_verifiable entries require a reason")
        return VerificationResult(
            id=entry_id,
            status=NOT_VERIFIABLE,
            manuscript_locator=manuscript_locator,
            expected=NOT_VERIFIABLE,
            reason=reason,
        )
    if expected_status != MATCH:
        raise VerificationError(f"{entry_id}: unsupported expected_status {expected_status!r}")
    if "expected" not in entry:
        raise VerificationError(f"{entry_id}: verifiable entries require expected")

    source = entry.get("source")
    if not isinstance(source, dict):
        raise VerificationError(f"{entry_id}: verifiable entries require source mapping")
    source_path = str(source.get("path", ""))
    pointer = str(source.get("pointer", ""))
    if not source_path or not pointer:
        raise VerificationError(f"{entry_id}: source.path and source.pointer are required")
    source_file = repo_root / source_path
    actual = _resolve_pointer(_load_structured_file(source_file), pointer)
    expected = entry["expected"]
    status = MATCH if _values_equal(expected, actual, tolerance=tolerance) else MISMATCH
    reason = None if status == MATCH else "expected value differs from source-of-record value"
    return VerificationResult(
        id=entry_id,
        status=status,
        manuscript_locator=manuscript_locator,
        expected=expected,
        actual=actual,
        source_path=source_path,
        pointer=pointer,
        reason=reason,
    )


def verify_declarations(path: Path, *, repo_root: Path, tolerance: float) -> dict[str, Any]:
    """Verify all declarations and return a JSON-serializable report payload."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise VerificationError("declarations file must be a mapping")
    if payload.get("schema_version") != DECLARATION_SCHEMA_VERSION:
        raise VerificationError(f"unsupported declaration schema: {payload.get('schema_version')}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise VerificationError("declarations require a non-empty entries list")
    results = [_verify_entry(entry, repo_root=repo_root, tolerance=tolerance) for entry in entries]
    counts = {
        MATCH: sum(result.status == MATCH for result in results),
        MISMATCH: sum(result.status == MISMATCH for result in results),
        NOT_VERIFIABLE: sum(result.status == NOT_VERIFIABLE for result in results),
        BLOCKED: 0,
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "issue": int(payload["issue"]),
        "claim_boundary": payload["claim_boundary"],
        "selection_assumption": payload["selection_assumption"],
        "declarations_path": str(path),
        "status_counts": counts,
        "overall_status": "pass" if counts[MISMATCH] == 0 and counts[BLOCKED] == 0 else "fail",
        "results": [result.__dict__ for result in results],
    }


def _format_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return "`" + json.dumps(value, sort_keys=True) + "`"
    return "`" + str(value) + "`"


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    """Write the committed AI-generated Markdown verification report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = report["status_counts"]
    lines = [
        "<!-- AI-GENERATED (robot_sf#4366, 2026-07-04) - NEEDS-REVIEW -->",
        "# Issue 4366 Manuscript-Asserted Number Verification",
        "",
        f"Schema: `{report['schema_version']}`",
        "",
        "Claim boundary: verification aid only. No manuscript edits, no claim changes, no full "
        "benchmark campaign, and no Slurm/GPU submission.",
        "",
        f"Declarations: `{report['declarations_path']}`",
        "",
        f"Selection assumption: {report['selection_assumption']}",
        "",
        "## Summary",
        "",
        f"- Overall status: `{report['overall_status']}`",
        f"- Matches: `{counts[MATCH]}`",
        f"- Mismatches: `{counts[MISMATCH]}`",
        f"- Not verifiable: `{counts[NOT_VERIFIABLE]}`",
        f"- Blocked: `{counts[BLOCKED]}`",
        "",
        "## Results",
        "",
        "| id | status | manuscript locator | expected | actual | source | reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in report["results"]:
        source = ""
        if result.get("source_path"):
            source = f"`{result['source_path']}#{result['pointer']}`"
        reason = result.get("reason") or ""
        actual = "" if result.get("actual") is None else _format_value(result["actual"])
        lines.append(
            "| {id} | `{status}` | {locator} | {expected} | {actual} | {source} | {reason} |".format(
                id=result["id"],
                status=result["status"],
                locator=_escape_table_cell(str(result["manuscript_locator"])),
                expected=_escape_table_cell(_format_value(result["expected"])),
                actual=_escape_table_cell(actual),
                source=_escape_table_cell(source),
                reason=_escape_table_cell(str(reason)),
            )
        )
    lines.extend(["", "<!-- /AI-GENERATED -->", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the command-line verifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declarations", type=Path, default=DEFAULT_DECLARATIONS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    try:
        report = verify_declarations(
            args.declarations, repo_root=repo_root, tolerance=args.tolerance
        )
        write_markdown_report(report, args.report)
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except VerificationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        "verified {total} declarations: {matches} match, {mismatches} mismatch, "
        "{not_verifiable} not_verifiable".format(
            total=len(report["results"]),
            matches=report["status_counts"][MATCH],
            mismatches=report["status_counts"][MISMATCH],
            not_verifiable=report["status_counts"][NOT_VERIFIABLE],
        )
    )
    return 0 if report["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
