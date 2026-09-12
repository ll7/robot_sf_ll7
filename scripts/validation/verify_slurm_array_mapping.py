#!/usr/bin/env python3
"""Deterministic pre-submission verifier for SLURM array-index to campaign-row mapping (#8854).

Verifies one-to-one and onto (bijective) mapping between an immutable expected-row ledger
and a scheduled SLURM array or scalar launcher configuration without submitting jobs.
Detects gaps, duplicates, off-by-one bounds, zero/one-based confusion, inconsistent chunk tails,
shard reorder, retry overlap, resume conflict, integer overflow, invalid concurrency, and output
path collisions.

CLI:
    uv run python scripts/validation/verify_slurm_array_mapping.py --check \\
        --ledger <ledger.json> --launcher <launcher.json> [--format json|csv] [--output <path>]

Exit codes:
    0: verified (exact valid bijective mapping)
    2: blocked (mapping error, gaps, duplicates, collisions, etc.)
    3: malformed (unreadable file, invalid JSON, schema mismatch, integer overflow)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

RECEIPT_SCHEMA = "robot_sf.slurm_array_mapping_receipt.v1"
EXPECTED_LEDGER_SCHEMA = "expected_row_ledger.v1"
LAUNCHER_SCHEMA = "slurm_array_launcher.v1"

CLAIM_BOUNDARY = (
    "Operational pre-submission scheduler verification only: valid array mapping does not "
    "guarantee Slurm cluster health, job execution success, resource availability, or scientific "
    "result validity."
)

EXIT_VERIFIED = 0
EXIT_BLOCKED = 2
EXIT_MALFORMED = 3

MAX_INT32 = 2_147_483_647
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PRIVATE_LOCATOR_RE = re.compile(r"(^/|^[A-Za-z]:[\\/]|~|\$|://|@github\.com)")

REASON_CODES = (
    "ledger_missing_or_unreadable",
    "launcher_missing_or_unreadable",
    "malformed_ledger",
    "malformed_launcher",
    "empty_ledger",
    "empty_launcher",
    "invalid_array_bounds",
    "off_by_one_bounds",
    "zero_one_based_confusion",
    "gap_in_scheduled_rows",
    "duplicate_scheduled_rows",
    "duplicate_expected_rows",
    "inconsistent_chunk_tails",
    "shard_reorder",
    "retry_overlap",
    "resume_conflict",
    "integer_overflow",
    "invalid_concurrency",
    "output_path_collision",
    "unmapped_expected_rows",
    "extraneous_scheduled_rows",
    "identity_mismatch",
)


@dataclass(frozen=True)
class ExpectedRow:
    """Single immutable row in the expected campaign ledger."""

    row_id: str
    output_path: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ScheduledPosition:
    """Specific scheduled slot inside an array task."""

    task_id: int
    intra_task_index: int
    scheduled_row_index: int
    row_id: str
    output_path: str


@dataclass(frozen=True)
class ReceiptMeta:
    """Metadata grouping for receipt construction."""

    ledger_path: Path
    launcher_path: Path
    ledger_data: Mapping[str, Any]
    launcher_data: Mapping[str, Any]
    repo_root: Path | None = None


def compute_sha256(path: Path) -> str:
    """Compute hex SHA-256 of file at path."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sanitize_path(raw_path: str | Path, repo_root: Path | None = None) -> str:
    """Normalize path to sanitized relative POSIX path, preventing private path leaks."""
    p_str = str(raw_path).strip()
    if PRIVATE_LOCATOR_RE.search(p_str) and repo_root is not None:
        try:
            rel = Path(p_str).resolve().relative_to(repo_root.resolve())
            return rel.as_posix()
        except ValueError:
            return Path(p_str).name
    posix = PurePosixPath(p_str)
    return posix.as_posix()


def _validate_safe_id(value: Any, name: str, problems: list[dict[str, str]]) -> str | None:
    """Validate a safe string identifier."""
    if not isinstance(value, str) or not SAFE_ID_RE.fullmatch(value):
        problems.append(
            {
                "code": "invalid_identifier",
                "location": name,
                "message": f"{name} must match {SAFE_ID_RE.pattern}, got {value!r}",
            }
        )
        return None
    return value


def _parse_row_entry(
    raw: Any, loc: str, seen_ids: set[str], problems: list[dict[str, str]]
) -> ExpectedRow | None:
    """Parse and validate one expected row dictionary."""
    if not isinstance(raw, Mapping):
        problems.append(
            {"code": "malformed_ledger", "location": loc, "message": "Row must be an object"}
        )
        return None

    raw_id = raw.get("row_id")
    row_id = _validate_safe_id(raw_id, f"{loc}.row_id", problems)
    if row_id is None:
        return None

    if row_id in seen_ids:
        problems.append(
            {
                "code": "duplicate_expected_rows",
                "location": loc,
                "message": f"Duplicate expected row_id {row_id!r}",
            }
        )
    seen_ids.add(row_id)

    raw_out = raw.get("output_path")
    if raw_out is None:
        raw_out = f"output/row_{row_id}.json"
    elif not isinstance(raw_out, str) or not raw_out.strip():
        problems.append(
            {
                "code": "malformed_ledger",
                "location": f"{loc}.output_path",
                "message": "output_path must be a non-empty string",
            }
        )
        return None
    elif ".." in raw_out or raw_out.startswith(("/", "\\")):
        problems.append(
            {
                "code": "malformed_ledger",
                "location": f"{loc}.output_path",
                "message": f"Escaped or absolute output_path not permitted: {raw_out!r}",
            }
        )
        return None

    return ExpectedRow(
        row_id=row_id,
        output_path=PurePosixPath(raw_out.strip()).as_posix(),
        metadata=dict(raw),
    )


def parse_expected_ledger(
    data: Mapping[str, Any], problems: list[dict[str, str]]
) -> list[ExpectedRow] | None:
    """Parse and validate expected campaign rows from the ledger."""
    schema = data.get("schema_version")
    if schema != EXPECTED_LEDGER_SCHEMA:
        problems.append(
            {
                "code": "malformed_ledger",
                "location": "schema_version",
                "message": f"Expected schema_version {EXPECTED_LEDGER_SCHEMA!r}, got {schema!r}",
            }
        )
        return None

    raw_rows = data.get("rows")
    if not isinstance(raw_rows, list):
        problems.append(
            {
                "code": "malformed_ledger",
                "location": "rows",
                "message": "Field 'rows' must be a list",
            }
        )
        return None

    if not raw_rows:
        problems.append(
            {
                "code": "empty_ledger",
                "location": "rows",
                "message": "Expected-row ledger contains no rows",
            }
        )
        return None

    expected_rows: list[ExpectedRow] = []
    seen_ids: set[str] = set()

    for idx, raw in enumerate(raw_rows):
        entry = _parse_row_entry(raw, f"rows[{idx}]", seen_ids, problems)
        if entry is not None:
            expected_rows.append(entry)

    return expected_rows


def _check_integer_bounds(name: str, val: Any, problems: list[dict[str, str]]) -> bool:
    """Check integer type and prevent 32-bit overflow."""
    if not isinstance(val, int):
        problems.append(
            {
                "code": "malformed_launcher",
                "location": f"array_spec.{name}",
                "message": f"{name} must be integer, got {val!r}",
            }
        )
        return False
    if val > MAX_INT32 or val < -MAX_INT32:
        problems.append(
            {
                "code": "integer_overflow",
                "location": f"array_spec.{name}",
                "message": f"{name} exceeds integer bounds: {val}",
            }
        )
        return False
    return True


def _validate_spec_bounds(
    min_idx: int,
    max_idx: int,
    step: int,
    chunk_size: int,
    total_expected: int,
    zero_based: bool,
    task_offset: int,
    problems: list[dict[str, str]],
) -> list[int] | None:
    """Validate bounds, task count, and return expanded task IDs."""
    if step < 1:
        problems.append(
            {
                "code": "malformed_launcher",
                "location": "array_spec.step",
                "message": f"step must be >= 1, got {step}",
            }
        )
        return None
    if chunk_size < 1:
        problems.append(
            {
                "code": "malformed_launcher",
                "location": "array_spec.chunk_size",
                "message": f"chunk_size must be >= 1, got {chunk_size}",
            }
        )
        return None
    if min_idx > max_idx:
        problems.append(
            {
                "code": "invalid_array_bounds",
                "location": "array_spec",
                "message": f"min_index ({min_idx}) > max_index ({max_idx})",
            }
        )
        return None
    if zero_based and min_idx != 0 and task_offset == 0:
        problems.append(
            {
                "code": "zero_one_based_confusion",
                "location": "array_spec.min_index",
                "message": f"zero_based is declared true but min_index is {min_idx} without task_offset",
            }
        )

    task_ids = list(range(min_idx, max_idx + 1, step))
    total_tasks = len(task_ids)
    required_tasks = math.ceil(total_expected / chunk_size) if total_expected > 0 else 0

    if total_tasks != required_tasks:
        code = (
            "off_by_one_bounds"
            if abs(total_tasks - required_tasks) == 1
            else "invalid_array_bounds"
        )
        problems.append(
            {
                "code": code,
                "location": "array_spec",
                "message": f"Array task count ({total_tasks}) != required tasks ({required_tasks})",
            }
        )
    return task_ids


def _expand_tasks(
    launcher: Mapping[str, Any], total_expected: int, problems: list[dict[str, str]]
) -> tuple[list[int], dict[str, Any]] | None:
    """Expand and validate scheduled task indices and configuration."""
    job_type = launcher.get("job_type", "array")
    if job_type == "scalar":
        task_offset = launcher.get("task_offset", 0)
        if not isinstance(task_offset, int) or task_offset < 0:
            problems.append(
                {
                    "code": "malformed_launcher",
                    "location": "task_offset",
                    "message": "task_offset must be non-negative integer",
                }
            )
            return None
        return [task_offset], {
            "job_type": "scalar",
            "chunk_size": total_expected,
            "has_partial_tail": False,
            "tail_size": total_expected,
            "zero_based": True,
            "task_offset": task_offset,
            "concurrency": 1,
        }

    spec = launcher.get("array_spec")
    if not isinstance(spec, Mapping):
        problems.append(
            {
                "code": "malformed_launcher",
                "location": "array_spec",
                "message": "Missing 'array_spec' object",
            }
        )
        return None

    min_idx = spec.get("min_index")
    max_idx = spec.get("max_index")
    step = spec.get("step", 1)
    chunk_size = spec.get("chunk_size", 1)
    task_offset = spec.get("task_offset", 0)
    concurrency = spec.get("concurrency")
    zero_based = spec.get("zero_based", True)

    for name, val in [
        ("min_index", min_idx),
        ("max_index", max_idx),
        ("step", step),
        ("chunk_size", chunk_size),
        ("task_offset", task_offset),
    ]:
        if not _check_integer_bounds(name, val, problems):
            return None

    if concurrency is not None and (not isinstance(concurrency, int) or concurrency <= 0):
        problems.append(
            {
                "code": "invalid_concurrency",
                "location": "array_spec.concurrency",
                "message": f"concurrency must be positive integer, got {concurrency!r}",
            }
        )

    task_ids = _validate_spec_bounds(
        min_idx, max_idx, step, chunk_size, total_expected, zero_based, task_offset, problems
    )
    if task_ids is None:
        return None

    tail_size = total_expected % chunk_size
    has_partial_tail = tail_size != 0

    summary = {
        "job_type": "array",
        "chunk_size": chunk_size,
        "has_partial_tail": has_partial_tail,
        "tail_size": tail_size if has_partial_tail else chunk_size,
        "zero_based": zero_based,
        "task_offset": task_offset,
        "concurrency": concurrency,
    }
    return task_ids, summary


def _check_identities(
    ledger_data: Mapping[str, Any], launcher_data: Mapping[str, Any], problems: list[dict[str, str]]
) -> None:
    """Verify campaign identity parameters agree between ledger and launcher."""
    for field in ("campaign_id", "source_tree_sha256"):
        val1 = ledger_data.get(field)
        val2 = launcher_data.get(field)
        if val1 and val2 and val1 != val2:
            problems.append(
                {
                    "code": "identity_mismatch",
                    "location": field,
                    "message": f"{field} mismatch: ledger={val1!r} vs launcher={val2!r}",
                }
            )


def _check_manifest_and_retries(
    mapping_spec: Mapping[str, Any],
    task_count: int,
    expected_rows: list[ExpectedRow],
    problems: list[dict[str, str]],
) -> None:
    """Verify manifest ordering, retry ranges, and resume invariants."""
    manifest_order = mapping_spec.get("manifest_order")
    if isinstance(manifest_order, list):
        expected_ids = [r.row_id for r in expected_rows]
        if manifest_order != expected_ids and sorted(manifest_order) == sorted(expected_ids):
            problems.append(
                {
                    "code": "shard_reorder",
                    "location": "mapping.manifest_order",
                    "message": "Launcher manifest order diverges from canonical ledger row sequence",
                }
            )

    retry_offset = mapping_spec.get("retry_offset", 0)
    retry_mode = mapping_spec.get("retry_mode", "none")
    if retry_mode != "none" and retry_offset > 0 and retry_offset < task_count:
        problems.append(
            {
                "code": "retry_overlap",
                "location": "mapping.retry_offset",
                "message": f"retry_offset {retry_offset} overlaps with active task range {task_count}",
            }
        )

    resume_mode = mapping_spec.get("resume_mode", "none")
    if resume_mode == "skip_existing":
        existing_rows = set(mapping_spec.get("existing_row_ids", []))
        if not existing_rows.issubset({r.row_id for r in expected_rows}):
            problems.append(
                {
                    "code": "resume_conflict",
                    "location": "mapping.existing_row_ids",
                    "message": "Resume existing_row_ids contains identities not in expected ledger",
                }
            )


def _resolve_output_path(
    target_row: ExpectedRow,
    task_id: int,
    intra_idx: int,
    output_pattern: str | None,
    problems: list[dict[str, str]],
) -> str:
    """Resolve destination output path using pattern or default row output path."""
    if not output_pattern or not isinstance(output_pattern, str):
        return target_row.output_path
    kwargs = dict(target_row.metadata)
    kwargs["row_id"] = target_row.row_id
    kwargs["task_id"] = task_id
    kwargs["intra_idx"] = intra_idx
    try:
        return output_pattern.format(**kwargs)
    except KeyError as exc:
        problems.append(
            {
                "code": "malformed_launcher",
                "location": "mapping.output_pattern",
                "message": f"output_pattern references missing key: {exc}",
            }
        )
        return target_row.output_path


def _map_scheduled_slots(
    task_ids: list[int],
    step: int,
    task_offset: int,
    chunk_size: int,
    expected_rows: list[ExpectedRow],
    output_pattern: str | None,
    problems: list[dict[str, str]],
) -> list[ScheduledPosition]:
    """Map array tasks and intra-task indices to expected rows."""
    positions: list[ScheduledPosition] = []
    total_expected = len(expected_rows)
    min_task_id = min(task_ids) if task_ids else 0

    for task_id in task_ids:
        norm_task_idx = (task_id - min_task_id - task_offset) // step if step > 0 else 0
        task_start_row = norm_task_idx * chunk_size

        if task_start_row >= total_expected:
            problems.append(
                {
                    "code": "extraneous_scheduled_rows",
                    "location": f"task[{task_id}]",
                    "message": f"Task {task_id} starts at row {task_start_row} beyond ledger size {total_expected}",
                }
            )
            continue

        for intra_idx in range(chunk_size):
            global_row_idx = task_start_row + intra_idx
            if global_row_idx >= total_expected:
                continue

            target_row = expected_rows[global_row_idx]
            out_path = _resolve_output_path(
                target_row, task_id, intra_idx, output_pattern, problems
            )
            positions.append(
                ScheduledPosition(
                    task_id=task_id,
                    intra_task_index=intra_idx,
                    scheduled_row_index=global_row_idx,
                    row_id=target_row.row_id,
                    output_path=out_path,
                )
            )

    return positions


def _check_bijectivity_and_collisions(
    positions: list[ScheduledPosition],
    expected_rows: list[ExpectedRow],
    problems: list[dict[str, str]],
) -> None:
    """Verify one-to-one, onto mapping and collision absence."""
    scheduled_row_ids = [p.row_id for p in positions]
    for rid, count in Counter(scheduled_row_ids).items():
        if count > 1:
            problems.append(
                {
                    "code": "duplicate_scheduled_rows",
                    "location": f"row_id[{rid}]",
                    "message": f"Row {rid!r} scheduled {count} times",
                }
            )

    unmapped = {r.row_id for r in expected_rows} - set(scheduled_row_ids)
    if unmapped:
        problems.append(
            {
                "code": "gap_in_scheduled_rows",
                "location": "mapping",
                "message": f"{len(unmapped)} expected rows have no scheduled task slot: {sorted(unmapped)[:5]}",
            }
        )
        problems.append(
            {
                "code": "unmapped_expected_rows",
                "location": "mapping",
                "message": f"Unmapped rows: {sorted(unmapped)[:5]}",
            }
        )

    for out_p, count in Counter(p.output_path for p in positions).items():
        if count > 1:
            problems.append(
                {
                    "code": "output_path_collision",
                    "location": f"output_path[{out_p}]",
                    "message": f"Output path {out_p!r} assigned to {count} different scheduled slots",
                }
            )


def verify_array_mapping(
    ledger_data: Mapping[str, Any],
    launcher_data: Mapping[str, Any],
    *,
    ledger_path: Path,
    launcher_path: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Execute complete deterministic mapping verification pass."""
    problems: list[dict[str, str]] = []
    meta = ReceiptMeta(
        ledger_path=ledger_path,
        launcher_path=launcher_path,
        ledger_data=ledger_data,
        launcher_data=launcher_data,
        repo_root=repo_root,
    )

    _check_identities(ledger_data, launcher_data, problems)
    expected_rows = parse_expected_ledger(ledger_data, problems)
    if expected_rows is None:
        status = (
            "malformed" if any(p["code"].startswith("malformed") for p in problems) else "blocked"
        )
        return _build_receipt(status, meta, 0, 0, 0, {}, problems, [])

    total_expected = len(expected_rows)
    task_res = _expand_tasks(launcher_data, total_expected, problems)
    if task_res is None:
        is_malformed = any(
            p["code"].startswith("malformed") or p["code"] == "integer_overflow" for p in problems
        )
        return _build_receipt(
            "malformed" if is_malformed else "blocked", meta, total_expected, 0, 0, {}, problems, []
        )

    task_ids, summary = task_res
    mapping_spec = launcher_data.get("mapping", {})
    _check_manifest_and_retries(mapping_spec, len(task_ids), expected_rows, problems)

    step = (
        launcher_data.get("array_spec", {}).get("step", 1) if summary["job_type"] == "array" else 1
    )
    positions = _map_scheduled_slots(
        task_ids,
        step,
        summary["task_offset"],
        summary["chunk_size"],
        expected_rows,
        mapping_spec.get("output_pattern"),
        problems,
    )

    _check_bijectivity_and_collisions(positions, expected_rows, problems)

    if summary.get("has_partial_tail"):
        final_rows = [p for p in positions if p.task_id == task_ids[-1]]
        if len(final_rows) != summary["tail_size"]:
            problems.append(
                {
                    "code": "inconsistent_chunk_tails",
                    "location": f"task[{task_ids[-1]}]",
                    "message": f"Final task has {len(final_rows)} rows, expected {summary['tail_size']}",
                }
            )

    status = "verified" if not problems else "blocked"
    if any(p["code"].startswith("malformed") or p["code"] == "integer_overflow" for p in problems):
        status = "malformed"

    return _build_receipt(
        status, meta, total_expected, len(task_ids), len(positions), summary, problems, positions
    )


def _build_receipt(
    status: str,
    meta: ReceiptMeta,
    expected_count: int,
    task_count: int,
    scheduled_count: int,
    summary: Mapping[str, Any],
    problems: list[dict[str, str]],
    positions: Sequence[ScheduledPosition],
) -> dict[str, Any]:
    """Assemble deterministic JSON receipt payload."""
    unique_codes = sorted({p["code"] for p in problems if p["code"] in REASON_CODES})
    first_error = problems[0]["message"] if problems else None

    ledger_sha = compute_sha256(meta.ledger_path) if meta.ledger_path.is_file() else None
    launcher_sha = compute_sha256(meta.launcher_path) if meta.launcher_path.is_file() else None

    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": status,
        "ledger_file": _sanitize_path(meta.ledger_path, meta.repo_root),
        "launcher_file": _sanitize_path(meta.launcher_path, meta.repo_root),
        "ledger_sha256": ledger_sha,
        "launcher_sha256": launcher_sha,
        "campaign_id": meta.ledger_data.get("campaign_id") or meta.launcher_data.get("campaign_id"),
        "expected_row_count": expected_count,
        "scheduled_task_count": task_count,
        "scheduled_row_count": scheduled_count,
        "mapping_summary": dict(summary),
        "reason_codes": unique_codes,
        "first_error": first_error,
        "problems": problems,
        "scheduled_positions": [
            {
                "task_id": p.task_id,
                "intra_task_index": p.intra_task_index,
                "scheduled_row_index": p.scheduled_row_index,
                "row_id": p.row_id,
                "output_path": p.output_path,
            }
            for p in positions
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def format_csv(receipt: Mapping[str, Any]) -> str:
    """Format scheduled positions as deterministic CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["task_id", "intra_task_index", "scheduled_row_index", "row_id", "output_path"])
    for pos in receipt.get("scheduled_positions", []):
        writer.writerow(
            [
                pos["task_id"],
                pos["intra_task_index"],
                pos["scheduled_row_index"],
                pos["row_id"],
                pos["output_path"],
            ]
        )
    return buf.getvalue()


def build_parser() -> argparse.ArgumentParser:
    """Build command line interface parser."""
    parser = argparse.ArgumentParser(
        description="Verify SLURM array-index to campaign-row mapping before submission."
    )
    parser.add_argument("--check", action="store_true", help="Perform pre-submission verification.")
    parser.add_argument(
        "--ledger", required=True, type=Path, help="Path to expected-row ledger JSON."
    )
    parser.add_argument(
        "--launcher", required=True, type=Path, help="Path to SLURM array launcher JSON."
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output serialization format (default: json).",
    )
    parser.add_argument("--output", type=Path, help="Optional output path to write receipt.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root for path sanitization.",
    )
    return parser


def _load_input_files(
    ledger_path: Path, launcher_path: Path, problems: list[dict[str, str]]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Safely load and parse input JSON files."""
    if not ledger_path.is_file():
        problems.append(
            {
                "code": "ledger_missing_or_unreadable",
                "location": str(ledger_path),
                "message": f"Ledger file not found: {ledger_path}",
            }
        )
    if not launcher_path.is_file():
        problems.append(
            {
                "code": "launcher_missing_or_unreadable",
                "location": str(launcher_path),
                "message": f"Launcher file not found: {launcher_path}",
            }
        )
    if problems:
        return None

    ledger_data: dict[str, Any] = {}
    launcher_data: dict[str, Any] = {}
    try:
        with ledger_path.open("r", encoding="utf-8") as f:
            ledger_data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        problems.append(
            {
                "code": "malformed_ledger",
                "location": str(ledger_path),
                "message": f"Could not parse ledger JSON: {exc}",
            }
        )

    try:
        with launcher_path.open("r", encoding="utf-8") as f:
            launcher_data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        problems.append(
            {
                "code": "malformed_launcher",
                "location": str(launcher_path),
                "message": f"Could not parse launcher JSON: {exc}",
            }
        )

    if problems:
        return None
    return ledger_data, launcher_data


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point returning standard status codes."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.check:
        parser.error("--check is required for verification.")

    problems: list[dict[str, str]] = []
    loaded = _load_input_files(args.ledger, args.launcher, problems)
    meta = ReceiptMeta(args.ledger, args.launcher, {}, {}, args.repo_root)

    if loaded is None:
        receipt = _build_receipt("malformed", meta, 0, 0, 0, {}, problems, [])
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return EXIT_MALFORMED

    ledger_data, launcher_data = loaded
    receipt = verify_array_mapping(
        ledger_data,
        launcher_data,
        ledger_path=args.ledger,
        launcher_path=args.launcher,
        repo_root=args.repo_root,
    )

    out_content = (
        format_csv(receipt)
        if args.format == "csv"
        else json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_content, encoding="utf-8")
    else:
        sys.stdout.write(out_content)

    status = receipt["status"]
    if status == "verified":
        return EXIT_VERIFIED
    if status == "blocked":
        return EXIT_BLOCKED
    return EXIT_MALFORMED


if __name__ == "__main__":
    sys.exit(main())
