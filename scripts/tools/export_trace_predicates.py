#!/usr/bin/env python3
"""CLI: export trace-level predicates from campaign episode records (issue #5593).

Reads a JSONL file or directory of episode records, extracts safety predicates
from event ledgers, and writes a versioned JSONL export with manifest and
coverage report.

Usage:
    python scripts/tools/export_trace_predicates.py <input> [options]

Examples:
    # Single JSONL file
    python scripts/tools/export_trace_predicates.py campaign.jsonl

    # Directory of JSONL files
    python scripts/tools/export_trace_predicates.py output/scenario_dir/

    # Custom output directory with specific predicate families
    python scripts/tools/export_trace_predicates.py campaign.jsonl \\
        --output-dir ./export --families oscillation late_evasive
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from robot_sf.benchmark.event_ledger import ensure_event_ledger
from robot_sf.benchmark.trace_predicate_export import (
    MOTIVATED_PREDICATE_FAMILIES,
    PredicateExportRecord,
    build_coverage_report,
    build_export_manifest,
    build_predicate_export_batch,
    export_to_jsonl,
    write_coverage_report,
    write_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load episode records from a JSONL file."""
    records: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning("Skipping malformed line %d in %s: %s", i + 1, path.name, e)
    return records


def _discover_jsonl_files(input_path: Path) -> list[Path]:
    """Find all JSONL files in a path (file or directory)."""
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in {".jsonl", ".json"} else []
    if input_path.is_dir():
        files = sorted(
            input_path.rglob("*.jsonl"),
        )
        return files
    return []


def _ensure_ledgers(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure every record has an event ledger, building if missing."""
    result: list[dict[str, Any]] = []
    for rec in records:
        rec_copy = deepcopy(rec)
        rec_copy.setdefault("event_ledger", {})
        ensure_event_ledger(rec_copy)
        result.append(rec_copy)
    return result


def _write_export_artifacts(
    export_records: list[PredicateExportRecord],
    all_records: list[dict[str, Any]],
    *,
    output_dir: Path,
    predicate_families: list[str] | None,
) -> None:
    """Write the export and dependent artifacts as one fail-closed stage."""
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "trace_predicates.jsonl"
    checksum = export_to_jsonl(export_records, jsonl_path)
    logger.info("Exported %d records to %s (sha256: %s)", len(export_records), jsonl_path, checksum)

    manifest = build_export_manifest(export_records)
    manifest_path = output_dir / "trace_predicates_manifest.json"
    write_manifest(manifest, manifest_path)
    logger.info("Wrote manifest to %s", manifest_path)

    coverage_rows = build_coverage_report(export_records, predicate_families=predicate_families)
    coverage_path = output_dir / "trace_predicates_coverage.md"
    write_coverage_report(coverage_rows, coverage_path, total_episodes=len(all_records))
    logger.info("Wrote coverage report to %s", coverage_path)


def _log_export_summary(export_records: list[PredicateExportRecord]) -> None:
    """Log exported and missing predicate families."""
    families_with_records = set()
    families_missing = set()
    for rec in export_records:
        families_with_records.update(rec.surrogate_events)
        families_missing.update(rec.missing_fields)

    logger.info(
        "Predicate families exported: %s", ", ".join(sorted(families_with_records)) or "(none)"
    )
    if families_missing:
        logger.info("Predicate families missing: %s", ", ".join(sorted(families_missing)))


def main(argv: list[str] | None = None) -> int:
    """Run the trace predicate export CLI.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description="Export trace-level predicates from campaign episode records."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="JSONL file or directory containing episode records.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: parent of input).",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        choices=MOTIVATED_PREDICATE_FAMILIES,
        help=f"Predicate families to export (default: all). "
        f"Choices: {', '.join(MOTIVATED_PREDICATE_FAMILIES)}",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Discover input files
    jsonl_files = _discover_jsonl_files(args.input)
    if not jsonl_files:
        logger.error("No JSONL files found at %s", args.input)
        return 1

    logger.info("Found %d JSONL file(s)", len(jsonl_files))

    # Load all records
    all_records: list[dict[str, Any]] = []
    for f in jsonl_files:
        records = _load_jsonl_records(f)
        logger.info("Loaded %d records from %s", len(records), f.name)
        all_records.extend(records)

    if not all_records:
        logger.error("No episode records loaded from input")
        return 1

    # Ensure event ledgers
    all_records = _ensure_ledgers(all_records)

    # Build export records
    try:
        export_records = build_predicate_export_batch(
            all_records,
            predicate_families=args.families,
        )
    except ValueError as e:
        logger.exception("Export failed: %s", e)
        return 2

    # Determine output directory
    output_dir = args.output_dir or args.input.parent if args.input.is_file() else args.input
    try:
        _write_export_artifacts(
            export_records,
            all_records,
            output_dir=output_dir,
            predicate_families=args.families,
        )
    except OSError as exc:
        logger.exception("Trace predicate export blocked by file I/O failure: %s", exc)
        return 3

    _log_export_summary(export_records)

    return 0


if __name__ == "__main__":
    sys.exit(main())
