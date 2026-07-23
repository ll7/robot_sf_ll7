#!/usr/bin/env python3
"""Deterministic coverage-constrained Pareto portfolio selection from a scenario archive.

Reads a generated scenario archive (JSON Lines or JSON list of catalog entries),
applies descriptor extraction, Pareto filtering, and deterministic max-min
coverage selection, then writes a schema-versioned selection manifest.

This is analysis tooling only. It does not modify any archive, run simulation
campaigns, or claim benchmark evidence.

Examples
--------
    # Select from a JSONL archive, write manifest to stdout.
    uv run python scripts/analysis/select_scenario_portfolio.py \
        --archive output/archive/entries.jsonl

    # Write manifest to a specific path with a custom max portfolio size.
    uv run python scripts/analysis/select_scenario_portfolio.py \
        --archive output/archive/entries.jsonl \
        --manifest-id portfolio-20260714 \
        --max-size 10 \
        --output docs/context/evidence/portfolio_selection.json

    # Use a JSON list instead of JSONL.
    uv run python scripts/analysis/select_scenario_portfolio.py \
        --archive output/archive/entries.json \
        --format json

    # Dry-run: validate only, no output file.
    uv run python scripts/analysis/select_scenario_portfolio.py \
        --archive output/archive/entries.jsonl \
        --dry-run
"""
# evidence-writer-exempt: references evidence paths but does not write to evidence tree; guarded by AST analysis

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

from robot_sf.benchmark.scenario_generation.portfolio_selector import (
    SCHEMA_VERSION,
    CoverageQuotas,
    DescriptorConfig,
    PortfolioSelectionError,
    SelectionConfig,
    select_portfolio,
    validate_selection_manifest,
    write_selection_manifest,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Coverage-constrained Pareto portfolio selection from a scenario archive.",
    )
    parser.add_argument(
        "--archive",
        type=str,
        required=True,
        help="Path to the scenario archive file (JSONL or JSON list of catalog entries).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=["auto", "jsonl", "json"],
        help="Input format: auto-detect from extension, jsonl, or json list. (default: auto)",
    )
    parser.add_argument(
        "--manifest-id",
        type=str,
        default=None,
        help="Unique manifest ID (default: auto-generated from archive name).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output manifest JSON path (default: stdout).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=20,
        help="Maximum portfolio size. (default: 20)",
    )
    parser.add_argument(
        "--min-per-topology",
        type=int,
        default=1,
        help="Minimum candidates per topology type. (default: 1)",
    )
    parser.add_argument(
        "--min-per-interaction",
        type=int,
        default=1,
        help="Minimum candidates per interaction class. (default: 1)",
    )
    parser.add_argument(
        "--max-same-generator",
        type=int,
        default=5,
        help="Maximum candidates from the same generator. (default: 5)",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="min_max",
        choices=["min_max", "z_score", "none"],
        help="Descriptor normalization method. (default: min_max)",
    )
    parser.add_argument(
        "--archive-hash",
        type=str,
        default="",
        help="Optional SHA-256 hash of the archive file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print summary without writing an output file.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional notes for the manifest provenance.",
    )
    return parser.parse_args(argv)


def _detect_format(path: Path) -> str:
    """Detect input format from file extension."""
    if path.suffix == ".jsonl":
        return "jsonl"
    if path.suffix == ".json":
        return "json"
    return "jsonl"  # default


def _load_entries(path: Path, fmt: str) -> list[dict[str, Any]]:
    """Load scenario entries from a file.

    Parameters
    ----------
    path : Path
        Path to the archive file.
    fmt : str
        Input format: "jsonl" or "json".

    Returns
    -------
    list[dict[str, Any]]
        Loaded scenario entries.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if fmt == "jsonl":
        entries: list[dict[str, Any]] = []
        for line in text.split("\n"):
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries

    # JSON format — expect a list
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Single entry wrapped
        return [data]
    msg = f"Unsupported JSON structure: expected list or dict, got {type(data).__name__}"
    raise ValueError(msg)


def _auto_manifest_id(archive_path: Path) -> str:
    """Generate a manifest ID from the archive filename."""
    from datetime import datetime

    stem = archive_path.stem.replace(".jsonl", "").replace(".json", "")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"portfolio-{stem}-{timestamp}"


def main(argv: list[str] | None = None) -> int:
    """Entry point for the portfolio selection CLI.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments (default: sys.argv[1:]).

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    args = _parse_args(argv)

    archive_path = Path(args.archive)
    if not archive_path.exists():
        print(f"Error: archive file not found: {archive_path}", file=sys.stderr)
        return 1

    fmt = args.format if args.format != "auto" else _detect_format(archive_path)
    try:
        entries = _load_entries(archive_path, fmt)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Error loading archive: {exc}", file=sys.stderr)
        return 1

    if not entries:
        print("Error: archive contains no entries.", file=sys.stderr)
        return 1

    manifest_id = args.manifest_id or _auto_manifest_id(archive_path)

    config = SelectionConfig(
        quotas=CoverageQuotas(
            min_per_topology=args.min_per_topology,
            min_per_interaction_class=args.min_per_interaction,
            max_from_same_generator=args.max_same_generator,
        ),
        descriptor_config=DescriptorConfig(
            normalization_method=args.normalization,  # type: ignore[arg-type]
        ),
        max_portfolio_size=args.max_size,
    )

    try:
        manifest = select_portfolio(
            entries=entries,
            manifest_id=manifest_id,
            config=config,
            archive_path=str(archive_path),
            archive_hash=args.archive_hash,
        )
    except PortfolioSelectionError as exc:
        print(f"Portfolio selection failed: {exc}", file=sys.stderr)
        return 1

    # Validate output
    try:
        validate_selection_manifest(manifest)
    except PortfolioSelectionError as exc:
        print(f"Generated manifest failed schema validation: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        pareto_size = manifest["pareto_analysis"]["front_size"]
        selected_count = len(manifest["selection_sequence"])
        excluded_count = len(manifest["exclusion_ledger"])
        print(f"Manifest ID: {manifest_id}")
        print(f"Eligible candidates: {len(entries)}")
        print(f"Pareto front size: {pareto_size}")
        print(f"Selected: {selected_count}")
        print(f"Excluded: {excluded_count}")
        print(f"Schema version: {SCHEMA_VERSION}")
        print("Dry-run: output file not written.")
        return 0

    if args.output:
        output_path = Path(args.output)
        write_selection_manifest(manifest, output_path)
        print(f"Selection manifest written to {output_path}", file=sys.stderr)
    else:
        json.dump(manifest, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
