"""Benchmark artifact publication helper (size report + DOI-ready bundle export)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from robot_sf.benchmark.artifact_publication import (
    export_publication_bundle,
    measure_artifact_size_ranges,
)
from robot_sf.common.artifact_paths import get_artifact_category_path

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for benchmark publication tooling."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    size_report = subparsers.add_parser(
        "size-report",
        help="Measure artifact size ranges across benchmark run directories.",
    )
    size_report.add_argument(
        "--benchmarks-root",
        type=Path,
        default=get_artifact_category_path("benchmarks"),
        help="Benchmark output root to scan (default: output/benchmarks).",
    )
    size_report.add_argument(
        "--include-videos",
        action="store_true",
        default=False,
        help="Include video files in size distribution calculations.",
    )
    size_report.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the size report JSON payload.",
    )

    export = subparsers.add_parser(
        "export",
        help="Export one benchmark run as a publication bundle with checksums.",
    )
    export.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Benchmark run directory to export.",
    )
    export.add_argument(
        "--out-dir",
        type=Path,
        default=get_artifact_category_path("benchmarks") / "publication",
        help="Destination directory for bundle folder + tar.gz archive.",
    )
    export.add_argument(
        "--bundle-name",
        type=str,
        default=None,
        help="Optional output bundle name (default: <run_dir_name>_publication_bundle).",
    )
    export.add_argument(
        "--include-videos",
        action="store_true",
        default=False,
        help="Include videos in the exported payload.",
    )
    export.add_argument(
        "--repository-url",
        type=str,
        default="https://github.com/ll7/robot_sf_ll7",
        help="Public repository URL used in publication metadata.",
    )
    export.add_argument(
        "--release-tag",
        type=str,
        default="{release_tag}",
        help="Release tag placeholder or concrete tag used in citation metadata.",
    )
    export.add_argument(
        "--doi",
        type=str,
        default="10.5281/zenodo.<record-id>",
        help="DOI placeholder or concrete DOI stored in publication metadata.",
    )
    export.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing bundle directory/archive with the same name.",
    )
    return parser


def _run_size_report(args: argparse.Namespace) -> int:
    """Execute the ``size-report`` subcommand."""
    report = measure_artifact_size_ranges(
        args.benchmarks_root,
        include_videos=bool(args.include_videos),
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("Size report written to {}", args.output_json)

    print(json.dumps(report, indent=2))
    return 0


def _run_export(args: argparse.Namespace) -> int:
    """Execute the ``export`` subcommand."""
    result = export_publication_bundle(
        args.run_dir,
        args.out_dir,
        bundle_name=args.bundle_name,
        include_videos=bool(args.include_videos),
        repository_url=args.repository_url,
        release_tag=args.release_tag,
        doi=args.doi,
        overwrite=bool(args.overwrite),
    )
    payload = {
        "bundle_dir": str(result.bundle_dir),
        "archive_path": str(result.archive_path),
        "manifest_path": str(result.manifest_path),
        "checksums_path": str(result.checksums_path),
        "file_count": result.file_count,
        "total_bytes": result.total_bytes,
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run benchmark publication helper CLI and return a POSIX exit code."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "size-report":
        return _run_size_report(args)
    if args.command == "export":
        return _run_export(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
