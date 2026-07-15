"""CLI: export trace-level safety predicates from a completed campaign.

Reads one or more campaign ``episodes.jsonl`` bundles (each episode already carries a
``safety_predicates`` block from robot_sf/benchmark/safety_predicates.py) and emits a
versioned, queryable export (JSON-lines), a manifest of predicate presence/gaps, and a
coverage report that enumerates exported-vs-motivated predicates for a release.

See issue #5593. The export fails closed on missing/degraded predicate fields: gaps are
recorded in the manifest and coverage report rather than silently dropped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import robot_sf.benchmark.trace_predicate_export as tpe

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        action="append",
        help="Path to a campaign episodes.jsonl bundle. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--release",
        required=True,
        help="Release/run label this export is produced for (used in manifest/coverage).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("trace_predicate_export"),
        help="Output directory. Defaults to ./trace_predicate_export.",
    )
    parser.add_argument(
        "--export-name",
        default="trace_predicate_export.jsonl",
        help="Export JSON-lines filename. Defaults to trace_predicate_export.jsonl.",
    )
    parser.add_argument(
        "--manifest-name",
        default="trace_predicate_manifest.json",
        help="Manifest JSON filename. Defaults to trace_predicate_manifest.json.",
    )
    parser.add_argument(
        "--coverage-name",
        default="trace_predicate_coverage.json",
        help="Coverage report JSON filename. Defaults to trace_predicate_coverage.json.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the export command."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    missing = [str(p) for p in args.bundle if not Path(p).is_file()]
    if missing:
        print(f"bundle path(s) not found: {', '.join(missing)}", file=sys.stderr)
        return 1

    try:
        export_rows, manifest, coverage, failed_sources = tpe.export_trace_predicates_from_bundle(
            args.bundle, release=args.release
        )
    except tpe.TracePredicateExportError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    export_jsonl, manifest_json, coverage_json = tpe.write_trace_predicate_export(
        export_rows=export_rows,
        manifest=manifest,
        coverage=coverage,
        export_jsonl=args.output_dir / args.export_name,
        manifest_json=args.output_dir / args.manifest_name,
        coverage_json=args.output_dir / args.coverage_name,
    )

    print(f"wrote export to {export_jsonl} ({len(export_rows)} episodes)")
    print(f"wrote manifest to {manifest_json} (complete={manifest['complete']})")
    print(
        f"wrote coverage to {coverage_json} "
        f"({coverage['summary']['exported_count']}/{coverage['summary']['motivated_count']} "
        "motivated predicates exported)"
    )
    if failed_sources:
        print("failed sources:", file=sys.stderr)
        for item in failed_sources:
            print(f"  - {item}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
