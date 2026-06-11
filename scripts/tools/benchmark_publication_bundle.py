"""Benchmark artifact publication helper (size report + DOI-ready bundle export)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from robot_sf.benchmark.artifact_publication import (
    DissertationArtifactSpec,
    export_dissertation_artifact_bundle,
    export_evidence_bundle,
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

    evidence = subparsers.add_parser(
        "evidence-bundle",
        help="Package selected compact evidence files with checksums and claim-boundary metadata.",
    )
    evidence.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Directory containing the compact evidence files to include.",
    )
    evidence.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination directory for the evidence bundle folder.",
    )
    evidence.add_argument(
        "--bundle-name",
        type=str,
        required=True,
        help="Output bundle directory name.",
    )
    evidence.add_argument(
        "--file",
        type=Path,
        action="append",
        default=[],
        help="Evidence file to include, relative to --source-root. Repeat for multiple files.",
    )
    evidence.add_argument(
        "--command",
        type=str,
        dest="evidence_command",
        required=True,
        help="Canonical command that produced or validates the evidence.",
    )
    evidence.add_argument(
        "--commit",
        type=str,
        required=True,
        help="Repository commit associated with the evidence.",
    )
    evidence.add_argument(
        "--claim-boundary",
        type=str,
        required=True,
        help="Conservative claim boundary for the evidence bundle.",
    )
    evidence.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing evidence bundle directory with the same name.",
    )

    dissertation = subparsers.add_parser(
        "dissertation-bundle",
        help="Export selected figure/table artifacts with dissertation-facing provenance.",
    )
    dissertation.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Directory containing the selected figure/table source artifacts.",
    )
    dissertation.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination directory for the dissertation artifact bundle.",
    )
    dissertation.add_argument(
        "--bundle-name",
        type=str,
        required=True,
        help="Output bundle directory name.",
    )
    dissertation.add_argument(
        "--artifact-spec",
        type=Path,
        required=True,
        help=(
            "JSON file containing an artifacts array with artifact_id, source_path, "
            "source_artifact, caption_draft, claim_boundary, recommended_manuscript_use, "
            "and fallback_degraded_summary fields."
        ),
    )
    dissertation.add_argument(
        "--command",
        type=str,
        dest="generation_command",
        required=True,
        help="Canonical command that generated or validates the selected artifacts.",
    )
    dissertation.add_argument(
        "--commit",
        type=str,
        required=True,
        help="Repository commit associated with the selected artifacts.",
    )
    dissertation.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing dissertation artifact bundle directory.",
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


def _run_evidence_bundle(args: argparse.Namespace) -> int:
    """Execute the ``evidence-bundle`` subcommand."""
    result = export_evidence_bundle(
        args.source_root,
        args.out_dir,
        bundle_name=args.bundle_name,
        files=list(args.file),
        command=str(args.evidence_command),
        commit=str(args.commit),
        claim_boundary=str(args.claim_boundary),
        overwrite=bool(args.overwrite),
    )
    payload = {
        "bundle_dir": str(result.bundle_dir),
        "manifest_path": str(result.manifest_path),
        "checksums_path": str(result.checksums_path),
        "file_count": result.file_count,
        "total_bytes": result.total_bytes,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _load_dissertation_artifacts(spec_path: Path) -> list[DissertationArtifactSpec]:
    """Load dissertation artifact rows from a JSON spec file."""
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    rows = payload.get("artifacts") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Dissertation artifact spec must be a list or contain an artifacts list")

    artifacts: list[DissertationArtifactSpec] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Artifact spec row {index} must be an object")
        try:
            artifacts.append(
                DissertationArtifactSpec(
                    artifact_id=str(row["artifact_id"]),
                    source_path=Path(str(row["source_path"])),
                    source_artifact=str(row["source_artifact"]),
                    caption_draft=str(row["caption_draft"]),
                    claim_boundary=str(row["claim_boundary"]),
                    recommended_manuscript_use=str(row["recommended_manuscript_use"]),
                    fallback_degraded_summary=str(row["fallback_degraded_summary"]),
                )
            )
        except KeyError as exc:
            raise ValueError(f"Artifact spec row {index} missing required field: {exc}") from exc
    return artifacts


def _run_dissertation_bundle(args: argparse.Namespace) -> int:
    """Execute the ``dissertation-bundle`` subcommand."""
    result = export_dissertation_artifact_bundle(
        args.source_root,
        args.out_dir,
        bundle_name=args.bundle_name,
        artifacts=_load_dissertation_artifacts(args.artifact_spec),
        command=str(args.generation_command),
        commit=str(args.commit),
        overwrite=bool(args.overwrite),
    )
    payload = {
        "bundle_dir": str(result.bundle_dir),
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
    if args.command == "evidence-bundle":
        return _run_evidence_bundle(args)
    if args.command == "dissertation-bundle":
        return _run_dissertation_bundle(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
