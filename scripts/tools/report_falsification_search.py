#!/usr/bin/env python3
"""Build a deterministic convergence report from persisted falsification search artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.adversarial.falsification_report import (
    build_convergence_report,
    write_convergence_report,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse report command arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        required=True,
        type=Path,
        help="Persisted adversarial-sampler-comparison.v3 JSON index.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for deterministic JSON, Markdown and PNG outputs.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root for resolving manifest and config paths (default: current directory).",
    )
    parser.add_argument(
        "--manifest-path-map",
        type=Path,
        help=(
            "Optional falsification_manifest_path_map.v1 JSON input for resolving unavailable "
            "declared manifest paths to digest-pinned repository-relative archived manifests."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the report without running a search or simulator."""
    args = parse_args(argv)
    report = build_convergence_report(
        args.comparison,
        repo_root=args.repo_root,
        manifest_path_map_path=args.manifest_path_map,
    )
    outputs = write_convergence_report(report, args.output_dir)
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "run_count": len(report["runs"]),
                "source_revision_status": report["provenance"]["source_revision"]["status"],
                "manifest_path_map_sha256": (
                    report["provenance"]["manifest_path_map"]["sha256"]
                    if report["provenance"]["manifest_path_map"] is not None
                    else None
                ),
                "outputs": outputs,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
