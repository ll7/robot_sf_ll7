"""CLI for the external pedestrian-prior extraction staging/preflight checker (issue #2918).

Checks a metadata-only extraction manifest against the canonical source-type,
prior-parameter, provenance, and authored-vs-dataset-backed contract, then prints
a structured report. It does NOT ingest, download, or read any external
trajectory data, stores no raw trajectories, and makes NO calibrated- or
representative-prior claim.

Example:
    uv run python scripts/tools/check_pedestrian_prior_extraction_manifest.py \
        --manifest configs/research/pedestrian_prior_extraction_manifest_issue_2918_example.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.pedestrian_prior_extraction_manifest import (
    CONTRACT_STATUS_READY,
    PedestrianPriorExtractionManifestError,
    check_pedestrian_prior_extraction_manifest,
    load_pedestrian_prior_extraction_manifest,
)

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "research"
    / "pedestrian_prior_extraction_manifest_issue_2918_example.yaml"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to a pedestrian_prior_extraction_manifest.v1 file (JSON or YAML).",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero unless the contract status is 'ready'.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the manifest check and print a JSON report.

    Returns:
        Process exit code (0 on success, 2 on manifest error, 1 when
        ``--require-ready`` is set and the contract is not ready).
    """
    args = _parse_args(argv)
    try:
        manifest = load_pedestrian_prior_extraction_manifest(args.manifest)
        report = check_pedestrian_prior_extraction_manifest(manifest, source=args.manifest)
    except (PedestrianPriorExtractionManifestError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if args.require_ready and report.contract_status != CONTRACT_STATUS_READY:
        print(
            f"contract status is {report.contract_status!r}, expected 'ready'",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
