"""Fail-closed provenance manifest checker for benchmark result runs.

Usage:
    uv run python scripts/validation/check_benchmark_result_provenance.py \\
        --manifest output/path/episodes.jsonl.provenance.json

Exit codes:
    0: manifest complete and valid
    2: missing required field, invalid schema_version, missing artifact,
       missing row link, malformed postprocessing entry
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robot_sf.benchmark.result_provenance import (
    ProvenanceValidationError,
    load_result_provenance_manifest,
    validate_result_provenance_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate a benchmark result provenance manifest.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to the provenance manifest JSON file.",
    )
    return parser


def main() -> None:
    """Validate the manifest and exit with code 0 or 2."""
    parser = build_arg_parser()
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest file not found: {manifest_path}", file=sys.stderr)
        sys.exit(2)

    try:
        payload = load_result_provenance_manifest(manifest_path)
        validate_result_provenance_manifest(payload)
        print(f"OK: {manifest_path} is valid", file=sys.stderr)
        sys.exit(0)
    except ProvenanceValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(2)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
