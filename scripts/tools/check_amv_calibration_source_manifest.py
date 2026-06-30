"""CLI checker for the AMV calibration source manifest (issue #1585).

The report is metadata-only. It does not collect external data, ingest traces,
calibrate AMV actuation values, or update benchmark/paper claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.amv_calibration_source_manifest import (
    AmvCalibrationSourceManifestError,
    check_amv_calibration_source_manifest,
    load_amv_calibration_source_manifest,
)

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "benchmarks"
    / "issue_1585_amv_calibration_source_manifest.yaml"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to an amv_calibration_source_manifest.v1 YAML or JSON file.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero unless source_status is ready and blocker-free.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run manifest check and print JSON report."""
    args = _parse_args(argv)
    try:
        manifest = load_amv_calibration_source_manifest(args.manifest)
        report = check_amv_calibration_source_manifest(manifest)
    except AmvCalibrationSourceManifestError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 2

    payload = report.to_dict()
    payload["ok"] = report.is_ready
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_ready and not report.is_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
