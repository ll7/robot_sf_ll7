"""CLI for the AMV actuation-latency / rider-coupling measurement-manifest checker (issue #3283).

Checks a metadata-only measurement intake manifest against the canonical
quantity, synchronization, provenance, and synthetic-vs-measured contract, then
prints a structured report. It does NOT collect, ingest, or read any real
command-response data and makes NO measured-value claim.

Example:
    uv run python scripts/tools/check_amv_actuation_latency_measurement_manifest.py \
        --manifest configs/benchmarks/issue_3283_amv_actuation_latency_measurement_manifest_example.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.actuation_latency_measurement_manifest import (
    CONTRACT_STATUS_READY,
    AmvActuationLatencyManifestError,
    check_amv_actuation_latency_measurement_manifest,
    load_amv_actuation_latency_measurement_manifest,
)

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "benchmarks"
    / "issue_3283_amv_actuation_latency_measurement_manifest_example.yaml"
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
        help="Path to an amv_actuation_latency_measurement_manifest.v1 file (JSON or YAML).",
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
        manifest = load_amv_actuation_latency_measurement_manifest(args.manifest)
        report = check_amv_actuation_latency_measurement_manifest(manifest, source=args.manifest)
    except (AmvActuationLatencyManifestError, ValueError) as exc:
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
