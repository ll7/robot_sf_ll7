"""Extract compact pedestrian-prior summaries from a local trajectory fixture.

This issue #2918 helper is CPU/local only. It reads an already-available
fixture or staged local file, writes bounded summaries, and never stores raw
trajectory samples in the output report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.pedestrian_prior_extraction import (
    PedestrianPriorExtractionError,
    extract_pedestrian_prior_report_from_file,
    write_pedestrian_prior_extraction_report,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="Local trajectory YAML/JSON input."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for deterministic compact JSON report.",
    )
    parser.add_argument(
        "--value-status",
        choices=("proxy-placeholder", "dataset-backed"),
        default="proxy-placeholder",
        help=(
            "Status stamped on extracted values. Use dataset-backed only after separate "
            "license-compatible staging and manifest admission."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run extraction CLI and return a process exit code."""

    args = _parse_args(argv)
    try:
        report = extract_pedestrian_prior_report_from_file(
            args.input,
            value_status=args.value_status,
        )
    except PedestrianPriorExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = report.to_dict()
    if args.output:
        write_pedestrian_prior_extraction_report(report, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
