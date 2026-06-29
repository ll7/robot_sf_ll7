#!/usr/bin/env python3
"""Check proposal-model rerun readiness for disjoint certified failure archives.

This issue #3275 validation gate reads archive metadata only. It does not run
benchmarks, proposal-model inference, SLURM jobs, or artifact publication.

Exit codes:
- ``0``: archive inputs are ready for a diagnostic rerun.
- ``2``: inputs pass leakage/certification checks, but supplied output is
  diagnostic-only and not benchmark evidence.
- ``3``: readiness is blocked by leakage, missing certification metadata, or
  malformed inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.failure_archive_rerun_readiness import (
    BLOCKED,
    DIAGNOSTIC_ONLY,
    READY,
    classify_failure_archive_rerun_readiness,
)

_EXIT_CODES = {READY: 0, DIAGNOSTIC_ONLY: 2, BLOCKED: 3}


def _build_parser() -> argparse.ArgumentParser:
    """Return CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-archive",
        type=Path,
        required=True,
        help="Failure archive used to fit/select proposal-model candidates.",
    )
    parser.add_argument(
        "--rerun-archive",
        type=Path,
        required=True,
        help="Disjoint certified failure archive intended for the rerun slice.",
    )
    parser.add_argument(
        "--rerun-output",
        type=Path,
        default=None,
        help="Optional rerun report JSON; diagnostic-only markers cap the verdict.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON readiness report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the readiness check and return a fail-closed exit code."""

    args = _build_parser().parse_args(argv)
    readiness = classify_failure_archive_rerun_readiness(
        args.source_archive,
        args.rerun_archive,
        rerun_output=args.rerun_output,
    )
    payload = readiness.to_payload()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return _EXIT_CODES.get(readiness.status, 3)


if __name__ == "__main__":
    sys.exit(main())
